# -*- coding: utf-8 -*-
"""
infer.py — inference for a single new image (+mask) → class + confidence

Usage from notebook/script:
    from infer import load_pipeline, predict_record, predict_file
    pack = load_pipeline("outputs/best_model.pt")
    res  = predict_file(pack, image_path="x.jpg", mask_png="x_mask.png")
    # or with COCO json: predict_file(pack, "x.jpg", coco_json="_annotations.coco.json",
    #                                 image_filename="x.jpg")

CLI:
    python infer.py --ckpt outputs/best_model.pt --image x.jpg --mask_png x_mask.png

Note: confidence is softmax of s·cos(θ) — useful for comparing relative
confidence between classes, but not a true calibrated probability.
"""
import argparse
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import (build_tensor_transform, mask_from_coco_segmentation,
                     measure_length_px, tip_crop_from_mask)
from model import arcface_logits


# class pairs that share the same true length → tip TTA matters most for them
TIP_PAIRS = [
    {"Needle_Holder", "Artery_Forceps"},
    {"Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150"},
]


def length_prior_probs(length_cm: Optional[float], classes: List[str],
                       real_length_cm: dict, sigma_cm: float = 1.2) -> np.ndarray:
    """
    Gaussian prior over classes from the measured physical length.
    p(c) ∝ exp(-(L_meas − L_c)² / (2σ²)) for classes with known real length;
    unknown-length classes get the max prior (uninformative).
    """
    K = len(classes)
    prior = np.ones(K, dtype=np.float64)
    if length_cm is None:
        return prior
    known = {c: real_length_cm.get(c) for c in classes if real_length_cm.get(c) is not None}
    if not known:
        return prior
    dists = {c: (length_cm - L) ** 2 for c, L in known.items()}
    worst = max(dists.values())
    for c, d2 in dists.items():
        prior[classes.index(c)] = np.exp(-d2 / (2 * sigma_cm ** 2))
    # classes with unknown length: neutral (max of known priors)
    neutral = max(prior[classes.index(c)] for c in known)
    for i in range(K):
        if classes[i] not in known:
            prior[i] = neutral
    return prior / prior.sum()


def load_pipeline(ckpt_path: str, device: Optional[torch.device] = None) -> dict:
    """Load checkpoint → pack (model, arcface head, transform, metadata)"""
    from evaluate import load_bundle
    pack = load_bundle(ckpt_path, device)
    pack["tensor_tf"] = build_tensor_transform(pack["cfg"].img_size)
    # Store cfg as dict so predict_array can read use_tta
    if hasattr(pack["cfg"], "to_dict"):
        # Keep original object as well for img_size
        pack["_cfg_obj"] = pack["cfg"]
        pack["cfg"] = pack["cfg"].to_dict()
    return pack


def _predict_once(pack: dict, image_rgb: np.ndarray, ln_norm: float) -> torch.Tensor:
    """Single forward pass → returns logits tensor (num_classes,)"""
    tensor = pack["tensor_tf"](image=image_rgb)["image"][None].to(pack["device"])
    ln = torch.tensor([ln_norm], dtype=torch.float32, device=pack["device"])
    emb = pack["model"](tensor, ln)
    return arcface_logits(pack["arcface"], emb.float())[0]

@torch.no_grad()
def predict_array(pack: dict, image_rgb: np.ndarray,
                  mask_gray: Optional[np.ndarray] = None) -> dict:
    """
    Predict 1 image with TTA (Test-Time Augmentation)
      image_rgb : uint8 (H,W,3) RGB
      mask_gray : binary mask of the instrument (H,W) — if not provided, the
                  training mean length is used instead (model can still predict
                  but is less accurate for class pairs that differ by size)

    TTA (v3):
      1. full view (orig + hflip)
      2. TIP view: crops of both instrument ends along the mask major axis —
         the ONLY signal separating same-length pairs (Needle_Holder↔
         Artery_Forceps, 23↔150): curved vs straight jaws. Tip logits get a
         2.5× weight when top-2 lands in a tip-critical pair.
      3. length prior (cm, when calibration_ratio known) multiplies the
         class probabilities.
    """
    ratio = pack.get("calibration_ratio")
    length = None
    if mask_gray is not None:
        length = measure_length_px(mask_gray)
        if ratio:
            length = length * ratio
    ln_norm = ((length if length is not None else pack["length_mean"])
               - pack["length_mean"]) / pack["length_std"]

    use_tta = pack.get("cfg", {}).get("use_tta", False) if isinstance(pack.get("cfg"), dict) else False
    logits_full = _predict_once(pack, image_rgb, ln_norm)
    if use_tta:
        img_flip = np.ascontiguousarray(image_rgb[:, ::-1, :])
        logits_full = (logits_full + _predict_once(pack, img_flip, ln_norm)) / 2.0

    # ---- tip TTA ----
    logits_tip = None
    if use_tta and mask_gray is not None:
        try:
            tip = tip_crop_from_mask(image_rgb, mask_gray, tip_frac=0.45, both_ends=True)
            if tip is not None and tip.shape[0] >= 16 and tip.shape[1] >= 16:
                t_orig = _predict_once(pack, tip, ln_norm)
                t_flip = _predict_once(pack, np.ascontiguousarray(tip[:, ::-1, :]), ln_norm)
                logits_tip = (t_orig + t_flip) / 2.0
        except Exception:
            logits_tip = None

    logits = logits_full
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    if logits_tip is not None:
        probs_tip = F.softmax(logits_tip, dim=-1).cpu().numpy()
        top2 = np.argsort(probs)[::-1][:2]
        pair = {pack["classes"][int(i)] for i in top2}
        tip_weight = 2.5 if any(pair == p for p in TIP_PAIRS) else 1.0
        blended = probs ** 1.0 * (probs_tip ** tip_weight)
        blended = blended / blended.sum()
        probs = blended
        # geometric mean variant:
        # probs = np.sqrt(probs * (probs_tip ** tip_weight)); probs /= probs.sum()

    # ---- length prior (cm) ----
    if ratio and mask_gray is not None:
        from config import REAL_LENGTH_CM
        length_cm = measure_length_px(mask_gray) * ratio
        prior = length_prior_probs(length_cm, pack["classes"], REAL_LENGTH_CM)
        probs = probs * prior
        probs = probs / probs.sum()

    top3 = np.argsort(probs)[::-1][:3]
    classes: List[str] = pack["classes"]
    return {
        "class": classes[int(top3[0])],
        "confidence": float(probs[top3[0]]),
        "top3": [{"class": classes[int(i)], "prob": float(probs[i])} for i in top3],
        "length_used": float(length if length is not None else pack["length_mean"]),
        "tta_used": use_tta,
        "tip_tta_used": logits_tip is not None,
    }


def predict_record(pack: dict, record: dict) -> dict:
    """Predict from a record returned by load_coco_records() (uses segmentation polygon in record)"""
    bgr = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = mask_from_coco_segmentation(record["segmentation"], record["height"], record["width"])
    res = predict_array(pack, img, mask)
    res["truth"] = record["class_name"]
    return res


def predict_file(pack: dict, image_path: str, mask_png: Optional[str] = None,
                 coco_json: Optional[str] = None,
                 image_filename: Optional[str] = None) -> dict:
    """
    Predict from files:
      - mask_png   : mask file (white/black) if available
      - coco_json  : or point to an annotation file and specify image_filename → uses the first polygon annotation for that image
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    mask = None
    if mask_png:
        m = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise IOError(f"Failed to read mask: {mask_png}")
        mask = ((m > 127).astype(np.uint8)) * 255
    elif coco_json:
        fname = image_filename or os.path.basename(image_path)
        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)
        images = {im["id"]: im for im in coco["images"]}
        target = next((im for im in coco["images"] if im["file_name"] == fname), None)
        if target is None:
            raise KeyError(f"{fname} not found in {coco_json}")
        ann = next((a for a in coco["annotations"] if a["image_id"] == target["id"]), None)
        if ann is None:
            raise KeyError(f"{fname} has no annotation")
        mask = mask_from_coco_segmentation(ann["segmentation"],
                                           int(images[target["id"]]["height"]),
                                           int(images[target["id"]]["width"]))
    return predict_array(pack, img, mask)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Inference DINOv2+ArcFace")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask_png", default=None)
    ap.add_argument("--coco_json", default=None)
    args = ap.parse_args(argv)

    pack = load_pipeline(args.ckpt)
    res = predict_file(pack, args.image, mask_png=args.mask_png, coco_json=args.coco_json)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
