# -*- coding: utf-8 -*-
"""
infer.py — inference ภาพใหม่ 1 ภาพ (+mask) → class + confidence

ใช้จาก notebook/สคริปต์:
    from infer import load_pipeline, predict_record, predict_file
    pack = load_pipeline("outputs/best_model.pt")
    res  = predict_file(pack, image_path="x.jpg", mask_png="x_mask.png")
    # หรือมี COCO json: predict_file(pack, "x.jpg", coco_json="_annotations.coco.json",
    #                                 image_filename="x.jpg")

CLI:
    python infer.py --ckpt outputs/best_model.pt --image x.jpg --mask_png x_mask.png

หมายเหตุ: confidence คือ softmax ของ s·cos(θ) — ใช้เปรียบเทียบ "ความมั่นใจสัมพัทธ์"
ระหว่าง class ได้ แต่ไม่ใช่ probability ที่ calibrated จริง
"""
import argparse
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_tensor_transform, mask_from_coco_segmentation, measure_length_px
from model import arcface_logits


def load_pipeline(ckpt_path: str, device: Optional[torch.device] = None) -> dict:
    """โหลด checkpoint → pack (model, arcface head, transform, metadata)"""
    from evaluate import load_bundle
    pack = load_bundle(ckpt_path, device)
    pack["tensor_tf"] = build_tensor_transform(pack["cfg"].img_size)
    # เก็บ cfg เป็น dict เพื่อให้ predict_array อ่าน use_tta ได้
    if hasattr(pack["cfg"], "to_dict"):
        pack["cfg"] = pack["cfg"].to_dict()
    return pack


def _predict_once(pack: dict, image_rgb: np.ndarray, ln_norm: float) -> torch.Tensor:
    """ทำนายครั้งเดียว → คืน logits tensor (num_classes,)"""
    tensor = pack["tensor_tf"](image=image_rgb)["image"][None].to(pack["device"])
    ln = torch.tensor([ln_norm], dtype=torch.float32, device=pack["device"])
    emb = pack["model"](tensor, ln)
    return arcface_logits(pack["arcface"], emb.float())[0]


@torch.no_grad()
def predict_array(pack: dict, image_rgb: np.ndarray,
                  mask_gray: Optional[np.ndarray] = None) -> dict:
    """
    ทำนาย 1 ภาพ พร้อม TTA (Test-Time Augmentation)
      image_rgb : uint8 (H,W,3) RGB
      mask_gray : binary mask ของเครื่องมือ (H,W) — ถ้าไม่มี จะใช้ค่าความยาวเฉลี่ย
                  ของ train แทน (โมเดลยังทำนายได้ แต่แม่นยำน้อยลงกับคู่ class ต่างขนาด)

    TTA: original + horizontal flip → average logits → เสถียรกว่าทำนายครั้งเดียว
    """
    ratio = pack.get("calibration_ratio")
    if mask_gray is not None:
        length = measure_length_px(mask_gray) * (ratio if ratio else 1.0)
    else:
        length = pack["length_mean"]  # neutral fallback

    ln = (length - pack["length_mean"]) / pack["length_std"]

    # TTA: original + horizontal flip → average logits
    use_tta = pack.get("cfg", {}).get("use_tta", False) if isinstance(pack.get("cfg"), dict) else False
    if use_tta:
        logits_orig = _predict_once(pack, image_rgb, ln)
        # horizontal flip ของ mask ด้วย (ถ้ามี) ไม่ต้อง flip ความยาว — ความยาวไม่เปลี่ยน
        img_flip = np.ascontiguousarray(image_rgb[:, ::-1, :])
        logits_flip = _predict_once(pack, img_flip, ln)
        logits = (logits_orig + logits_flip) / 2.0
    else:
        logits = _predict_once(pack, image_rgb, ln)

    probs = F.softmax(logits, dim=-1).cpu()

    top3 = probs.topk(3)
    classes: List[str] = pack["classes"]
    return {
        "class": classes[int(top3.indices[0])],
        "confidence": float(top3.values[0]),
        "top3": [{"class": classes[int(i)], "prob": float(p)}
                 for p, i in zip(top3.values, top3.indices)],
        "length_used": float(length),
        "tta_used": use_tta,
    }


def predict_record(pack: dict, record: dict) -> dict:
    """ทำนายจาก record ที่ได้จาก load_coco_records() (ใช้ segmentation polygon ใน record)"""
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
    ทำนายจากไฟล์:
      - mask_png   : ไฟล์ mask (ขาว/ดำ) ถ้ามี
      - coco_json  : หรือชี้ไฟล์ annotation แล้วระบุ image_filename → ใช้ polygon ann แรกของรูปนั้น
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"อ่านภาพไม่สำเร็จ: {image_path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    mask = None
    if mask_png:
        m = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise IOError(f"อ่าน mask ไม่สำเร็จ: {mask_png}")
        mask = ((m > 127).astype(np.uint8)) * 255
    elif coco_json:
        fname = image_filename or os.path.basename(image_path)
        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)
        images = {im["id"]: im for im in coco["images"]}
        target = next((im for im in coco["images"] if im["file_name"] == fname), None)
        if target is None:
            raise KeyError(f"ไม่พบ {fname} ใน {coco_json}")
        ann = next((a for a in coco["annotations"] if a["image_id"] == target["id"]), None)
        if ann is None:
            raise KeyError(f"{fname} ไม่มี annotation")
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
