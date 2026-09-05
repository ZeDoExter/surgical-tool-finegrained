# -*- coding: utf-8 -*-
"""
dataset.py — Load images + segmentation masks from COCO format (Roboflow export)

Main responsibilities:
1) parse ``_annotations.coco.json`` → records (path, polygon, label)
2) rasterize polygon → binary mask
3) measure instrument length from mask (minAreaRect) as a single auxiliary feature
4) task-safe augmentation:
   - focus on photometric ops (brightness/contrast/gamma/CLAHE) to simulate
     specular reflections on metal and varying illumination on the green cloth
   - no crop/zoom that would destroy aspect ratio or absolute scale
     (size is a key discriminative feature!)
   - no cutout / random erasing over the instrument
   - horizontal flip can be toggled per class (some classes have handedness
     and must not be flipped)

Notes:
- Background is green surgical cloth; shadows cast on the cloth move with
  instrument placement / light direction and make bounding/segmentation harder
  (harder than a high-contrast silver tray). Photometric + shadow simulation
  targets this difficulty.
- Experimental defaults found useful elsewhere in the project: image size 560
  and bbox margin ~0.15. Defaults in this module are kept for compatibility;
  see Dataset docstring.
"""
import json
import math
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)


# ============================================================ Length measurement from mask
def measure_length_px(mask: np.ndarray) -> float:
    """
    Return the maximum length of the instrument in pixels (from a single binary mask).

    Uses ``cv2.minAreaRect`` because instruments are often placed diagonally
    rather than axis-aligned — the minimum-area rotated rectangle enclosing the
    contour gives a long side that approximates the true instrument length.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # empty mask (annotation error) → return 0 to avoid crash
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)  # ((cx,cy), (w,h), angle)
    (w, h) = rect[1]
    return float(max(w, h))


def get_length_cm(mask: np.ndarray, calibration_ratio: float) -> float:
    """Convert pixel length → cm using calibration_ratio (cm/pixel) from a reference object."""
    return measure_length_px(mask) * calibration_ratio


def mask_from_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """
    Convert COCO segmentation → binary mask (uint8, values 0/255).

    Supports polygon (standard Roboflow format) and RLE (requires pycocotools).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):  # RLE format
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "Found RLE segmentation but pycocotools is not installed (pip install pycocotools)"
            ) from exc
        rle = segmentation
        if isinstance(rle.get("counts"), list):  # uncompressed RLE → convert to compressed first
            rle = mask_utils.frPyObjects(rle, height, width)
        return (mask_utils.decode(rle) * 255).astype(np.uint8)
    for poly in segmentation:  # list of polygons [[x1,y1,x2,y2,...], ...]
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


# ============================================================ COCO parsing
def _is_head_class(class_name: str) -> bool:
    """True for zoomed-tip classes added in newer Roboflow exports (e.g.
    'Artery_Forceps_Head'). Whole-instrument flow uses only the 14 full
    classes — head-only photos are excluded by default."""
    return class_name.endswith("_Head") or class_name.endswith("-Head")


def load_coco_records(data_dir: str, split: str,
                      include_head_classes: bool = False) -> Tuple[List[dict], List[str]]:
    """
    Read a split folder ("train"/"valid"/"test") containing ``_annotations.coco.json``.

    Returns ``(records, class_names)`` where each record is a dict with
    ``image_path / segmentation / width / height / class_name / label``.

    - label = index from **sorted class names** (stable regardless of
      category_id ordering in the json).
    - ``*_Head`` categories (zoomed-tip photos) are DROPPED unless
      ``include_head_classes=True`` — keeps the 14-class flow unchanged.
    - 1 annotation = 1 sample → if one image contains multiple instruments
      it yields multiple samples (recommend using bbox_margin > 0 in
      Dataset to crop per instance).
    """
    split_dir = os.path.join(data_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    return _load_coco_split(split_dir, ann_path, include_head_classes)


def load_coco_records_multi(data_dirs: List[str], split: str,
                            include_head_classes: bool = False) -> Tuple[List[dict], List[str]]:
    """
    Load one split from MULTIPLE dataset folders and concatenate the records
    (e.g. dataset/ + dataset_extra/). Class names must be the SAME sorted list
    in every folder — enforced, since the label space must stay consistent.
    """
    all_records: List[dict] = []
    ref_classes: List[str] = None
    for d in data_dirs:
        recs, classes = load_coco_records(d, split, include_head_classes)
        if ref_classes is None:
            ref_classes = classes
        elif classes != ref_classes:
            raise ValueError(
                f"Class lists differ across dataset folders.\n"
                f"  {data_dirs[0]}: {ref_classes}\n  {d}: {classes}\n"
                f"Every folder must contain the same 14 class names (in train at least)."
            )
        all_records.extend(recs)
    return all_records, ref_classes


def _load_coco_split(split_dir: str, ann_path: str,
                     include_head_classes: bool) -> Tuple[List[dict], List[str]]:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    # Filter to categories that actually appear (skip dummy super-category like id 0)
    used_names = {cat_id_to_name[ann["category_id"]] for ann in coco["annotations"]}
    if not include_head_classes:
        used_names = {n for n in used_names if not _is_head_class(n)}
    class_names = sorted(used_names)
    name_to_label = {n: i for i, n in enumerate(class_names)}

    records: List[dict] = []
    for ann in coco["annotations"]:
        cname = cat_id_to_name[ann["category_id"]]
        if not include_head_classes and _is_head_class(cname):
            continue
        im = images[ann["image_id"]]
        records.append({
            "image_path": os.path.join(split_dir, im["file_name"]),
            "segmentation": ann["segmentation"],
            "width": int(im["width"]),
            "height": int(im["height"]),
            "class_name": cname,
            "label": name_to_label[cname],
            "coco_json": ann_path,
        })
    # v3 reality fix: same tool annotated as multiple touching polygons
    # (e.g. Cotton_Piler jaws) -> merge into ONE record, else the detector
    # learns "each tooth = one instrument" and answers 1 tool as N boxes
    records = merge_split_annotations(records)
    return records, class_names


def segmentation_bbox(segmentation, width: int, height: int) -> Tuple[int, int, int, int]:
    """Bounding box (x1,y1,x2,y2) enclosing all polygons — used when cropping per instance (bbox_margin > 0)."""
    xs: List[float] = []
    ys: List[float] = []
    for poly in segmentation if isinstance(segmentation, list) else []:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        xs += [float(pts[:, 0].min()), float(pts[:, 0].max())]
        ys += [float(pts[:, 1].min()), float(pts[:, 1].max())]
    if not xs:  # RLE or empty polygon → use full image
        return 0, 0, width, height
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


_COCO_CACHE: dict = {}


def load_coco_annotations_for_image(record: dict) -> List[dict]:
    """
    All annotations (instruments) of the image that `record` belongs to — used by
    the detector dataset when a real photo contains multiple instruments
    (the "mix" dataset). Each returned dict has segmentation / class_name /
    label (label = sorted class index, same numbering as load_coco_records).
    Falls back to the record itself when the original COCO json is unavailable.
    The parsed json is cached by path+mtime (a 2,600-image patch-paste json is
    several MB — re-parsing per sample is slow).
    """
    ann_path = record.get("coco_json")
    if not ann_path or not os.path.exists(ann_path):
        return [record]
    mtime = os.path.getmtime(ann_path)
    entry = _COCO_CACHE.get(ann_path)
    if entry is None or entry[0] != mtime:
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        used_names = sorted({cat_id_to_name[a["category_id"]] for a in coco["annotations"]
                             if a["category_id"] in cat_id_to_name
                             and not _is_head_class(cat_id_to_name[a["category_id"]])})
        name_to_label = {n: i for i, n in enumerate(used_names)}
        img_id_by_file = {im["file_name"]: im["id"] for im in coco["images"]}
        anns_by_img: dict = {}
        for a in coco["annotations"]:
            cname = cat_id_to_name.get(a["category_id"])
            if cname is None or _is_head_class(cname) or cname not in name_to_label:
                continue
            anns_by_img.setdefault(a["image_id"], []).append({
                "segmentation": a["segmentation"],
                "class_name": cname,
                "label": name_to_label[cname],
            })
        entry = (mtime, anns_by_img, img_id_by_file)
        _COCO_CACHE[ann_path] = entry
    _, anns_by_img, img_id_by_file = entry
    iid = img_id_by_file.get(os.path.basename(record["image_path"]))
    out = anns_by_img.get(iid, [])
    return out if out else [record]


# ============================================================ Mask-Aware Patch-Paste (Copy-Paste) Augmentation
def extract_instrument_patch(record: dict, pad: int = 3) -> Optional[dict]:
    """
    Extract instrument foreground RGB + binary mask from a COCO record.
    Returns a dictionary with the cropped RGB, mask, label, and class_name.
    """
    img = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mask = mask_from_coco_segmentation(record["segmentation"], record["height"], record["width"])
    x1, y1, x2, y2 = segmentation_bbox(record["segmentation"], record["width"], record["height"])

    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    patch_rgb = img[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy()

    return {
        "rgb": patch_rgb,
        "mask": patch_mask,
        "class_name": record["class_name"],
        "label": record["label"],
        "width": w,
        "height": h,
    }


def transform_instrument_patch(
    patch: dict,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    allow_flip: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random scaling, full 360-degree rotation with expanded bounding box,
    and optional flipping to an instrument patch.
    """
    rng = rng or random
    p_rgb = patch["rgb"]
    p_mask = patch["mask"]
    h, w = p_rgb.shape[:2]

    scale = rng.uniform(scale_range[0], scale_range[1])
    angle = rng.uniform(-180.0, 180.0)

    nh = max(int(h * scale), 4)
    nw = max(int(w * scale), 4)
    scaled_rgb = cv2.resize(p_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(p_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Rotate with adjusted bounding box
    cx, cy = nw / 2.0, nh / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = max(int((nh * sin) + (nw * cos)), 4)
    new_h = max(int((nh * cos) + (nw * sin)), 4)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rot_rgb = cv2.warpAffine(
        scaled_rgb, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    rot_mask = cv2.warpAffine(
        scaled_mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    if allow_flip and rng.random() < 0.5:
        rot_rgb = np.ascontiguousarray(rot_rgb[:, ::-1, :])
        rot_mask = np.ascontiguousarray(rot_mask[:, ::-1])

    return rot_rgb, rot_mask


def patch_paste_augment(
    target_img: np.ndarray,
    target_mask: Optional[np.ndarray],
    patch_pool: List[dict],
    target_label: Optional[int] = None,
    flip_flags: Optional[List[bool]] = None,
    max_pastes: int = 2,
    max_overlap: float = 0.20,
    blend_feather: int = 3,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Mask-aware Patch Paste (Copy-Paste):
    Pastes 1 to max_pastes secondary instrument patches onto target_img (green cloth).

    Key features:
    - Extracts & rotates exact instrument foregrounds (via COCO polygon segmentation).
    - Preserves primary target tool visibility by bounding overlap with target_mask.
    - Feathered Gaussian edge blending for natural lighting integration on green cloth.
    - Simulates multi-instrument surgical scenes to prevent background overfitting.
    """
    if not patch_pool:
        return target_img
    rng = rng or random
    out_img = target_img.copy()
    h_dst, w_dst = out_img.shape[:2]

    num_pastes = rng.randint(1, max_pastes)
    candidate_pool = [p for p in patch_pool if target_label is None or p.get("label") != target_label]
    if not candidate_pool:
        candidate_pool = patch_pool

    for _ in range(num_pastes):
        patch = rng.choice(candidate_pool)
        label = patch.get("label", 0)
        allow_flip = flip_flags[label] if flip_flags is not None and label < len(flip_flags) else True

        rot_rgb, rot_mask = transform_instrument_patch(
            patch, scale_range=(0.85, 1.15), allow_flip=allow_flip, rng=rng
        )
        ph, pw = rot_rgb.shape[:2]

        if ph >= h_dst or pw >= w_dst:
            scale_down = min((h_dst - 10) / ph, (w_dst - 10) / pw) * rng.uniform(0.6, 0.9)
            if scale_down <= 0:
                continue
            new_ph = max(int(ph * scale_down), 4)
            new_pw = max(int(pw * scale_down), 4)
            rot_rgb = cv2.resize(rot_rgb, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)
            rot_mask = cv2.resize(rot_mask, (new_pw, new_ph), interpolation=cv2.INTER_NEAREST)
            ph, pw = new_ph, new_pw

        if ph >= h_dst or pw >= w_dst or ph < 4 or pw < 4:
            continue

        best_x, best_y = 0, 0
        placed = False
        target_area = np.sum(target_mask > 0) if target_mask is not None else 0

        for _attempt in range(15):
            x = rng.randint(0, w_dst - pw)
            y = rng.randint(0, h_dst - ph)

            if target_mask is not None and target_area > 0:
                target_roi_mask = target_mask[y:y + ph, x:x + pw]
                overlap = np.sum((rot_mask > 0) & (target_roi_mask > 0))
                overlap_ratio = overlap / float(target_area)
                if overlap_ratio <= max_overlap:
                    best_x, best_y = x, y
                    placed = True
                    break
            else:
                best_x, best_y = x, y
                placed = True
                break

        if not placed:
            best_x = rng.randint(0, w_dst - pw)
            best_y = rng.randint(0, h_dst - ph)

        alpha = (rot_mask > 0).astype(np.float32)
        if blend_feather > 0:
            k = blend_feather if blend_feather % 2 == 1 else blend_feather + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = alpha[..., None]

        brightness_factor = rng.uniform(0.85, 1.15)
        pasted_rgb = np.clip(rot_rgb.astype(np.float32) * brightness_factor, 0, 255)

        roi = out_img[best_y:best_y + ph, best_x:best_x + pw].astype(np.float32)
        blended = (1.0 - alpha) * roi + alpha * pasted_rgb
        out_img[best_y:best_y + ph, best_x:best_x + pw] = np.clip(blended, 0, 255).astype(np.uint8)

    return out_img


def cutmix_augment(img: np.ndarray, pool: list,
                   rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Backward-compatible wrapper for patch-paste / CutMix augmentation.
    """
    if not pool:
        return img
    if isinstance(pool[0], dict) and "rgb" in pool[0]:
        return patch_paste_augment(img, None, pool, rng=rng)
    # Fallback to simple random crop paste if pool contains raw images
    rng = rng or random
    h, w = img.shape[:2]
    src = rng.choice(pool)
    src_h, src_w = src.shape[:2]
    pw, ph = min(w // 2, src_w), min(h // 2, src_h)
    if pw < 2 or ph < 2:
        return img
    x1, y1 = rng.randint(0, w - pw), rng.randint(0, h - ph)
    sx1, sy1 = rng.randint(0, src_w - pw), rng.randint(0, src_h - ph)
    res = img.copy()
    res[y1:y1 + ph, x1:x1 + pw] = src[sy1:sy1 + ph, sx1:sx1 + pw]
    return res


# ============================================================ Augmentation
def tip_crop_from_mask(img: np.ndarray, mask: np.ndarray, tip_frac: float = 0.42,
                      both_ends: bool = False) -> np.ndarray:
    """
    Crop a square around an instrument TIP along the mask's major axis.

    Why: Needle_Holder vs Artery_Forceps (and Forceps 23 vs 150) share the same
    length — the ONLY reliable signal is the tip shape (curved vs straight
    jaws). A full 560×560 resize shrinks the tip to ~70px; this crop keeps
    ~230px of tip detail.

    mask : binary (H,W) uint8/bool — instrument mask (from COCO polygon or
           detector output on the Pi)
    Returns an RGB crop (tip view). Falls back to the full image if the mask
    is degenerate.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    length = max(rw, rh)
    if length < 8:
        return img
    # major-axis direction (minAreaRect convention: angle is of the `w` side)
    if rw >= rh:
        dirv = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    else:
        dirv = np.array([-math.sin(math.radians(angle)), math.cos(math.radians(angle))])
    dirv = dirv / (np.linalg.norm(dirv) + 1e-8)
    half = length * 0.5
    p1 = np.array([cx, cy]) - dirv * half
    p2 = np.array([cx, cy]) + dirv * half
    end = p1 if not both_ends else None
    # pick the end (tip) — mask coverage check: the tip end has less mask
    # coverage in its neighborhood (tips are thin) — simply take both ends
    # when both_ends, else choose the end whose local mask area is SMALLER
    # (tips are thin → smaller local area)
    def local_area(pt: np.ndarray) -> float:
        r = max(int(length * 0.12), 8)
        x0, x1 = max(int(pt[0]) - r, 0), min(int(pt[0]) + r, mask.shape[1])
        y0, y1 = max(int(pt[1]) - r, 0), min(int(pt[1]) + r, mask.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(np.sum(mask[y0:y1, x0:x1] > 0))

    ends = [p1, p2]
    if not both_ends:
        end = ends[0] if local_area(ends[0]) <= local_area(ends[1]) else ends[1]
        ends = [end]

    h, w = img.shape[:2]
    crops = []
    side = max(int(length * tip_frac), 24)
    for pt in ends:
        x0 = int(max(pt[0] - side / 2, 0)); x1 = int(min(pt[0] + side / 2, w))
        y0 = int(max(pt[1] - side / 2, 0)); y1 = int(min(pt[1] + side / 2, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = img[y0:y1, x0:x1]
        crops.append(crop)
    if not crops:
        return img
    out = crops[0]
    if len(crops) > 1:  # both ends → stack vertically (fixed shape for the CNN)
        h2 = max(c.shape[0] for c in crops)
        out = np.vstack([cv2.copyMakeBorder(c, 0, h2 - c.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
                         for c in crops])
    return out


def maybe_tip_zoom(img: np.ndarray, mask: Optional[np.ndarray], prob: float,
                   tip_frac: float = 0.42, rng: Optional[random.Random] = None) -> np.ndarray:
    """With `prob`, replace the crop with a zoomed tip view (training-time only)."""
    if mask is None or prob <= 0:
        return img
    r = rng or random
    if r.random() >= prob:
        return img
    try:
        tip = tip_crop_from_mask(img, mask, tip_frac=tip_frac)
        if tip is None or tip.shape[0] < 16 or tip.shape[1] < 16:
            return img
        return tip
    except Exception:
        return img


def rotate_image_mask(img: np.ndarray, mask: Optional[np.ndarray],
                      angle_deg: float, border_value: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rotate image (and its mask, if given) together around the mask centroid
    with expanded bounding box — nothing is cut off.

    Why: the classifier training only saw axis-aligned crops + flips, so the
    live camera mixing up tools at rotated angles was expected. Rotating
    image+mask together keeps the instrument and its length feature intact.
    Length is recomputed AFTER rotation by the caller (minAreaRect is
    rotation-invariant anyway).
    """
    h, w = img.shape[:2]
    if mask is not None and mask.shape[0] == h and mask.shape[1] == w:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            M0 = cv2.moments(max(cnts, key=cv2.contourArea))
            if M0["m00"] > 0:
                center = (M0["m10"] / M0["m00"], M0["m01"] / M0["m00"])
            else:
                center = (w / 2, h / 2)
        else:
            center = (w / 2, h / 2)
    else:
        center = (w / 2, h / 2)

    # expanded canvas so nothing is clipped
    cos_a, sin_a = abs(math.cos(math.radians(angle_deg))), abs(math.sin(math.radians(angle_deg)))
    new_w = int(w * cos_a + h * sin_a) + 1
    new_h = int(w * sin_a + h * cos_a) + 1
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    # pad source to new canvas first so warp doesn't sample outside
    pw, ph = max(new_w - w, 0), max(new_h - h, 0)
    img_p = cv2.copyMakeBorder(img, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if mask is not None:
        mask_p = cv2.copyMakeBorder(mask, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                                    cv2.BORDER_CONSTANT, value=0)
    else:
        mask_p = None
    # recompute center on padded coords
    center_p = (center[0] + pw // 2, center[1] + ph // 2)
    M = cv2.getRotationMatrix2D(center_p, angle_deg, 1.0)
    rot_h, rot_w = img_p.shape[:2]
    img_r = cv2.warpAffine(img_p, M, (rot_w, rot_h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_r = None
    if mask_p is not None:
        mask_r = cv2.warpAffine(mask_p, M, (rot_w, rot_h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, mask_r


def merge_split_annotations(records: List[dict], min_contact_px: int = 8) -> List[dict]:
    """
    v3 reality: one physical tool is sometimes annotated as MULTIPLE polygons
    (e.g. Cotton_Piler jaws labeled as 2 separate "teeth" polygons). The
    detector then learns "each tooth = one instrument" and the live feed
    answers 1 tool as 3 boxes.

    Fix at the SOURCE: merge records of the same class on the same image
    whose masks touch/overlap (dilated by min_contact_px) into ONE record
    with concatenated polygons — a merged mask gives the correct length via
    minAreaRect and one bbox for cropping.
    """
    from collections import defaultdict
    by_img = defaultdict(list)
    for r in records:
        by_img[r["image_path"]].append(r)
    out: List[dict] = []
    for path, rs in by_img.items():
        used = [False] * len(rs)
        for i, r in enumerate(rs):
            if used[i]:
                continue
            group = [r]
            used[i] = True
            m_i = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
            for j in range(i + 1, len(rs)):
                if used[j] or rs[j]["class_name"] != r["class_name"]:
                    continue
                m_j = mask_from_coco_segmentation(rs[j]["segmentation"], rs[j]["height"], rs[j]["width"])
                k = np.ones((min_contact_px * 2 + 1, min_contact_px * 2 + 1), np.uint8)
                near = cv2.dilate(m_i, k) > 0
                if np.any(near & (m_j > 0)):
                    group.append(rs[j])
                    used[j] = True
                    m_i = np.maximum(m_i, m_j)
            if len(group) == 1:
                out.append(r)
            else:
                merged_seg = []
                for g in group:
                    if isinstance(g["segmentation"], list):
                        merged_seg.extend(g["segmentation"])
                base = dict(group[0])
                base["segmentation"] = merged_seg
                out.append(base)
    return out


def build_photometric_aug() -> A.Compose:
    """
    Photometric augmentation for training — intentionally stronger than usual
    because metal reflections vary per capture, and the green cloth background
    with shifting shadows makes illumination inconsistent.

    Critically, there is *no* crop/scale/cutout because "size and shape"
    are what the model must learn to separate visually similar classes.

    v3 additions (targeting live-camera failure modes):
      - GaussNoise: webcam sensor noise at low light
      - ImageCompression: MJPEG stream artifacts (we serve JPEG quality 80)
      - Downscale: camera feed softness / slight defocus at distance
      - MotionBlur: slight camera/reflection movement between frames
    All photometric-only — scale/length features stay intact.
    """
    return A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),             # boost local contrast (metal on green cloth has low contrast; shadows worsen it)
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),  # stronger-than-usual jitter
        A.RandomGamma(gamma_limit=(70, 150), p=0.7),                        # simulate different exposure / lighting
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),                           # slight defocus blur
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),                        # sensor noise (v3)
        A.ImageCompression(quality_range=(60, 95), p=0.25),                # JPEG/MJPEG artifacts (v3)
        A.Downscale(scale_range=(0.6, 0.95), p=0.15),                       # soft focus / low-res feed (v3)
        A.MotionBlur(blur_limit=(3, 7), p=0.1),                             # slight motion between frames (v3)
    ])


def build_tensor_transform(img_size: int) -> A.Compose:
    """Fixed-size resize + ImageNet normalization + conversion to tensor (used for train/eval/infer)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def simulate_shadow(img: np.ndarray, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Simulate "shadow" on the green cloth background — shadows shift with
    instrument placement / light direction.

    Draws 1-2 soft-edged dark blobs (ellipse + Gaussian falloff) multiplied
    onto the image. Implemented in numpy instead of A.RandomShadow because
    albumentations' signature changes frequently between 1.x ↔ 2.x — avoids
    version coupling.

    Green cloth shadows are the main difficulty for bounding/segmentation
    (not reflections on a silver tray).
    """
    rng = rng or random
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax = rng.uniform(w * 0.2, w * 0.7)
        ay = rng.uniform(h * 0.2, h * 0.7)
        strength = rng.uniform(0.35, 0.65)          # darkest shadow ~35-65%
        d2 = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
        mask *= 1.0 - strength * np.exp(-d2)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================ Length statistics + split
def record_lengths(records: List[dict], calibration_ratio: Optional[float]) -> List[float]:
    """Measure length of every record (px or cm if ratio given) — rasterize directly from polygon."""
    out = []
    for r in records:
        mask = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
        L = measure_length_px(mask)
        out.append(L if calibration_ratio is None else L * calibration_ratio)
    return out


def compute_length_stats(records: List[dict], calibration_ratio: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute mean/std of lengths — **must be computed from train only**
    and the same values reused to normalize val/test/inference (avoid data leakage).
    """
    L = np.asarray(record_lengths(records, calibration_ratio), dtype=np.float64)
    mean = float(L.mean())
    std = max(float(L.std()), 1e-6)
    return mean, std


def stratified_split(records: List[dict], val_fraction: float = 0.2, seed: int = 42):
    """Stratified train/val split (preserve class proportions) — needed when data is scarce (~30 images/class)."""
    if val_fraction <= 0 or len(records) < 10:
        return records, []
    from sklearn.model_selection import train_test_split
    y = [r["label"] for r in records]
    tr, va = train_test_split(records, test_size=val_fraction, random_state=seed, stratify=y)

def _seg_key(seg) -> bytes:
    """Stable content key for a COCO segmentation so a record can recognize
    its own polygons among the image's other annotations."""
    if not isinstance(seg, list):
        return b""
    return b"|".join(np.asarray(p, dtype=np.float64).tobytes() for p in seg)

# ============================================================ PyTorch Dataset
class SurgicalInstrumentDataset(Dataset):
    """
    Dataset for surgical instrument classification — returns a dict:
      ``image``  : FloatTensor (3, H, W) normalized
      ``length`` : scalar float = (length − mean) / std  ← auxiliary feature
      ``label``  : int64 class index

    Important notes:
    - Length is measured from the *original* mask (before augmentation) because
      photometric ops / flip should not change the true physical length.
    - ``flip_flags[label]`` must be True for that class to receive horizontal flip.
    - ``bbox_margin`` > 0 when one image contains multiple instruments → crop
      around the bbox of that instance (preserves aspect/scale, not a free zoom).
      Experiments found bbox_margin ≈ 0.15 effective; image size 560 was the
      best-performing default in experiments (this class defaults to 224 for
      backward compatibility — pass 504 explicitly to reproduce those results).
    - Background is green cloth; shadows on the cloth make tight bounding harder,
      which is why photometric + shadow augmentation is used.
    """

    def __init__(self, records: List[dict], length_stats: Tuple[float, float],
                 img_size: int = 224, calibration_ratio: Optional[float] = None,
                 flip_flags: Optional[List[bool]] = None, training: bool = True,
                 bbox_margin: float = 0.0, cutmix_prob: float = 0.0,
                 patch_paste_prob: float = 0.0, patch_paste_max_objects: int = 2,
                 hard_classes: Optional[List[str]] = None, hard_tip_zoom_prob: float = 0.7,
                 erase_neighbors: bool = False):
        self.records = records
        self.length_mean, self.length_std = length_stats
        self.training = training
        self.bbox_margin = bbox_margin
        self.img_size = img_size
        self.erase_neighbors = erase_neighbors  # default OFF: measured R1/R2/R3
                                                # (0.9215/0.8848/0.9319) — erasing
                                                # train crops hurts transfer
        # Support both patch_paste_prob and legacy cutmix_prob
        self.patch_paste_prob = patch_paste_prob if patch_paste_prob > 0 else cutmix_prob if training else 0.0
        self.patch_paste_max_objects = patch_paste_max_objects
        self.patch_paste_max_overlap = patch_paste_max_overlap
        self.cutmix_prob = self.patch_paste_prob
        self.tip_zoom_prob = tip_zoom_prob if training else 0.0
        self.tip_zoom_size = tip_zoom_size
        self.coco_json = coco_json
        self.rotate_prob = rotate_prob if training else 0.0
        self.rotate_range = rotate_range
        from config import HARD_CLASSES
        self.hard_classes = set(hard_classes if hard_classes is not None else HARD_CLASSES)
        self.hard_tip_zoom_prob = hard_tip_zoom_prob
        # oversample hard classes x3: repeat their records in the index pool
        if training:
            base = list(range(len(self.records)))
            extra = [i for i, r in enumerate(self.records)
                     if r["class_name"] in self.hard_classes] * 2
            self._index_pool = base + extra
        else:
            self._index_pool = list(range(len(self.records)))
        self.tensor_tf = build_tensor_transform(img_size)
        self.aug = build_photometric_aug() if training else None
        self.flip_flags = flip_flags
        # Measure length once at dataset creation (rasterizing polygons in memory is very fast)
        self._lengths = record_lengths(records, calibration_ratio)
        # Pre-load a pool of instrument patches for Mask-Aware Patch-Paste (Copy-Paste)
        self._patch_pool: List[dict] = []
        if self.patch_paste_prob > 0 and len(records) > 1:
            pool_size = min(64, len(records))
            pool_indices = random.sample(range(len(records)), pool_size)
            for pi in pool_indices:
                p = extract_instrument_patch(records[pi])
                if p is not None:
                    self._patch_pool.append(p)

    def __len__(self) -> int:
        return len(self._index_pool)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _erase_neighbors(self, img: np.ndarray, r: dict) -> np.ndarray:
        """
        Paint neighboring instruments (other annotations in the same image)
        with green cloth so this instrument's crop never contains a tip of a
        different tool that would poison the class label (mix dataset).
        """
        if not self.erase_neighbors:
            return img
        H, W = img.shape[:2]
        self_key = _seg_key(r["segmentation"])
        canvas = np.zeros((H, W), np.uint8)
        n_others = 0
        for a in load_coco_annotations_for_image(r):
            if a["class_name"] == r["class_name"] and \
                    _seg_key(a["segmentation"]) == self_key:
                continue  # this record itself (or its merged duplicate)
            polys = a["segmentation"] if isinstance(a["segmentation"], list) else []
            for poly in polys:
                pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                cv2.fillPoly(canvas, [pts.astype(np.int32)], 1)
                n_others += 1
        if n_others:
            img = img.copy()
            img[canvas > 0] = (60, 120, 70)  # cloth green (RGB)
        return img

    def _maybe_crop(self, img: np.ndarray, r: dict) -> np.ndarray:
        """Validation-time crop around this instance's bbox (no neighbor erase
        here — erase already ran on the full image before cropping)."""
        m = self.bbox_margin
        x1, y1, x2, y2 = segmentation_bbox(r["segmentation"], r["width"], r["height"])
        dx, dy = int((x2 - x1) * m), int((y2 - y1) * m)
        x1, y1 = max(x1 - dx, 0), max(y1 - dy, 0)
        x2, y2 = min(x2 + dx, r["width"]), min(y2 + dy, r["height"])
        return img[y1:y2, x1:x2]

    def __getitem__(self, idx: int) -> dict:
        r = self.records[self._index_pool[idx]]
        img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {r['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._erase_neighbors(img, r)

        if self.training:
            target_mask = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
            # ROTATION: live camera sees tools at any angle — train on it.
            # Crop FIRST (small canvas), THEN rotate image+mask together so the
            # rotated canvas stays small. minAreaRect length is
            # rotation-invariant so the length feature stays correct.
            x1, y1, x2, y2 = self._mask_bbox(target_mask)
            m = self.bbox_margin
            dx, dy = int((x2 - x1) * m), int((y2 - y1) * m)
            cx1, cy1 = max(x1 - dx, 0), max(y1 - dy, 0)
            cx2, cy2 = min(x2 + dx, r["width"]), min(y2 + dy, r["height"])
            img = img[cy1:cy2, cx1:cx2]
            target_mask = target_mask[cy1:cy2, cx1:cx2].copy()
            if self.rotate_prob > 0 and random.random() < self.rotate_prob:
                angle = random.uniform(*self.rotate_range)
                img, target_mask = rotate_image_mask(img, target_mask, angle)
            # Mask-Aware Patch-Paste (Copy-Paste): paste secondary tools onto green cloth
            if self.patch_paste_prob > 0 and self._patch_pool and random.random() < self.patch_paste_prob:
                img = patch_paste_augment(
                    img, target_mask, self._patch_pool,
                    target_label=r["label"],
                    flip_flags=self.flip_flags,
                    max_pastes=self.patch_paste_max_objects,
                    max_overlap=self.patch_paste_max_overlap,
                    blend_feather=3
                )
            # TIP-ZOOM on the cropped view. Hard classes
            # (Suture/Artery/Needle) get a much higher rate.
            tip_p = self.tip_zoom_prob if r["class_name"] not in self.hard_classes \
                else self.hard_tip_zoom_prob
            if tip_p > 0 and random.random() < tip_p:
                img = maybe_tip_zoom(img, target_mask, 1.0, self.tip_zoom_size)
            if random.random() < 0.5:               # green-cloth shadow
                img = simulate_shadow(img)
            img = self.aug(image=img)["image"]
            # per-class horizontal flip (only for classes without handedness issues)
            if self.flip_flags is not None and self.flip_flags[r["label"]] and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
        else:
            if self.bbox_margin > 0:
                img = self._maybe_crop(img, r)

        # main tensor
        tensor = self.tensor_tf(image=img)["image"]
        length_norm = (self._lengths[self._index_pool[idx]] - self.length_mean) / self.length_std
        out = {
            "image": tensor,
            "length": torch.tensor(length_norm, dtype=torch.float32),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }
        return out


# ============================================================ Visualization (debug)
def visualize_records(records: List[dict], calibration_ratio: Optional[float] = None,
                      n: int = 6, cols: int = 3, seed: int = 0):
    """
    Display image + mask outline + measured length — use to verify that
    annotation / length measurement is correct before real training
    (returns a matplotlib figure).
    """
    import matplotlib.pyplot as plt
    rng = random.Random(seed)
    picks = rng.sample(records, min(n, len(records)))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    unit = "cm" if calibration_ratio else "px"
    for ax, r in zip(axes, picks):
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 40, 40), 3)
        L = measure_length_px(mask) * (calibration_ratio if calibration_ratio else 1.0)
        ax.imshow(img)
        ax.set_title(f"{r['class_name']} | {L:.1f} {unit}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


def visualize_patch_paste_samples(records: List[dict], n: int = 6, cols: int = 3,
                                  max_pastes: int = 2, seed: int = 42):
    """
    Generate and display n sample images with Mask-Aware Patch-Paste augmentation applied.
    Use in notebooks/Colab to visually inspect how secondary instruments are blended
    onto the green cloth before starting full training.
    """
    import matplotlib.pyplot as plt
    rng = random.Random(seed)
    # Pre-extract patch pool
    pool = []
    for r in records[:60]:
        p = extract_instrument_patch(r)
        if p is not None:
            pool.append(p)

    picks = rng.sample(records, min(n, len(records)))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")

    for ax, r in zip(axes, picks):
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target_mask = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])

        aug_img = patch_paste_augment(
            img, target_mask, pool,
            target_label=r["label"],
            max_pastes=max_pastes,
            max_overlap=0.20,
            blend_feather=3,
            rng=rng,
        )

        # Highlight target instrument with yellow border
        cnts, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(aug_img, cnts, -1, (255, 220, 0), 2)

        ax.imshow(aug_img)
        ax.set_title(f"Target: {r['class_name']}\n(yellow border = main label)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig
