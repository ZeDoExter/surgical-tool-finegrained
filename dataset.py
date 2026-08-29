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
- Experimental defaults found useful elsewhere in the project: image size 504
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
def load_coco_records(data_dir: str, split: str) -> Tuple[List[dict], List[str]]:
    """
    Read a split folder ("train"/"valid"/"test") containing ``_annotations.coco.json``.

    Returns ``(records, class_names)`` where each record is a dict with
    ``image_path / segmentation / width / height / class_name / label``.

    - label = index from **sorted class names** (stable regardless of
      category_id ordering in the json).
    - 1 annotation = 1 sample → if one image contains multiple instruments
      it yields multiple samples (recommend using bbox_margin > 0 in
      Dataset to crop per instance).
    """
    split_dir = os.path.join(data_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    # Filter to categories that actually appear (skip dummy super-category like id 0)
    used_names = {cat_id_to_name[ann["category_id"]] for ann in coco["annotations"]}
    class_names = sorted(used_names)
    name_to_label = {n: i for i, n in enumerate(class_names)}

    records: List[dict] = []
    for ann in coco["annotations"]:
        im = images[ann["image_id"]]
        cname = cat_id_to_name[ann["category_id"]]
        records.append({
            "image_path": os.path.join(split_dir, im["file_name"]),
            "segmentation": ann["segmentation"],
            "width": int(im["width"]),
            "height": int(im["height"]),
            "class_name": cname,
            "label": name_to_label[cname],
        })
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


# ============================================================ Augmentation
def build_photometric_aug() -> A.Compose:
    """
    Photometric augmentation for training — intentionally stronger than usual
    because metal reflections vary per capture, and the green cloth background
    with shifting shadows makes illumination inconsistent.

    Critically, there is *no* crop/scale/cutout because "size and shape"
    are what the model must learn to separate visually similar classes.
    """
    return A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),             # boost local contrast (metal on green cloth has low contrast; shadows worsen it)
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),  # stronger-than-usual jitter
        A.RandomGamma(gamma_limit=(70, 150), p=0.7),                        # simulate different exposure / lighting
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),                           # slight defocus blur
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
    return tr, va


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
      Experiments found bbox_margin ≈ 0.15 effective; image size 504 was the
      best-performing default in experiments (this class defaults to 224 for
      backward compatibility — pass 504 explicitly to reproduce those results).
    - Background is green cloth; shadows on the cloth make tight bounding harder,
      which is why photometric + shadow augmentation is used.
    """

    def __init__(self, records: List[dict], length_stats: Tuple[float, float],
                 img_size: int = 224, calibration_ratio: Optional[float] = None,
                 flip_flags: Optional[List[bool]] = None, training: bool = True,
                 bbox_margin: float = 0.0):
        self.records = records
        self.length_mean, self.length_std = length_stats
        self.training = training
        self.bbox_margin = bbox_margin
        self.img_size = img_size
        self.tensor_tf = build_tensor_transform(img_size)
        self.aug = build_photometric_aug() if training else None
        self.flip_flags = flip_flags
        # Measure length once at dataset creation (rasterizing polygons in memory is very fast)
        self._lengths = record_lengths(records, calibration_ratio)

    def __len__(self) -> int:
        return len(self.records)

    def _maybe_crop(self, img: np.ndarray, r: dict) -> np.ndarray:
        m = self.bbox_margin
        x1, y1, x2, y2 = segmentation_bbox(r["segmentation"], r["width"], r["height"])
        dx, dy = int((x2 - x1) * m), int((y2 - y1) * m)
        x1, y1 = max(x1 - dx, 0), max(y1 - dy, 0)
        x2, y2 = min(x2 + dx, r["width"]), min(y2 + dy, r["height"])
        return img[y1:y2, x1:x2]

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {r['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.bbox_margin > 0:
            img = self._maybe_crop(img, r)
        if self.training:
            if random.random() < 0.5:               # green-cloth shadow (real difficulty: lighting/shadow is the main challenge, not a silver tray)
                img = simulate_shadow(img)
            img = self.aug(image=img)["image"]
            # per-class horizontal flip (only for classes without handedness issues)
            if self.flip_flags is not None and self.flip_flags[r["label"]] and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])

        # main tensor
        tensor = self.tensor_tf(image=img)["image"]
        length_norm = (self._lengths[idx] - self.length_mean) / self.length_std
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
