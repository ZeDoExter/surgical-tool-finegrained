# -*- coding: utf-8 -*-
"""
det_dataset.py — Training dataset for the DINOv2 detector (no YOLO)

Problem: the COCO export has 1 instrument per image (441 images) — a detector
needs MULTI-object scenes. Waiting for a hand-made mixed dataset is slow, so
this Dataset synthesizes scenes ON THE FLY every epoch:

  1. pick a real green-cloth background (random real photo without instruments
     is ideal; here: real photo, instrument pixels erased via its mask) OR a
     procedural green cloth (HSV noise + weave + vignette + shadows)
  2. paste 2-5 instrument foregrounds (extracted via COCO polygons with
     `transform_instrument_patch` — random rotation/scale/flip)
  3. render the SAME photometric/shadow augmentation the classifier uses
  4. targets: per-pixel (1 + num_classes) at patch-grid resolution 40×40
     (each patch takes the class of the instrument covering its CENTER;
      background patches = class 0 = background)

Also supports real multi-annotation images (future "mix" dataset): any COCO
image with >1 annotation is used as-is (targets rasterized the same way).

Inference-time domain gap is handled by: same green cloth, same lighting
simulation, real instrument patches (not synthetic), and scale jitter.
"""
import math
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import (
    IMAGENET_MEAN, IMAGENET_STD, build_photometric_aug,
    extract_instrument_patch, load_coco_records, load_coco_records_multi,
    mask_from_coco_segmentation, simulate_shadow, transform_instrument_patch,
)

# background channel index (class 0); instrument classes start at 1
BG = 0

# ---------------- procedural green cloth ----------------
def render_green_cloth(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Procedural green surgical cloth: base HSV noise + weave texture + vignette."""
    hue = rng.uniform(55, 75)            # green hue (OpenCV H: 0-179)
    sat = rng.uniform(90, 140)
    val = rng.uniform(150, 200)
    h_n = np.random.default_rng(rng.randrange(1 << 30))
    s_noise = h_n.integers(-12, 12, (h, w)).astype(np.int16)
    v_noise = h_n.integers(-18, 18, (h, w)).astype(np.int16)
    hsv = np.empty((h, w, 3), dtype=np.int16)
    hsv[..., 0] = int(hue)
    hsv[..., 1] = int(sat) + s_noise
    hsv[..., 2] = int(val) + v_noise
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    cloth = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # weave: faint diagonal texture (scaled random noise, blurred)
    wv = (np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (h // 4, w // 4)) * 255 * 0.03)
    wv = cv2.resize(wv.astype(np.float32), (w, h))
    cloth = np.clip(cloth.astype(np.float32) + wv[..., None], 0, 255).astype(np.uint8)
    # vignette (uneven lighting across the rig)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = 1.0 - 0.18 * r ** 2
    cloth = np.clip(cloth.astype(np.float32) * vig[..., None], 0, 255).astype(np.uint8)
    return cloth


def erase_with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace instrument pixels with procedural cloth (in-painting-lite)."""
    h, w = img.shape[:2]
    cloth = render_green_cloth(h, w, random.Random(random.randrange(1 << 30)))
    out = img.copy()
    dil = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    out[dil > 0] = cloth[dil > 0]
    return out


# ---------------- scene synthesis ----------------
def synth_scene(patch_pool: List[dict], h: int, w: int, rng: random.Random,
                min_objects: int = 2, max_objects: int = 5,
                scale_range: Tuple[float, float] = (0.75, 1.25),
                same_class_prob: float = 0.15, max_overlap: float = 0.20,
                green_prob: float = 0.5, bg_pool: Optional[List[np.ndarray]] = None,
                shadows: bool = True) -> Tuple[np.ndarray, List[dict]]:
    """
    Compose a training scene. Returns (rgb, instances) where each instance is
    {mask:(H,W) uint8 0/255, label:int(≥1), class_name:str}.
    """
    # --- background ---
    if bg_pool and rng.random() > green_prob:
        bg = rng.choice(bg_pool).copy()
        if bg.shape[0] != h or bg.shape[1] != w:
            bg = cv2.resize(bg, (w, h))
    else:
        bg = render_green_cloth(h, w, rng)

    n = rng.randint(min_objects, max_objects)
    # unique-class first, then allow same-class duplicates with prob
    pool = [p for p in patch_pool if p["rgb"].shape[0] < h and p["rgb"].shape[1] < w]
    if not pool:
        return bg, []
    used_labels = set()

    # --- greedy non-overlap placement (mask-aware) ---
    occ = np.zeros((h, w), dtype=np.uint8)      # cumulative occupancy
    instances: List[dict] = []
    attempts_left = n * 12
    while len(instances) < n and attempts_left > 0 and pool:
        attempts_left -= 1
        patch = rng.choice(pool)
        if patch["label"] in used_labels and rng.random() >= same_class_prob:
            continue  # same class only when allowed (rare duplicates)
        allow_flip = True
        rot_rgb, rot_mask = transform_instrument_patch(
            patch, scale_range=scale_range, allow_flip=allow_flip, rng=rng)
        ph, pw = rot_rgb.shape[:2]
        if ph >= h or pw >= w or ph < 8 or pw < 8:
            continue
        # choose position with overlap control
        placed = False
        x = y = 0
        for _try in range(12):
            x = rng.randint(0, w - pw)
            y = rng.randint(0, h - ph)
            roi = occ[y:y + ph, x:x + pw]
            ov = np.sum((rot_mask > 0) & (roi > 0)) / max(np.sum(rot_mask > 0), 1)
            if ov <= max_overlap:
                placed = True
                break
        if not placed:
            continue

        # paste with feathered alpha
        alpha = (rot_mask > 0).astype(np.float32)
        k = 5
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)[..., None]
        pasted = np.clip(rot_rgb.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255)
        roi_img = bg[y:y + ph, x:x + pw].astype(np.float32)
        bg[y:y + ph, x:x + pw] = np.clip((1 - alpha) * roi_img + alpha * pasted, 0, 255).astype(np.uint8)
        occ[y:y + ph, x:x + pw] = np.maximum(occ[y:y + ph, x:x + pw], (rot_mask > 0).astype(np.uint8) * 255)
        m = np.zeros((h, w), dtype=np.uint8)
        m[y:y + ph, x:x + pw] = (rot_mask > 0).astype(np.uint8) * 255
        instances.append({"mask": m, "label": patch["label"], "class_name": patch["class_name"]})
        used_labels.add(patch["label"])

    if shadows:
        bg = simulate_shadow(bg, rng)
    return bg, instances


def build_patch_pool(data_dir: str, min_area: int = 120,
                     max_pool: int = 400,
                     extra_data_dirs: Optional[List[str]] = None) -> Tuple[List[dict], List[str]]:
    """
    Extract instrument foreground patches + class list (labels start at 1).

    Uses ONLY original photos (files not named aug_patchpaste_*) — the
    patch-paste composites contain the same instruments again and there are
    thousands of them after augment_dataset, which made this step take
    ~10x longer for zero new information.
    """
    dirs = [data_dir] + list(extra_data_dirs or [])
    tr_records, classes = load_coco_records_multi(dirs, "train")
    records = [r for r in tr_records
               if "aug_patchpaste_" not in os.path.basename(r["image_path"])]
    for d in dirs:
        va_path = os.path.join(d, "valid", "_annotations.coco.json")
        if os.path.exists(va_path):
            va_records, _ = load_coco_records(d, "valid")
            records += [r for r in va_records
                        if "aug_patchpaste_" not in os.path.basename(r["image_path"])]
    print(f"[patch pool] extracting from {len(records)} original photos "
          f"across {len(dirs)} dataset folder(s) ...", flush=True)
    pool: List[dict] = []
    for i, r in enumerate(records):
        if i % 50 == 0:
            print(f"  [patch pool] {i}/{len(records)}", flush=True)
        p = extract_instrument_patch(r)
        if p is None:
            continue
        if np.sum(p["mask"] > 0) < min_area:
            continue
        p["label"] = r["label"] + 1  # +1: label 0 = background in detector targets
        pool.append(p)
    if len(pool) > max_pool:
        pool = random.Random(42).sample(pool, max_pool)
    print(f"[patch pool] done: {len(pool)} patches", flush=True)
    return pool, classes


def build_bg_pool(data_dir: str, max_n: int = 24,
                  extra_data_dirs: Optional[List[str]] = None) -> List[np.ndarray]:
    """Real backgrounds: original photos with the instrument erased (cloth visible).
    Skips patch-paste composites (their mask only covers one of several tools)."""
    dirs = [data_dir] + list(extra_data_dirs or [])
    records, _ = load_coco_records_multi(dirs, "train")
    records = [r for r in records
               if "aug_patchpaste_" not in os.path.basename(r["image_path"])]
    rng = random.Random(0)
    picks = records if len(records) <= max_n else rng.sample(records, max_n)
    out: List[np.ndarray] = []
    for r in picks:
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
        out.append(erase_with_mask(rgb, m))
    print(f"[bg pool] {len(out)} cloth backgrounds", flush=True)
    return out


# ---------------- targets ----------------
def make_grid_targets(instances: List[dict], h: int, w: int, grid: int,
                      num_classes: int) -> np.ndarray:
    """Full-res targets (1+C, H, W): ch0 = fg, ch1..C = class (label is 1-indexed)."""
    t = np.zeros((1 + num_classes, h, w), dtype=np.float32)
    for inst in instances:
        m = inst["mask"]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        hit = m > 0
        t[0][hit] = 1.0
        lab = int(inst["label"])
        if 1 <= lab <= num_classes:
            t[lab][hit] = 1.0
    return t


# ---------------- Dataset ----------------
class DetectorSynthDataset(Dataset):
    """
    Each __getitem__: new random scene (infinite variety) → (image, target).
    Also mixes in real images (single-instrument photos as 1-object scenes and
    real multi-annotation photos when the future mix dataset arrives).
    """

    def __init__(self, patch_pool: List[dict], classes: List[str],
                 img_size: int = 560, grid: int = 40, num_classes: int = 14,
                 training: bool = True, bg_pool: Optional[List[np.ndarray]] = None,
                 synth_min_objects: int = 2, synth_max_objects: int = 5,
                 synth_same_class_prob: float = 0.15,
                 synth_scale_range: Tuple[float, float] = (0.75, 1.25),
                 synth_green_prob: float = 0.5, synth_shadows: bool = True,
                 synth_max_overlap: float = 0.20, min_mask_area_px: int = 120,
                 real_records: Optional[List[dict]] = None,
                 real_prob: float = 0.35, aug: bool = True):
        self.pool = patch_pool
        self.classes = classes
        self.h = self.w = img_size
        self.grid = grid
        self.num_classes = num_classes
        self.training = training
        self.bg_pool = bg_pool or []
        self.min_objects = synth_min_objects
        self.max_objects = synth_max_objects
        self.same_class_prob = synth_same_class_prob
        self.scale_range = synth_scale_range
        self.green_prob = synth_green_prob
        self.shadows = synth_shadows
        self.max_overlap = synth_max_overlap
        self.min_area = min_mask_area_px
        self.real_records = real_records or []
        self.real_prob = real_prob
        self.aug = aug
        self.photometric = build_photometric_aug() if aug else None
        self._by_class: dict = {}
        for p in self.pool:
            self._by_class.setdefault(p["label"], []).append(p)

    def __len__(self) -> int:
        # virtual length: enough steps per epoch; __getitem__ re-synthesizes each time
        return max(len(self.pool) * 6, 64)

    def _real_scene(self, idx: int) -> Tuple[np.ndarray, List[dict]]:
        r = self.real_records[idx % len(self.real_records)]
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read: {r['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.w, self.h))
        # ALL annotations of this image (mix dataset → multiple)
        from dataset import load_coco_annotations_for_image
        anns = load_coco_annotations_for_image(r)
        instances = []
        sx, sy = self.w / r["width"], self.h / r["height"]
        for a in anns:
            m = mask_from_coco_segmentation(a["segmentation"], r["height"], r["width"])
            m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            if np.sum(m > 0) < self.min_area:
                continue
            cname = a["class_name"]
            try:
                lab = self.classes.index(cname) + 1
            except ValueError:
                continue
            instances.append({"mask": m, "label": lab, "class_name": cname})
        return rgb, instances

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random()  # fresh randomness every visit
        use_real = self.real_records and rng.random() < self.real_prob
        if use_real:
            img, instances = self._real_scene(rng.randrange(len(self.real_records)))
            if self.aug and self.training and rng.random() < 0.3:
                img = simulate_shadow(img, rng)
        else:
            img, instances = synth_scene(
                self.pool, self.h, self.w, rng,
                min_objects=self.min_objects, max_objects=self.max_objects,
                scale_range=self.scale_range, same_class_prob=self.same_class_prob,
                max_overlap=self.max_overlap, green_prob=self.green_prob,
                bg_pool=self.bg_pool, shadows=self.shadows and self.training,
            )
        if self.aug and self.training:
            img = self.photometric(image=img)["image"]

        target = make_grid_targets(instances, self.h, self.w, self.grid, self.num_classes)

        # to tensors (ImageNet norm, CHW) — keep float32: IMAGENET stats are
        # plain tuples, and (float32 - tuple) would silently promote to float64
        img_f = img.astype(np.float32) / np.float32(255.0)
        _m = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        _s = np.asarray(IMAGENET_STD, dtype=np.float32)
        img_f = (img_f - _m) / _s
        image = torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1),
                                                      dtype=np.float32))
        target_t = torch.from_numpy(np.ascontiguousarray(target))
        return {"image": image, "target": target_t, "n_instances": len(instances)}
