# -*- coding: utf-8 -*-
"""
dataset.py — โหลดภาพ + segmentation mask จาก COCO format (Roboflow export)

หน้าที่หลัก:
1) parse ``_annotations.coco.json`` → records (path, polygon, label)
2) rasterize polygon → binary mask
3) วัดความยาวเครื่องมือจาก mask (minAreaRect) เป็น auxiliary feature 1 ค่า
4) augmentation ที่ปลอดภัยกับงานนี้:
   - เน้น photometric (brightness/contrast/gamma/CLAHE) จำลองแสงสะท้อนบนโลหะ
   - ไม่มี crop/zoom ที่ทำลาย aspect ratio หรือ scale ของวัตถุ (ขนาดคือฟีเจอร์สำคัญ!)
   - ไม่มี cutout/random erasing กลางวัตถุ
   - horizontal flip เปิด/ปิดได้ "ต่อ class" (บาง class มี handedness ห้าม flip)
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


# ============================================================ การวัดความยาวจาก mask
def measure_length_px(mask: np.ndarray) -> float:
    """
    คืนความยาวสูงสุดของเครื่องมือ หน่วย pixel (จาก binary mask ชิ้นเดียว)

    ใช้ ``cv2.minAreaRect`` เพราะเครื่องมือมักวาง "เฉียง" ไม่ตรงแนวแกน —
    กล่องหมุนที่มี area เล็กสุดที่ล้อม contour จะให้ด้านยาวสุด ≈ ความยาวจริงของเครื่องมือ
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # mask ว่าง (annotation ผิดพลาด) → คืน 0 กัน crash
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)  # ((cx,cy), (w,h), angle)
    (w, h) = rect[1]
    return float(max(w, h))


def get_length_cm(mask: np.ndarray, calibration_ratio: float) -> float:
    """แปลงความยาว pixel → cm ด้วย calibration_ratio (cm/pixel) จาก object อ้างอิง"""
    return measure_length_px(mask) * calibration_ratio


def mask_from_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """
    แปลง segmentation ของ COCO → binary mask (uint8, ค่า 0/255)

    รองรับ polygon (รูปแบบมาตรฐานของ Roboflow) และ RLE (ต้องมี pycocotools)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):  # RLE format
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "เจอ segmentation แบบ RLE แต่ยังไม่ได้ติดตั้ง pycocotools (pip install pycocotools)"
            ) from exc
        rle = segmentation
        if isinstance(rle.get("counts"), list):  # uncompressed RLE → แปลงเป็น compressed ก่อน
            rle = mask_utils.frPyObjects(rle, height, width)
        return (mask_utils.decode(rle) * 255).astype(np.uint8)
    for poly in segmentation:  # list ของ polygon [[x1,y1,x2,y2,...], ...]
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


# ============================================================ COCO parsing
def load_coco_records(data_dir: str, split: str) -> Tuple[List[dict], List[str]]:
    """
    อ่านโฟลเดอร์ split ("train"/"valid"/"test") ที่มี ``_annotations.coco.json``

    คืน ``(records, class_names)`` โดย record = dict ที่มี
    ``image_path / segmentation / width / height / class_name / label``

    - label = index จากการ **sort ชื่อ class** (คงที่เสมอ ไม่ว่า category_id ใน json จะเรียงแบบไหน)
    - 1 annotation = 1 sample → ถ้า 1 ภาพมีหลายเครื่องมือ จะได้หลาย sample
      (แนะนำใช้ bbox_margin > 0 ใน Dataset เพื่อ crop รายชิ้น)
    """
    split_dir = os.path.join(data_dir, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ไม่เจอไฟล์ annotation: {ann_path}")
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
    """bounding box (x1,y1,x2,y2) ครอบ polygon ทั้งหมด — ใช้ตอน crop รายชิ้น (bbox_margin > 0)"""
    xs: List[float] = []
    ys: List[float] = []
    for poly in segmentation if isinstance(segmentation, list) else []:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        xs += [float(pts[:, 0].min()), float(pts[:, 0].max())]
        ys += [float(pts[:, 1].min()), float(pts[:, 1].max())]
    if not xs:  # RLE หรือ polygon ว่าง → ใช้ทั้งภาพ
        return 0, 0, width, height
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# ============================================================ Augmentation
def build_photometric_aug() -> A.Compose:
    """
    Augmentation ฝั่ง "แสง" สำหรับ train — ตั้งใจให้แรงกว่าปกติ เพราะโลหะสะท้อนแสง
    ต่างกันทุกครั้งที่ถ่าย แต่ *ไม่มี* การ crop/scale/cutout ใดๆ เพราะ
    "ขนาดและรูปทรง" คือสิ่งที่โมเดลต้องเรียนรู้เพื่อแยก class ที่เหมือนกัน
    """
    return A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),             # เพิ่ม local contrast (โลหะบนถาดโลหะ contrast ต่ำ)
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),  # jitter แรงกว่าปกติ
        A.RandomGamma(gamma_limit=(70, 150), p=0.7),                        # จำลอง exposure/แสงไฟต่างกัน
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),                           # เบลอเล็กน้อยแบบ defocus
    ])


def build_tensor_transform(img_size: int) -> A.Compose:
    """resize ขนาดคงที่ + normalize ด้วยค่า ImageNet + แปลงเป็น tensor (ใช้ทั้ง train/eval/infer)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def simulate_shadow(img: np.ndarray, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    จำลอง "เงา" บนพื้นหลังสีเขียว (green mat) — เงาเคลื่อนตามตำแหน่งวางเครื่องมือ/ทิศไฟ

    วาด blob มืดขอบนุ่ม 1-2 จุด (ellipse + Gaussian falloff) คูณลงภาพ
    เขียน numpy เองแทน A.RandomShadow เพราะ signature ของ albumentations
    เปลี่ยนบ่อยระหว่าง 1.x ↔ 2.x — ไม่อยากผูกกับเวอร์ชัน
    """
    rng = rng or random
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax = rng.uniform(w * 0.2, w * 0.7)
        ay = rng.uniform(h * 0.2, h * 0.7)
        strength = rng.uniform(0.35, 0.65)          # เงาดำสุด ~35-65%
        d2 = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
        mask *= 1.0 - strength * np.exp(-d2)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================ สถิติความยาว + split
def record_lengths(records: List[dict], calibration_ratio: Optional[float]) -> List[float]:
    """วัดความยาวของทุก record (px หรือ cm ถ้ามี ratio) — rasterize จาก polygon โดยตรง"""
    out = []
    for r in records:
        mask = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
        L = measure_length_px(mask)
        out.append(L if calibration_ratio is None else L * calibration_ratio)
    return out


def compute_length_stats(records: List[dict], calibration_ratio: Optional[float] = None) -> Tuple[float, float]:
    """
    คำนวณ mean/std ของความยาว — **ต้องคำนวณจาก train เท่านั้น**
    แล้วใช้ค่าเดียวกันนี้ normalize ทั้ง val/test/inference (กัน data leakage)
    """
    L = np.asarray(record_lengths(records, calibration_ratio), dtype=np.float64)
    mean = float(L.mean())
    std = max(float(L.std()), 1e-6)
    return mean, std


def stratified_split(records: List[dict], val_fraction: float = 0.2, seed: int = 42):
    """แบ่ง train/val แบบ stratified (คงสัดส่วน class) — จำเป็นเมื่อข้อมูลน้อย ~30 ภาพ/class"""
    if val_fraction <= 0 or len(records) < 10:
        return records, []
    from sklearn.model_selection import train_test_split
    y = [r["label"] for r in records]
    tr, va = train_test_split(records, test_size=val_fraction, random_state=seed, stratify=y)
    return tr, va


# ============================================================ PyTorch Dataset
class SurgicalInstrumentDataset(Dataset):
    """
    Dataset สำหรับงานจำแนกเครื่องมือผ่าตัด — คืน dict:
      ``image``  : FloatTensor (3, H, W) normalize แล้ว
      ``length`` : scalar float = (ความยาว − mean) / std  ← auxiliary feature
      ``label``  : int64 class index

    หมายเหตุสำคัญ:
    - ความยาววัดจาก mask "ต้นฉบับ" (ก่อน augment) เพราะ photometric/flip ไม่ควรเปลี่ยนความยาวจริง
    - ``flip_flags[label]`` เป็น True เท่านั้นที่จะโดน horizontal flip
    - ``bbox_margin`` > 0 เมื่อ 1 ภาพมีหลายเครื่องมือ → crop รอบ bbox ของชิ้นนั้น
      (crop คงสัดส่วน/scale เดิม ไม่ใช่การ zoom อิสระ)
    """

    def __init__(self, records: List[dict], length_stats: Tuple[float, float],
                 img_size: int = 224, calibration_ratio: Optional[float] = None,
                 flip_flags: Optional[List[bool]] = None, training: bool = True,
                 bbox_margin: float = 0.0):
        self.records = records
        self.length_mean, self.length_std = length_stats
        self.training = training
        self.bbox_margin = bbox_margin
        self.tensor_tf = build_tensor_transform(img_size)
        self.aug = build_photometric_aug() if training else None
        self.flip_flags = flip_flags
        # วัดความยาวครั้งเดียวตอนสร้าง dataset (rasterize polygon ใน memory เร็วมาก)
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
            raise IOError(f"อ่านภาพไม่สำเร็จ: {r['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.bbox_margin > 0:
            img = self._maybe_crop(img, r)
        if self.training:
            if random.random() < 0.5:               # เงาพื้นเขียว (โจทย์จริง: แสง/เงาเป็นปัญหาหลัก)
                img = simulate_shadow(img)
            img = self.aug(image=img)["image"]
            # horizontal flip แบบ per-class (เฉพาะ class ที่ไม่มีปัญหา handedness)
            if self.flip_flags is not None and self.flip_flags[r["label"]] and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])

        tensor = self.tensor_tf(image=img)["image"]
        length_norm = (self._lengths[idx] - self.length_mean) / self.length_std
        return {
            "image": tensor,
            "length": torch.tensor(length_norm, dtype=torch.float32),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


# ============================================================ Visualization (debug)
def visualize_records(records: List[dict], calibration_ratio: Optional[float] = None,
                      n: int = 6, cols: int = 3, seed: int = 0):
    """
    แสดงภาพ + เส้น outline ของ mask + ความยาวที่วัดได้ — ใช้เช็คว่า
    annotation/การวัดความยาวถูกต้อง ก่อนเทรนจริง (คืน matplotlib figure)
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
