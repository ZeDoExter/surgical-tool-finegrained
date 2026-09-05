# -*- coding: utf-8 -*-
"""
augment_dataset.py — Generate Mask-Aware Patch-Paste (Copy-Paste) Multi-Instrument Images

Usage:
    python augment_dataset.py --data_dir dataset --num_aug 2 --max_pastes 2 --seed 42

What this does:
1) Reads train/_annotations.coco.json and extracts clean foreground patches + polygon masks
   for all instruments.
2) For each training image, generates `num_aug` new synthetic multi-instrument images by
   pasting 1 to `max_pastes` other instruments onto the green cloth with realistic rotation,
   scale jitter, and feathered edge blending.
3) Generates exact COCO annotations (segmentation polygons, bounding boxes, category IDs)
   for ALL instruments in the new images and merges them into _annotations.coco.json.

This eliminates the "1 image = 1 instrument" bottleneck and teaches models to recognize
instruments in realistic cluttered multi-tool scenes.
"""
import argparse
import json
import os
import random
from typing import List, Tuple

import cv2
import numpy as np

from dataset import (
    extract_instrument_patch,
    load_coco_records,
    mask_from_coco_segmentation,
    transform_instrument_patch,
)


def paste_instrument_with_annotation(
    canvas_img: np.ndarray,
    canvas_mask: np.ndarray,
    patch: dict,
    max_overlap: float = 0.20,
    blend_feather: int = 3,
    rng: random.Random = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Pastes an instrument patch onto canvas_img, returns updated image, updated mask,
    and a new COCO annotation dict for the pasted tool.
    """
    rng = rng or random
    h_dst, w_dst = canvas_img.shape[:2]

    rot_rgb, rot_mask = transform_instrument_patch(
        patch, scale_range=(0.85, 1.15), allow_flip=True, rng=rng
    )
    ph, pw = rot_rgb.shape[:2]

    # Resize if patch exceeds canvas
    if ph >= h_dst or pw >= w_dst:
        scale_down = min((h_dst - 10) / ph, (w_dst - 10) / pw) * rng.uniform(0.6, 0.9)
        if scale_down <= 0:
            return canvas_img, canvas_mask, None
        new_ph = max(int(ph * scale_down), 4)
        new_pw = max(int(pw * scale_down), 4)
        rot_rgb = cv2.resize(rot_rgb, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)
        rot_mask = cv2.resize(rot_mask, (new_pw, new_ph), interpolation=cv2.INTER_NEAREST)
        ph, pw = new_ph, new_pw

    if ph >= h_dst or pw >= w_dst or ph < 4 or pw < 4:
        return canvas_img, canvas_mask, None

    # Search for position with acceptable overlap
    best_x, best_y = 0, 0
    placed = False
    canvas_occupied = np.sum(canvas_mask > 0)

    for _ in range(20):
        x = rng.randint(0, w_dst - pw)
        y = rng.randint(0, h_dst - ph)

        if canvas_occupied > 0:
            target_roi_mask = canvas_mask[y:y + ph, x:x + pw]
            overlap = np.sum((rot_mask > 0) & (target_roi_mask > 0))
            if overlap / float(canvas_occupied) <= max_overlap:
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

    # Edge feathering
    alpha = (rot_mask > 0).astype(np.float32)
    if blend_feather > 0:
        k = blend_feather if blend_feather % 2 == 1 else blend_feather + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = alpha[..., None]

    brightness = rng.uniform(0.85, 1.15)
    pasted_rgb = np.clip(rot_rgb.astype(np.float32) * brightness, 0, 255)

    roi = canvas_img[best_y:best_y + ph, best_x:best_x + pw].astype(np.float32)
    blended = (1.0 - alpha) * roi + alpha * pasted_rgb
    canvas_img[best_y:best_y + ph, best_x:best_x + pw] = np.clip(blended, 0, 255).astype(np.uint8)

    # Update cumulative mask
    canvas_mask[best_y:best_y + ph, best_x:best_x + pw] = np.maximum(
        canvas_mask[best_y:best_y + ph, best_x:best_x + pw], rot_mask
    )

    # Extract polygon coordinates for COCO annotation
    bin_mask = (rot_mask > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 50:
            continue
        pts = c.reshape(-1, 2) + np.array([best_x, best_y])
        if len(pts) >= 3:
            polys.append(pts.flatten().tolist())

    if not polys:
        return canvas_img, canvas_mask, None

    ann_dict = {
        "segmentation": polys,
        "bbox": [int(best_x), int(best_y), int(pw), int(ph)],
        "area": float(np.sum(bin_mask > 0)),
        "category_id": patch.get("category_id", 1),
        "iscrowd": 0,
    }
    return canvas_img, canvas_mask, ann_dict


def augment_dataset(data_dir: str, num_aug: int = 2, max_pastes: int = 2,
                     max_overlap: float = 0.20, seed: int = 42,
                     max_base_anns: int = 6) -> None:
    """
    Generate mask-aware patch-paste augmented training data and update COCO json.

    - ``*_Head`` annotations are NEVER copied into new images (head-only crops
      are not wanted anywhere in the flow).
    - base images that already have >= ``max_base_anns`` annotations are
      SKIPPED (they are already rich multi-tool scenes; pasting more would
      only create unrealistic clutter and cover real tools, since the
      placement fallback ignores overlap when it can't find a spot).
    """
    train_dir = os.path.join(data_dir, "train")
    ann_path = os.path.join(train_dir, "_annotations.coco.json")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Category mappings
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    name_to_cat_id = {c["name"]: c["id"] for c in categories}

    print(f"Loading {len(images)} images and extracting instrument patches...")
    records, _ = load_coco_records(data_dir, "train")

    patch_pool: List[dict] = []
    for r in records:
        p = extract_instrument_patch(r)
        if p is not None:
            p["category_id"] = name_to_cat_id.get(r["class_name"], 1)
            patch_pool.append(p)

    print(f"Extracted {len(patch_pool)} clean instrument patches.")

    rng = random.Random(seed)
    max_img_id = max(im["id"] for im in images)
    max_ann_id = max(ann["id"] for ann in annotations) if annotations else 0

    new_images = []
    new_annotations = []
    total_generated = 0

    # Group original annotations by image_id (drop *_Head: head-only crops
    # must not propagate into generated images)
    anns_by_img = {}
    for a in annotations:
        if cat_id_to_name.get(a["category_id"], "").endswith("_Head"):
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for im in images:
        img_path = os.path.join(train_dir, im["file_name"])
        base_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if base_bgr is None:
            continue
        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        orig_anns = anns_by_img.get(im["id"], [])
        if len(orig_anns) >= max_base_anns:
            continue  # already a rich multi-tool scene — don't clutter it further

        for aug_idx in range(num_aug):
            max_img_id += 1
            aug_canvas = base_rgb.copy()
            h, w = aug_canvas.shape[:2]

            # Build base mask from existing annotations
            base_mask = np.zeros((h, w), dtype=np.uint8)
            for a in orig_anns:
                m = mask_from_coco_segmentation(a["segmentation"], h, w)
                base_mask = np.maximum(base_mask, m)

            # Copy original annotations for new image
            current_image_anns = []
            for a in orig_anns:
                max_ann_id += 1
                new_a = dict(a)
                new_a["id"] = max_ann_id
                new_a["image_id"] = max_img_id
                current_image_anns.append(new_a)

            # Paste 1 to max_pastes secondary tools
            num_pastes = rng.randint(1, max_pastes)
            for _ in range(num_pastes):
                p = rng.choice(patch_pool)
                aug_canvas, base_mask, ann_dict = paste_instrument_with_annotation(
                    aug_canvas, base_mask, p, max_overlap=max_overlap,
                    blend_feather=3, rng=rng
                )
                if ann_dict is not None:
                    max_ann_id += 1
                    ann_dict["id"] = max_ann_id
                    ann_dict["image_id"] = max_img_id
                    current_image_anns.append(ann_dict)

            # Save augmented image
            aug_filename = f"aug_patchpaste_{im['id']:04d}_{aug_idx:02d}.jpg"
            aug_path = os.path.join(train_dir, aug_filename)
            cv2.imwrite(
                aug_path, cv2.cvtColor(aug_canvas, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            # Record new image
            new_im_entry = {
                "id": max_img_id,
                "file_name": aug_filename,
                "width": w,
                "height": h,
            }
            new_images.append(new_im_entry)
            new_annotations.extend(current_image_anns)
            total_generated += 1

    # Merge into COCO structure
    coco["images"] = images + new_images
    coco["annotations"] = annotations + new_annotations

    # Backup original annotation file first if not backed up
    bak_path = os.path.join(train_dir, "_annotations.coco.json.bak")
    if not os.path.exists(bak_path):
        import shutil
        shutil.copy2(ann_path, bak_path)

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {total_generated} new multi-instrument augmented images.")
    print(f"Total training images: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Updated annotation saved to: {ann_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate Mask-Aware Patch-Paste Multi-Instrument Images")
    ap.add_argument("--data_dir", default="dataset", help="Dataset root directory")
    ap.add_argument("--num_aug", type=int, default=2,
                    help="Number of augmented copies per original image (default: 2)")
    ap.add_argument("--max_pastes", type=int, default=2,
                    help="Max number of secondary tools pasted per image (default: 2)")
    ap.add_argument("--max_overlap", type=float, default=0.20,
                    help="Max allowed overlap fraction with existing tools (default: 0.20)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    augment_dataset(args.data_dir, args.num_aug, args.max_pastes, args.max_overlap, args.seed)


if __name__ == "__main__":
    main()
