# -*- coding: utf-8 -*-
"""
tools/coco_to_yolo.py — COCO-seg dataset -> Ultralytics YOLO detection format.

Class ids come from dataset.load_coco_records (sorted class names), so the
mapping is IDENTICAL to the DINO pipeline and yolo_meta.json classes.
Boxes are the tight polygon bbox (segmentation_bbox), normalized cxcywh.

Usage:
    python tools/coco_to_yolo.py --data_dir dataset --out_dir dataset_yolo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_coco_records, segmentation_bbox


def convert_split(split_dir: str, out_img: str, out_lbl: str) -> int:
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    recs, classes = load_coco_records(
        os.path.dirname(split_dir.rstrip("/\\")) or ".",
        os.path.basename(split_dir.rstrip("/\\")),
    )
    # group annotations by image
    by_img = {}
    for r in recs:
        by_img.setdefault(r["image_path"], []).append(r)
    n_ann = 0
    for img_path, anns in by_img.items():
        base = os.path.splitext(os.path.basename(img_path))[0]
        lines = []
        W = anns[0]["width"]
        H = anns[0]["height"]
        for a in anns:
            x1, y1, x2, y2 = segmentation_bbox(
                a["segmentation"], a["width"], a["height"])
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            lines.append(f"{a['label']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_ann += 1
        with open(os.path.join(out_lbl, base + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        # images are symlinked (same box) / copied at deploy time
        link = os.path.join(out_img, os.path.basename(img_path))
        if not os.path.exists(link):
            try:
                os.symlink(os.path.abspath(img_path), link)
            except OSError:
                import shutil
                shutil.copy2(img_path, link)
    return n_ann, classes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out_dir", default="dataset_yolo")
    args = ap.parse_args(argv)

    splits = {}
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        n, classes = convert_split(
            split_dir,
            os.path.join(args.out_dir, "images", split),
            os.path.join(args.out_dir, "labels", split),
        )
        splits[split] = n
        print(f"{split}: {n} boxes")

    names = {i: c for i, c in enumerate(classes)}
    yaml_path = os.path.join(args.out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out_dir)}\n")
        f.write("train: images/train\nval: images/valid\ntest: images/test\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, c in names.items():
            f.write(f"  {i}: {c}\n")
    print(f"wrote {yaml_path} nc={len(classes)}")
    print("classes:", classes)


if __name__ == "__main__":
    main()
