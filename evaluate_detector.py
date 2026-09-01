# -*- coding: utf-8 -*-
"""
evaluate_detector.py — instance P/R/F1 + overlay gallery on real photos

    python evaluate_detector.py --ckpt outputs_detector/best_detector.pt --data_dir dataset
"""
import argparse
import os

import cv2
import numpy as np
import torch

from det_model import load_detector
from det_postprocess import draw_instances, instances_from_logits
from train_detector import build_real_val_records, validate_instances


def save_overlays(bundle, records, out_dir: str, n: int = 12) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, classes, device = bundle["model"], bundle["cfg"], bundle["classes"], bundle["device"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for i, r in enumerate(records[:n]):
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        x = torch.from_numpy(
            ((cv2.resize(rgb, (cfg.img_size, cfg.img_size)).astype(np.float32) / 255.0 - mean) / std
             ).transpose(2, 0, 1)[None]
        ).to(device)
        with torch.no_grad():
            logits = model(x)[0].float().cpu().numpy()
        insts = instances_from_logits(
            logits, classes, frame_rgb=cv2.resize(rgb, (cfg.img_size, cfg.img_size)),
            mask_threshold=cfg.mask_threshold, min_instance_area=cfg.min_instance_area,
            nms_iou=cfg.nms_iou, conf_min_score=cfg.conf_min_score, want_tip_crops=False,
        )
        sx, sy = W / cfg.img_size, H / cfg.img_size
        for inst in insts:
            x1, y1, x2, y2 = inst["bbox"]
            inst["bbox_frame"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        vis = draw_instances(bgr, insts, use_frame_coords=True)
        cv2.imwrite(os.path.join(out_dir, f"det_{i:02d}_{r['class_name']}.jpg"), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out_dir", default="outputs_detector/eval")
    args = ap.parse_args()

    bundle = load_detector(args.ckpt)
    recs = build_real_val_records(args.data_dir)
    if not recs:
        raise SystemExit("no val/test records")
    m = validate_instances(bundle["model"], recs, bundle["classes"], bundle["cfg"], bundle["device"],
                           max_images=len(recs))
    print(f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"IoU={m['mean_iou']:.3f}  n_gt={m['n_gt']} n_det={m['n_det']} n_tp={m['n_tp']}")
    save_overlays(bundle, recs, args.out_dir)
    print(f"overlays → {args.out_dir}")


if __name__ == "__main__":
    main()
