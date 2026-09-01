# -*- coding: utf-8 -*-
"""
export_detector_onnx.py — export the DINOv2 detector to ONNX for Raspberry Pi 5

Run on PC (torch/transformers/peft needed), NOT on the Pi:
    python export_detector_onnx.py --ckpt outputs_detector/best_detector.pt --out_dir pi_final_v3/onnx_export

Outputs:
  - detector_dino.onnx   : pixel_values (1,3,560,560) → logits (1,15,560,560)
                           (ch0 = fg, ch1..14 = classes; sigmoid + instances in
                            det_postprocess.py on the Pi)
  - detector_meta.json   : classes, img_size, post-processing thresholds,
                           real lengths (cm), calibration_ratio (from calibrate.py)
LoRA is merged into the backbone before export → clean Dinov2Model graph.
"""
import argparse
import json
import os

import numpy as np
import torch

from config import REAL_LENGTH_CM
from det_model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="pi_final_v3/onnx_export")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/3] loading checkpoint ...")
    bundle = load_detector(args.ckpt, device=torch.device("cpu"))
    model = bundle["model"]
    cfg = bundle["cfg"]
    classes = bundle["classes"]

    print("[2/3] merging LoRA + export ONNX ...")
    if hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
        print("      LoRA merged into backbone")
    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, dtype=torch.float32)
    onnx_path = os.path.join(args.out_dir, "detector_dino.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"      saved -> {onnx_path}")

    # sanity: same outputs after export?
    with torch.no_grad():
        ref = model(dummy)[0, :3, :3, :3].numpy()
    print(f"      sanity ref logits corner: {ref.flatten()[:3]}")

    print("[3/3] writing metadata ...")
    meta = {
        "classes": classes,
        "img_size": int(cfg.img_size),
        "mask_threshold": float(cfg.mask_threshold),
        "min_instance_area": int(cfg.min_instance_area),
        "nms_iou": float(cfg.nms_iou),
        "conf_min_score": float(cfg.conf_min_score),
        "real_length_cm": REAL_LENGTH_CM,
        "calibration_ratio": None,  # filled by calibrate.py (cm per pixel at the rig)
        "model": "DINOv2 detector (seg head) — YOLO-free",
    }
    meta_path = os.path.join(args.out_dir, "detector_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      saved -> {meta_path}")
    print("\nDone — copy the whole folder to the Pi (pi_final_v3/onnx_export).")


if __name__ == "__main__":
    main()
