# -*- coding: utf-8 -*-
"""
export_to_onnx.py — รันบนเครื่อง PC/Colab เท่านั้น (ต้องมี torch, transformers, peft ครบ)
ไม่ต้องรันบน Raspberry Pi

ทำ 2 อย่าง:
1) โหลด checkpoint (.pt) -> merge LoRA เข้า backbone -> export เป็น .onnx
   (กราฟ ONNX จะรับ pixel_values + length_feat -> คืน embedding 384-d
    ไม่รวม ArcFace classification head ไว้ในกราฟ เพราะ ArcFace ตอน inference
    คือแค่ cosine similarity ธรรมดา ทำในโค้ด numpy ฝั่ง Pi ได้เร็วกว่าและง่ายกว่า)
2) export ArcFace weight matrix (W) เป็น .npy + metadata (classes, length_mean/std,
   scale, img_size) เป็น .json ให้ฝั่ง Pi ใช้คำนวณ cosine similarity เอง

รัน:
    python export_to_onnx.py --ckpt best_model.pt --out_dir onnx_export
"""
import argparse
import json
import os

import numpy as np
import torch

from evaluate import load_bundle


def export_all(ckpt: str, out_dir: str = "onnx_export", opset: int = 17) -> str:
    """Callable export (used by train_all.py --export) — returns onnx path."""
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] loading checkpoint + building model ...")
    bundle = load_bundle(ckpt, device=torch.device("cpu"))  # export บน CPU พอ ไม่ต้องใช้ GPU
    model = bundle["model"]
    arcface = bundle["arcface"]
    cfg = bundle["cfg"]

    print("[2/4] merging LoRA into backbone (ถ้ามี) ...")
    if hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
        print("      merged LoRA -> backbone กลายเป็น Dinov2Model ธรรมดาแล้ว")
    else:
        print("      ไม่มี LoRA ให้ merge (finetune_mode != 'lora') ข้ามขั้นตอนนี้")
    model.eval()

    print(f"[3/4] exporting ONNX (img_size={cfg.img_size}) ...")
    dummy_px = torch.randn(1, 3, cfg.img_size, cfg.img_size, dtype=torch.float32)
    dummy_len = torch.zeros(1, dtype=torch.float32)

    onnx_path = os.path.join(out_dir, "surgical_dino_fusion.onnx")
    # FIXED batch=1 (no dynamic axes) — faster on ARM/Pi runtimes
    torch.onnx.export(
        model,
        (dummy_px, dummy_len),
        onnx_path,
        input_names=["pixel_values", "length_feat"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"      saved -> {onnx_path}")

    print("[4/4] exporting ArcFace weight + metadata ...")
    # pytorch_metric_learning ArcFaceLoss เก็บ W เป็น shape (embedding_size, num_classes)
    W = arcface.W.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(out_dir, "arcface_W.npy"), W)

    from config import REAL_LENGTH_CM
    meta = {
        "classes": bundle["classes"],
        "length_mean": float(bundle["length_mean"]),
        "length_std": float(bundle["length_std"]),
        "calibration_ratio": bundle["calibration_ratio"],
        "scale": float(cfg.scale),
        "img_size": int(cfg.img_size),
        "real_length_cm": REAL_LENGTH_CM,
    }
    with open(os.path.join(out_dir, "classifier_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nเสร็จแล้ว — เอาไฟล์ทั้งหมดในโฟลเดอร์ '{out_dir}' ไปวางที่ Raspberry Pi:")
    print("  - surgical_dino_fusion.onnx")
    print("  - arcface_W.npy")
    print("  - classifier_meta.json")
    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path ของ best_model.pt")
    ap.add_argument("--out_dir", default="onnx_export")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export_all(args.ckpt, args.out_dir, args.opset)


if __name__ == "__main__":
    main()
