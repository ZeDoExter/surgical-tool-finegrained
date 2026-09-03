# -*- coding: utf-8 -*-
"""
export_detector_onnx.py — export the DINOv2 detector to ONNX for Raspberry Pi 5

Run on PC (torch/transformers/peft needed), NOT on the Pi:
    python export_detector_onnx.py --ckpt outputs_detector/best_detector.pt --out_dir pi_final_v3/onnx_export

Outputs:
  - detector_dino.onnx   : pixel_values (1,3,560,560) -> logits (1,15,560,560)
                           (ch0 = fg, ch1..14 = classes; sigmoid + instances in
                            det_postprocess.py on the Pi)
  - detector_meta.json   : classes, img_size, post-processing thresholds,
                           real lengths (cm), calibration_ratio (from calibrate.py)
LoRA is merged into the backbone before export -> clean Dinov2Model graph.
"""
import argparse
import json
import os

import numpy as np
import torch

from config import REAL_LENGTH_CM
from det_model import load_detector


def export_int8(fp32_path: str, out_dir: str, data_dir: str = "dataset") -> str:
    """
    Post-training static INT8 quantization for CPU (Pi 5).

    Calibrates on synthetic scenes from the SAME generator the detector
    trained on (domain-matched), then writes detector_dino_int8.onnx.
    Reads the model input size from the ONNX graph (works for 560/448).
    Falls back gracefully (returns fp32 path) if quantization deps missing.
    """
    try:
        import onnx
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                              QuantType, quantize_static)
    except ImportError:
        print("[int8] onnxruntime.quantization unavailable -> skip")
        return fp32_path

    # model input size from the graph itself
    model = onnx.load(fp32_path)
    in_dim = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    img_size = int(in_dim[-1]) if len(in_dim) >= 4 and in_dim[-1] > 0 else 560
    print(f"[int8] model input size: {img_size}")

    # calibration data: 40 synthetic multi-tool scenes through real preprocessing
    import cv2
    import numpy as np
    import random
    from det_dataset import build_patch_pool, synth_scene
    from det_postprocess import preprocess_frame

    pool, classes = build_patch_pool(data_dir, min_area=80)
    rng = random.Random(0)

    class SceneReader(CalibrationDataReader):
        def __init__(self, n=40):
            self.i = 0
            self.n = n
            self.data = []
            for _ in range(n):
                img, _ = synth_scene(pool, img_size, img_size, rng, min_objects=2, max_objects=4)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.data.append({"pixel_values": preprocess_frame(img_bgr, img_size)})

        def get_next(self):
            if self.i >= self.n:
                return None
            item = self.data[self.i]
            self.i += 1
            return item

    int8_path = os.path.join(out_dir, "detector_dino_int8.onnx")
    quantize_static(
        fp32_path, int8_path,
        SceneReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"[int8] saved -> {int8_path}")
    return int8_path


def export_all(ckpt_path: str, out_dir: str, opset: int = 17,
               int8: bool = True, data_dir: str = "dataset") -> str:
    """Callable export (used by train_all.py --export) — returns onnx path."""
    args = argparse.Namespace(ckpt=ckpt_path, out_dir=out_dir, opset=opset)

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

    if int8:
        export_int8(onnx_path, args.out_dir, data_dir=data_dir)
    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="pi_final_v3/onnx_export")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no_int8", action="store_true", help="skip INT8 quantization")
    args = ap.parse_args()
    export_all(args.ckpt, args.out_dir, args.opset, int8=not args.no_int8)
    print("\nDone — copy the whole folder to the Pi (pi_final_v3/onnx_export).")


if __name__ == "__main__":
    main()
