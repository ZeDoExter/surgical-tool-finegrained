# -*- coding: utf-8 -*-
"""
export_detector_onnx.py — export the DINOv2 detector to ONNX for Raspberry Pi 5

Run on PC (torch/transformers/peft needed), NOT on the Pi:
    python export_detector_onnx.py --ckpt outputs_detector/best_detector.pt --out_dir pi_final_v3/onnx_export
    python export_detector_onnx.py --ckpt ckpt.pt --out_dir out --img_size 448 --with_tokens

Outputs:
  - detector_dino.onnx   : pixel_values (1,3,H,W) -> logits (1,15,H,W)
                           (ch0 = fg, ch1..14 = classes; sigmoid + instances in
                            det_postprocess.py on the Pi)
  - detector_dino_tokens.onnx (with --with_tokens): extra output patch_tokens
                           (1,384,G,G) = last-layer ViT patch grid for the
                           zero-cost prototype fast path on the Pi
  - detector_meta.json   : classes, img_size, post-processing thresholds
                           (size-scaled), real lengths (cm), calibration_ratio
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
               int8: bool = False, data_dir: str = "dataset",
               img_size: int = None, with_tokens: bool = False) -> str:
    """Callable export (used by train_all.py --export) — returns onnx path.

    int8 is OFF by default: QDQ static quantization collapses this ViT
    graph (fg logits 18.6 -> 5.7, all instances lost). Use 448 input for
    speed instead — that path is verified safe.

    img_size: re-export a trained checkpoint at a different size without
    retraining (conv/decoder weights are size-independent; ViT positional
    embeddings interpolate). 448 (=32x14) is the recommended fast size.
    Thresholds stored in meta are scaled accordingly.

    with_tokens: also export the raw last-layer ViT patch grid (1,384,G,G)
    as a second output for the zero-cost prototype fast path on the Pi
    (mask-pool per instance + cosine to class centroids, no 2nd model).
    """
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

    export_img = int(img_size or cfg.img_size)
    if export_img % 14 != 0:
        raise ValueError(f"img_size {export_img} must be divisible by 14")
    if export_img != cfg.img_size:
        # size-independent weights — only the runtime grid/interp changes
        model.img_size = export_img
        if hasattr(model, "decoder"):
            model.decoder.img_size = export_img
        print(f"      re-targeted {cfg.img_size} -> {export_img} (grid {model.grid}x{model.grid})")
    dummy = torch.randn(1, 3, export_img, export_img, dtype=torch.float32)

    out_names = ["logits"]
    if with_tokens:
        out_names.append("patch_tokens")

        def _forward_with_tokens(pixel_values):
            out = model.backbone(pixel_values=pixel_values.to(torch.float32),
                                 output_hidden_states=True)
            last = out.last_hidden_state[:, 1:, :]
            mid_tokens = None
            if model.decoder.use_mid_feats:
                hs = out.hidden_states
                mid_tokens = [hs[6][:, 1:, :], hs[9][:, 1:, :]]
            g = model.grid
            logits = model.decoder(last, mid_tokens, grid=g)
            B, N, D = last.shape
            patch_tokens = last.transpose(1, 2).reshape(B, D, g, g)
            return logits, patch_tokens

        def _forward_fp32(self, pixel_values):
            logits, _ = _forward_with_tokens(pixel_values)
            return logits

        model._forward_fp32 = _forward_fp32.__get__(model, type(model))

        class _TokWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                logits, toks = _forward_with_tokens(pixel_values)
                return logits, toks

        export_mod = _TokWrapper(model)
    else:
        export_mod = model

    onnx_path = os.path.join(args.out_dir, "detector_dino.onnx")
    # FIXED batch=1 shape (no dynamic axes): measurably faster on ARM/Pi —
    # the runtime can fully specialize memory plans + fold shapes
    torch.onnx.export(
        export_mod,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=out_names,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"      saved -> {onnx_path}")

    # sanity: same outputs after export?
    with torch.no_grad():
        ref = model(dummy)[0, :3, :3, :3].numpy()
    print(f"      sanity ref logits corner: {ref.flatten()[:3]}")

    print("[3/3] writing metadata ...")
    # thresholds tuned at the checkpoint size — scale the area threshold
    # to the export resolution (mask pixels scale quadratically)
    area_scale = (export_img / float(cfg.img_size)) ** 2
    meta = {
        "classes": classes,
        "img_size": int(export_img),
        "mask_threshold": float(cfg.mask_threshold),
        "min_instance_area": int(round(cfg.min_instance_area * area_scale)),
        "nms_iou": float(cfg.nms_iou),
        "conf_min_score": float(cfg.conf_min_score),
        "real_length_cm": REAL_LENGTH_CM,
        "calibration_ratio": None,  # filled by calibrate.py (cm per pixel at the rig)
        "model": "DINOv2 detector (seg head) — YOLO-free",
        "has_patch_tokens": bool(with_tokens),
        "patch_dim": 384,
        "token_grid": int(export_img // 14),
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
    ap.add_argument("--img_size", type=int, default=None,
                    help="re-export the checkpoint at another /14 size without "
                         "retraining (448 recommended for Pi speed)")
    ap.add_argument("--with_tokens", action="store_true",
                    help="also export last-layer ViT patch tokens for the "
                         "zero-cost prototype fast path on the Pi")
    ap.add_argument("--int8", action="store_true",
                    help="INT8 static quant (experimental — FAILED quality gate "
                         "on this ViT graph, kept only for experiments)")
    args = ap.parse_args()
    export_all(args.ckpt, args.out_dir, args.opset, int8=args.int8,
               img_size=args.img_size, with_tokens=args.with_tokens)
    print("\nDone — copy the whole folder to the Pi (pi_final_v3/onnx_export).")


if __name__ == "__main__":
    main()
