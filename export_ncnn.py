#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_ncnn.py — export DINOv2 + LoRA + CAHM model to ONNX → NCNN

Usage:
  python export_ncnn.py --ckpt model/model_0_4/best_model.pt --out model_0_4
  python export_ncnn.py --ckpt model/model_0_7/best_model.pt --out model_0_7
  # then convert ONNX → NCNN (needs onnx2ncnn from ncnn):
  #   onnx2ncnn model_0_4.onnx model_0_4.param model_0_4.bin

Input for deployment:
  - image: (1,3,560,560) float32, normalized with ImageNet mean/std (same as training)
  - length: (1,) float32, normalized: (length_px - mean)/std
    where length_px = max(w,h) from minAreaRect(mask), mean/std from checkpoint
    If mask not available, use length = mean (i.e. 0 after normalization) — model still works but loses length cue for Needle↔Artery.

Output:
  - logits: (1,14) float32, argmax = predicted class
  - classes order is in checkpoint['classes'] and also saved as {out}.classes.json
"""
import argparse
import json
import pathlib
import torch
import torch.nn as nn

from model import SurgicalDinoFusion, arcface_logits
from pytorch_metric_learning.losses import ArcFaceLoss


class Wrapper(nn.Module):
    def __init__(self, backbone: SurgicalDinoFusion, arcface: ArcFaceLoss):
        super().__init__()
        self.backbone = backbone
        self.arcface = arcface

    def forward(self, image: torch.Tensor, length: torch.Tensor):
        # image: (B,3,H,W)  length: (B,) or (B,1)
        if length.dim() == 2:
            length = length.squeeze(1)
        emb = self.backbone(image, length)
        logits = arcface_logits(self.arcface, emb)
        return logits


def export_one(ckpt_path: str, out_prefix: str, opset: int = 17):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    classes = ckpt["classes"]
    print(f"[load] {ckpt_path}  img={cfg['img_size']}  classes={len(classes)}  epoch={ckpt['epoch']} acc={ckpt['val_acc']:.4f}")

    # build backbone
    model = SurgicalDinoFusion(
        backbone_name=cfg["backbone_name"],
        finetune_mode=cfg["finetune_mode"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        partial_last_blocks=cfg["partial_last_blocks"],
        head_dropout=cfg["head_dropout"],
        use_attention_pool=cfg.get("use_attention_pool", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # merge LoRA (base + lora_A/B -> single Linear) for lighter ONNX graph
    if hasattr(model.backbone, "merge_and_unload"):
        try:
            model.backbone = model.backbone.merge_and_unload()
            print(f"[lora] merged -> {type(model.backbone).__name__}")
        except Exception as e:
            print(f"[lora] merge skip: {e}")

    arcface = ArcFaceLoss(num_classes=len(classes), embedding_size=model.embed_dim, margin=cfg["margin"], scale=cfg["scale"])
    wrapper = Wrapper(model, arcface)
    wrapper.eval()

    # dummy inputs
    H = W = int(cfg["img_size"])
    dummy_img = torch.randn(1, 3, H, W)
    dummy_len = torch.randn(1)

    out_path = pathlib.Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_path.with_suffix(".onnx"))

    # verify forward once
    with torch.no_grad():
        out = wrapper(dummy_img, dummy_len)
        print(f"[verify] logits {out.shape}  sample {out[0,:3].tolist()}")

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_len),
        onnx_path,
        input_names=["image", "length"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "length": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[onnx] saved {onnx_path}  ({pathlib.Path(onnx_path).stat().st_size/1024/1024:.1f} MB)")

    # save classes and stats for runtime
    (out_path.parent / f"{out_path.name}.classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "img_size": H,
        "length_mean": float(ckpt["length_mean"]),
        "length_std": float(ckpt["length_std"]),
        "calibration_ratio": ckpt.get("calibration_ratio"),
        "classes": classes,
    }
    (out_path.parent / f"{out_path.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] {out_path.name}.classes.json / .meta.json")
    print(f"[next] onnx2ncnn {onnx_path} {out_path.with_suffix('.param')} {out_path.with_suffix('.bin')}")
    print(f"       (download onnx2ncnn from https://github.com/Tencent/ncnn/releases)")

    # quick onnx check
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print("[onnx] checker ok")
    except Exception as e:
        print(f"[onnx] checker skip: {e}")

    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_model.pt")
    ap.add_argument("--out", required=True, help="output prefix, e.g. model_0_4/ncnn")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export_one(args.ckpt, args.out, args.opset)


if __name__ == "__main__":
    main()
