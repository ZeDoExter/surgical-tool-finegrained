# -*- coding: utf-8 -*-
"""
build_prototypes.py — per-class centroids of detector backbone features (PC/Colab).

Principle (DINOv2 linear probing): last-layer patch tokens are linearly
separable by class. For each train instance we mean-pool the patch grid over
its GT mask, then average per class -> (384, 14) normalized prototypes.

On the Pi this gives a ZERO-cost label for every detected instance (no 2nd
ViT forward): fast_emb = mean(pool(tokens, mask)); class = argmax cos.
Ambiguous instances fall back to the ArcFace classifier (cascade).

Usage:
    python build_prototypes.py --ckpt outputs_detector/best_detector.pt \
        --data_dir dataset --img_size 448 --out_dir pi_final_v3/onnx_export
"""
import argparse
import json
import os

import numpy as np
import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--out_dir", default="pi_final_v3/onnx_export")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    from det_model import load_detector
    from dataset import load_coco_records, mask_from_coco_segmentation

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_detector(args.ckpt, device=device)
    model = bundle["model"]
    classes = bundle["classes"]
    C = len(classes)
    model.eval()

    grid = args.img_size // 14
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    import cv2
    recs, _ = load_coco_records(args.data_dir, "train")
    sums = {c: np.zeros(384, dtype=np.float64) for c in classes}
    counts = {c: 0 for c in classes}

    @torch.no_grad()
    def pooled_emb(rgb, mask):
        x = cv2.resize(rgb, (args.img_size, args.img_size)).astype(np.float32) / 255.0
        x = (x - mean) / std
        x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(device)
        out = model.backbone(pixel_values=x.to(torch.float32),
                             output_hidden_states=True)
        last = out.last_hidden_state[:, 1:, :].float()          # (1, G*G, 384)
        toks = last.transpose(1, 2).reshape(1, 384, grid, grid)  # (1,384,G,G)
        mm = cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_NEAREST) > 0
        if mm.sum() < 3:
            return None
        v = toks[0][:, mm].mean(dim=1).cpu().numpy()
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    for i, r in enumerate(recs):
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = mask_from_coco_segmentation(r["segmentation"], r["height"], r["width"])
        v = pooled_emb(rgb, m)
        if v is None:
            continue
        sums[r["class_name"]] += v
        counts[r["class_name"]] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(recs)}")

    P = np.zeros((384, C), dtype=np.float32)
    for j, c in enumerate(classes):
        if counts[c] > 0:
            v = sums[c] / counts[c]
            P[:, j] = v / (np.linalg.norm(v) + 1e-8)
        print(f"  {c:36s} n={counts[c]}")

    proto_path = os.path.join(args.out_dir, "class_prototypes.npy")
    np.save(proto_path, P)
    print(f"saved -> {proto_path} {P.shape}")

    meta_path = os.path.join(args.out_dir, "detector_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["prototypes"] = "class_prototypes.npy"
        meta["prototypes_built_at"] = args.img_size
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("meta updated with prototypes entry")


if __name__ == "__main__":
    main()
