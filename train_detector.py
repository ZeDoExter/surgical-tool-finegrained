# -*- coding: utf-8 -*-
"""
train_detector.py — train the DINOv2 detector (NO YOLO)

Loss per patch (40×40 grid):
  - CE over (1+C) channels with background down-weighted (bg_weight=0.25):
    most patches are background — plain CE would teach "predict background
    everywhere"; down-weighting balances FG recall.
  - Dice on the binary fg channel (works well with small FG fraction).

Validation = instance-level on REAL photos: run the full post-processing
(connected components + NMS) on real single-instrument photos (and real
multi-annotation photos when the mix dataset exists) and match to GT via
mask IoU ≥ 0.3 + class match. This is the metric that matters on the Pi.

Usage:
    python train_detector.py --data_dir dataset --epochs 60
    python train_detector.py --validate_only --ckpt outputs_detector/best_detector.pt
"""
import argparse
import math
import os
import random
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import cv2

from config import DetectorConfig
from det_dataset import (BG, DetectorSynthDataset, build_bg_pool,
                         build_patch_pool, make_grid_targets)
from det_model import SurgicalDinoDetector, count_trainable
from det_postprocess import instances_from_logits
from dataset import load_coco_records, mask_from_coco_segmentation


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = min((step - warmup) / max(1, total - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------- losses ----------------
def detector_loss(logits: torch.Tensor, target: torch.Tensor,
                  bg_weight: float = 0.25, ce_weight: float = 1.0,
                  dice_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    logits: (B, 1+C, g, g) | target: (B, 1+C, g, g) in {0,1}
    ch0 = fg mask; ch1..C = per-class one-hot (bg patch → all zeros)
    """
    B, K, g, _ = logits.shape
    # ---- per-patch CE over (bg + C classes) ----
    # build class index target: bg=0, class c → c+1
    cls_tgt = torch.argmax(target[:, 1:], dim=1) + 1        # (B,g,g) in 1..C
    cls_tgt = torch.where(target[:, 0] > 0.5, cls_tgt, torch.zeros_like(cls_tgt))
    ce_map = F.cross_entropy(logits, cls_tgt, reduction="none")  # (B,g,g)
    weights = torch.ones_like(ce_map)
    weights[cls_tgt == 0] = bg_weight
    ce = (ce_map * weights).mean()

    # ---- Dice on fg ----
    fg_logits = logits[:, 0]
    fg_tgt = target[:, 0]
    fg_prob = torch.sigmoid(fg_logits)
    inter = (fg_prob * fg_tgt).sum(dim=(1, 2))
    union = fg_prob.sum(dim=(1, 2)) + fg_tgt.sum(dim=(1, 2))
    dice = 1.0 - ((2 * inter + 1.0) / (union + 1.0)).mean()

    loss = ce_weight * ce + dice_weight * dice
    return loss, {"ce": ce.item(), "dice": dice.item()}


# ---------------- real-photo validation ----------------
def build_real_val_records(data_dir: str) -> List[dict]:
    """Real photos for instance-level validation (test split; falls back to valid)."""
    for split in ("test", "valid"):
        p = os.path.join(data_dir, split, "_annotations.coco.json")
        if os.path.exists(p):
            recs, _ = load_coco_records(data_dir, split)
            return recs
    return []


@torch.no_grad()
def validate_instances(model, records: List[dict], classes: List[str],
                       cfg: DetectorConfig, device, max_images: int = 40) -> dict:
    """
    Full-pipeline validation on real photos: forward → post-process → match.
    Greedy IoU≥0.30 matching (mask-wise), class counted only for matched pairs.
    Returns {"precision", "recall", "f1", "mean_iou", "confusion", counts}.
    """
    model.eval()
    C = len(classes)
    n_det = n_gt = n_tp = 0
    ious = []
    conf = np.zeros((C, C), dtype=np.int64)
    for r in records[:max_images]:
        bgr = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(
            ((cv2.resize(rgb, (cfg.img_size, cfg.img_size)).astype(np.float32) / 255.0
              - np.array([0.485, 0.456, 0.406], dtype=np.float32))
             / np.array([0.229, 0.224, 0.225], dtype=np.float32)).transpose(2, 0, 1)[None]
        ).to(device)
        logits = model(x)[0].float().cpu().numpy()
        insts = instances_from_logits(
            logits, classes, frame_rgb=None,
            mask_threshold=cfg.mask_threshold,
            min_instance_area=cfg.min_instance_area,
            nms_iou=cfg.nms_iou,
            conf_min_score=cfg.conf_min_score,
            want_tip_crops=False,
        )
        # GT: ALL annotations of this image (single- or multi-instrument photos);
        # label = sorted class index (same numbering as the classes list)
        from dataset import load_coco_annotations_for_image
        gt_insts = []
        for a in load_coco_annotations_for_image(r):
            m = mask_from_coco_segmentation(a["segmentation"], r["height"], r["width"])
            if np.sum(m > 0) < cfg.min_mask_area_px:
                continue
            m_rs = cv2.resize(m, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_NEAREST)
            gt_insts.append({"mask": m_rs, "label": a["label"]})
        n_gt += len(gt_insts)

        # greedy match detections → GT
        pairs = []
        for di, inst in enumerate(insts):
            dm = inst["mask"] > 0
            for gi, gt in enumerate(gt_insts):
                gm = gt["mask"] > 0
                inter = np.logical_and(dm, gm).sum()
                union = np.logical_or(dm, gm).sum()
                iou = inter / max(union, 1)
                if iou >= 0.30:
                    pairs.append((iou, di, gi))
        pairs.sort(reverse=True)
        used_d, used_g = set(), set()
        for iou, di, gi in pairs:
            if di in used_d or gi in used_g:
                continue
            used_d.add(di); used_g.add(gi)
            n_tp += 1
            ious.append(iou)
            conf[gt_insts[gi]["label"], insts[di]["label"]] += 1
        n_det += len(insts)

    prec = n_tp / max(n_det, 1)
    rec = n_tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "confusion": conf, "n_gt": n_gt, "n_det": n_det, "n_tp": n_tp}


# ---------------- main ----------------
def run_training(cfg: DetectorConfig, val_records: Optional[List[dict]] = None) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[data] building patch pool + backgrounds ...")
    pool, classes = build_patch_pool(cfg.data_dir, min_area=cfg.min_mask_area_px)
    bg_pool = build_bg_pool(cfg.data_dir, max_n=24)
    print(f"[data] patches={len(pool)} classes={len(classes)} bg_pool={len(bg_pool)}")
    if not pool:
        raise RuntimeError("patch pool is empty — check dataset/train")

    # real records for the "real-scene" mix inside training (and val on real photos)
    tr_records, _ = load_coco_records(cfg.data_dir, "train")
    va_records, _ = (load_coco_records(cfg.data_dir, "valid")
                     if os.path.exists(os.path.join(cfg.data_dir, "valid", "_annotations.coco.json"))
                     else ([], None))
    real_records = list(tr_records) + list(va_records or [])
    if val_records is None:
        val_records = build_real_val_records(cfg.data_dir)
    if not val_records:
        val_records = real_records[:40]
    print(f"[data] real records for scene mix={len(real_records)} | val={len(val_records)}")

    grid = cfg.img_size // 14
    ds = DetectorSynthDataset(
        pool, classes, img_size=cfg.img_size, grid=grid, num_classes=len(classes),
        training=True, bg_pool=bg_pool,
        synth_min_objects=cfg.synth_min_objects, synth_max_objects=cfg.synth_max_objects,
        synth_same_class_prob=cfg.synth_same_class_prob,
        synth_scale_range=cfg.synth_scale_range, synth_green_prob=cfg.synth_green_prob,
        synth_shadows=cfg.synth_shadows, synth_max_overlap=cfg.synth_max_overlap,
        min_mask_area_px=cfg.min_mask_area_px,
        real_records=real_records, real_prob=0.35,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                    collate_fn=None)

    model = SurgicalDinoDetector(
        backbone_name=cfg.backbone_name, finetune_mode=cfg.finetune_mode,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        partial_last_blocks=cfg.partial_last_blocks,
        num_classes=len(classes), img_size=cfg.img_size,
        decoder_dim=cfg.decoder_dim, decoder_mid_dim=cfg.decoder_mid_dim,
        decoder_dropout=cfg.decoder_dropout, use_mid_feats=cfg.use_mid_feats,
    ).to(device)
    print(f"[model] trainable={count_trainable(model):,} (mode={cfg.finetune_mode})")

    lr_bb = cfg.lr_backbone if cfg.finetune_mode == "partial" else (
        cfg.lr_lora if cfg.finetune_mode == "lora" else None)
    optimizer = AdamW(model.param_groups(cfg.lr_head, lr_bb), weight_decay=cfg.weight_decay)
    total_steps = max(1, len(dl)) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: warmup_cosine_factor(s, warmup_steps, total_steps))
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    best = {"f1": -1.0, "epoch": -1, "state": None}
    bad = 0
    history = {"train_loss": [], "val_f1": [], "val_prec": [], "val_rec": []}
    ckpt_path = os.path.join(cfg.output_dir, "best_detector.pt")

    print(f"[train] steps/epoch={len(dl)} epochs={cfg.epochs}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        seen = 0
        for batch in dl:
            px = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast("cuda"):
                    logits = model(px)
                    loss, parts = detector_loss(logits, tgt, cfg.bg_weight,
                                                cfg.ce_weight, cfg.dice_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(px)
                loss, parts = detector_loss(logits, tgt, cfg.bg_weight,
                                            cfg.ce_weight, cfg.dice_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()
            tot += loss.item() * px.size(0)
            seen += px.size(0)
        tr_loss = tot / max(seen, 1)

        metrics = validate_instances(model, val_records, classes, cfg, device)
        history["train_loss"].append(tr_loss)
        history["val_f1"].append(metrics["f1"])
        history["val_prec"].append(metrics["precision"])
        history["val_rec"].append(metrics["recall"])
        improved = metrics["f1"] > best["f1"] + 1e-4
        star = "  *best*" if improved else ""
        if improved:
            best.update(f1=metrics["f1"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad = 0
            torch.save({
                "model_state": best["state"],
                "classes": classes,
                "cfg": cfg.to_dict(),
                "epoch": epoch,
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            }, ckpt_path)
        else:
            bad += 1
        print(f"[epoch {epoch:03d}/{cfg.epochs}] loss={tr_loss:.4f} "
              f"(ce={parts['ce']:.3f} dice={parts['dice']:.3f}) "
              f"val: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} IoU={metrics['mean_iou']:.3f}{star}", flush=True)
        if bad >= cfg.patience:
            print(f"[early stop] patience {cfg.patience}")
            break

    # final save if never improved (keep last state so export always possible)
    if best["state"] is None:
        torch.save({
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "classes": classes, "cfg": cfg.to_dict(), "epoch": epoch,
            "val_f1": 0.0, "val_precision": 0.0, "val_recall": 0.0,
        }, ckpt_path)
    print(f"[done] best epoch={best['epoch']} F1={best['f1']:.4f} → {ckpt_path}")
    return ckpt_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train DINOv2 detector (YOLO-free)")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--finetune_mode", choices=["lora", "partial", "frozen"], default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = replace(DetectorConfig(), **overrides)
    run_training(cfg)


if __name__ == "__main__":
    main()
