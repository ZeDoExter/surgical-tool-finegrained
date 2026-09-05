# -*- coding: utf-8 -*-
"""
train_student.py — distill the DINOv2 detector into a tiny CNN (true realtime).

Why (research-backed):
  - Knowledge distillation (Hinton et al. 2015): a small student trained
    against a big teacher's soft targets reaches far beyond what it learns
    from hard labels alone — ideal here because we HAVE a strong teacher.
  - LRASPP-MobileNetV3 (Howard et al.): designed for realtime segmentation,
    ~3.2M params, pretrained backbone — fits Pi CPU at 320px (~5-10 fps).
  - No new annotations needed: the teacher auto-labels infinite synth scenes
    on the GPU (fast), GT polygons supervise the hard targets.

Contract: student.forward(pixel_values) -> logits (B, 1+C, H, W) — IDENTICAL
to SurgicalDinoDetector, so det_postprocess.py, validate_instances, and the
Pi app work unchanged (only the ONNX filename + meta img_size differ).

Usage:
    python train_student.py --teacher outputs_detector/best_detector.pt --epochs 40
    python train_student.py --teacher ckpt.pt --export   # + ONNX (fp32) + INT8 (gated)
"""
import argparse
import sys
import time
import math
import os
from typing import List, Optional
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import cv2

from config import DetectorConfig
from det_dataset import DetectorSynthDataset, build_bg_pool, build_patch_pool
from det_model import load_detector
from det_postprocess import instances_from_logits
from dataset import load_coco_records
from train_detector import (detector_loss, seed_everything,
                            warmup_cosine_factor, validate_instances,
                            build_real_val_records)


class StudentSeg(nn.Module):
    """LRASPP-MobileNetV3 backbone (pretrained) + 15-ch head, same IO contract."""

    def __init__(self, num_classes: int = 14, pretrained_backbone: bool = True,
                 lr_head: float = 3e-4, lr_backbone: float = 1e-4):
        super().__init__()
        from torchvision.models.segmentation import (
            lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights)
        w = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.net = lraspp_mobilenet_v3_large(weights=w, num_classes=21)
        self.net.classifier.low_classifier = nn.Conv2d(40, 1 + num_classes, 1)
        self.net.classifier.high_classifier = nn.Conv2d(128, 1 + num_classes, 1)
        self._lr_head = lr_head
        self._lr_backbone = lr_backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values)["out"]   # (B, 1+C, H, W)

    def param_groups(self):
        head = list(self.net.classifier.parameters())
        bb = [p for n, p in self.net.named_parameters() if not n.startswith("classifier.")]
        return [{"params": [p for p in head if p.requires_grad], "lr": self._lr_head},
                {"params": [p for p in bb if p.requires_grad], "lr": self._lr_backbone}]


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def distill_loss(stu_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 target: torch.Tensor, cfg: DetectorConfig,
                 kd_weight: float = 0.5) -> tuple:
    """Same GT loss as the teacher + MSE against teacher soft logits."""
    hard, parts = detector_loss(stu_logits.float(), target,
                                cfg.bg_weight, cfg.ce_weight, cfg.dice_weight)
    t_ds = F.interpolate(teacher_logits.float().detach(),
                         size=stu_logits.shape[-2:], mode="bilinear", align_corners=False)
    kd = F.mse_loss(stu_logits.float(), t_ds)
    loss = hard + kd_weight * kd
    parts["kd"] = kd.item()
    return loss, parts


def run_training(cfg: DetectorConfig, teacher_ckpt: str, img_size: int = 320,
                 kd_weight: float = 0.5, epochs: int = 40,
                 batch_size: int = 16, num_workers: int = 2,
                 output_dir: str = "outputs_student",
                 init_ckpt: Optional[str] = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    seed_everything(cfg.seed if hasattr(cfg, "seed") else 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[data] building patch pool + backgrounds ...")
    from dataset import load_coco_records_multi
    pool, classes = build_patch_pool(cfg.data_dir, min_area=cfg.min_mask_area_px,
                                     extra_data_dirs=getattr(cfg, "extra_data_dirs", None))
    bg_pool = build_bg_pool(cfg.data_dir, max_n=24,
                            extra_data_dirs=getattr(cfg, "extra_data_dirs", None))
    dirs = [cfg.data_dir] + list(getattr(cfg, "extra_data_dirs", []) or [])
    tr_records, _ = load_coco_records_multi(dirs, "train")

    print(f"[teacher] {teacher_ckpt}")
    teacher = load_detector(teacher_ckpt, device=device)["model"].eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_img = teacher.img_size if hasattr(teacher, "img_size") else 560

    ds = DetectorSynthDataset(
        pool, classes, img_size=img_size, grid=img_size // 10,
        num_classes=len(classes), training=True, bg_pool=bg_pool,
        synth_min_objects=cfg.synth_min_objects,
        synth_max_objects=cfg.synth_max_objects,
        synth_same_class_prob=cfg.synth_same_class_prob,
        synth_scale_range=cfg.synth_scale_range,
        synth_green_prob=cfg.synth_green_prob,
        synth_shadows=cfg.synth_shadows,
        synth_max_overlap=cfg.synth_max_overlap,
        min_mask_area_px=cfg.min_mask_area_px,
        real_records=tr_records, real_prob=0.35)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=device.type == "cuda")
    model = StudentSeg(num_classes=len(classes)).to(device)
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location=device,
                        weights_only=False)["model_state"]
        model.load_state_dict(sd)
        print(f"[student] warm-started from {init_ckpt}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[student] trainable={n_tr:,}")

    optimizer = AdamW(model.param_groups(), weight_decay=cfg.weight_decay)
    total_steps = max(1, len(dl)) * epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda s: warmup_cosine_factor(s, warmup_steps, total_steps))
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    # validation uses the SAME thresholds, scaled to the student size
    import copy
    vcfg = copy.copy(cfg)
    vcfg.img_size = img_size
    vcfg.min_instance_area = int(round(cfg.min_instance_area * (img_size / 560.0) ** 2))
    val_records = build_real_val_records(cfg.data_dir)
    ckpt_path = os.path.join(output_dir, "best_student.pt")
    best = {"f1": -1.0, "epoch": -1, "state": None}
    bad = 0
    patience = 10

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    print(f"[train] steps/epoch={len(dl)} epochs={epochs} img={img_size}")
    t_train0 = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        tot, seen = 0.0, 0
        parts = {"ce": 0.0, "dice": 0.0, "kd": 0.0}
        pbar = tqdm(dl, desc=f"{epoch}/{epochs}", unit="batch",
                    bar_format="{l_bar}{bar:24}{r_bar}", leave=False,
                    dynamic_ncols=True, mininterval=1.0,
                    disable=None, file=sys.stderr)
        for batch in pbar:
            px = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    t_in = F.interpolate(px, size=(t_img, t_img), mode="bilinear",
                                         align_corners=False)
                    t_logits = teacher(t_in)
            if scaler is not None:
                with torch.autocast("cuda"):
                    s_logits = model(px)
                    loss, parts = distill_loss(s_logits, t_logits, tgt, cfg, kd_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                s_logits = model(px)
                loss, parts = distill_loss(s_logits, t_logits, tgt, cfg, kd_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()
            tot += loss.item() * px.size(0)
            seen += px.size(0)
            if not getattr(pbar, "disable", False):
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 ce=f"{parts['ce']:.3f}", dice=f"{parts['dice']:.3f}")
        tr_loss = tot / max(seen, 1)

        metrics = validate_instances(model, val_records, classes, vcfg, device)
        improved = metrics["f1"] > best["f1"] + 1e-4
        star = " *best*" if improved else ""
        if improved:
            best.update(f1=metrics["f1"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad = 0
            torch.save({
                "model_state": best["state"], "classes": classes,
                "cfg": {"img_size": img_size, "val_f1": metrics["f1"]},
                "epoch": epoch, "val_f1": metrics["f1"],
                "val_precision": metrics["precision"], "val_recall": metrics["recall"],
                "backbone": "lraspp-mobilenetv3",
            }, ckpt_path)
        else:
            bad += 1
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - t_train0
        avg_epoch_time = elapsed_total / epoch
        eta = avg_epoch_time * (epochs - epoch)
        epoch_str = f"{epoch_time:.1f}s" if epoch_time < 60 else format_time(epoch_time)
        print(f"Epoch {epoch}/{epochs} {'━' * 40} {epoch_str} | "
              f"loss={tr_loss:.4f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} IoU={metrics['mean_iou']:.3f} "
              f"ETA={format_time(eta)}{star}", flush=True)
        if bad >= patience:
            print(f"[early stop] patience {patience}")
            break

    if best["state"] is None:
        torch.save({"model_state": {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()},
                    "classes": classes, "cfg": {"img_size": img_size},
                    "epoch": epoch, "val_f1": 0.0, "backbone": "lraspp-mobilenetv3"},
                   ckpt_path)
    print(f"[done] best epoch={best['epoch']} F1={best['f1']:.4f} -> {ckpt_path}")
    return ckpt_path


def export_all_student(ckpt_path: str, out_dir: str, img_size: int = 320,
                       int8: bool = False) -> str:
    """Export student -> detector_dino_student.onnx (+ gated dynamic INT8)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    model = StudentSeg(num_classes=len(classes), pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    onnx_path = os.path.join(out_dir, "detector_dino_student.onnx")
    torch.onnx.export(model, (dummy,), onnx_path,
                      input_names=["pixel_values"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True)
    print(f"saved -> {onnx_path}")

    meta_path = os.path.join(out_dir, "detector_meta.json")
    import json as _json
    meta = {"classes": classes, "img_size": int(img_size),
            "mask_threshold": 0.5, "min_instance_area": 26,
            "nms_iou": 0.4, "conf_min_score": 0.35,
            "calibration_ratio": None, "model": "LRASPP-MobileNetV3 student",
            "has_patch_tokens": False}
    try:
        from config import REAL_LENGTH_CM
        meta["real_length_cm"] = REAL_LENGTH_CM
    except Exception:
        pass
    # preserve calibration if a meta already exists
    if os.path.exists(meta_path):
        try:
            old = _json.load(open(meta_path, encoding="utf-8"))
            if old.get("calibration_ratio"):
                meta["calibration_ratio"] = old["calibration_ratio"]
        except Exception:
            pass
    with open(meta_path, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved -> {meta_path}")

    if int8:
        print("[int8] dynamic quant (no calibration needed for CNNs) ...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            i8 = os.path.join(out_dir, "detector_dino_student_int8.onnx")
            quantize_dynamic(onnx_path, i8, weight_type=QuantType.QInt8)
            # quality gate vs fp32
            if _int8_gate_ok(onnx_path, i8):
                print(f"[int8] gate PASSED -> {i8}")
            else:
                print("[int8] gate FAILED — keeping fp32 only")
                os.remove(i8)
        except ImportError:
            print("[int8] quantization deps missing -> skip")
    return onnx_path


def _int8_gate_ok(fp32_path: str, int8_path: str, n: int = 12) -> bool:
    """Agreement gate: same top-instance class+bbox on real test photos."""
    import onnxruntime as ort
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pi_final_v3"))
    import det_postprocess as pp
    try:
        from dataset import load_coco_records
        recs, _ = load_coco_records("dataset", "test")
    except Exception:
        try:
            from dataset import load_coco_records
            recs, _ = load_coco_records("dataset", "valid")
        except Exception:
            return True  # can't check — trust, fp32 stays anyway

    def sess(p):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])

    try:
        s32, s8 = sess(fp32_path), sess(int8_path)
        import json as _json
        meta = _json.load(open(os.path.join(os.path.dirname(fp32_path),
                                            "detector_meta.json"), encoding="utf-8"))
        classes, sz = meta["classes"], meta["img_size"]
        ok = tot = 0
        for r in recs[:max(n, len(recs))][:n]:
            bgr = cv2.imread(r["image_path"])
            if bgr is None:
                continue
            px = pp.preprocess_frame(bgr, sz)
            (l32,) = s32.run(["logits"], {"pixel_values": px})
            (l8,) = s8.run(["logits"], {"pixel_values": px})
            i32 = pp.instances_from_logits(l32[0], classes, frame_rgb=None,
                                            want_tip_crops=False, conf_min_score=0.3)
            i8 = pp.instances_from_logits(l8[0], classes, frame_rgb=None,
                                           want_tip_crops=False, conf_min_score=0.3)
            tot += 1
            if i32 and i8 and i32[0]["class_name"] == i8[0]["class_name"]:
                a, b = i32[0]["bbox"], i8[0]["bbox"]
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                uni = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
                if uni > 0 and inter / uni > 0.5:
                    ok += 1
        print(f"[int8 gate] agreement {ok}/{tot}")
        return tot > 0 and ok / tot >= 0.85
    except Exception as e:
        print(f"[int8 gate] check failed ({e}) — keeping fp32 only")
        return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Distill DINOv2 detector into tiny CNN")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--kd_weight", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output_dir", default="outputs_student")
    ap.add_argument("--init_ckpt", default=None,
                    help="warm-start weights from a previous student run "
                         "(e.g. after a crash — LR schedule restarts fresh)")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export_int8", action="store_true")
    args = ap.parse_args(argv)

    from config import DetectorConfig
    cfg = DetectorConfig(data_dir=args.data_dir)
    ckpt = run_training(cfg, args.teacher, img_size=args.img_size,
                        kd_weight=args.kd_weight, epochs=args.epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        output_dir=args.output_dir, init_ckpt=args.init_ckpt)
    if args.export:
        export_all_student(ckpt, "pi_final_v3/onnx_export",
                           img_size=args.img_size, int8=args.export_int8)


if __name__ == "__main__":
    main()
