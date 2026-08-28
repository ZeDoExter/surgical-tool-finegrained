# -*- coding: utf-8 -*-
"""
evaluate.py — ประเมิน checkpoint ที่เทรนแล้ว

ใช้จาก notebook/สคริปต์:
    from evaluate import evaluate_checkpoint
    metrics = evaluate_checkpoint("outputs/best_model.pt")

ได้ทั้ง accuracy รวม, classification report ราย class, confusion matrix
(heatmap) และ "คู่ class ที่สับสนบ่อยสุด" ซึ่งมักเป็นคู่ที่ต่างกันแค่ขนาด
"""
import os
from dataclasses import replace
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)

from config import TrainConfig
from dataset import SurgicalInstrumentDataset
from model import SurgicalDinoFusion, arcface_logits
from pytorch_metric_learning.losses import ArcFaceLoss
from train import resolve_records, torch_load_compat



def load_bundle(ckpt_path: str, device: Optional[torch.device] = None) -> dict:
    """
    โหลด checkpoint → สร้าง model + ArcFace head ให้พร้อม inference
    (cfg ถูกเก็บไว้ใน checkpoint เวลาเทรน → reproduce โครงสร้างได้เป๊ะ)
    """
    ckpt = torch_load_compat(ckpt_path)
    cfg = TrainConfig(**ckpt["cfg"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ต้องสร้างด้วย finetune_mode เดิม เพื่อให้ชื่อ key ของ state_dict ตรงกัน
    model = SurgicalDinoFusion(
        backbone_name=cfg.backbone_name, finetune_mode=cfg.finetune_mode,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        partial_last_blocks=cfg.partial_last_blocks, head_dropout=cfg.head_dropout,
        use_attention_pool=getattr(cfg, "use_attention_pool", True),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    classes: List[str] = ckpt["classes"]
    arcface = ArcFaceLoss(num_classes=len(classes), embedding_size=model.embed_dim,
                          margin=cfg.margin, scale=cfg.scale)
    arcface.load_state_dict(ckpt["arcface_state"])
    arcface.to(device)

    return {"model": model, "arcface": arcface, "classes": classes, "cfg": cfg,
            "device": device,
            "length_mean": float(ckpt["length_mean"]), "length_std": float(ckpt["length_std"]),
            "calibration_ratio": ckpt.get("calibration_ratio")}


@torch.no_grad()
def predict_all(bundle: dict, records: List[dict]):
    """รันทั้ง validation set → (y_true, y_pred, confidence ของ class ที่ทำนาย)"""
    cfg = bundle["cfg"]
    device = bundle["device"]
    length_stats = (bundle["length_mean"], bundle["length_std"])
    ds = SurgicalInstrumentDataset(records, length_stats, cfg.img_size,
                                   cfg.calibration_ratio, flip_flags=None, training=False,
                                   bbox_margin=getattr(cfg, "bbox_margin", 0.0))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    y_true, y_pred, y_conf = [], [], []
    for batch in dl:
        px = batch["image"].to(device)
        ln = batch["length"].to(device)
        emb = bundle["model"](px, ln)
        logits = arcface_logits(bundle["arcface"], emb.float())
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        y_pred += pred.cpu().tolist()
        y_conf += conf.cpu().tolist()
        y_true += batch["label"].tolist()
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None, figsize=(12, 10)):
    """heatmap ของ confusion matrix (seaborn ถ้ามี, ไม่มีก็ matplotlib ล้วน)"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    except ImportError:
        im = ax.imshow(cm, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[log] บันทึก confusion matrix → {save_path}")
    return fig


def print_top_confused(cm: np.ndarray, class_names: List[str], top_n: int = 10) -> None:
    """พิมพ์คู่ class ที่สับสนบ่อยสุด (จริง → ทำนาย) — จุดที่ต้องแก้ต่อ เช่น เพิ่มฟีเจอร์ขนาด"""
    pairs = [(int(cm[i, j]), i, j)
             for i in range(len(class_names)) for j in range(len(class_names))
             if i != j and cm[i, j] > 0]
    if not pairs:
        print("ไม่มีความสับสนข้าม class เลย 🎉")
        return
    pairs.sort(reverse=True)
    print("\nคู่ class ที่สับสนบ่อยสุด (จริง → ทำนาย):")
    for cnt, i, j in pairs[:top_n]:
        print(f"  {class_names[i]} → {class_names[j]} : {cnt} ครั้ง")


def evaluate_checkpoint(ckpt_path: str, data_dir: Optional[str] = None,
                        show_plot: bool = True, save_dir: Optional[str] = None) -> dict:
    """
    ประเมิน checkpoint บน validation set → dict ที่มี
      accuracy / balanced_accuracy / report / confusion_matrix / cm_path / fig
    """
    bundle = load_bundle(ckpt_path)
    cfg = bundle["cfg"]
    if data_dir:
        cfg = replace(cfg, data_dir=data_dir)
    _, va_records, _ = resolve_records(cfg)
    if len(va_records) == 0:
        raise ValueError("validation set ว่าง — เช็ค data_dir/val_fraction")

    y_true, y_pred, _ = predict_all(bundle, va_records)
    classes = bundle["classes"]
    acc = float((y_true == y_pred).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    present = sorted(set(y_true.tolist()))
    names_present = [classes[i] for i in present]
    print(f"\n===== Evaluation ({len(va_records)} samples) =====")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\n" + classification_report(
        [classes[i] for i in y_true], [classes[i] for i in y_pred],
        labels=names_present, digits=3, zero_division=0))

    out_dir = save_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig = plot_confusion_matrix(cm, classes, save_path=cm_path)
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    print_top_confused(cm, classes)

    return {"accuracy": acc, "balanced_accuracy": bal_acc,
            "report": classification_report(
                [classes[i] for i in y_true], [classes[i] for i in y_pred],
                output_dict=True, zero_division=0),
            "confusion_matrix": cm, "cm_path": cm_path, "fig": fig}
