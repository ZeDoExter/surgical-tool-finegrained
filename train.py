# -*- coding: utf-8 -*-
"""
train.py — training loop for DINOv2 + length fusion + ArcFace

Dataset context: instruments on green cloth background — shadows make tight
bounding challenging (not silver tray). Defaults img_size=504 and
bbox_margin=0.15 come from experiments (must be divisible by 14 for ViT patch size).

Usage from notebook/script:
    from config import TrainConfig
    from train import run_training
    best_ckpt = run_training(TrainConfig(data_dir="/content/dataset"))

Or via CLI:
    python train.py --data_dir dataset --epochs 50 --finetune_mode lora
    python train.py --data_dir dataset --kfold 5     # Stratified k-fold CV
"""
import argparse
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pytorch_metric_learning.losses import ArcFaceLoss

from config import TrainConfig
from dataset import (SurgicalInstrumentDataset, compute_class_length_means,
                     compute_length_stats, compute_lgms_margins,
                     load_coco_records, stratified_split)
from model import AdaptiveArcFaceLoss, SurgicalDinoFusion, arcface_logits, count_trainable


# ============================================================ utils
def seed_everything(seed: int) -> None:
    """Fix seeds for all RNGs to ensure reproducibility (critical with small data — different splits change results)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Mixup: random interpolation between randomly paired samples
    x: image tensor (B,3,H,W) | y: label (B,)
    return: mixed_x, y_a, y_b, lam (lambda = interpolation ratio)

    Important for small datasets: helps regularization by "blending" between classes
    alpha=0.4 → lam ~ Beta(0.4, 0.4) usually near 0 or 1 (not in the middle)
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 so labels don't swap
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def torch_load_compat(path: str) -> dict:
    """torch.load compatible with both old and new torch (default weights_only changed in torch 2.6)"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_flip_flags(class_names: List[str], cfg: TrainConfig) -> List[bool]:
    """Only classes listed in cfg.flip_allowed can be flipped (None = all classes can be flipped)"""
    allowed = set(cfg.flip_allowed) if cfg.flip_allowed is not None else None
    return [True if allowed is None else n in allowed for n in class_names]


def resolve_records(cfg: TrainConfig) -> Tuple[List[dict], List[dict], List[str]]:
    """
    Load records from data_dir:
      - if valid/_annotations.coco.json exists → use it directly
      - otherwise → stratified split from train using val_fraction/seed from config
    """
    tr, classes = load_coco_records(cfg.data_dir, "train")
    valid_ann = os.path.join(cfg.data_dir, "valid", "_annotations.coco.json")
    if os.path.exists(valid_ann):
        va, classes_valid = load_coco_records(cfg.data_dir, "valid")
        if classes_valid != classes:
            raise ValueError(f"Class lists differ between train/valid:\n{classes}\n{classes_valid}")
    else:
        tr, va = stratified_split(tr, cfg.val_fraction, cfg.seed)
        print(f"[data] no valid/ folder → stratified split {len(tr)}/{len(va)} (seed={cfg.seed})")
    return tr, va, classes

def warmup_cosine_factor(step: int, warmup: int, total: int) -> float:
    """LR schedule: linear warmup → cosine decay to near 0 by the end"""
    if step < warmup:
        return step / max(1, warmup)
    t = min((step - warmup) / max(1, total - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * t))


# ============================================================ CAHM helpers
def _cahm_d_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Compute pair difficulty from confusion matrix:
      d(i,j) = C[i,j] + C[j,i]  (i!=j), normalized by max
    """
    C = cm.astype(np.float64)
    n = C.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = C[i, j] + C[j, i]
    m = d.max()
    if m > 1e-9:
        d = d / m
    return d


def _cahm_weights(labels: torch.Tensor, d_t: np.ndarray, alpha: float, device) -> torch.Tensor:
    """w = 1 + alpha * max_j d_t[y,j]  (per-sample)"""
    if d_t is None:
        return torch.ones_like(labels, dtype=torch.float32)
    d = torch.as_tensor(d_t, device=device, dtype=torch.float32)  # (C,C)
    # row-wise max (diagonal is already 0, so no need to exclude)
    row_max = d.max(dim=1).values  # (C,)
    w = 1.0 + alpha * row_max[labels]
    return w


@torch.no_grad()
def _eval_confusion(model, loss_fn, loader, device, num_classes: int) -> np.ndarray:
    """Run over the full validation set to build a confusion matrix for CAHM"""
    from sklearn.metrics import confusion_matrix
    model.eval()
    ys_true, ys_pred = [], []
    for batch in loader:
        px = batch["image"].to(device, non_blocking=True)
        ln = batch["length"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        edge = batch.get("edge_map")
        if edge is not None:
            edge = edge.to(device, non_blocking=True)
        emb = model(px, ln, edge)
        logits = arcface_logits(loss_fn, emb.float())
        pred = logits.argmax(dim=1).cpu().numpy()
        ys_true.extend(y.cpu().numpy().tolist())
        ys_pred.extend(pred.tolist())
    cm = confusion_matrix(ys_true, ys_pred, labels=list(range(num_classes)))
    return cm
# ============================================================ epochs
def train_one_epoch(model, loss_fn, loader, optimizer, scheduler, scaler, device, cfg,
                    cahm_d: Optional[np.ndarray] = None) -> float:
    """Train for 1 epoch → return average loss (ArcFace on embeddings from the fusion head)"""
    model.train()
    total, seen = 0.0, 0
    mixup_alpha = getattr(cfg, "mixup_alpha", 0.0)
    use_cahm = bool(getattr(cfg, "use_cahm", False)) and cahm_d is not None
    cahm_alpha = float(getattr(cfg, "cahm_alpha", 2.0))
    is_adaptive = hasattr(loss_fn, "compute_loss_dict")
    for batch in loader:
        px = batch["image"].to(device, non_blocking=True)
        ln = batch["length"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        edge = batch.get("edge_map")
        if edge is not None:
            edge = edge.to(device, non_blocking=True)

        # Mixup: randomly interpolate between 2 samples (regularization for small datasets)
        use_mixup = mixup_alpha > 0 and model.training and not use_cahm  # disable mixup when using CAHM so weights remain clear
        if use_mixup:
            px, y_a, y_b, lam = mixup_data(px, y, mixup_alpha)
        else:
            y_a, y_b, lam = y, y, 1.0

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:  # GPU → mixed precision
            with torch.autocast("cuda"):
                emb = model(px, ln, edge)
                if use_mixup:
                    loss = lam * loss_fn(emb.float(), y_a) + (1 - lam) * loss_fn(emb.float(), y_b)
                elif use_cahm:
                    # CAHM weighted loss — retrieve per-sample loss and multiply by w
                    if is_adaptive:
                        d = loss_fn.compute_loss_dict(emb.float(), y)
                        per = d["losses"]  # (B,)
                    else:
                        # must pass ref_emb = embeddings to pass PML identity check
                        ef = emb.float()
                        ld = loss_fn.compute_loss(ef, y, None, ef, y)
                        per = ld["loss"]["losses"]  # (B,)
                    w = _cahm_weights(y, cahm_d, cahm_alpha, device)
                    loss = (per * w).mean()
                else:
                    loss = loss_fn(emb.float(), y)  # cast to fp32 before ArcFace for stability
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:  # CPU → fp32
            emb = model(px, ln, edge)
            if use_mixup:
                loss = lam * loss_fn(emb, y_a) + (1 - lam) * loss_fn(emb, y_b)
            elif use_cahm:
                if is_adaptive:
                    d = loss_fn.compute_loss_dict(emb.float(), y)
                    per = d["losses"]
                else:
                    ef = emb.float()
                    ld = loss_fn.compute_loss(ef, y, None, ef, y)
                    per = ld["loss"]["losses"]
                w = _cahm_weights(y, cahm_d, cahm_alpha, device)
                loss = (per * w).mean()
            else:
                loss = loss_fn(emb, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        scheduler.step()  # per-step schedule (warmup+cosine)

        total += loss.item() * y.size(0)
        seen += y.size(0)
    return total / max(seen, 1)


@torch.no_grad()
def validate(model, loss_fn, loader, device) -> Tuple[float, float]:
    """
    validation -> (val_loss, val_acc)
    - val_loss: ArcFace loss on the val set (logged; tiebreak for best-model selection)
    - val_acc : argmax over s*cos(theta) logits (true inference mode, no margin)
    """
    model.eval()
    embs, ys = [], []
    for batch in loader:
        px = batch["image"].to(device, non_blocking=True)
        ln = batch["length"].to(device, non_blocking=True)
        edge = batch.get("edge_map")
        if edge is not None:
            edge = edge.to(device, non_blocking=True)
        embs.append(model(px, ln, edge).float().cpu())
        ys.append(batch["label"])
    E = torch.cat(embs).to(device)
    Y = torch.cat(ys).to(device)
    loss = loss_fn(E, Y).item()
    acc = (arcface_logits(loss_fn, E).argmax(dim=1) == Y).float().mean().item()
    return loss, acc

def save_history_plot(history: dict, out_png: str) -> None:
    """Plot loss/accuracy — may fail (headless) without crashing training"""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("ArcFace loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(history["val_acc"], color="tab:green")
        ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[log] saved history plot → {out_png}")
    except Exception as e:  # noqa: BLE001 — plotting should not crash training
        print(f"(skipping history plot: {e})")


# ============================================================ main training entry
def run_training(cfg: TrainConfig,
                 records_train: Optional[List[dict]] = None,
                 records_valid: Optional[List[dict]] = None,
                 tag: str = "") -> str:
    """
    Train once (single split) — returns path of the best checkpoint (highest val accuracy)

    checkpoint contains: model_state, arcface_state, classes, length_mean/std,
                         cfg (dict), epoch, val_loss, val_acc
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data ----------
    if records_train is None or records_valid is None:
        records_train, records_valid, _ = resolve_records(cfg)
    all_recs = list(records_train) + list(records_valid)
    label2name = {r["label"]: r["class_name"] for r in all_recs}
    class_names = [label2name[i] for i in range(max(label2name) + 1)]  # indices always sorted

    # length mean/std ← from train only (prevent leakage)
    length_stats = compute_length_stats(records_train, cfg.calibration_ratio)
    print(f"[data] train={len(records_train)} val={len(records_valid)} "
          f"classes={len(class_names)} length_mean={length_stats[0]:.2f} std={length_stats[1]:.2f}")

    flip_flags = build_flip_flags(class_names, cfg)
    use_sef = bool(getattr(cfg, "use_sef", False))
    ds_train = SurgicalInstrumentDataset(records_train, length_stats, cfg.img_size,
                                         cfg.calibration_ratio, flip_flags, training=True,
                                         bbox_margin=cfg.bbox_margin, use_sef=use_sef)
    ds_val = SurgicalInstrumentDataset(records_valid, length_stats, cfg.img_size,
                                       cfg.calibration_ratio, flip_flags=None, training=False,
                                       bbox_margin=cfg.bbox_margin, use_sef=use_sef)
    pin = device.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=pin)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=pin)
    # ---------- model + loss ----------
    model = SurgicalDinoFusion(
        backbone_name=cfg.backbone_name, finetune_mode=cfg.finetune_mode,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        partial_last_blocks=cfg.partial_last_blocks, head_dropout=cfg.head_dropout,
        use_attention_pool=getattr(cfg, "use_attention_pool", True),
        use_sef=use_sef,
    ).to(device)
    print(f"[model] trainable params = {count_trainable(model):,} (mode={cfg.finetune_mode}, sef={use_sef})")

    # ArcFace — standard or LGMS (per-class margin)
    if bool(getattr(cfg, "use_lgms", False)):
        class_means = compute_class_length_means(records_train, cfg.calibration_ratio)
        margins = compute_lgms_margins(class_means, len(class_names),
                                       m_base=cfg.margin, gamma=cfg.lgms_gamma, k=cfg.lgms_k)
        print(f"[LGMS] margins per class: {[f'{m:.1f}' for m in margins]}")
        loss_fn = AdaptiveArcFaceLoss(num_classes=len(class_names), embedding_size=model.embed_dim,
                                      margin_per_class=margins, scale=cfg.scale).to(device)
    else:
        # ArcFace: margin in degrees (~28.6° = 0.5 rad), s=64 — forces embeddings to have
        # tight intra-class / wide inter-class separation, suitable for classes with very similar shapes
        loss_fn = ArcFaceLoss(num_classes=len(class_names), embedding_size=model.embed_dim,
                              margin=cfg.margin, scale=cfg.scale).to(device)
    if cfg.finetune_mode == "partial":
        lr_bb = cfg.lr_backbone
    elif cfg.finetune_mode == "lora":
        lr_bb = cfg.lr_lora
    else:
        lr_bb = None
    optimizer = AdamW(model.param_groups(cfg.lr_head, lr_bb), weight_decay=cfg.weight_decay)

    total_steps = max(1, len(dl_train)) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: warmup_cosine_factor(s, warmup_steps, total_steps))

    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")   # torch >= 2.3
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()    # fallback for old torch
    else:
        scaler = None

    # ---------- loop + early stopping ----------
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    # Select best by "val_acc" (val_loss as tiebreak) — simulations show that on small datasets
    # ArcFace val_loss and val_acc conflict (lowest loss != best model); original spec used val loss
    best = {"val_loss": float("inf"), "val_acc": -1.0, "epoch": -1, "model": None, "arcface": None}
    bad_epochs = 0
    cahm_d = None  # (C,C) EMA state for CAHM
    use_cahm = bool(getattr(cfg, "use_cahm", False))
    cahm_start = int(getattr(cfg, "cahm_start_epoch", 10))
    cahm_beta = float(getattr(cfg, "cahm_beta", 0.9))

    for epoch in range(1, cfg.epochs + 1):
        # pass cahm_d to this epoch if start time has been reached
        cur_cahm = cahm_d if (use_cahm and epoch > cahm_start and cahm_d is not None) else None
        tl = train_one_epoch(model, loss_fn, dl_train, optimizer, scheduler, scaler, device, cfg, cahm_d=cur_cahm)
        vl, va = validate(model, loss_fn, dl_val, device)
        history["train_loss"].append(tl); history["val_loss"].append(vl); history["val_acc"].append(va)

        # CAHM: update difficulty after validation (for next epoch)
        if use_cahm and epoch >= cahm_start:
            try:
                cm = _eval_confusion(model, loss_fn, dl_val, device, len(class_names))
                d_cur = _cahm_d_from_cm(cm)
                if cahm_d is None:
                    cahm_d = d_cur
                else:
                    cahm_d = cahm_beta * cahm_d + (1 - cahm_beta) * d_cur
                # log top-1 confused pair for debugging
                flat = [(i, j, cahm_d[i, j]) for i in range(len(class_names)) for j in range(len(class_names)) if i != j]
                flat.sort(key=lambda x: -x[2])
                if flat:
                    i, j, v = flat[0]
                    print(f"  [CAHM] top confused: {class_names[i]}↔{class_names[j]} d={v:.3f}")
            except Exception as e:
                print(f"  [CAHM] skip update: {e}")

        improved = va > best["val_acc"] + 1e-4 or \
            (va >= best["val_acc"] - 1e-4 and vl < best["val_loss"] - 1e-4)
        star = ""
        if improved:
            best.update(val_loss=vl, val_acc=va, epoch=epoch,
                        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        arcface={k: v.detach().cpu().clone() for k, v in loss_fn.state_dict().items()})
            bad_epochs = 0
            star = "  *best*"
            # save immediately — checkpoint survives interruption (file exists even with early stop)
            try:
                ckpt_immediate = os.path.join(cfg.output_dir, f"best_model{tag}.pt")
                torch.save({
                    "model_state": best["model"],
                    "arcface_state": best["arcface"],
                    "classes": class_names,
                    "length_mean": float(length_stats[0]),
                    "length_std": float(length_stats[1]),
                    "calibration_ratio": cfg.calibration_ratio,
                    "cfg": cfg.to_dict(),
                    "epoch": best["epoch"],
                    "val_loss": float(best["val_loss"]),
                    "val_acc": float(best["val_acc"]),
                }, ckpt_immediate)
            except Exception as e:
                print(f"(skipping immediate save: {e})")
        else:
            bad_epochs += 1
        cur_lr = scheduler.get_last_lr()[0]
        print(f"{tag}[epoch {epoch:03d}/{cfg.epochs}] train={tl:.4f} val={vl:.4f} "
              f"val_acc={va:.4f} lr={cur_lr:.2e}{star}", flush=True)

        if bad_epochs >= cfg.patience:
            print(f"[early stop] no improvement for {cfg.patience} epochs — stopping at epoch {epoch}")
            break

    # ---------- save best checkpoint ----------
    ckpt_path = os.path.join(cfg.output_dir, f"best_model{tag}.pt")
    torch.save({
        "model_state": best["model"],
        "arcface_state": best["arcface"],
        "classes": class_names,
        "length_mean": float(length_stats[0]),
        "length_std": float(length_stats[1]),
        "calibration_ratio": cfg.calibration_ratio,
        "cfg": cfg.to_dict(),
        "epoch": best["epoch"],
        "val_loss": float(best["val_loss"]),
        "val_acc": float(best["val_acc"]),
    }, ckpt_path)
    save_history_plot(history, os.path.join(cfg.output_dir, f"history{tag}.png"))

    print(f"[done] best epoch={best['epoch']} val_loss={best['val_loss']:.4f} "
          f"val_acc={best['val_acc']:.4f} → {ckpt_path}")
    return ckpt_path


# ============================================================ k-fold CV
def run_kfold(cfg: TrainConfig, k: Optional[int] = None) -> Tuple[List[str], List[float]]:
    """
    Stratified k-fold CV over all data (train∪valid) — more reliable than single split
    when there are only ~30 images/class; returns (paths, val_acc per fold)
    """
    from sklearn.model_selection import StratifiedKFold
    tr, va, _ = resolve_records(cfg)
    all_recs = tr + va
    y = [r["label"] for r in all_recs]
    skf = StratifiedKFold(n_splits=k or cfg.kfold, shuffle=True, random_state=cfg.seed)

    paths, accs = [], []
    for i, (idx_tr, idx_va) in enumerate(skf.split(all_recs, y)):
        rec_tr = [all_recs[j] for j in idx_tr]
        rec_va = [all_recs[j] for j in idx_va]
        print(f"\n========== Fold {i + 1}/{skf.n_splits} "
              f"(train={len(rec_tr)} val={len(rec_va)}) ==========")
        path = run_training(cfg, rec_tr, rec_va, tag=f"_fold{i + 1}")
        paths.append(path)
        ckpt = torch_load_compat(path)
        accs.append(float(ckpt["val_acc"]))

    print("\n===== K-FOLD SUMMARY =====")
    for i, a in enumerate(accs):
        print(f"fold {i + 1}: val_acc={a:.4f}")
    print(f"mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return paths, accs


# ============================================================ CLI
def main(argv=None) -> None:
    from dataclasses import replace
    ap = argparse.ArgumentParser(description="Train DINOv2+length-fusion+ArcFace for surgical instrument classification")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--finetune_mode", choices=["lora", "partial", "frozen"], default=None)
    ap.add_argument("--kfold", type=int, default=None, help="e.g. 5 → Stratified 5-fold CV")
    ap.add_argument("--calibration_ratio", type=float, default=None,
                    help="cm/pixel from reference object (omit = use pixels)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    # CAHM / LGMS / SEF — toggle auxiliary algorithms
    ap.add_argument("--use_cahm", action="store_true", help="enable CAHM (confusion-aware hard mining)")
    ap.add_argument("--use_lgms", action="store_true", help="enable LGMS (length-gated margin scaling)")
    ap.add_argument("--use_sef", action="store_true", help="enable SEF (Scharr edge fusion)")
    ap.add_argument("--cahm_alpha", type=float, default=None)
    ap.add_argument("--cahm_beta", type=float, default=None)
    ap.add_argument("--lgms_gamma", type=float, default=None)
    ap.add_argument("--lgms_k", type=int, default=None)
    args = ap.parse_args(argv)

    overrides = {}
    for k, v in vars(args).items():
        if k == "calibration_ratio":
            continue
        if v is None:
            continue
        # store_true flags: False means not set → don't override (keep default False)
        if isinstance(v, bool) and not v:
            continue
        overrides[k] = v
    if args.calibration_ratio is not None:
        overrides["calibration_ratio"] = args.calibration_ratio
    cfg = replace(TrainConfig(), **overrides)
    if cfg.kfold and cfg.kfold > 1:
        run_kfold(cfg)
    else:
        path = run_training(cfg)
        print("checkpoint:", path)


if __name__ == "__main__":
    main()
