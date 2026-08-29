# -*- coding: utf-8 -*-
"""
config.py — central defaults for the whole pipeline

Edit values here, or override when creating the object: TrainConfig(data_dir=..., epochs=...)
"""
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class TrainConfig:
    # ---------------- Data ----------------
    data_dir: str = "dataset"          # folder with train/ and valid/ (Roboflow COCO Segmentation export)
    img_size: int = 504                # must be divisible by 14 (504=36×14) — from kNN probe experiments
    val_fraction: float = 0.2          # used when valid/ folder is missing → stratified split from train
    calibration_ratio: Optional[float] = None  # cm/pixel — measured from a reference object of known size
                                               # (camera rig is fixed, so one ratio works for all images)
                                               # None = use pixel units, normalized by train mean/std
    flip_allowed: Optional[List[str]] = None   # class names that are allowed to be flipped
                                               # None = all classes can be flipped
                                               # remove handedness classes (left/right) from the list
    num_workers: int = 2               # Colab/Linux can use 2 — on Windows set to 0 if it hangs
    bbox_margin: float = 0.15          # crop margin around bbox per instance (0=no crop, from experiments)

    # ---------------- Model ----------------
    backbone_name: str = "facebook/dinov2-small"  # ViT-S/14, hidden dim = 384 (~21M params)
    finetune_mode: str = "lora"        # "lora" (recommended for small data) | "partial" | "frozen"
    partial_last_blocks: int = 2       # for mode="partial": unfreeze last N ViT blocks + final LayerNorm
    lora_r: int = 16                   # from experiments (r16 better than r8 on this data)
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    head_dropout: float = 0.1          # dropout for fusion head
    use_attention_pool: bool = True    # True = AttentionPooling instead of CLS-only (v2)
                                      # False = use CLS token (backward compat)
    mixup_alpha: float = 0.4           # Mixup alpha for regularization (0.0 = off)
                                      # smaller = stronger, 0.4 suits ~30 samples/class
    use_tta: bool = True               # enable TTA at inference (flip + multi-scale)

    # ---------------- ArcFace ----------------
    margin: float = 28.6               # additive angular margin in degrees (≈ 0.5 rad)
                                       # — pytorch-metric-learning expects degrees and converts to radians
    scale: float = 64.0                # s: scale cosine before softmax so gradients don't vanish

    # ---------------- CAHM (Confusion-Aware Hard Mining) ----------------
    use_cahm: bool = True
    cahm_alpha: float = 2.0            # extra weight for confused pairs
    cahm_beta: float = 0.9             # EMA smoothing for difficulty score
    cahm_start_epoch: int = 10         # start after this epoch (let confusion stabilize)

    # ---------------- Training ----------------
    batch_size: int = 32               # fills T4 15GB at 504px with DINOv2-S + LoRA
    epochs: int = 50
    patience: int = 12                 # early stopping: stop when val accuracy doesn't improve for N epochs
    lr_head: float = 3e-4              # learning rate for fusion head
    lr_backbone: float = 5e-6          # for finetune_mode="partial"
    lr_lora: float = 1e-4              # for finetune_mode="lora"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1          # first 10% of total steps is linear warmup, then cosine decay
    grad_clip: float = 1.0
    seed: int = 42
    kfold: Optional[int] = None        # e.g. 5 = Stratified 5-fold CV (more reliable for small data)
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        """Convert config to dict (saved to checkpoint for reproducibility at evaluate/infer)"""
        return asdict(self)
