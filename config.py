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
    img_size: int = 560                # must be divisible by 14 (560=40×14) — from kNN probe experiments (best on full-res data)
    val_fraction: float = 0.2          # used when valid/ folder is missing → stratified split from train
    calibration_ratio: Optional[float] = None  # cm/pixel — measured from a reference object of known size
                                               # (camera rig is fixed, so one ratio works for all images)
                                               # None = use pixel units, normalized by train mean/std
    flip_allowed: Optional[List[str]] = None   # class names that are allowed to be flipped
                                               # None = all classes can be flipped
                                               # remove handedness classes (left/right) from the list
    num_workers: int = 2               # Colab/Linux can use 2 — on Windows set to 0 if it hangs
    bbox_margin: float = 0.15          # crop margin around bbox per instance (0=no crop, from experiments)
    tip_zoom_prob: float = 0.35        # prob of replacing the crop with a zoomed TIP view (tip-shape
                                       # is the true signal for Needle↔Artery / 23↔150 — same length!)
    tip_zoom_size: float = 0.42        # tip crop length as fraction of the instrument's long axis
    cutmix_prob: float = 0.0           # legacy alias for patch_paste_prob
    patch_paste_prob: float = 0.4      # probability of applying Mask-Aware Patch-Paste (Copy-Paste)
                                       # 0.0 = off, 0.4 = ~40% of samples get mixed with secondary tools
                                       # creates realistic multi-tool scenes on green cloth
    patch_paste_max_objects: int = 2   # max number of secondary instruments pasted per image
    patch_paste_max_overlap: float = 0.20  # max allowable overlap with target instrument (keeps target >=80% visible)

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
    batch_size: int = 32               # fills T4 15GB at 560px with DINOv2-S + LoRA
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


# ============================================================ Detector
# Real-world instrument lengths in cm (measured with a ruler, shared by
# detector post-processing and the classifier's length prior). Classes not
# listed → unknown length (prior disabled for them).
REAL_LENGTH_CM: dict = {
    # Only user-measured lengths. Equal-length pairs are NOT listed — a length
    # prior cannot separate them (tip TTA must). Add more after you measure.
    "Root_Elevators": 15.5,
    "Root_Tip_Elevator_Straight": 14.5,
}

# Class pairs that share the same true length — the ONLY reliable signal is
# the tip shape (curved vs straight jaws / beak form), not size.
TIP_CRITICAL_PAIRS: List[List[str]] = [
    ["Needle_Holder", "Artery_Forceps"],
    ["Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150"],
]

# Confusion-prone classes (from live-camera errors): these get EXTRA
# augmentation (rotation, tip-zoom, oversampling) during training.
# Root_Tip_Pick added after live testing (slow/low-confidence answers).
HARD_CLASSES: List[str] = [
    "Suture_Scissors",
    "Artery_Forceps",
    "Needle_Holder",
    "Root_Tip_Pick",
]


@dataclass
class DetectorConfig:
    """DINOv2-based single-stage detector (no YOLO): DINOv2 backbone + light seg head.

    Input 560×560 → 40×40 patch grid → decoder → per-patch (1+num_classes) logits
    → mask+label per instance via connected components → bbox/label/conf like YOLO.
    """
    # ---------------- Data ----------------
    data_dir: str = "dataset"            # same Roboflow COCO Segmentation export as classifier
    img_size: int = 560                 # must be divisible by 14 (560=40×14)
    num_workers: int = 2                # set 0 on Windows if DataLoader hangs
    # scene synthesis: paste 2-5 instruments per 560×560 canvas each step
    synth_min_objects: int = 2
    synth_max_objects: int = 5
    synth_same_class_prob: float = 0.15 # probability of pasting a same-class duplicate
    synth_scale_range: tuple = (0.75, 1.25)  # scale jitter (absolute scale varies with placement)
    synth_bg_source: str = "mixed"      # "mixed" = green-cloth render OR real background
    synth_green_prob: float = 0.5       # prob of procedural green cloth vs real-crop background
    synth_shadows: bool = True          # simulate cloth shadows (main difficulty on the rig)
    synth_max_overlap: float = 0.20     # max mask overlap between pasted instruments
    min_mask_area_px: int = 120         # drop instances whose mask < this many px (annotation noise)
    use_real_mixed_images: bool = True   # also feed real COCO images that have >1 annotation (if any)

    # ---------------- Model ----------------
    backbone_name: str = "facebook/dinov2-small"
    finetune_mode: str = "lora"         # "lora" | "partial" | "frozen"
    partial_last_blocks: int = 2
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    decoder_dim: int = 192              # conv decoder width
    decoder_mid_dim: int = 256          # pixel-shuffle mid width
    decoder_dropout: float = 0.1
    use_mid_feats: bool = True           # fuse hidden_states[6] and [9] into the decoder

    # ---------------- Training ----------------
    batch_size: int = 4                 # GTX 1650 4GB; raise to 16 on a T4/Colab
    epochs: int = 60
    patience: int = 12
    lr_head: float = 3e-4
    lr_backbone: float = 5e-6           # "partial" mode
    lr_lora: float = 8e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    bg_weight: float = 0.25             # CE class weight for background patch (set <1 to
                                         # counter patch imbalance: ~90% of patches are bg)
    dice_weight: float = 1.0            # Dice on binary instrument mask (FG vs BG)
    ce_weight: float = 1.0              # multi-class CE over (bg + classes) per patch
    output_dir: str = "outputs_detector"

    # ---------------- Instance post-processing ----------------
    mask_threshold: float = 0.5         # per-patch prob threshold to call a patch FG-of-class
    min_instance_area: int = 80         # connected components smaller than this (px, at 40×40 scale → ×196 for 560) are dropped
    nms_iou: float = 0.40               # IoU (mask-wise) NMS between same-class instances
    conf_min_score: float = 0.35        # drop instances with mean class prob < this

    def to_dict(self) -> dict:
        return asdict(self)
