# -*- coding: utf-8 -*-
"""
config.py — ค่าตั้งต้นกลางของทั้ง pipeline

แก้ค่าในไฟล์นี้ หรือ override ตอนสร้าง object: TrainConfig(data_dir=..., epochs=...)
"""
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class TrainConfig:
    # ---------------- ข้อมูล ----------------
    data_dir: str = "dataset"          # โฟลเดอร์ที่มี train/ และ valid/ (Roboflow COCO Segmentation export)
    img_size: int = 224                # ต้องหารด้วย 14 ลงตัว (ViT patch size ของ DINOv2)
    val_fraction: float = 0.2          # ใช้เมื่อไม่มีโฟลเดอร์ valid/ → stratified split จาก train
    calibration_ratio: Optional[float] = None  # cm/pixel — วัดจาก object อ้างอิงที่รู้ขนาดจริง
                                               # (กล้องระยะคงที่ตลอด จึงใช้ค่าเดียวได้ทุกภาพ)
                                               # None = ใช้หน่วย pixel แล้ว normalize ด้วย mean/std ของ train
    flip_allowed: Optional[List[str]] = None   # รายชื่อ class ที่ "flip ซ้าย-ขวาได้"
                                               # None = flip ได้ทุก class
                                               # class ที่มี handedness (ของซ้าย/ขวามือ) → ตัดชื่อออกจากลิสต์
    num_workers: int = 2               # Colab/Linux ใช้ 2 ได้ — Windows ถ้าค้างให้เปลี่ยนเป็น 0
    bbox_margin: float = 0.15          # crop รอบ bbox ต่อชิ้น (0=ไม่ crop, 0.15 ดีสุด 0.9688)

    # ---------------- โมเดล ----------------
    backbone_name: str = "facebook/dinov2-small"  # ViT-S/14, hidden dim = 384 (~21M params)
    finetune_mode: str = "lora"        # "lora" (แนะนำสำหรับข้อมูลน้อย) | "partial" | "frozen"
    partial_last_blocks: int = 2       # ใช้เมื่อ mode="partial": ปลดล็อก N ViT block ท้าย + final LayerNorm
    lora_r: int = 16                   # ★ ดีสุดจาก tune 0.9688 (r8 ได้ 0.9467) — r16 capacity เยอะขึ้น
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    head_dropout: float = 0.1          # dropout ของ fusion head
    use_attention_pool: bool = True    # True = AttentionPooling แทน CLS-only (v2)
                                      # False = ใช้ CLS token เหมือนเดิม (backward compat)
    mixup_alpha: float = 0.4           # Mixup alpha สำหรับ regularization (0.0 = ปิด)
                                      # ยิ่งน้อยยิ่งแรง, 0.4 เหมาะกับ ~30 samples/class
    use_tta: bool = True               # เปิด TTA ตอน inference (flip + multi-scale)

    # ---------------- ArcFace ----------------
    margin: float = 28.6               # additive angular margin หน่วย "องศา" (≈ 0.5 rad)
                                       # — pytorch-metric-learning รับหน่วยองศาแล้วแปลงเป็นเรเดียนเอง
    scale: float = 64.0                # s: ขยาย cosine ก่อน softmax เพื่อไม่ให้ gradient แบน

    # ---------------- การเทรน ----------------
    batch_size: int = 16
    epochs: int = 50
    patience: int = 12                 # early stopping: stop when val accuracy stops improving for N epochs
    lr_head: float = 3e-4              # learning rate ของ fusion head
    lr_backbone: float = 5e-6          # ใช้เมื่อ finetune_mode="partial"
    lr_lora: float = 1e-4              # ใช้เมื่อ finetune_mode="lora"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1          # 10% แรกของ total steps เป็น linear warmup แล้วค่อย cosine decay
    grad_clip: float = 1.0
    seed: int = 42
    kfold: Optional[int] = None        # เช่น 5 = Stratified 5-fold CV (แม่นกว่าเมื่อข้อมูลน้อย)
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        """แปลง config เป็น dict (เก็บลง checkpoint เพื่อ reproduce ตอน evaluate/infer)"""
        return asdict(self)
