# surgical-tool-finegrained

Fine-grained classification of **14 surgical/dental instruments** on a silver tray — where several classes differ *only* by length. Small data (~400 images, ~30/class), low contrast (silver on silver), fixed camera rig.

**Architecture:** `DINOv2-S/14 (ViT-S)` + **LoRA (r=16)** + **length fusion** (minAreaRect → normalized) → **ArcFace (m=28.6°, s=64)**

```
image → DINOv2 ──→ 384-d ──┐
                          ├─ concat(385) → Linear → 384 → ArcFace
mask  → length (px/cm) ───┘
```

## Results

| Split | Method | Val Acc | Note |
|---|---|---|---|
| 1032/244 single split | LoRA r8 50e (224) | 0.9467 | best epoch 22 |
| 1032/244 single split (15e) | LoRA r8 (224) | 0.9180 | baseline |
| 1032/244 single split (15e) | **LoRA r16 (224)** | **0.9221** | **best tune** |
| 1020/256 k-fold Fold1 | LoRA r16 23e (224) | **0.9688** | 8/256 errors |
| 1032/244 single split | LoRA r16 34e (**546**, batch32) | **0.9549** | Bal 0.9580, 11/244 errors |

Top confused: `Needle_Holder ↔ Artery_Forceps` (5+3 @224) → (3+2 @546) — 546px ช่วยเห็นปลายคีมชัดขึ้น

## Quick Start

```bash
pip install -r requirements.txt

# Train (LoRA, 50 epochs, early stop 12)
python train.py --data_dir dataset --epochs 50 --batch_size 8 --img_size 224
# Colab T4 large-image: --img_size 546 --batch_size 32 (~7.5/15GB)

# Evaluate
python evaluate.py --checkpoint outputs/best_model.pt --data_dir dataset

# Single image (+ mask polygon or PNG)
python infer.py --checkpoint outputs/best_model.pt --image path.jpg --mask_json path.json --ann_id 0
```

Dataset: Roboflow export **COCO Segmentation** with structure:

```
dataset/
  train/_annotations.coco.json + *.jpg
  valid/_annotations.coco.json + *.jpg
```

`1 annotation = 1 sample` — use `bbox_margin=0.15` in `config.py` to crop per-instance when one image has multiple tools.

## Colab

Open `DentalInstrument_DINOv2_ArcFace.ipynb` — it `%%writefile`s all modules, so the notebook is self-contained. Default `lora_r=16` (best, 0.9688). For larger images: `img_size=546` (39×14) + `batch_size=32` on T4.

## Config

Edit `config.py` or override:

```python
from config import TrainConfig
cfg = TrainConfig(
    data_dir="dataset",
    img_size=224,      # 224 or 546 (must be divisible by 14)
    batch_size=8,
    bbox_margin=0.15,  # 0 = no crop, 0.15 = per-instance crop
    lora_r=16,         # 8 or 16 (16 best on this data)
    calibration_ratio=None,  # cm/px if you have a ruler reference, else pixel
)
```

## License

MIT
