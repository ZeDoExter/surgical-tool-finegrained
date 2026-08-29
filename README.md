# surgical-tool-finegrained

Fine-grained classification of **14 surgical instruments** on a **green cloth** — where several classes differ *only* by length. Small data (~400 images, ~30/class), fixed camera rig. The main difficulty is **shadows from lighting** that make bounding the instrument hard (more than background color).

**Architecture:** `DINOv2-S/14 (ViT-S)` + `LoRA r=16` + `length fusion` + `ArcFace` + `CAHM` (kept)

```
image → DINOv2 → 384-d ─┐
                        ├─ concat → Linear → 384 → ArcFace
mask  → length (px/cm) ─┘
```

## Usage

```bash
pip install -r requirements.txt

# Train (CAHM on by default, 504 from experiments)
python train.py --data_dir dataset --epochs 50 --batch_size 32 --img_size 504

# Evaluate
python evaluate.py --checkpoint outputs/best_model.pt --data_dir dataset

# Predict single image (+ mask)
python infer.py --checkpoint outputs/best_model.pt --image path.jpg --mask_json path.json --ann_id 0
```

Dataset: **COCO Segmentation** export from Roboflow

```
dataset/
  train/_annotations.coco.json + *.jpg  # 1032 samples
  valid/_annotations.coco.json + *.jpg  # 244 samples
```

`1 annotation = 1 sample` — use `bbox_margin=0.15` to crop per instance when one image has multiple tools.

## Colab

Open `DentalInstrument_DINOv2_ArcFace.ipynb` — self-contained (writes all modules via `%%writefile`)

- Cell `3.5` checks `kNN probe` before training
- Cell `4` trains (`504 batch32` on T4, CAHM on by default)
- Cell `5` evaluates

See `ALGORITHM.md` for full algorithm

## Config

```python
from config import TrainConfig
cfg = TrainConfig(
    data_dir="dataset",
    img_size=504,      # from kNN probe experiments (must be divisible by 14)
    batch_size=32,     # T4 15GB, use 16 for GTX 1650 4GB
    bbox_margin=0.15,  # from experiments
    lora_r=16,
    calibration_ratio=None,  # cm/px if you have a ruler reference
    # use_cahm=True by default (from ablation)
)
```

Edit `config.py` or override when creating `TrainConfig`

## Experimental Results

**kNN probe (frozen, no training) — used to pick img_size before training**

| img_size | kNN probe |
|---|---|
| 224 | 0.7131 |
| 504 | 0.7582 |
| 518 | 0.7459 |
| 560 | 0.7336 |
| 546 | 0.7295 |
| 616 | 0.7295 |

`bbox_margin` at `504`: `0.0 → 0.3074` / `0.10 → 0.7377` / `0.15 → 0.7582` / `0.20 → 0.7336`

**Training (LoRA r16, 50 epochs, early-stop, 504)**

| Config | Val Acc | Balanced | Needle↔Artery | Note |
|---|---|---|---|---|
| baseline | 0.9467 | 0.9506 | 5+3=8 |  |
| cahm | **0.9590** | **0.9606** | **2+1=3** | **kept — enabled by default** |
| lgms | 0.9385 | 0.9348 | 3+6=9 | removed — worse than baseline |
| sef | 0.9549 | 0.9461 | 5+2=7 | removed — not as good as cahm |
| cahm_lgms | 0.9508 | 0.9510 | 3+3=6 | removed |
| all | 0.9508 | 0.9567 | 2+3=5 | removed — not better than cahm alone |

Only **CAHM** is kept (`use_cahm=True` by default). `LGMS`/`SEF` were tried and removed — see `ALGORITHM.md` and git history.
Ablation was run on `1032/244` single split at `504` (same as `kNN` best). `LGMS`/`SEF` code and flags are deleted.

## License

MIT
