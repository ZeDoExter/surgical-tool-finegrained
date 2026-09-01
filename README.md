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

# Train (CAHM on by default, 560 from experiments)
python train.py --data_dir dataset --epochs 50 --batch_size 32 --img_size 560

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

See `ALGORITHM.md` for full algorithm

## Config

```python
from config import TrainConfig
cfg = TrainConfig(
    data_dir="dataset",
    img_size=560,      # from kNN probe experiments (must be divisible by 14)
    batch_size=32,     # T4 15GB, use 16 for GTX 1650 4GB
    bbox_margin=0.15,  # from experiments
    lora_r=16,
    calibration_ratio=None,  # cm/px if you have a ruler reference
    # use_cahm=True by default (from ablation)
)
```

Edit `config.py` or override when creating `TrainConfig`

## Experimental Results

### v2 dataset (full resolution, no Roboflow resize)

**kNN probe (frozen, no training) — used to pick img_size before training**

| img_size | kNN probe | random baseline |
|---|---|---|
| 448 | 0.8514 | 0.071 |
| 504 | 0.8378 | 0.071 |
| 560 | **0.878** | 0.071 |
| 616 | 0.8514 | 0.071 |

Default: `img_size=560` (40×14, best on full-res data)

### Exported models (from Colab, v2 dataset, img_size=560)

Pre-trained models exported to ONNX are in `from_colab/onnx_export/`:
- `surgical_dino_fusion.onnx` — full pipeline (classifier + ArcFace)
- `detector_dino.onnx` — detector (seg head, YOLO-free)
- `classifier_meta.json` / `detector_meta.json` — class lists and config

Classifier checkpoint: `from_colab/outputs_classifier/best_model.pt`
Detector checkpoint: `from_colab/outputs_detector/best_detector.pt`

**Classifier eval (74 val samples, 14 classes):**

| Metric | Score |
|---|---|
| Accuracy | **1.0000** |
| Balanced Accuracy | **1.0000** |

**Detector eval (38 GT instances):**

| Metric | Score |
|---|---|
| Precision | 0.974 |
| Recall | 1.000 |
| F1 | **0.987** |
| IoU | 0.683 |

### v1 dataset (512px resize from Roboflow — historical)

**kNN probe (frozen, no training)**

| img_size | kNN probe |
|---|---|
| 224 | 0.7131 |
| 504 | 0.7582 |
| 518 | 0.7459 |
| 560 | 0.7336 |
| 546 | 0.7295 |
| 616 | 0.7295 |

**Training (LoRA r16, 50 epochs, early-stop, 504)**

| Config | Val Acc | Balanced | Needle↔Artery | Note |
|---|---|---|---|---|
| baseline | 0.9467 | 0.9506 | 5+3=8 |  |
| cahm | **0.9590** | **0.9606** | **2+1=3** | **kept — enabled by default** |

Only **CAHM** is kept (`use_cahm=True` by default). `LGMS`/`SEF` were tried and removed — see `ALGORITHM.md` and git history.

## License

MIT
