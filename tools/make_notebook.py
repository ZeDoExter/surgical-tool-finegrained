# -*- coding: utf-8 -*-
"""tools/make_notebook.py — Colab notebook that embeds the current pipeline.

Run: python tools/make_notebook.py
Output: DentalInstrument_DINOv2_DetectClassify.ipynb
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "DentalInstrument_DINOv2_DetectClassify.ipynb")

MODULES = [
    "config.py",
    "dataset.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "infer.py",
    "augment_dataset.py",
    "det_model.py",
    "det_dataset.py",
    "det_postprocess.py",
    "train_detector.py",
    "evaluate_detector.py",
    "export_to_onnx.py",
    "export_detector_onnx.py",
    "calibrate.py",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md("""# DINOv2 detector + fine-grained classifier (no YOLO)

Pipeline for **14 dental/surgical instruments** on green cloth.

```
frame 560x560
  -> DINOv2-S + light seg decoder     -> mask per instance
  -> connected components             -> bbox + label + mask length (minAreaRect)
  -> crop + tip crops + length (cm)   -> DINOv2-S + ArcFace classifier
```

No YOLO. Mix-dataset photos are **not required** — this notebook generates
multi-instrument scenes with **Mask-Aware Patch-Paste** (lots of them) plus
on-the-fly synthesis during detector training.

Hard pairs (same physical length, differ only at the tip):
- `Needle_Holder` vs `Artery_Forceps` (curved vs straight tip)
- `Mandibular_Universal_Forceps_23` vs `Maxillary_Universal_Forceps_150`
Handled by **tip-zoom augmentation + tip TTA**.

Length still helps: `Root_Elevators` 15.5 cm vs `Root_Tip_Elevator_Straight` 14.5 cm
(after `calibrate.py` on the Pi rig).

**Runtime:** Colab T4 GPU. Upload `dataset.zip` (Roboflow COCO Segmentation) first.
"""))

cells.append(md("## 0) Install"))
cells.append(code("""%pip install -q -U torchao peft transformers pytorch-metric-learning albumentations \\
    opencv-python-headless scikit-learn seaborn tqdm onnxruntime
print("deps ready")
"""))

cells.append(md("## 1) Dataset — upload `dataset.zip` then run"))
cells.append(code("""import os, pathlib, zipfile

found_zip = None
for zname in [
    "dataset.zip", "/content/dataset.zip",
    "Dental Instrument v2.v2i.coco.zip",
    "/content/Dental Instrument v2.v2i.coco.zip",
]:
    if os.path.exists(zname):
        found_zip = zname
        break

if found_zip is not None and not os.path.exists("/content/dataset/train/_annotations.coco.json"):
    print("extracting", found_zip)
    os.makedirs("/content/dataset", exist_ok=True)
    with zipfile.ZipFile(found_zip, "r") as z:
        z.extractall("/content/dataset")

DATA_DIR = None
for c in ["/content/dataset", "dataset", "/content", "."]:
    if os.path.exists(os.path.join(c, "train", "_annotations.coco.json")):
        DATA_DIR = c
        break
if DATA_DIR is None:
    raise FileNotFoundError("upload dataset.zip then re-run this cell")

print("DATA_DIR =", DATA_DIR)
for sp in ["train", "valid", "test"]:
    p = pathlib.Path(DATA_DIR, sp)
    if p.exists():
        n_img = sum(1 for _ in list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))
        print(f"  {sp:6s}: {n_img:4d} images")
"""))

cells.append(md("## 2) Write modules"))
for mod in MODULES:
    path = os.path.join(ROOT, mod)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if not src.endswith("\n"):
        src += "\n"
    cells.append(code(f"%%writefile {mod}\n{src}"))

cells.append(md("""## 3) Sanity check + patch-paste preview

Confirm class counts, mask overlays, and that copy-paste composites look real
on the green cloth.
"""))
cells.append(code("""CALIB_RATIO = None   # set after calibrate.py, e.g. 0.025 cm/px

import sys
sys.path.insert(0, "/content")
from collections import Counter
from dataset import load_coco_records, visualize_records, visualize_patch_paste_samples, compute_length_stats

train_recs, class_names = load_coco_records(DATA_DIR, "train")
dist = Counter(r["class_name"] for r in train_recs)
print(f"classes={len(class_names)}  train images={len(train_recs)}")
for n in class_names:
    print(f"  {n:36s} {dist[n]}")
stats = compute_length_stats(train_recs, CALIB_RATIO)
print(f"length mean={stats[0]:.1f} std={stats[1]:.1f}  unit={'cm' if CALIB_RATIO else 'px'}")

fig1 = visualize_records(train_recs, calibration_ratio=CALIB_RATIO, n=6, seed=7)
print("--- patch-paste preview (3 extra tools / image) ---")
fig2 = visualize_patch_paste_samples(train_recs, n=6, max_pastes=3, seed=42)
"""))

cells.append(md("""## 4) Generate lots of multi-instrument images (patch-paste)

Does **not** wait for a hand-made mix dataset. Writes extra COCO images into
`train/` (original json is backed up to `_annotations.coco.json.bak`).

`num_aug=8` x ~329 train photos ≈ 2600 extra scenes, 2–4 tools each.
Re-run this cell only once (it skips if the backup already exists).
"""))
cells.append(code("""import os, shutil
from augment_dataset import augment_dataset

ann = os.path.join(DATA_DIR, "train", "_annotations.coco.json")
bak = ann + ".bak"
if os.path.exists(bak):
    print("already augmented (found .bak) — skip. Delete the .bak to regenerate.")
else:
    augment_dataset(
        DATA_DIR,
        num_aug=8,
        max_pastes=4,
        max_overlap=0.20,
        seed=42,
    )
    print("done. train images now include patch-paste composites.")

from pathlib import Path
n = len(list(Path(DATA_DIR, "train").glob("*.jpg"))) + len(list(Path(DATA_DIR, "train").glob("*.png")))
print("train images on disk:", n)
"""))

cells.append(md("""## 5) Frozen kNN probe (optional, ~2 min)

If accuracy >> 1/14, DINOv2 features already separate the classes — worth training.
"""))
cells.append(code("""import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import TrainConfig
from dataset import SurgicalInstrumentDataset, compute_length_stats
from model import SurgicalDinoFusion
from train import resolve_records

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
probe = SurgicalDinoFusion(finetune_mode="frozen").to(device).eval()
cfg0 = TrainConfig(data_dir=DATA_DIR)
tr_recs, va_recs, probe_classes = resolve_records(cfg0)
probe_stats = compute_length_stats(tr_recs, CALIB_RATIO)

@torch.no_grad()
def embed(recs, training):
    ds = SurgicalInstrumentDataset(
        recs, probe_stats, cfg0.img_size, CALIB_RATIO,
        None, training, bbox_margin=cfg0.bbox_margin,
    )
    E, Y = [], []
    for b in DataLoader(ds, batch_size=32, num_workers=2):
        out = probe.backbone(pixel_values=b["image"].to(device)).last_hidden_state
        E.append(out[:, 0].cpu())
        Y.append(b["label"])
    return torch.cat(E), torch.cat(Y)

Etr, ytr = embed(tr_recs, False)
Eva, yva = embed(va_recs, False)
pred = ytr[(F.normalize(Eva, dim=1) @ F.normalize(Etr, dim=1).T).argmax(dim=1)]
acc = (pred == yva).float().mean().item()
print(f"kNN probe acc={acc:.3f}  (random={1/len(probe_classes):.3f})")
"""))

cells.append(md("""## 6) Train classifier (DINOv2 + length fusion + ArcFace + tip-zoom)

`tip_zoom_prob=0.35` forces the model to see instrument **tips** — the only
signal for Needle_Holder vs Artery_Forceps and Forceps 23 vs 150 (same length).
"""))
cells.append(code("""from config import TrainConfig
from train import run_training

cfg = TrainConfig(
    data_dir=DATA_DIR,
    img_size=560,
    batch_size=32,
    finetune_mode="lora",
    epochs=50,
    num_workers=2,
    use_cahm=True,
    patch_paste_prob=0.4,
    patch_paste_max_objects=3,
    tip_zoom_prob=0.35,
    tip_zoom_size=0.42,
    calibration_ratio=CALIB_RATIO,
    output_dir="outputs",
)
try:
    best_ckpt = run_training(cfg)
except RuntimeError as e:
    if "out of memory" not in str(e).lower():
        raise
    import torch
    torch.cuda.empty_cache()
    cfg.batch_size = 16
    best_ckpt = run_training(cfg)
print("classifier ckpt:", best_ckpt)
"""))

cells.append(md("## 7) Evaluate classifier"))
cells.append(code("""from evaluate import evaluate_checkpoint
metrics = evaluate_checkpoint(best_ckpt, data_dir=DATA_DIR)
print(f"acc={metrics['accuracy']:.4f}  bal={metrics['balanced_accuracy']:.4f}")
"""))

cells.append(md("""## 8) Train detector (DINOv2 seg head, no YOLO)

On-the-fly multi-tool scenes every step (2–5 instruments on green cloth +
shadows) plus the real photos (including the patch-paste images from step 4).
Best checkpoint is picked by **instance F1** on real val/test photos.
"""))
cells.append(code("""from dataclasses import replace
from config import DetectorConfig
from train_detector import run_training as run_detector

dcfg = DetectorConfig(
    data_dir=DATA_DIR,
    img_size=560,
    batch_size=8,
    finetune_mode="lora",
    epochs=60,
    num_workers=2,
    synth_min_objects=2,
    synth_max_objects=5,
    synth_max_overlap=0.20,
    output_dir="outputs_detector",
)
try:
    det_ckpt = run_detector(dcfg)
except RuntimeError as e:
    if "out of memory" not in str(e).lower():
        raise
    import torch
    torch.cuda.empty_cache()
    dcfg = replace(dcfg, batch_size=4)
    det_ckpt = run_detector(dcfg)
print("detector ckpt:", det_ckpt)
"""))

cells.append(md("## 9) Evaluate detector (P/R/F1 + overlays)"))
cells.append(code("""from evaluate_detector import main as _eval_det
import sys
sys.argv = ["evaluate_detector.py", "--ckpt", det_ckpt, "--data_dir", DATA_DIR,
            "--out_dir", "outputs_detector/eval"]
_eval_det()
"""))

cells.append(md("""## 10) Export ONNX (classifier + detector)

Copy the `onnx_export/` folder to Raspberry Pi 5 (`pi_final_v3/onnx_export/`).
Calibrate cm/px later on the rig with `calibrate.py`.
"""))
cells.append(code("""from export_to_onnx import main as export_cls
from export_detector_onnx import main as export_det
import sys

sys.argv = ["export_to_onnx.py", "--ckpt", best_ckpt, "--out_dir", "onnx_export"]
export_cls()
sys.argv = ["export_detector_onnx.py", "--ckpt", det_ckpt, "--out_dir", "onnx_export"]
export_det()
print("onnx_export contents:")
import os
print(os.listdir("onnx_export"))
"""))

cells.append(md("## 11) Download checkpoints + ONNX"))
cells.append(code("""from google.colab import files
import os, zipfile

zip_path = "/content/dino_detect_classify_export.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ["outputs", "outputs_detector", "onnx_export"]:
        if not os.path.isdir(folder):
            continue
        for root, _, fnames in os.walk(folder):
            for fn in fnames:
                p = os.path.join(root, fn)
                z.write(p, p)
print("zip", zip_path, os.path.getsize(zip_path))
files.download(zip_path)
"""))

cells.append(md("""---
### After Colab — realtime on Raspberry Pi 5

1. Unzip into `pi_final_v3/onnx_export/`
2. One-time length calibration (photograph a ruler on the cloth):
   `python calibrate.py --image ruler.jpg --known_cm 15.5 --detector_meta pi_final_v3/onnx_export/detector_meta.json --classifier_meta pi_final_v3/onnx_export/classifier_meta.json`
3. Realtime server on Pi 5 — **production (gunicorn)**:

   ```bash
   cd pi_final_v3
   pip install onnxruntime opencv-python flask flask-cors numpy gunicorn
   gunicorn --workers 1 --threads 8 --worker-class gthread --timeout 0 \\
       --bind 0.0.0.0:8000 app:app
   ```

   - `--workers 1` ห้ามมากกว่านี้ (กล้องเปิดได้ process เดียว; background threads
     start on import)
   - ห้ามใช้ `--preload` (threads จะ start ใน master แล้วตายตอน fork)
   - `--timeout 0` กัน stream MJPEG โดนตัด
   - หรือ dev mode: `python app.py`

4. ดูผล realtime: `http://<pi-ip>:8000/video_feed?token=<API_KEY>`
   และ JSON: `/detects?token=<API_KEY>`

เมื่อ mix dataset จริงมา วาง COCO ลง `dataset/` แล้วรัน cell 3 เป็นต้นไปใหม่ —
loader รองรับหลาย annotation ต่อภาพอยู่แล้ว (patch-paste อุดช่องว่างไปก่อน)
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [],
}
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
    nb["cells"].append(c)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes, {len(cells)} cells)")
