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
    "train_all.py",
    "build_prototypes.py",
    "train_student.py",
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

**Runtime (pick one — auto-detected):**
- **Google Colab** (T4) — upload `dataset.zip` first
- **Local machine via VS Code** (Colab/notebook extension, own GPU as kernel) —
  expects `dataset/` in the repo; batch sizes auto-scale to your VRAM,
  no package upgrades, Windows-safe workers

Upload `dataset.zip` (Roboflow COCO Segmentation) when on Colab.
"""))

cells.append(md("## 0) Environment + dependencies"))
cells.append(code('''import sys, subprocess, importlib.util

# ── detect where we are ─────────────────────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print("environment:", "Colab (cloud T4)" if IN_COLAB else "LOCAL machine (own GPU)")

# ── verify deps; install ONLY what is missing ───────────────────
# (local: never -U — upgrading torch/transformers can break a working venv)
required = {
    "torch": "torch", "transformers": "transformers", "peft": "peft",
    "pytorch_metric_learning": "pytorch-metric-learning",
    "albumentations": "albumentations", "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn", "onnxruntime": "onnxruntime", "tqdm": "tqdm",
}
missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    cmd = [sys.executable, "-m", "pip", "install"] + (["-q"] if not IN_COLAB else ["-q", "-U"]) + missing
    print("installing:", missing)
    subprocess.run(cmd, check=True)
else:
    print("deps OK")

if IN_COLAB:
    # peft 0.17+ needs torchao >= 0.16 (Colab images ship 0.10)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=False)
    print(">>> Colab: if peft/torchao upgraded, Runtime -> Restart session, then continue from cell 1")
'''))

cells.append(md("## 1) Dataset"))
cells.append(code('''import os, pathlib, zipfile

DATA_DIR = None
if IN_COLAB:
    # upload dataset.zip via the left file panel (or it sits in /content)
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
    candidates = ["/content/dataset", "dataset", "/content", "."]
else:
    # local machine: dataset/ of this repo (or a dataset.zip next to it)
    here = os.path.dirname(os.path.abspath("__file__")) if False else os.getcwd()
    if not os.path.exists(os.path.join(here, "dataset", "train", "_annotations.coco.json")) \\
            and os.path.exists(os.path.join(here, "dataset.zip")):
        print("extracting local dataset.zip")
        with zipfile.ZipFile(os.path.join(here, "dataset.zip")) as z:
            z.extractall(here)
    candidates = [here, os.path.join(here, ".."), "."]

for c in candidates:
    if os.path.exists(os.path.join(c, "train", "_annotations.coco.json")):
        DATA_DIR = c
        break
if DATA_DIR is None:
    raise FileNotFoundError("no train/_annotations.coco.json — put the COCO export in ./dataset (local) or upload dataset.zip (Colab)")

print("DATA_DIR =", os.path.abspath(DATA_DIR))
for sp in ["train", "valid", "test"]:
    p = pathlib.Path(DATA_DIR, sp)
    if p.exists():
        n_img = sum(1 for _ in list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))
        print(f"  {sp:6s}: {n_img:4d} images")
'''))

cells.append(md("""## 2) Write modules

Colab: writes the 16 modules to /content via %%writefile.
Local: **copies nothing** — the repo .py files ARE the modules (this cell only
makes sure they are importable and prints a quick inventory).
"""))
for mod in MODULES:
    path = os.path.join(ROOT, mod)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if not src.endswith("\n"):
        src += "\n"
    cells.append(code(f"%%writefile {mod}\n{src}"))
cells.append(code('''# ── make modules importable ──────────────────────────────────────
import os, sys

if IN_COLAB:
    sys.path.insert(0, "/content")
else:
    # local: cwd should be the repo root (notebook sits next to the .py files)
    repo = os.getcwd()
    sys.path.insert(0, repo)
    missing = [m for m in ["config", "dataset", "model", "train", "det_model",
                           "det_dataset", "train_all"] if not os.path.exists(os.path.join(repo, m + ".py"))]
    if missing:
        raise FileNotFoundError(f"run the notebook from the repo root (missing {missing})")

import config, dataset, model, train, det_model, det_dataset, train_all  # noqa
print("modules import OK from", os.path.abspath(config.__file__))
'''))

cells.append(md("""## 3) Sanity check + patch-paste preview

Confirm class counts, mask overlays, and that copy-paste composites look real
on the green cloth.
"""))
cells.append(code('''CALIB_RATIO = None   # set after calibrate.py, e.g. 0.025 cm/px

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
'''))

cells.append(md("""## 4) Generate lots of multi-instrument images (patch-paste)

Writes extra COCO images into `train/` (original json is backed up to
`_annotations.coco.json.bak`). Re-running is safe — the cell skips when
the backup exists (delete `.bak` + `aug_*` files to regenerate).

Recommended for v3 (already has real mix photos): `num_aug=2, max_pastes=2`
→ ~700 new scenes. (The old 8/4 recipe was for v1/v2 single-tool photos.)

Built-in guards: `*_Head` annotations are never copied, and base photos
that already have ≥6 tools are skipped (no unrealistic clutter).
Set `EXTRA_DIRS` to augment + train on more dataset folders at once.
"""))
cells.append(code('''import os, shutil

# extra COCO dataset folders (same 14 classes) — merged into training
EXTRA_DIRS = []

from augment_dataset import augment_dataset

for _d in [DATA_DIR] + EXTRA_DIRS:
    ann = os.path.join(_d, "train", "_annotations.coco.json")
    bak = ann + ".bak"
    if os.path.exists(bak):
        print(f"{_d}: already augmented (found .bak) — skip. Delete the .bak to regenerate.")
        continue
    augment_dataset(
        _d,
        num_aug=2,
        max_pastes=2,
        max_overlap=0.20,
        seed=42,
    )
    print(f"{_d}: done. train images now include patch-paste composites.")

from pathlib import Path
n = len(list(Path(DATA_DIR, "train").glob("*.jpg"))) + len(list(Path(DATA_DIR, "train").glob("*.png")))
print("train images on disk:", n)
'''))

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
import platform as _plat
_NW = 0 if _plat.system() == "Windows" else 2  # spawn workers hang notebooks on Windows
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
    for b in DataLoader(ds, batch_size=32, num_workers=_NW):
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

cells.append(md("""## 6) Train — one command (train_all.py: detector + classifier)

`train_all.py` runs BOTH trainings sequentially with YOLO-style progress
(tqdm per epoch + ETA + summary tables):

- Stage 1 DETECTOR: DINOv2 + seg head, on-the-fly multi-tool scenes,
  instance-F1 checkpoint selection
- Stage 2 CLASSIFIER: DINOv2 + length fusion + ArcFace + CAHM + tip-zoom
  (`tip_zoom_prob=0.35` — forces the model to learn instrument TIPS, the
  only signal for Needle_Holder vs Artery_Forceps and 23 vs 150, same length)

Optional: reuse an existing classifier checkpoint — upload `outputs.zip` and
run the RESUME cell below first, then set `SKIP_CLASSIFIER=True`.

Stuck with no output for >10 min? The process hung before printing (usually
DataLoader workers or a stalled download) — NOT slow training. Do this:
1. Runtime → Interrupt execution
2. Run the SMOKE cell (6b) below — if it finishes, workers/GPU are fine
3. Re-run cell 6 with `FORCE_WORKERS = 0` at the top
4. After any interrupt/crash, open `train_all.log` — every line is mirrored there
"""))
cells.append(code('''# ── RESUME (optional): reuse a saved classifier from a previous run ──
# Colab: upload outputs.zip (left panel). Local: outputs.zip next to the repo.
# Extracts the checkpoint and points best_ckpt at it.
import zipfile, os, glob

found = None
for zname in ["outputs.zip", "outputs.zip", "../outputs.zip"]:
    z = os.path.abspath(zname)
    if os.path.exists(z):
        found = z
        break

target_dir = "/content" if IN_COLAB else os.getcwd()
if found and not glob.glob(os.path.join(target_dir, "outputs", "best_model*.pt")) \\
        and not glob.glob(os.path.join(target_dir, "outputs*", "**", "best_model*.pt"), recursive=True):
    print("extracting", found)
    with zipfile.ZipFile(found) as z:
        z.extractall(target_dir)

cands = sorted(glob.glob(os.path.join(target_dir, "outputs", "best_model*.pt"))
               + glob.glob(os.path.join(target_dir, "outputs*", "**", "best_model*.pt"), recursive=True))
if cands:
    best_ckpt = cands[0]
    print("reuse classifier:", best_ckpt)
else:
    print("no saved classifier found - classifier will be trained fresh")

SKIP_CLASSIFIER = bool(cands)   # True if reusing
'''))

cells.append(code('''# ── MAIN TRAINING — one command, progress like YOLO ──
# detector @448 (fast on Pi) + classifier @560 (tip detail)
# auto-scales batch sizes to the GPU at hand; Windows local -> workers 0
# Output streams line-by-line below (plus full copy in train_all.log).
# If NOTHING appears for >10 min -> the process is stuck, not slow:
#   Runtime -> Interrupt, then set FORCE_WORKERS = 0 and re-run this cell.
import subprocess, sys, torch

FORCE_WORKERS = None   # None = auto (0 on Windows, 2 elsewhere); set 0 to debug hangs

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU?!'} ({vram_gb:.1f} GB)", flush=True)

if vram_gb >= 14:   bs_det, bs_cls = 16, 32   # T4-safe (24 risks OOM spikes)
elif vram_gb >= 7:  bs_det, bs_cls = 8, 16
elif vram_gb >= 3:  bs_det, bs_cls = 4, 8
else:               bs_det, bs_cls = 2, 4

# Windows kernels hang with spawn workers; Linux/WSL/Colab want 2
import platform as _plat
WORKERS = FORCE_WORKERS if FORCE_WORKERS is not None else (0 if _plat.system() == "Windows" else 2)
print(f"batch_det={bs_det} batch_cls={bs_cls} workers={WORKERS}", flush=True)

# extra dataset folders from cell 4 (EXTRA_DIRS) are merged into training
_extra = EXTRA_DIRS if "EXTRA_DIRS" in dir() else []

cmd = [
    sys.executable, "-u", "train_all.py",
    "--data_dir", DATA_DIR,
    "--epochs_det", "60",
    "--epochs_cls", "50",
    "--batch_det", str(bs_det),
    "--batch_cls", str(bs_cls),
    "--img_size_det", "448",
    "--img_size_cls", "504",
    "--num_workers", str(WORKERS),
    "--log_file", "train_all.log",
] + (["--data_dirs"] + _extra if _extra else [])
if SKIP_CLASSIFIER:
    cmd.append("--skip_classifier")
print("cmd:", " ".join(cmd), flush=True)
print("tip: full copy of this output is also saved to train_all.log", flush=True)

# stream child output line-by-line (never silent — proves the process is alive)
proc = subprocess.Popen(cmd, cwd=os.getcwd() if not IN_COLAB else None,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print(f"\\n[train_all] exit code: {rc}", flush=True)
if rc != 0:
    raise SystemExit(f"train_all failed (exit {rc}) — see train_all.log")
'''))

cells.append(md("""## 6b) SMOKE TEST (~5 min, run this first if cell 6 ever hangs)

Trains the classifier for exactly 1 epoch. If this finishes, imports +
GPU + DataLoader workers + checkpoint saving all work — the full run is
safe to start. If THIS hangs with no output, the problem is environmental
(workers/download), not the training code.
"""))
cells.append(code('''# ── SMOKE: 1-epoch classifier run — proves the stack works ──
import subprocess, sys

smoke_cmd = [
    sys.executable, "-u", "train_all.py",
    "--data_dir", DATA_DIR,
    "--skip_detector",
    "--epochs_cls", "1",
    "--batch_cls", "16",
    "--img_size_cls", "560",
    "--num_workers", str(WORKERS),
    "--log_file", "train_smoke.log",
]
print("cmd:", " ".join(smoke_cmd), flush=True)
proc = subprocess.Popen(smoke_cmd, cwd=os.getcwd() if not IN_COLAB else None,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print(f"\\n[smoke] exit code: {rc} (0 = stack OK, start cell 6)", flush=True)
'''))

cells.append(md("""## 7) Evaluate classifier"""))
cells.append(code('''import glob, os
cands = sorted(glob.glob(os.path.join(os.getcwd(), "outputs", "best_model*.pt"))
               + glob.glob(os.path.join(os.getcwd(), "outputs*", "best_model*.pt")))
if "best_ckpt" not in dir() or not best_ckpt or not os.path.exists(best_ckpt):
    if not cands:
        raise FileNotFoundError("no classifier checkpoint — train first or use the RESUME cell")
    best_ckpt = cands[0]

from evaluate import evaluate_checkpoint
metrics = evaluate_checkpoint(best_ckpt, data_dir=DATA_DIR)
print(f"acc={metrics['accuracy']:.4f}  bal={metrics['balanced_accuracy']:.4f}")
'''))

cells.append(md("""## 8) (Optional) Detector-only training / re-training

Only needed if you want to retrain the detector separately with different
settings. Stage 1 of the previous cell already trained it.
"""))
cells.append(code("""# ── OPTIONAL — detector only ──
# from dataclasses import replace
# from config import DetectorConfig
# from train_detector import run_training as run_detector
# dcfg = DetectorConfig(data_dir=DATA_DIR, img_size=560, batch_size=16,
#                       epochs=60, finetune_mode="lora", num_workers=2)
# det_ckpt = run_detector(dcfg)
# print("detector ckpt:", det_ckpt)
"""))

cells.append(md("## 9) Evaluate detector (P/R/F1 + overlays)"))
cells.append(code('''import glob, os, sys
cands = sorted(glob.glob(os.path.join(os.getcwd(), "outputs_detector", "best_detector.pt"))
               + glob.glob(os.path.join(os.getcwd(), "outputs_detector*", "best_detector.pt")))
if not cands:
    raise FileNotFoundError("no detector checkpoint in outputs_detector/")
det_ckpt = cands[0]

from evaluate_detector import main as _eval_det
sys.argv = ["evaluate_detector.py", "--ckpt", det_ckpt, "--data_dir", DATA_DIR,
            "--out_dir", "outputs_detector/eval"]
_eval_det()
'''))

cells.append(md("""## 10) Export ONNX (classifier + detector)

Copy the `onnx_export/` folder to Raspberry Pi 5 (`pi_final_v3/onnx_export/`).
Calibrate cm/px later on the rig with `calibrate.py`.
"""))
cells.append(code('''import glob, os
cands = sorted(glob.glob(os.path.join(os.getcwd(), "outputs", "best_model*.pt"))
               + glob.glob(os.path.join(os.getcwd(), "outputs*", "best_model*.pt")))
det_cands = sorted(glob.glob(os.path.join(os.getcwd(), "outputs_detector", "best_detector.pt")))
if not cands or not det_cands:
    raise FileNotFoundError("missing checkpoints — train first (or RESUME cell for classifier)")
best_ckpt = cands[0]
det_ckpt = det_cands[0]

import export_to_onnx as _ec
import export_detector_onnx as _ed
_ec.export_all(best_ckpt, "onnx_export")
# with_tokens=True: extra patch_tokens output → zero-cost prototype labels on Pi
_ed.export_all(det_ckpt, "onnx_export", with_tokens=True)

import build_prototypes as _bp
_bp.main(["--ckpt", det_ckpt, "--data_dir", DATA_DIR, "--out_dir", "onnx_export"])

print("onnx_export contents:", os.listdir("onnx_export"))
'''))

cells.append(md("""## 10b) (Recommended for Pi realtime) Train the tiny student

Distills the DINOv2 detector into LRASPP-MobileNetV3 (~3.2M params) at 320px.
Verified locally: **15ms vs 705ms per frame (47x)** for the same contract
(bbox+label via identical post-processing). The DINOv2 detector stays as
teacher/fallback; the Pi prefers the student file automatically.
Takes ~1-2h on T4.
"""))
cells.append(code('''import glob, os
tcands = sorted(glob.glob(os.path.join(os.getcwd(), "outputs_detector", "best_detector.pt")))
if not tcands:
    raise FileNotFoundError("train the detector first (cell 6)")
teacher_ckpt = tcands[0]

from train_student import run_training as run_student, export_all_student
from config import DetectorConfig
_extra = EXTRA_DIRS if "EXTRA_DIRS" in dir() else []
scfg = DetectorConfig(data_dir=DATA_DIR, extra_data_dirs=_extra or None)
stud_ckpt = run_student(scfg, teacher_ckpt, img_size=320, epochs=40,
                        batch_size=16, num_workers=2,
                        output_dir="outputs_student")
export_all_student(stud_ckpt, "onnx_export", img_size=320, int8=False)
print("student onnx ready — Pi will auto-prefer detector_dino_student.onnx")
'''))

cells.append(md("## 11) Download checkpoints + ONNX"))
cells.append(code('''import os, zipfile

zip_path = os.path.join(os.getcwd(), "dino_detect_classify_export.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ["outputs", "outputs_detector", "onnx_export"]:
        if not os.path.isdir(folder):
            continue
        for root, _, fnames in os.walk(folder):
            for fn in fnames:
                p = os.path.join(root, fn)
                z.write(p, p)
print("zip ready:", zip_path, os.path.getsize(zip_path), "bytes")

# browser Colab → auto-download; VS Code / other kernels → grab it from the
# file explorer (right-click -> Download) or copy to Drive
try:
    from google.colab import files
    files.download(zip_path)
    print("downloading in browser ...")
except Exception:
    print("not in browser-Colab — download it manually from the file explorer:")
    print("  ", zip_path)
'''))

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
