# -*- coding: utf-8 -*-
"""tools/make_notebook.py — build Colab notebook (.ipynb) that embeds all modules via %%writefile

Run: python tools/make_notebook.py
Output: DentalInstrument_DINOv2_ArcFace.ipynb (upload to Google Colab)
Guaranteed that notebook content = actual .py files in the repo (read from disk at generate time)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["config.py", "dataset.py", "model.py", "train.py", "evaluate.py", "infer.py"]
OUT = os.path.join(ROOT, "DentalInstrument_DINOv2_ArcFace.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# ---------------------------------------------------------------- intro
cells.append(md("""# 🦷 Fine-grained Classification of Surgical Instruments — DINOv2 + Length Fusion + ArcFace

Pipeline for classifying **14 surgical instrument classes** (~400 images) on a green cloth with shadows. Some classes differ only by length:

```
image ──► DINOv2 ViT-S/14 ──► CLS embedding (384-dim)──┐
                                                      ├─ concat(385) ─► Linear ─► 384 ─► ArcFace loss
mask ──► minAreaRect ──► length (normalized) ─────────┘
```

**What this notebook does:** install deps → write all 6 modules (`config/dataset/model/train/evaluate/infer`) → sanity-check data → train (LoRA + warmup/cosine + early stopping) → confusion matrix → inference demo

**Data to prepare:** Roboflow export as *COCO Segmentation* with structure
`DATA_DIR/train/_annotations.coco.json` + `DATA_DIR/valid/_annotations.coco.json`
"""))

# ---------------------------------------------------------------- install
cells.append(md("## 0) Install dependencies"))
cells.append(code("%pip install -q -U torchao peft transformers pytorch-metric-learning albumentations "
                  "opencv-python-headless scikit-learn seaborn tqdm\nprint('✅ deps ready')"))

# ---------------------------------------------------------------- data
cells.append(md("""## 1) Prepare data — pick **one method** and run this cell"""))
cells.append(code('''# ── Set DATA_DIR to where your dataset lives ──────────────────────────────
DATA_DIR = "/content/dental_dataset"

# ▸ Method A: Upload a .zip file (exported from Roboflow and zipped)
# from google.colab import files
# up = files.upload()                       # select dataset.zip
# !unzip -o -q "{list(up.keys())[0]}" -d /content/
# DATA_DIR = "/content/dataset"             # ← adjust to the folder name inside the zip

# ▸ Method B: Dataset already on Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = "/content/drive/MyDrive/path/to/dataset"

# ▸ Method C: Pull directly from Roboflow (set API_KEY / WORKSPACE / PROJECT / VERSION)
# %pip install -q roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# rf.workspace("WORKSPACE").project("PROJECT").version(VERSION)\\
#   .download("coco-segmentation", location=DATA_DIR)

import pathlib
assert pathlib.Path(DATA_DIR, "train", "_annotations.coco.json").exists(), \\
    f"not found {DATA_DIR}/train/_annotations.coco.json — run the data-import cell first"
for sp in ["train", "valid", "test"]:
    p = pathlib.Path(DATA_DIR, sp)
    if p.exists():
        n_img = len(list(p.glob('*.jpg'))) + len(list(p.glob('*.png'))) + len(list(p.glob('*.jpeg')))
        n_json = len(list(p.glob('*annotations*.json')))
        print(f"{sp:6s}: {n_img:4d} images, {n_json} annotation file(s)")'''))

# ---------------------------------------------------------------- modules
cells.append(md("""## 2) Write module files (%%writefile)

Each cell creates a .py file by role — you can edit the code from the file editor on the left side of Colab after running"""))

for mod in MODULES:
    with open(os.path.join(ROOT, mod), encoding="utf-8") as f:
        src = f.read()
    cells.append(code(f"%%writefile {mod}\n{src}" if not src.endswith("\n") else f"%%writefile {mod}\n{src}"))

# also write tools/evaluate_ablation.py for Phase 3 ablation comparison
cells.append(code("!mkdir -p tools"))
import pathlib as _pl
_tool_path = os.path.join(ROOT, "tools", "evaluate_ablation.py")
if os.path.exists(_tool_path):
    with open(_tool_path, encoding="utf-8") as _f:
        _tool_src = _f.read()
    cells.append(code(f"%%writefile tools/evaluate_ablation.py\n{_tool_src}"))

# ---------------------------------------------------------------- sanity check
cells.append(md("""## 3) Data sanity check

Check: images per class, mask overlay correctness, whether measured lengths look sensible — **always before training**"""))
cells.append(code('''CALIB_RATIO = None   # ← cm/pixel if you have a reference object (e.g. 0.05); None = use pixels

import sys; sys.path.insert(0, "/content")
from collections import Counter

from dataset import load_coco_records, visualize_records, compute_length_stats

train_recs, class_names = load_coco_records(DATA_DIR, "train")
dist = Counter(r["class_name"] for r in train_recs)
print(f"Total classes: {len(class_names)}")
for n in class_names:
    print(f"  {n:30s}: {dist[n]} images")

stats = compute_length_stats(train_recs, CALIB_RATIO)
unit = "cm" if CALIB_RATIO else "px"
print(f"\\nlength stats (train): mean={stats[0]:.1f} {unit}, std={stats[1]:.1f}")

fig = visualize_records(train_recs, calibration_ratio=CALIB_RATIO, n=6, seed=7)'''))

cells.append(md("""## 3.5) Check whether DINOv2 weights are ready (frozen kNN probe — no training needed)

Use frozen DINOv2 to extract features → predict with 1-nearest-neighbor:
- Accuracy far above random (1/14 ≈ 7%) (e.g. >40-50%) = features already separate classes well, worth training further
- Accuracy near random = domain gap too large → try `img_size=518` (sharper) or a larger backbone"""))
cells.append(code('''# ponytail: probe trains for zero steps — check feature quality before investing in full training
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import SurgicalInstrumentDataset, compute_length_stats
from model import SurgicalDinoFusion
from train import resolve_records

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
probe_model = SurgicalDinoFusion(finetune_mode="frozen").to(device).eval()

cfg0 = TrainConfig(data_dir=DATA_DIR)
tr_recs, va_recs, probe_classes = resolve_records(cfg0)
probe_stats = compute_length_stats(tr_recs, CALIB_RATIO)
flip_all = [True] * len(probe_classes)

@torch.no_grad()
def embed(recs, training):
    ds = SurgicalInstrumentDataset(recs, probe_stats, cfg0.img_size, CALIB_RATIO,
                                   flip_all if training else None, training,
                                   bbox_margin=cfg0.bbox_margin)
    E, Y = [], []
    for b in DataLoader(ds, batch_size=32, num_workers=2):
        out = probe_model.backbone(pixel_values=b["image"].to(device)).last_hidden_state
        E.append(out[:, 0].cpu())          # raw CLS — not through the fusion head (head is still randomly initialized)
        Y.append(b["label"])
    return torch.cat(E), torch.cat(Y)

Etr, ytr = embed(tr_recs, True)            # training-side augmentation enabled = free data augmentation for the probe
Eva, yva = embed(va_recs, False)
sim = F.normalize(Eva, dim=1) @ F.normalize(Etr, dim=1).T
pred = ytr[sim.argmax(dim=1)]
acc = (pred == yva).float().mean().item()
print(f"kNN probe accuracy: {acc:.3f}   (random = {1 / len(probe_classes):.3f})")
if acc < 0.30:
    print("⚠️ features barely separate classes — try img_size=518 or dinov2-base before full training")
else:
    print("✅ features look good — ready to train")'''))

# ---------------------------------------------------------------- train
cells.append(md("""## 4) Train

- `finetune_mode="lora"` → train only LoRA adapters (very few parameters) to prevent overfitting
- Early stopping on validation loss; the best checkpoint is saved automatically
- For cross-validation evaluation → set `kfold=5`"""))
cells.append(code('''from config import TrainConfig
from train import run_training, run_kfold

cfg = TrainConfig(
    data_dir=DATA_DIR,
    img_size=504,            # 504=36×14 — from experiments (divisible by 14, best size in our tests)
    batch_size=32,
    finetune_mode="lora",     # "lora" (recommended) | "partial" | "frozen"
    epochs=50,
    num_workers=2,
    use_cahm=True,
    # ---- optional extras ----
    # calibration_ratio=CALIB_RATIO,
    # flip_allowed=["class_a", "class_b"],  # only classes that may be flipped (others are not flipped)
    # kfold=5,                              # Stratified 5-fold CV
    # use_cahm=False,  # to run baseline without CAHM
)

if cfg.kfold:
    paths, accs = run_kfold(cfg)
    best_ckpt = paths[accs.index(max(accs))]   # pick the best fold
else:
    best_ckpt = run_training(cfg)

print("\\nBest checkpoint:", best_ckpt)'''))

# ---------------------------------------------------------------- evaluate
cells.append(md("""## 5) Evaluate — accuracy + confusion matrix

Bright off-diagonal cells in the heatmap = class pairs the model confuses (often pairs that differ only in size)"""))
cells.append(code('''from evaluate import evaluate_checkpoint

metrics = evaluate_checkpoint(best_ckpt)
print(f"\\nAccuracy: {metrics['accuracy']:.4f}")'''))

# ---------------------------------------------------------------- ablation training (Phase 3)
cells.append(md("""## 5.5) Phase 3 — Ablation (baseline vs CAHM)

CAHM is kept (`use_cahm=True` by default). Run both to reproduce the table in README.
Each variant early-stops at ~20-35 epochs (patience 12) | 504 px batch 32 on T4 ~7-8 GB — if OOM, batch is automatically reduced to 16.
"""))
cells.append(code('''from config import TrainConfig
from train import run_training
import torch

ablation = {
    "baseline":  {"use_cahm": False},
    "cahm":      {"use_cahm": True},
}

ckpt_map = {}
for name, extra in ablation.items():
    print(f"\\n{'='*20} {name} {extra} {'='*20}")
    cfg_ab = TrainConfig(
        data_dir=DATA_DIR,
        img_size=504, batch_size=32,
        finetune_mode="lora", epochs=50, num_workers=2,
        output_dir=f"outputs_ablation/{name}",
        **extra
    )
    try:
        ckpt = run_training(cfg_ab)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[OOM] Reducing batch 32 → 16 and retrying")
            torch.cuda.empty_cache()
            cfg_ab.batch_size = 16
            ckpt = run_training(cfg_ab)
        else:
            raise
    ckpt_map[name] = ckpt
    print(f"[{name}] ✅ {ckpt}")

print("\\n--- ckpt_map ---")
for k,v in ckpt_map.items():
    print(f'{k}: \"{v}\"')'''))


cells.append(md("""## 5.6) Compare ablations — Phase 3 table

Measure Val Acc / Balanced Acc / Needle_Holder↔Artery_Forceps error for baseline vs CAHM
via `tools/evaluate_ablation.py` (runnable both in the notebook and as CLI)
"""))
cells.append(code('''from tools.evaluate_ablation import compare_checkpoints

# Use ckpt_map from the previous cell; if you haven't run ablations, set paths manually:
# ckpt_map = {
#     "baseline": "outputs_ablation/baseline/best_model.pt",
#     "cahm": "outputs_ablation/cahm/best_model.pt",
# }
results = compare_checkpoints(ckpt_map, data_dir=DATA_DIR, save_csv="outputs_ablation/ablation_results.csv")
print("\\n--- Summary ---")
for r in results:
    if r.get("found"):
        print(f'{r["name"]:12s} acc={r["accuracy"]:.4f} bal={r["balanced_acc"]:.4f} {r["needle_detail"]}')'''))

# ---------------------------------------------------------------- inference
cells.append(md("""## 6) Inference demo — new image + mask → class + confidence"""))
cells.append(code('''from infer import load_pipeline, predict_record
from train import resolve_records

pack = load_pipeline(best_ckpt)
_, valid_recs, _ = resolve_records(cfg)   # try 3 validation images

for r in valid_recs[:3]:
    res = predict_record(pack, r)
    ok = "✅" if res["class"] == res["truth"] else "❌"
    tops = ", ".join(f'{t["class"]}:{t["prob"]:.2f}' for t in res["top3"])
    print(f'{ok} truth={res["truth"]:20s} pred={res["class"]:20s} conf={res["confidence"]:.3f}')
    print("      top3:", tops)'''))

# ---------------------------------------------------------------- save
cells.append(md("""## 7) Save model back to Drive / Download"""))
cells.append(code('''# ▸ Download to local machine
from google.colab import files
files.download(best_ckpt)

# ▸ Or copy to Drive (mount first if not already mounted)
# !cp "{best_ckpt}" /content/drive/MyDrive/'''))
cells.append(md("""---
### 💡 Tips
- **calibration_ratio**: place an object of known real length L (cm) under the same camera rig → `ratio = L / measure_length_px(mask)` — the length feature will be in cm and easier to interpret
- **handedness**: for left/right-handed classes → list only the classes that may be flipped in `flip_allowed`

### 🎯 Playbook for near-indistinguishable class pairs (e.g. Universal forceps 150 vs 151, Curved Root elevator vs Straight Root-tip elevator)
1. **Increase resolution first** — the head of forceps at 224 px is only ~20 px; set `img_size=518` (=37×14, accepted by DINOv2) and patch tokens will carry ~2.3× more "head" detail; if GPU memory is tight, reduce batch to 8
2. **Pairs that differ in length** (short curved elevator vs long straight) — the length feature already handles this, but the mask must cover the true tip (shadow included in mask = length error = feature error)
3. **Collect targeted extra data**: check the confusion matrix in step 5 → add more shots only for the confused pairs, rotating yaw by ~30° each time (some angles hide the head = you need angles that reveal the distinguishing point)
4. **Still heavily confused?** Plan B: two-stage — first separate coarse groups (forceps/elevator/probe…) then a pair-specific classifier on a crop around the head

### 📋 Checklist for the next annotation round (Roboflow COCO Segmentation)
- Polygon must cover **the whole instrument** including head and tip, but **exclude shadows** (shadows stretch minAreaRect)
- 1 annotation per instrument (multiple per image allowed — pipeline crops per instance via bbox_margin=0.15 from experiments)
- Do not rename classes after they are set (label = sorted name order)
- Keep ~25+ images per class and spread shooting angles evenly
- Green cloth: pipeline already has shadow augmentation, but shooting with even lighting is still best

- GPU: Runtime → Change runtime type → T4 GPU (CPU works but is slow); `img_size=518` + T4 + batch 8 ≈ fits
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
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
