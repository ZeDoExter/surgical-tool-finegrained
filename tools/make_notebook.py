# -*- coding: utf-8 -*-
"""tools/make_notebook.py — สร้าง Colab notebook (.ipynb) ที่ฝังโมดูลทั้งหมดผ่าน %%writefile

รัน: python tools/make_notebook.py
ผลลัพธ์: DentalInstrument_DINOv2_ArcFace.ipynb (อัปโหลดขึ้น Google Colab ได้เลย)
การันตีว่าเนื้อหาใน notebook = ไฟล์ .py จริงใน repo (อ่านจากดิสก์ตอน generate)
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
cells.append(md("""# 🦷 Fine-grained Classification เครื่องมือผ่าตัด — DINOv2 + Length Fusion + ArcFace

Pipeline สำหรับจำแนก **เครื่องมือผ่าตัด 14 classes** (~400 ภาพ) โลหะสีเงินบนถาดสีเงิน บางคู่ class ต่างกันแค่ความยาว:

```
ภาพ ──► DINOv2 ViT-S/14 ──► CLS embedding (384-dim)──┐
                                                      ├─ concat(385) ─► Linear ─► 384 ─► ArcFace loss
mask ──► minAreaRect ──► ความยาว (normalize) ─────────┘
```

**สิ่งที่ notebook นี้ทำ:** ติดตั้ง deps → เขียนโมดูลทั้ง 6 ไฟล์ (`config/dataset/model/train/evaluate/infer`) → sanity-check ข้อมูล → เทรน (LoRA + warmup/cosine + early stopping) → confusion matrix → inference demo

**ข้อมูลที่ต้องเตรียม:** Roboflow export แบบ *COCO Segmentation* ที่มีโครงสร้าง
`DATA_DIR/train/_annotations.coco.json` + `DATA_DIR/valid/_annotations.coco.json`
"""))

# ---------------------------------------------------------------- install
cells.append(md("## 0) ติดตั้ง dependencies"))
cells.append(code("%pip install -q -U torchao peft transformers pytorch-metric-learning albumentations "
                  "opencv-python-headless scikit-learn seaborn tqdm\nprint('✅ deps ready')"))

# ---------------------------------------------------------------- data
cells.append(md("""## 1) เตรียมข้อมูล — เลือก **วิธีใดวิธีหนึ่ง** แล้วรันเซลล์นี้"""))
cells.append(code('''# ── กำหนด DATA_DIR ให้ตรงกับที่ข้อมูลอยู่ ──────────────────────────────
DATA_DIR = "/content/dental_dataset"

# ▸ วิธี A: อัปโหลดไฟล์ .zip (export จาก Roboflow แล้ว zip ไว้)
# from google.colab import files
# up = files.upload()                       # เลือก dataset.zip
# !unzip -o -q "{list(up.keys())[0]}" -d /content/
# DATA_DIR = "/content/dataset"             # ← แก้ตามชื่อโฟลเดอร์ข้างใน zip

# ▸ วิธี B: ไฟล์อยู่บน Google Drive อยู่แล้ว
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = "/content/drive/MyDrive/path/to/dataset"

# ▸ วิธี C: ดึงตรงจาก Roboflow (แก้ API_KEY / WORKSPACE / PROJECT / VERSION)
# %pip install -q roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# rf.workspace("WORKSPACE").project("PROJECT").version(VERSION)\\
#   .download("coco-segmentation", location=DATA_DIR)

import pathlib
assert pathlib.Path(DATA_DIR, "train", "_annotations.coco.json").exists(), \\
    f"ไม่พบ {DATA_DIR}/train/_annotations.coco.json — รันเซลล์ import ข้อมูลก่อน"
for sp in ["train", "valid", "test"]:
    p = pathlib.Path(DATA_DIR, sp)
    if p.exists():
        n_img = len(list(p.glob('*.jpg'))) + len(list(p.glob('*.png'))) + len(list(p.glob('*.jpeg')))
        n_json = len(list(p.glob('*annotations*.json')))
        print(f"{sp:6s}: {n_img:4d} รูป, {n_json} annotation file")'''))

# ---------------------------------------------------------------- modules
cells.append(md("""## 2) เขียนโมดูลโค้ดลงไฟล์ (%%writefile)

แต่ละเซลล์สร้างไฟล์ .py ตามหน้าที่ — แก้โค้ดได้จากไฟล์ editor ด้านซ้ายของ Colab หลังจากรันแล้ว"""))

for mod in MODULES:
    with open(os.path.join(ROOT, mod), encoding="utf-8") as f:
        src = f.read()
    cells.append(code(f"%%writefile {mod}\n{src}" if not src.endswith("\n") else f"%%writefile {mod}\n{src}"))

# ---------------------------------------------------------------- sanity check
cells.append(md("""## 3) Sanity check ข้อมูล

เช็ค: จำนวนภาพต่อ class, mask overlay ถูกไหม, ความยาวที่วัดได้ sensible ไหม — **ก่อนเทรนเสมอ**"""))
cells.append(code('''CALIB_RATIO = None   # ← cm/pixel ถ้ามี object อ้างอิง (เช่น 0.05) ; None = ใช้ pixel

import sys; sys.path.insert(0, "/content")
from collections import Counter

from dataset import load_coco_records, visualize_records, compute_length_stats

train_recs, class_names = load_coco_records(DATA_DIR, "train")
dist = Counter(r["class_name"] for r in train_recs)
print(f"class ทั้งหมด: {len(class_names)}")
for n in class_names:
    print(f"  {n:30s}: {dist[n]} ภาพ")

stats = compute_length_stats(train_recs, CALIB_RATIO)
unit = "cm" if CALIB_RATIO else "px"
print(f"\\nlength stats (train): mean={stats[0]:.1f} {unit}, std={stats[1]:.1f}")

fig = visualize_records(train_recs, calibration_ratio=CALIB_RATIO, n=6, seed=7)'''))

cells.append(md("""## 3.5) เช็คว่า weight ของ DINOv2 พร้อมหรือยัง (frozen kNN probe — ไม่ต้องเทรน)

ใช้ DINOv2 **frozen** ดึง feature → ทำนายด้วย 1-nearest-neighbor:
- accuracy สูงกว่า random (1/14 ≈ 7%) **มากๆ** (เช่น >40-50%) = feature แยก class ได้อยู่แล้ว เทรนต่อคุ้ม
- accuracy ต่ำเฉียด random = domain gap แรงเกิน → ลอง `img_size=518` (คมขึ้น) หรือ backbone ใหญ่ขึ้น"""))
cells.append(code('''# ponytail: probe ไม่เทรนสัก step — เช็คคุณภาพ feature ก่อนลงทุนเทรนจริง
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
                                   flip_all if training else None, training)
    E, Y = [], []
    for b in DataLoader(ds, batch_size=32, num_workers=2):
        out = probe_model.backbone(pixel_values=b["image"].to(device)).last_hidden_state
        E.append(out[:, 0].cpu())          # CLS ดิบ — ไม่ผ่าน fusion head (head ยังสุ่มอยู่)
        Y.append(b["label"])
    return torch.cat(E), torch.cat(Y)

Etr, ytr = embed(tr_recs, True)            # train-side aug เปิดได้ = ฟรี data augmentation ของ probe
Eva, yva = embed(va_recs, False)
sim = F.normalize(Eva, dim=1) @ F.normalize(Etr, dim=1).T
pred = ytr[sim.argmax(dim=1)]
acc = (pred == yva).float().mean().item()
print(f"kNN probe accuracy: {acc:.3f}   (random = {1 / len(probe_classes):.3f})")
if acc < 0.30:
    print("⚠️ feature แยก class ค่อนข้างไม่ได้ — ลอง img_size=518 หรือ dinov2-base ก่อนเทรนจริง")
else:
    print("✅ feature ใช้ได้ — ไปเทรนต่อได้")'''))

# ---------------------------------------------------------------- train
cells.append(md("""## 4) เทรน

- `finetune_mode="lora"` → เทรนเฉพาะ LoRA adapter (~พารามิเตอร์น้อยมาก) กัน overfitting
- early stopping จาก validation loss, checkpoint ที่ดีที่สุดถูกเซฟอัตโนมัติ
- อยากได้ผลประเมินแบบ cross-validation → ตั้ง `kfold=5`"""))
cells.append(code('''from config import TrainConfig
from train import run_training, run_kfold

cfg = TrainConfig(
    data_dir=DATA_DIR,
    img_size=616,            # 616=44×14 — เต็มที่ batch 32 บน T4
    batch_size=32,
    finetune_mode="lora",     # "lora" (แนะนำ) | "partial" | "frozen"
    epochs=50,
    num_workers=2,
    # ---- ตัวเลือกเสริม ----
    # calibration_ratio=CALIB_RATIO,
    # flip_allowed=["class_a", "class_b"],  # เฉพาะ class ที่ flip ได้ (ที่เหลือห้าม flip)
    # kfold=5,                              # Stratified 5-fold CV
)

if cfg.kfold:
    paths, accs = run_kfold(cfg)
    best_ckpt = paths[accs.index(max(accs))]   # เลือก fold ที่ดีที่สุด
else:
    best_ckpt = run_training(cfg)

print("\\ncheckpoint ที่ดีที่สุด:", best_ckpt)'''))

# ---------------------------------------------------------------- evaluate
cells.append(md("""## 5) ประเมินผล — accuracy + confusion matrix

ดู heatmap ช่อง off-diagonal สีสว่าง = คู่ class ที่โมเดลสับสน (มักเป็นคู่ต่างกันแค่ขนาด)"""))
cells.append(code('''from evaluate import evaluate_checkpoint

metrics = evaluate_checkpoint(best_ckpt)
print(f"\\nAccuracy: {metrics['accuracy']:.4f}")'''))

# ---------------------------------------------------------------- inference
cells.append(md("""## 6) Inference ทดลอง — ภาพใหม่ + mask → class + confidence"""))
cells.append(code('''from infer import load_pipeline, predict_record
from train import resolve_records

pack = load_pipeline(best_ckpt)
_, valid_recs, _ = resolve_records(cfg)   # เอาภาพ val มาลอง 3 ตัว

for r in valid_recs[:3]:
    res = predict_record(pack, r)
    ok = "✅" if res["class"] == res["truth"] else "❌"
    tops = ", ".join(f'{t["class"]}:{t["prob"]:.2f}' for t in res["top3"])
    print(f'{ok} truth={res["truth"]:20s} pred={res["class"]:20s} conf={res["confidence"]:.3f}')
    print("      top3:", tops)'''))

# ---------------------------------------------------------------- save
cells.append(md("""## 7) บันทึกโมเดลกลับ Drive / ดาวน์โหลด"""))
cells.append(code('''# ▸ ดาวน์โหลดกลับเครื่อง
from google.colab import files
files.download(best_ckpt)

# ▸ หรือคัดลอกไป Drive (mount ก่อนถ้ายังไม่ได้ mount)
# !cp "{best_ckpt}" /content/drive/MyDrive/'''))
cells.append(md("""---
### 💡 Tips
- **calibration_ratio**: วาง object ที่รู้ความยาวจริง L (cm) ใต้กล้อง rig เดิม → `ratio = L / measure_length_px(mask)` — feature ความยาวจะมีหน่วย cm ตีความง่าย
- **handedness**: class ของซ้าย/ขวามือ → ใส่เฉพาะชื่อ class ที่ flip ได้ใน `flip_allowed`

### 🎯 Playbook คู่ class ที่แทบแยกไม่ออก (เช่น Universal forceps 150 vs 151, Root elevator งอ vs Root-tip elevator ตรง)
1. **เพิ่ม resolution ก่อนเป็นอย่างแรก** — head ของ forceps ที่ 224px เหลือ ~20px ตั้ง `img_size=518` (=37×14, DINOv2 รับได้) patch token จะพกรายละเอียด "หัว" เยอะขึ้น ~2.3×; GPU ไม่พอลด batch เป็น 8
2. **คู่ต่างความยาว** (elevator งอสั้น vs straight ยาว) — length feature จัดการอยู่แล้ว แต่ mask ต้องครอบปลายจริง (mask โดนเงา = ความยาวเพี้ยน = ฟีเจอร์เพี้ยน)
3. **ถ่ายข้อมูลเพิ่มแบบเจาะจง**: ดู confusion matrix ในขั้น 5 → ถ่ายเพิ่มเฉพาะคู่ที่สับสน โดยหมุนมุม yaw ทีละ ~30° (บางมุมหัวซ่อน = ต้องมีภาพมุมที่เห็นจุดต่าง)
4. **ยังสับสนหนัก?** แผน B: สองขั้น — แยกกลุ่มหยาบ (forceps/elevator/probe…) ก่อน แล้ว classifier เฉพาะคู่บน crop บริเวณหัว

### 📋 เช็คลิสต์ตอน annotate รอบใหม่ (Roboflow COCO Segmentation)
- polygon ครอบ **ทั้งชิ้น** รวมหัว-ปลาย แต่ **ไม่เอาเงา** (เงาทำ minAreaRect ยืวขึ้น)
- 1 annotation ต่อ 1 ชิ้นเครื่องมือ (1 ภาพหลายชิ้นได้ — pipeline crop รายชิ้นด้วย bbox_margin)
- ชื่อ class ตั้งแล้วอย่าเปลี่ยนทีหลัง (label = ลำดับการ sort ชื่อ)
- จำนวนต่อ class ห้ามหลุด ~25+ ภาพ และกระจายมุมถ่ายให้ทั่ว
- พื้นเขียว: pipeline มี shadow augmentation ให้แล้ว แต่ถ่ายให้แสงสม่ำเสมอที่ทำได้ยังดีที่สุด

- GPU: Runtime → Change runtime type → T4 GPU (CPU ได้แต่ช้า); `img_size=518` + T4 + batch 8 ≈ ไหว
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
