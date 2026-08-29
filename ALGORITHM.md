# Algorithm — Surgical-Tool Fine-grained Classification

> Short, runnable — baseline first, then why the three add-ons are added

---

## 0) Overview

```
COCO (image + polygon) ──► Dataset ──► Model(baseline) ──► Loss ──► Train ──► Checkpoint
                              │              │               │         │
                    kNN probe (frozen)   length(1)     ArcFace    Evaluate / Infer
                                            │
                         Add-ons (enable when needed): CAHM | LGMS | SEF
```

**Input:** image on green cloth + polygon mask for 1 instrument → **Output:** class name (14) + confidence  
**Challenge:** green cloth background, but **shadows from lighting** make bounding hard; some classes differ only by length; ~30 images/class

---

## 1) Baseline — how it normally works

### 1.1 Data
```
1. Parse COCO: records = [{image_path, segmentation, width, height, label}]
   label = index of sorted class_name (0..13)

2. Measure length: mask = fillPoly(polygons) → minAreaRect → length = max(w,h)
   length_norm = (length - mean_train)/std_train
   # mean/std from train only

3. Crop per instance: bbox of polygon expanded by bbox_margin=0.15
   crop = image[y1-dy : y2+dy, x1-dx : x2+dx]  # 0.15 from kNN probe experiments

4. Augment (train): simulate_shadow, CLAHE, BrightnessContrast, Gamma, Hue, Blur, Flip per class
   No RandomResizedCrop/Cutout (would destroy scale)
   Resize 504×504 (from kNN probe, must be divisible by 14) → Normalize → ToTensor
```

### 1.2 kNN Probe (check before training, no training)
```
model = DINOv2 frozen, CLS token only
Etr = CLS(train) , Eva = CLS(val)
pred = 1-NN (cosine)
acc = mean(pred==true)  # if <0.30, try larger img_size before training
```

### 1.3 Model (baseline)
```
image (3,504,504) → DINOv2 ViT-S/14 → tokens (B,257,384)
  → AttentionPooling (6 heads) → e (B,384)
  → concat(e, length_norm) → (B,385) → LayerNorm → Linear→GELU→Linear → emb (B,384)

finetune: lora r=16 on query/value (~1.47M trainable) | or frozen / partial last 2 blocks
```

### 1.4 Loss (baseline)
```
cos = normalize(emb) @ normalize(W)   # W (384,14)
logits = 64 * cos(theta + 28.6°)      # ArcFace m=28.6°, s=64
loss = CrossEntropy(logits, label)
```

### 1.5 Training (baseline)
```
for epoch 1..50:
  train_one_epoch → tl
  validate → vl, va, cm
  best = max va → save best_model.pt
  early-stop patience 12
optimizer: AdamW (head 3e-4, LoRA 1e-4) + warmup 10% → cosine + GradScaler + mixup 0.4
```

### 1.6 Evaluate / Infer (baseline)
```
load checkpoint → emb → logits = 64*cos → softmax → top3
acc, balanced_acc, cm 14×14, top confused (e.g. Needle↔Artery)
TTA: (logits + logits_flip)/2 if use_tta
```

---

## 2) Three Add-ons — why they are added

Baseline already runs end-to-end — the three below are enabled only to fix remaining errors (mostly Needle↔Artery)

### 2.1 CAHM — Confusion-Aware Hard Mining (`use_cahm`)

**Why:** make loss focus on pairs the model confuses often

**How:**
```
d(i,j) = (C[i,j] + C[j,i]) / max   # C = confusion 14×14
d = 0.9*d_prev + 0.1*d_cur          # EMA
w = 1 + 2.0 * max_j d[y,j]          # α=2.0, start after epoch 10
loss = mean( per_sample_loss * w )
# mixup is disabled when CAHM is on
```

### 2.2 LGMS — Length-Gated Margin Scaling (`use_lgms`)

**Why:** some pairs differ only by length — give larger margin to pairs with similar length so they are pushed farther apart

**How:**
```
len_mean[c] = mean length per class (from train)
sim(i,j) = 1 - |len_i - len_j| / max_diff
twin[y] = 2 classes with highest sim
m(y) = 28.6 + 10 * mean(sim_y)   # γ=10°
loss uses m(y) instead of fixed m (AdaptiveArcFaceLoss)
```

### 2.3 SEF — Scharr Edge Fusion (`use_sef`)

**Why:** green cloth + metallic instruments + shadows → low contrast — add an edge branch to emphasize shape

**How:**
```
gray = RGB2GRAY(image after augment)
gx = Scharr(gray, dx=1), gy = Scharr(gray, dy=1)
edge = magnitude(gx,gy)/max → [0,1] → resize 504×504 → (1,504,504)
edge_feat = CNN 3 layers + GAP → (B,64)
fusion: concat(e 384, length 1, edge 64) → (B,449) → Linear → emb 384
```

**Enable/disable:** all three are flags in `config.py` — baseline has them off, enable any you need

```python
cfg = TrainConfig(data_dir="dataset", img_size=504, batch_size=32,
                  use_cahm=True)  # or use_lgms / use_sef
```

---

## 3) Default Config (from kNN probe)

```
img_size=504 (36×14)
bbox_margin=0.15
batch_size=32 (T4) / 16 (1650 4GB fallback)
finetune_mode=lora r=16
# chosen from kNN probe experiments before training
```

---

## 4) Run on Colab

```
1. Upload DentalInstrument_DINOv2_ArcFace.ipynb
2. Choose DATA_DIR (Drive/zip)
3. Run all — 3.5 probe should be ~0.75
4. Cell 4 train baseline → 5 evaluate → 5.5 train ablation 6 configs → 5.6 compare table
```

See experimental results in `README.md`
