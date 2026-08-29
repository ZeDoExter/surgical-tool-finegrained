# Algorithm ระบบ Surgical-Tool Fine-grained Classification

> DINOv2-S + LoRA + Length Fusion + ArcFace + CAHM / LGMS / SEF — อธิบายแบบสั้น อ่านจบแล้วรันตามได้

---

## 0) ภาพรวม

```
COCO (image + polygon) ──► Dataset ──► Model ──► Loss ──► Train ──► Checkpoint ──► Evaluate / Infer
                              │            │         │
                    kNN probe (frozen)  length  ArcFace / Adaptive
```

**Input:** ภาพถาด + polygon mask ของเครื่องมือ 1 ชิ้น  
**Output:** ชื่อ class (14) + confidence + top3

---

## 1) Data — `dataset.py`

**1.1 Parse COCO**
```
records = [{image_path, segmentation=[x1,y1,...], width,height, label, class_name}]
label = index ของ class_name ที่ sort แล้ว (0..13)
```

**1.2 วัดความยาว (aux feature)**
```
mask = fillPoly(polygons)  # 0/255
contour = findContours(mask) → minAreaRect → (w,h)
length_px = max(w,h)
length_cm = length_px * calibration_ratio (ถ้ามี)
length_norm = (length - mean_train) / std_train
# mean/std คิดจาก train เท่านั้น กัน leakage
```

**1.3 Crop รายชิ้น (bbox_margin)**
```
x1,y1,x2,y2 = bbox ของ polygon
dx,dy = (x2-x1)*m, (y2-y1)*m ; m=0.15 (จากการทดลองให้ผลดี จึงใช้เป็น default)
crop = image[y1-dy : y2+dy , x1-dx : x2+dx]
```
**1.4 Augmentation (train เท่านั้น)**
```
simulate_shadow (เงา blob นุ่ม) p0.5
CLAHE p0.5 | RandomBrightnessContrast p0.7 | RandomGamma 70-150 p0.7
HueSaturationValue p0.3 | GaussianBlur p0.2
HorizontalFlip p0.5 ต่อเมื่อ flip_flags[label]==True
# ห้าม: RandomResizedCrop / Cutout กลางวัตถุ (ทำลาย scale)
Resize(img_size, img_size)  # 504=36×14 (จากการทดลองให้ผลดี, ต้องหาร 14 ลงตัว)
Normalize(ImageNet) → ToTensorV2
```
**1.5 Scharr Edge (ถ้า use_sef)**
```
gray = RGB2GRAY(image หลัง augment/flip)
gx = Scharr(gray, dx=1) ; gy = Scharr(gray, dy=1)
edge = magnitude(gx,gy) / max → [0,1]
edge_resized = resize(edge, (504,504))
→ tensor (1,504,504)
```

## 2) kNN Probe — `cell 3.5` (ไม่เทรน)

```
model = DINOv2 frozen (CLS token อย่างเดียว)
Etr = CLS(train) ; Eva = CLS(val)
sim = normalize(Eva) @ normalize(Etr).T
pred = label[ argmax(sim) ]
acc = mean(pred==yva)
# ถ้า acc ต่ำกว่า 0.30 ควรปรับ img_size หรือ backbone ก่อนเทรน
```
---

## 3) Model — `model.py` + `edge_branch.py`

```
image (3,504,504) → DINOv2 ViT-S/14 → tokens (B,257,384)
  ├─ AttentionPooling (6 head) → e (B,384)   # ถ้า use_attention_pool=False ใช้ CLS
  └─ length_norm (B,1) + edge_feat (B,64) ถ้า SEF

fusion:
  aux = concat(length)                     # (B,1)        ถ้าไม่ใช้ SEF
  aux = concat(length, edge_branch(edge_map)) # (B,65)    ถ้า use_sef
  x = LayerNorm( concat(e, aux) )          # (B,385) หรือ (B,449)
  x = Linear→GELU→Dropout→Linear→Dropout → emb (B,384)

backbone finetune:
  frozen: freeze หมด
  partial: ปลด 2 block ท้าย + LayerNorm
  lora: LoRA r=16 บน query/value (train ~1.47M)
```

---

## 4) Loss

**4.1 ArcFace ปกติ**
```
cos = normalize(emb) @ normalize(W)   # W (384,14)
logits = s * cos(theta+m)  # m=28.6° , s=64
loss = CrossEntropy(logits, label)
```

**4.2 LGMS — Length-Gated Margin Scaling (`use_lgms`)**
```
len_mean[c] = เฉลี่ย length ต่อ class (จาก train)
sim_len(i,j) = 1 - |len_i - len_j| / max_diff
twin_pool[y] = k=2 class ที่ sim สูงสุด (ใกล้กันสุด)
m(y) = 28.6 + γ * mean(sim_y) ; γ=10°
# ตัวอย่าง 504: m ≈ 35-36° ทุก class
loss ใช้ m(y) แทน m คงที่ (AdaptiveArcFaceLoss)
```

---

## 5) Training — `train.py`

```
for epoch 1..50:
  # CAHM: ถ้า epoch >10 และ use_cahm
  w = 1 + α * max_j d[y,j] ; d จาก confusion epoch ก่อน (EMA β=0.9)
  loss = mean( per_sample_loss * w )   # คู่สับสนโดนคูณหนัก
  # ปิด mixup ตอนใช้ CAHM เพื่อให้ w ชัด

  train_one_epoch → tl
  validate → vl, va, cm
  # อัปเดต CAHM: d_cur = (C[i,j]+C[j,i])/max ; d = β*d + (1-β)*d_cur

  best = max va (vl เป็น tiebreak) → save best_model.pt
  early-stop patience 12
```

**Optimizer:** AdamW (head lr 3e-4, LoRA 1e-4) + warmup 10% → cosine decay + GradScaler + clip 1.0 + mixup α=0.4 (ปิดเมื่อใช้ CAHM)

---

## 6) Evaluation — `evaluate.py` + `tools/evaluate_ablation.py`

```
bundle = load checkpoint (model + W + cfg + length stats)
pred = argmax( s * cos ) ; prob = softmax
acc, balanced_acc, cm (14×14), report
plot confusion_matrix.png
top confused = sort cm[i,j] (i≠j) → เช่น Needle↔Artery 3+3=6

# ablation 6 สูตร (Phase 3)
compare_checkpoints({
  baseline:{}, cahm:{use_cahm}, lgms:{use_lgms}, sef:{use_sef},
  cahm_lgms:{both}, all:{all}
})
→ ตาราง | Config | Val Acc | Balanced | Needle↔Artery |
→ เกณฑ์: ดีขึ้นเฉลี่ย + ลด Needle↔Artery จริง + ถ้าแย่ลงตัดออก
```

---

## 7) Inference — `infer.py`

```
mask → length → ln ; image → tensor (+ edge_map ถ้า SEF)
emb = model(tensor, ln, edge)
logits = s*cos ; prob = softmax → top3
TTA ถ้า use_tta: (logits + logits_flip)/2
```

---
## 8) Config เริ่มต้น (จากการทดลอง)

```
img_size=504 (36×14)
bbox_margin=0.15
batch_size=32 (T4) / 16 (1650 4GB fallback)
finetune_mode=lora r=16
# ค่าเหล่านี้ได้จากการทดลอง kNN probe ก่อนเทรน
```

## 9) รันบน Colab

```
1. Upload DentalInstrument_DINOv2_ArcFace.ipynb
2. เลือก DATA_DIR (Drive/zip)
3. Run all — 3.5 probe ควรได้ ~0.75 (ถ้า 0.32 คือ bbox_margin หลุด)
4. เซลล์ 4 เทรน baseline → 5 ประเมิน → 5.5 เทรน ablation 6 สูตร → 5.6 เทียบตาราง
```
