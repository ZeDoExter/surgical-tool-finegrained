# Algorithm ระบบ Surgical-Tool Fine-grained Classification

> อ่านจบแล้วรันตามได้ — แยก baseline ปกติกับส่วนเสริมที่เพิ่มทีหลัง

---

## 0) ภาพรวม

```
COCO (image + polygon) ──► Dataset ──► Model(baseline) ──► Loss ──► Train ──► Checkpoint
                              │              │               │         │
                    kNN probe (frozen)   length(1)     ArcFace    Evaluate / Infer
                                            │
                         ส่วนเสริม (เปิดเมื่อต้องการ): CAHM | LGMS | SEF
```

**Input:** ภาพถาด + polygon mask 1 ชิ้น → **Output:** ชื่อ class (14) + confidence

---

## 1) Baseline ปกติ — ทำยังไง

### 1.1 Data
```
1. อ่าน COCO: records = [{image_path, segmentation, width, height, label}]
   label = index ของ class_name ที่ sort แล้ว (0..13)

2. วัดความยาว: mask = fillPoly(polygons) → minAreaRect → length = max(w,h)
   length_norm = (length - mean_train)/std_train
   # mean/std คิดจาก train เท่านั้น

3. Crop รายชิ้น: bbox ของ polygon ขยายด้วย bbox_margin=0.15
   crop = image[y1-dy : y2+dy, x1-dx : x2+dx]  # 0.15 มาจากการทดลอง

4. Augment (train): simulate_shadow, CLAHE, BrightnessContrast, Gamma, Hue, Blur, Flip ต่อ class
   ห้าม RandomResizedCrop/Cutout (ทำลาย scale)
   Resize 504×504 (จากการทดลอง, ต้องหาร 14 ลงตัว) → Normalize → ToTensor
```

### 1.2 kNN Probe (เช็คก่อนเทรน, ไม่ได้เทรน)
```
model = DINOv2 frozen, ใช้ CLS token อย่างเดียว
Etr = CLS(train) , Eva = CLS(val)
pred = 1-NN (cosine)
acc = mean(pred==true)  # ถ้าต่ำกว่า 0.30 ควรปรับ img_size ก่อน
```

### 1.3 Model (baseline)
```
image (3,504,504) → DINOv2 ViT-S/14 → tokens (B,257,384)
  → AttentionPooling (6 head) → e (B,384)
  → concat(e, length_norm) → (B,385) → LayerNorm → Linear→GELU→Linear → emb (B,384)

finetune: lora r=16 บน query/value (train ~1.47M) | หรือ frozen / partial 2 block ท้าย
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
acc, balanced_acc, cm 14×14, top confused (เช่น Needle↔Artery)
TTA: (logits + logits_flip)/2 ถ้าเปิด use_tta
```

---

## 2) ส่วนเสริม 3 ตัว — เพิ่มไปทำไม

baseline ก็รันได้จบแล้ว — 3 ตัวนี้เปิดเมื่ออยากแก้จุดที่ baseline ยังพลาด (ส่วนใหญ่กระจุกที่คู่ Needle↔Artery)

### 2.1 CAHM — Confusion-Aware Hard Mining (`use_cahm`)

**ทำไปทำไม:** ให้ loss หนักขึ้นกับคู่ class ที่สับสนบ่อย (เช่น Needle↔Artery) โดยดูจาก confusion ของ epoch ก่อน

**ทำยังไง:**
```
d(i,j) = (C[i,j] + C[j,i]) / max   # C = confusion 14×14
d = 0.9*d_prev + 0.1*d_cur          # EMA กันกระโดด
w = 1 + 2.0 * max_j d[y,j]          # α=2.0, เริ่มใช้หลัง epoch 10
loss = mean( per_sample_loss * w )
# ปิด mixup ตอนใช้ CAHM เพื่อให้ w ชัด
```

### 2.2 LGMS — Length-Gated Margin Scaling (`use_lgms`)

**ทำไปทำไม:** บางคู่ต่างกันแค่ความยาว (เช่น Needle ยาวใกล้ Artery) — อยากให้ margin กว้างขึ้นเฉพาะคู่ที่ยาวใกล้กัน จะได้แยกห่างกว่าเดิม

**ทำยังไง:**
```
len_mean[c] = เฉลี่ย length ต่อ class (จาก train)
sim(i,j) = 1 - |len_i - len_j| / max_diff
twin[y] = 2 class ที่ sim สูงสุด
m(y) = 28.6 + 10 * mean(sim_y)   # γ=10°
loss ใช้ m(y) แทน m คงที่ (AdaptiveArcFaceLoss)
```

### 2.3 SEF — Scharr Edge Fusion (`use_sef`)

**ทำไปทำไม:** เครื่องมือสีเงินบนถาดสีเงิน contrast ต่ำ — เพิ่ม branch ดูขอบโดยตรง จะได้เห็นปลายคีม/ฟันเลื่อยชัดขึ้น

**ทำยังไง:**
```
gray = RGB2GRAY(image หลัง augment)
gx = Scharr(gray, dx=1), gy = Scharr(gray, dy=1)
edge = magnitude(gx,gy)/max → [0,1] → resize 504×504 → (1,504,504)
edge_feat = CNN 3 ชั้น + GAP → (B,64)
fusion: concat(e 384, length 1, edge 64) → (B,449) → Linear → emb 384
```

**เปิด/ปิด:** ทั้ง 3 ตัวเป็น flag ใน `config.py` — baseline ปิดหมด, อยากลองตัวไหนเปิดตัวนั้น

```python
cfg = TrainConfig(data_dir="dataset", img_size=504, batch_size=32,
                  use_cahm=True)  # หรือ use_lgms / use_sef
```

---

## 3) Config เริ่มต้น

```
img_size=504 (36×14)      # จากการทดลอง kNN probe
bbox_margin=0.15          # จากการทดลอง
batch_size=32 (T4) / 16 (1650 4GB fallback)
finetune_mode=lora r=16
```

---

## 4) รันบน Colab

```
1. Upload DentalInstrument_DINOv2_ArcFace.ipynb
2. เลือก DATA_DIR (Drive/zip)
3. Run all — 3.5 probe ควรได้ ~0.75
4. เซลล์ 4 เทรน baseline → 5 ประเมิน → 5.5 เทรน ablation 6 สูตร → 5.6 เทียบตาราง
```

ดูผลการทดลองทั้งหมดใน `README.md`
