# Plan: Next-Phase Improvements for `surgical-tool-finegrained`

ส่งให้ agent เพื่อทำวิจัย (literature search) + ออกแบบ algorithm ใหม่ + implement ต่อบน repo
`https://github.com/ZeDoExter/surgical-tool-finegrained`

---

## 0. บริบทปัจจุบัน (Baseline ที่มีอยู่แล้ว — ห้ามรื้อ ต่อยอดเท่านั้น)

- Architecture: `DINOv2-S/14` + `LoRA (r=16)` + length fusion (จาก `minAreaRect` ของ mask) → `ArcFace (m=28.6°, s=64)`
- ผลปัจจุบัน (best): 95.49% val acc (balanced 95.80%), 11/244 error
- **ปัญหาที่เหลืออยู่ชัดเจน:** error ส่วนใหญ่ (5/11) กระจุกอยู่ที่คู่ `Needle_Holder ↔ Artery_Forceps` เท่านั้น
- ไฟล์ในโครงสร้าง repo: `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`, `config.py`
- ข้อจำกัดข้อมูล: ~30 ภาพ/class, 14 classes รวม ~400 ภาพ, val set มีแค่ 244 ภาพ (ต้องระวัง overfit ต่อ validation set เอง ถ้าปรับ hyperparameter ตาม val accuracy ตรงๆ)

**กฎเหล็ก:** ทุก experiment ใหม่ต้องเทียบผลกับ baseline นี้ด้วย **k-fold cross-validation** (ไม่ใช่ single split) ก่อนสรุปว่า "ดีขึ้นจริง" เพราะ val set เล็กเกินกว่าจะเชื่อผลจาก single split ได้

---

## Phase 0 — Literature Research (ทำก่อน implement อะไรทั้งหมด)

### เป้าหมาย
หา paper/เทคนิคที่ต่อยอดได้จริงกับปัญหา 2 อย่าง: (1) fine-grained classification ที่ต่างกันแค่รายละเอียดเล็ก และ (2) training บนข้อมูลจำกัดมาก (~30 samples/class)

### คำค้นหาที่ต้องใช้ (ค้นทีละคำ อย่ารวมกันในคำค้นเดียว)
1. `fine-grained visual classification survey`
2. `metric learning medical image classification`
3. `hard negative mining fine-grained recognition`
4. `few-shot fine-grained classification`
5. `class imbalance dynamic loss reweighting`
6. `ArcFace adaptive margin`
7. `edge-aware feature fusion classification`
8. `surgical instrument recognition deep learning`

### แหล่งค้นหา
- Google Scholar (scholar.google.com) — ใช้ "Cited by" และ "Related articles" ไล่ตาม
- Papers With Code (paperswithcode.com) — filter ตาม task "Fine-Grained Image Classification"
- Connected Papers (connectedpapers.com) — ใส่ paper หลักที่เจอจาก Scholar เป็นจุดเริ่มต้น ดู citation graph

### Deliverable ของ Phase 0
สร้างไฟล์ `research_notes.md` สรุปเป็นตารางแบบนี้:

| Paper | ปีที่ตีพิมพ์ | เทคนิคหลัก | นำมาปรับใช้กับ repo นี้ได้ยังไง | ระดับความคุ้ม (สูง/กลาง/ต่ำ) |

**เกณฑ์คัดเลือก:** เอาเฉพาะ paper ที่ตอบโจทย์ตรงกับปัญหา "ข้อมูลน้อย + class คล้ายกันมาก" เท่านั้น ไม่ต้องสรุปทุก paper ที่เจอ

---

## Phase 1 — Algorithm Design (3 อัลกอริทึมใหม่ของเรา)

ตั้งชื่อ + ออกแบบ formula ไว้ให้แล้ว ให้ agent implement ตามสเปคนี้ และถ้า literature research ใน Phase 0 เจอเทคนิคที่ดีกว่า ให้ปรับ spec ตรงนี้ได้ (แต่ต้องมี citation อ้างอิงกำกับ)

### Algorithm 1: **CAHM (Confusion-Aware Hard Mining)**
*อัลกอริทึมของเรา — ให้ loss โฟกัสที่คู่ class ที่โมเดลสับสนบ่อย โดยอัปเดตอัตโนมัติทุก epoch (ออกแบบเฉพาะสำหรับปัญหาคู่ Needle_Holder ↔ Artery_Forceps)*

**Input:** confusion matrix `C` (14×14) จาก validation set ของ epoch ก่อนหน้า

**ขั้นตอน:**
1. คำนวณ pair difficulty score:
   ```
   d(i,j) = C[i,j] + C[j,i]     สำหรับ i ≠ j
   d_norm(i,j) = d(i,j) / max(d)   # normalize ให้อยู่ช่วง [0,1]
   ```
2. Smooth ด้วย EMA (exponential moving average) กันค่ากระโดดเกินไประหว่าง epoch:
   ```
   d_t = β · d_(t-1) + (1-β) · d_norm      # β = 0.9 (ค่าเริ่มต้น)
   ```
3. ต่อ sample แต่ละตัวที่ label จริงคือ class `y` ให้คำนวณน้ำหนัก loss:
   ```
   w(sample) = 1 + α · max_j( d_t(y, j) )     สำหรับ j ที่ไม่ใช่ y
   ```
   (α = hyperparameter เริ่มที่ 2.0 — ปรับได้)
4. Loss ที่แก้ไขแล้ว:
   ```
   L_total = (1/N) · Σ w(sample_i) · L_ArcFace(sample_i)
   ```

**Implementation:** เพิ่ม hook ใน `train.py` คำนวณ confusion matrix จาก validation loop ทุก epoch → เก็บ state `d_t` ไว้ใน trainer object → ส่งเข้า loss function ของ epoch ถัดไป

**Guard เพื่อไม่ให้ overfit:** เริ่มใช้ CAHM หลัง epoch ที่ 10 เท่านั้น (ให้โมเดล converge คร่าวๆ ก่อน ไม่งั้น confusion matrix ช่วงแรกจะ noisy เกินไป)

---

### Algorithm 2: **LGMS (Length-Gated Margin Scaling)**
*อัลกอริทึมของเรา — ต่อยอดจาก length-aware ArcFace margin ที่ออกแบบไว้สำหรับ repo นี้ ทำให้เป็น algorithm ที่ implement ได้จริง*

**แนวคิด:** class ที่ length ใกล้เคียงกันมาก (เช่น Needle_Holder vs Artery_Forceps ถ้าตัวอย่างจริงมีความยาวใกล้กัน) ควรถูกบังคับให้ margin ในการแยกกว้างกว่าปกติ เพราะรู้ล่วงหน้าแล้วว่าเป็นคู่เสี่ยงสับสน

**Input:** `length_cm` เฉลี่ยของแต่ละ class (คำนวณจาก training set ทั้งหมดของ class นั้น, เก็บไว้ล่วงหน้าเป็น lookup table)

**ขั้นตอน:**
1. คำนวณ length similarity ระหว่างทุกคู่ class:
   ```
   sim_len(i,j) = 1 - |len_i - len_j| / max_diff_all_pairs
   ```
2. หา top-k class ที่ length ใกล้ class `y` มากที่สุด (k=2 เริ่มต้น) เรียกกลุ่มนี้ว่า "length-twin pool" ของ y
3. คำนวณ margin เฉพาะของ class y (แทนที่จะใช้ margin คงที่ 28.6° เท่ากันทุก class):
   ```
   m(y) = m_base + γ · mean_(j ∈ twin_pool)( sim_len(y,j) )
   ```
   (m_base = 28.6° ตามเดิม, γ = hyperparameter เริ่มที่ 10°)
4. ใช้ `m(y)` แทน `m` คงที่ตอนคำนวณ ArcFace loss ของ sample ที่ label เป็น y

**Implementation:** แก้ `model.py` ส่วน ArcFace loss ให้รับ margin เป็น array ต่อ class แทนค่าคงที่ตัวเดียว (ต้อง pre-compute `length-twin pool` และ `m(y)` ก่อนเริ่มเทรน จาก training set)

---

### Algorithm 3: **SEF (Scharr Edge Fusion)**
*อัลกอริทึมของเรา — ต่อยอดจากไอเดีย Scharr edge heatmap เป็น aux channel ที่ออกแบบไว้สำหรับ repo นี้*

**แนวคิด:** เพิ่ม branch เล็กๆ ที่ประมวลผล edge map ของภาพ (จาก Scharr filter) แยกจาก DINOv2 แล้วเอามา fuse กับ embedding หลัก เพื่อเน้นรูปทรง/ขอบให้ชัดกว่าที่ backbone เห็นเองจากภาพสี

**ขั้นตอน:**
1. คำนวณ Scharr edge magnitude จากภาพ grayscale:
   ```python
   gx = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
   gy = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
   edge_map = cv2.magnitude(gx, gy)   # ขนาดเท่าภาพต้นฉบับ
   ```
2. ส่ง `edge_map` ผ่าน small CNN branch แยกต่างหาก (3 conv layers + global average pooling) ได้ vector ขนาด 64-dim
3. Fusion กับ embedding หลัก:
   ```
   fused = concat(dino_embedding[384], length_feature[1], edge_feature[64])   # รวม 449-dim
   final_embedding = Linear(449 → 384)(fused)
   ```
4. `final_embedding` เข้า ArcFace loss ตามปกติ (ต่อจาก LGMS ถ้าใช้ทั้งคู่ร่วมกัน)

**Implementation:** เพิ่มไฟล์ใหม่ `edge_branch.py` แยกจาก `model.py` หลัก, แก้ `dataset.py` ให้คำนวณ edge_map ควบคู่กับตอนโหลด mask (เพราะต้องใช้ grayscale crop เดียวกับที่ crop ตาม mask)

---

## Phase 2 — Implementation Checklist (mapping ไปยังไฟล์ที่มีอยู่)

| ไฟล์เดิม | สิ่งที่ต้องแก้/เพิ่ม |
|---|---|
| `dataset.py` | เพิ่ม `get_edge_map()` method, คำนวณ + cache `length-twin pool` ต่อ class ตอน init dataset |
| `model.py` | แก้ ArcFace loss ให้รับ margin แบบ per-class array (สำหรับ LGMS), เพิ่ม fusion layer รับ edge feature (สำหรับ SEF) |
| `train.py` | เพิ่ม confusion-matrix tracking + EMA state (สำหรับ CAHM), เพิ่ม flag เปิด/ปิดแต่ละ algorithm แยกกันได้ (`--use_cahm`, `--use_lgms`, `--use_sef`) |
| `edge_branch.py` (ไฟล์ใหม่) | small CNN สำหรับ SEF |
| `evaluate.py` | เพิ่มการรายงานผลแบบ ablation คือรันทุก combination ของ 3 algorithm เทียบกับ baseline |
| `config.py` | เพิ่ม hyperparameter: `alpha` (CAHM), `gamma` (LGMS), `beta` (EMA smoothing) |

**ลำดับการ implement ที่แนะนำ:** CAHM ก่อน (ง่ายสุด ไม่แก้ architecture) → LGMS (แก้ loss แต่ไม่เพิ่ม branch ใหม่) → SEF (ซับซ้อนสุด เพิ่ม branch ใหม่ทั้งหมด)

---

## Phase 3 — Evaluation Criteria

ต้องรายงานผลแบบ ablation study ทดสอบทีละตัวและรวมกัน:

| Config | Val Acc | Balanced Acc | Needle_Holder↔Artery_Forceps error | หมายเหตุ |
|---|---|---|---|---|
| Baseline (ปัจจุบัน) | 95.49% | 95.80% | 5/11 | — |
| + CAHM | ? | ? | ? | — |
| + LGMS | ? | ? | ? | — |
| + SEF | ? | ? | ? | — |
| + CAHM + LGMS | ? | ? | ? | — |
| + ทั้งหมด | ? | ? | ? | — |

**เกณฑ์ตัดสินว่า "คุ้มเก็บไว้":**
1. ต้อง**ดีขึ้นจาก k-fold เฉลี่ย** ไม่ใช่ single split เดียว
2. ต้องลด error เฉพาะคู่ Needle_Holder↔Artery_Forceps ได้จริง ไม่ใช่แค่ตัวเลขรวมดีขึ้นแต่คู่นี้เท่าเดิม
3. ถ้า algorithm ไหนทำให้ accuracy โดยรวมแย่ลงแม้เล็กน้อย ให้ตัดออก ไม่ต้องฝืนเก็บไว้ทั้งหมด — เป้าหมายคือแก้จุดที่เจาะจง ไม่ใช่เพิ่มความซับซ้อนให้มากที่สุด

---

## หมายเหตุสำหรับ Agent

- ห้ามแก้ baseline architecture เดิม (DINOv2+LoRA+length fusion+ArcFace) ทั้ง 3 algorithm ใหม่ต้องเป็น "เสริม" ที่เปิด/ปิดได้ผ่าน config flag เท่านั้น
- ทำ Phase 0 (research) ให้เสร็จและสรุปเป็นไฟล์ก่อน แล้วค่อยดูว่าต้องปรับ spec ของ Phase 1 ตาม paper ที่เจอไหม ก่อนเริ่ม implement จริง
- ถ้า literature research เจอชื่อ/สูตรที่ดีกว่าที่ออกแบบไว้ในเอกสารนี้ ให้เสนอทางเลือกใหม่พร้อมเหตุผลก่อน ไม่ต้อง implement ตามสเปคเดิมแบบไม่ยืดหยุ่น
