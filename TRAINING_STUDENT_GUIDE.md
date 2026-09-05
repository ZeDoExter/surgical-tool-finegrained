# Student Detector — Training Guide

สำหรับเทรน student detector (ตัวเร็ว 20ms บน Pi) ให้แม่นขึ้นกว่า v1 (F1 0.66)

## ปัญหาที่พบบน Pi จริง (จากการทดสอบ v1)

1. **Triple Syringe กลืน Cotton Piler** — เครื่องมือเล็กไปโดนกล่องเพื่อน
   หรือตอบคลาสผิดเพราะเห็นเพียงบางส่วนของอีกตัว
2. **Dental Mirror ยาว แตกเป็น 2 กล่อง** — หัวกระจกกับด้ามจับถูกตีกรอบแยก
   สองอัน เป็นคลาสต่างกันไปเลย
3. **Root Tip Pick มองไม่เห็นปลาย / ตีกรอบแค่ด้าม** — ของเล็กมาก
   ที่ 320px ปลายเครื่องมือเหลือไม่กี่ pixel
4. **กล่องไม่ครอบทั้งเครื่องมือ** — เห็นแค่ส่วนกลาง (เป็นผลจาก 3 ข้อบน)

สาเหตุหลัก: **ความละเอียด 320px ต่ำเกินไปสำหรับเครื่องมือเล็ก** ทุกอย่าง
ลงมาจากตรงนี้ ตามด้วยสูตร distill ที่ยังเบา (kd_weight=0.5, 40 epochs
early-stop ที่ 13)

## ทางแก้ตามลำดับผลต่อ F1 (เทรนที่เครื่อง GPU แทนได้ทั้งหมด)

### ขั้น 1: 448px แทน 320px (สำคัญสุด — แก่ปัญหาเล็ก/แตกกล่องโดยตรง)

```bash
.venv/Scripts/python.exe train_student.py ^
  --teacher from_colab/outputs_detector/best_detector.pt ^
  --img_size 448 --epochs 60 --batch_size 12 --num_workers 0 ^
  --export --export_int8
```

- ค่า expected: F1 จาก 0.66 → ~0.75-0.85; เวลา/เฟรม บน Pi เพิ่มจาก
  ~20ms เป็น ~35-45ms (ยังเร็วกว่า YOLO26n ที่ 54ms)
- หมายเหตุ: 448 ไม่ใช่ 560 เพราะ student เป็น CNN (stride ได้ทุกค่า)
  แต่ต้องหาร 32 ลงตัว (grid = img/10 ใน `train_student.py`)
  — 448 ✓ (448/32=14) ให้ feature เล็กสุด 14x14 เหมือนเดิม
- batch ลดเป็น 12 กัน OOM บนการ์ด 4GB (teacher 560 + student 448
  รันพร้อมกันใน VRAM)

### ขั้น 2: kd_weight ปรับตาม (แก้ปัญหา "ตอบเป็นเครื่องมืออื่น")

```bash
  --kd_weight 1.0
```

- v1 ใช้ 0.5 — GT loss (ce+dice) กับ KD (MSE กับ teacher logits) พอ
  ๆ กัน ทำให้ student บางครั้งเห็น patch คล้าย Cotton Piler ก็ตอบ
  Triple Syringe ไป (คลาสเพื่อนบ้าน) เพราะให้น้ำหนัก soft-target น้อยไป
- 1.0 = ตาม teacher มากขึ้น = กระจายความมั่นใจแบบเดียวกับ ViT
- ถ้าเห็นว่าโอเวอร์ติด teacher จนเรียน GT น้อยไป ลดกลับ 0.7

### ขั้น 3: synth_scale_range ขยาย (แก้ Root Tip Pick มองไม่เห็น)

โค้ดปัจจุบัน `config.py` ใช้ `synth_scale_range` ตายตัว ไม่มี flag
— ถ้าอยากได้เครื่องมือเล็กชัดขึ้นใน synthetic data ต้องแก้ใน
`config.py` (หา `synth_scale_range` แล้วขยายล่างให้เล็กลง เช่น
`[0.08, 0.42]` → `[0.05, 0.42]` เพื่อให้มีตัวอย่างเครื่องมือเล็กใน
training มากขึ้น) หรือเพิ่ม `synth_min_objects` ให้เยอะขึ้นเพื่อให้
มี scene ที่แน่นขึ้น

### สิ่งที่จะได้หลังเทรน (ไปวางบน Pi)

- `outputs_student/best_student.pt`
- `pi_final_v3/onnx_export/detector_dino_student.onnx` (export อัตโนมัติ)
- `detector_meta.json` (img_size=448 อัตโนมัติ)
- INT8 gate ถ้าผ่าน (≥85% agreement) จะได้ int8 ด้วย — เร็วขึ้นอีก ~2x

**คัดลอกไป `pi_final_v5/onnx_export/`** (สังเกต: export ที่ `pi_final_v3`
เพราะโค้ด hardcode — อย่าลืมย้าย) แล้วรัน:

```bash
gunicorn --workers 1 --threads 8 --worker-class gthread --timeout 0 \
    --bind 0.0.0.0:8000 app_student:app
```

`app_student.py` จะเลือก student ใหม่อัตโนมัติ (มีกล่องแล้วหยุดตรวจ
เฟรมเก่า ๆ สมเพชอยู่แล้ว)

### เช็คผลหลังเทรน

หลังจบ ดูบรรทัดสุดท้าย `[done] best epoch=N F1=X.XXXX`:
- **F1 ≥ 0.75** → ใช้งานได้ วาดไวมาก แม่นขึ้นเรื่องเครื่องมือเล็ก
- **F1 < 0.70** → ปัญหาอยู่ที่ architecture/data ไม่ใช่ epochs ไป
  ขั้น 3 หรือลองโยน 640px เข้าไปแล้วเทียบ

### ปัญหาที่ต้องระวัง (ทราบล่วงหน้า)

- **ทดสอบบน Pi ที่ไหน**: หลังเทรนเสร็จ ทดสอบการวาดกล่องบนรูปจริง
  ก่อนเอาไปใช้งานจริง (มี HUD FPS + ส่วนต่างเวลาแสดงอยู่แล้ว)
- อย่าเพิง export INT8 หาก gate ไม่ผ่าน — ของ v1 ตก (50%) เพราะ
  LRASPP มั่วกับ 320px หลัง quantize มาก ที่ 448px มีโอกาสผ่าน
  มากกว่า (สัญญาณ feature ชัดขึ้น)
- ค่า `min_instance_area` ใน meta จะอัปเดตตาม img_size อัตโนมัติ
  (`int(cfg.min_instance_area * (img_size/560)^2)` — คูณแล้วได้ 26 ที่
  320px → 51 ที่ 448px) ถ้า Root Tip Pick ยังหายอยู่ ลองลดค่านี้ใน
  `detector_meta.json` บน Pi ลงครึ่งหนึ่งแล้วทดสอบ

## สรุปสั้น ๆ (ถ้าจะเทรนวันนี้)

```bash
.venv/Scripts/python.exe train_student.py \
  --teacher from_colab/outputs_detector/best_detector.pt \
  --img_size 448 --epochs 60 --batch_size 12 --num_workers 0 \
  --kd_weight 1.0 --export --export_int8
```

เสร็จแล้วคัดลอก `detector_dino_student.onnx` + `detector_meta.json`
จาก `pi_final_v3/onnx_export/` ไป `pi_final_v5/onnx_export/` แล้วรัน
`app_student.py` บน Pi เหมือนเดิม
