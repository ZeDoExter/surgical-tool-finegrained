pi_final_v7 — FINAL RELEASE for Raspberry Pi 5
==============================================

อัปเดต (5 ก.ย. 2026) — โมเดลใหม่ทั้งคู่ + โหมดเต็มสปีด:

โมเดลในแพ็กนี้:
1. DETECTOR ใหม่: DINOv2 v2+aug (เทรนด้วย dataset_v2 + offline augment
   9,503 annotations, ไม่มีรูป mix) -> F1=0.962, Recall=1.000
   + มี patch tokens + class_prototypes.npy (fast path)
2. CLASSIFIER ใหม่: ใช้ dataset เดียวกัน -> val_acc=1.0000,
   ทดสอบบน test จริง (รวม mix scenes) 29/30 = 0.967
3. STUDENT CNN (สำรอง): F1=0.68, ~38ms สำหรับโหมดเร็วสุด

app.py คือตัวหลัก (ตามสเปกใหม่) — DINO detector เต็ม + classifier:
- กล่อง + ป้าย class คร่าว ๆ มาพร้อมกัน (จาก detector + prototypes)
- ป้าย ... = รอ ViT ยืนยัน (คู่ยาก Needle/Artery ฯลฯ) แล้วเติมทีหลัง
- sticky cache: ตอบแล้วแน่ใจ = ไม่ถามซ้ำ 5 นาที

การตั้งค่าความเร็ว (บรรทัดบนสุดของ app.py):
- DETECT_THREADS=4, CLASSIFY_THREADS=4   <- ใช้ Pi ทั้ง 4 cores (FULL mode)
  ถ้ากระตุก/เฟรม drop: ลดเป็น 3/3 หรือ 3/2 ก่อน
- CLASSIFY_MAX_PER_PASS=3  จำนวน track ที่ ViT ตอบต่อรอบ
- CLASSIFY_REFRESH_SEC=3   รอบ retry ของ track ที่ยังไม่แน่ใจ
- STICKY_CONFIRMED_SEC=300  ป้ายที่ confirmed อยู่ได้นานแค่ไหน

วิธีติดตั้งบน Pi:
  sudo apt update && sudo apt install -y python3-pip python3-venv libatlas3-base
  python3 -m venv .venv && source .venv/bin/activate
  pip install flask flask-cors gunicorn opencv-python-headless numpy onnxruntime

รัน (ตัวหลัก):
  gunicorn --workers 1 --threads 8 --worker-class gthread --timeout 0 \
      --bind 0.0.0.0:8000 app:app

  เพิ่มความเร็วได้อีก (ให้ ORT ใช้ neon + ทุก core):
  OMP_NUM_THREADS=4 gunicorn --workers 1 --threads 8 ...   (เหมือนเดิม + env)

ทางเลือก:
  app_student:app   = student CNN (เร็วสุด ~38ms, mask จริง, แม่นน้อยกว่า)
  app_dinoyolo:app  = YOLO26n กล่อง + ViT ป้าย (แบบเดิมจาก v6)
  app_yolo:app      = YOLO เปล่า (กล่องเท่านั้น)

หน้าเว็บ: http://<pi-ip>:8000/video_feed?token=<API_KEY>
API:       http://<pi-ip>:8000/detects?token=<API_KEY>

หมายเหตุ:
- detector_meta.json ในแพ็กนี้ตรงกับ detector_dino.onnx (560px + tokens)
  อย่าสลับกับ meta ของ student
- gunicorn ต้อง --workers 1 เสมอ (กล้องเปิดได้ทีเดียว)
- ถ้าช้าลงเรื่อย ๆ หลังใช้นาน: ป้าย confirmed จะ sticky 5 นาทีแล้ว
  ปกติไม่ควรเกิดอีก; ถ้าเกิดบอกผม
