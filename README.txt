pi/ — final Pi package (pkg6 + YOLO nomix)
=========================================
YOLO detect ทุกคลาส; 7 ตัวนี้ป้าย YOLO = final ไม่ส่ง ViT:
Cotton_Piler / Root_Tip_Pick / Dental_Mirror / Triple_Syringe /
Scalpel_Handle / Root_Tip_Elevator_LR / Cartridge_Syringe
(YOLO_FINAL_CLASSES ใน app_dinoyolo.py) คลาสที่เหลือวิ่ง cascade/ViT เดิม

ไฟล์:
  app_dinoyolo.py         ตัวแอป
  yolo_detector_ncnn.py   YOLO NCNN wrapper
  dino_classifier_onnx.py DINOv2 cls (อีก 7 คลาสที่ยังใช้ ViT)
  onnx_export/            weights nomix (yolo26n_v10_nomix_512/) + cls
                          (ไม่ขึ้น github, อยู่เฉพาะเครื่องนี้กับ Pi)

รันบน Pi (ก๊อปทั้งโฟลเดอร์นี้ไป):
  YOLO_MODEL=nomix OMP_NUM_THREADS=4 gunicorn --workers 1 --threads 8 --worker-class gthread \
      --timeout 0 --bind 0.0.0.0:8000 app_dinoyolo:app
เช็ค: boot log version=2026-09-07-v9pkg6 model=nomix
