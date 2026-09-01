pi_final_v3 — DINOv2 detector + classifier (NO YOLO) on Raspberry Pi 5
======================================================================
Detector : DINOv2-S + light seg head  → bbox + mask + label (ONNX)
Classifier: existing DINOv2+ArcFace    → fine-grained class (ONNX)
            + tip TTA (Needle↔Artery, 23↔150)
            + mask minAreaRect length (not bbox max(w,h))
            + cm prior after calibrate.py (Root Elevators 15.5 vs Straight 14.5)

On PC (train + export):
  python train_detector.py --data_dir dataset --epochs 60 --num_workers 0
  python export_detector_onnx.py --ckpt outputs_detector/best_detector.pt --out_dir pi_final_v3/onnx_export
  python export_to_onnx.py --ckpt model/model_0_7/best_model.pt --out_dir pi_final_v3/onnx_export
  python calibrate.py --image ruler.jpg --known_cm 15.5 --detector_meta pi_final_v3/onnx_export/detector_meta.json --classifier_meta pi_final_v3/onnx_export/classifier_meta.json

Copy onto Pi:
  detector_dino.onnx, detector_meta.json
  surgical_dino_fusion.onnx, arcface_W.npy, classifier_meta.json
  det_postprocess.py, detector_onnx.py, dino_classifier_onnx.py, app.py

On Pi (realtime MJPEG + /detects API):
  pip install onnxruntime opencv-python flask flask-cors numpy gunicorn

  # production serve (recommended):
  gunicorn --workers 1 --threads 8 --worker-class gthread --timeout 0 \
      --bind 0.0.0.0:8000 app:app

  # or dev mode:
  python app.py

  # http://<pi-ip>:8000/video_feed?token=<API_KEY>

  gunicorn notes:
    - MUST be --workers 1 (the camera can be opened by only one process;
      background threads start on module import, one worker = one set)
    - do NOT use --preload (threads would start in master and die on fork)
    - --timeout 0 keeps long-lived MJPEG streams from being killed
    - --threads 8 lets several clients watch /video_feed at once

No ultralytics / no supervision / no YOLO.
