# surgical-tool-finegrained

Fine-grained classification of **14 surgical instruments** on a **green cloth**
— where several classes differ *only* by length. Fixed camera rig. The main
difficulty is **shadows from lighting** that make bounding the instrument hard
(more than background color).

## Pipeline

```
frame → DETECTOR (DINOv2-S seg head → bbox + mask + coarse label)
      → CLASSIFIER (DINOv2-S + length fusion + ArcFace → fine label)
```

- **Detector** (`train_detector.py`): DINOv2-S/14 + light seg head, LoRA r=16,
  CE + Dice loss, instance-F1 checkpoint selection on real photos.
- **Classifier** (`train.py`): DINOv2-S + instrument length fusion + ArcFace
  margin loss + CAHM (confusion-aware hard mining). `train_all.py` runs both.
- **Student** (`train_student.py`): distills the detector into
  LRASPP-MobileNetV3 (+ gated INT8 export) for realtime Pi inference.
- **YOLO alt** (`tools/coco_to_yolo.py` + Ultralytics): same data in YOLO
  format for a nano-detector baseline.

## Datasets

COCO Segmentation exports from Roboflow. `1 annotation = 1 sample`
(use `bbox_margin=0.15` to crop per instance when one image holds many tools).

| Dataset | train | valid | test | notes |
|---|---|---|---|---|
| `dataset/` (current) | 349 imgs / 639 anns | multi | multi | 329 original + 20 `WIN_*` mix photos |
| `dataset_v2/` (zip: `Dental Instrument v2.v2i.coco.zip`) | 329 / 329 | 74 / 74 | 38 / 38 | original single-instrument, no mix |

Multi-annotation images share one file across records; the detector merges
touching same-tool polygons (`merge_split_annotations`).

## Usage

```bash
pip install -r requirements.txt        # or requirements-rocm.txt on AMD

# Full pipeline (detector + classifier, CAHM on, classifier @504)
python train_all.py --data_dir dataset --epochs_det 60 --epochs_cls 50

# Single stages
python train_detector.py --data_dir dataset --epochs 60 --img_size 448
python train.py --data_dir dataset --epochs 50 --img_size 504

# Distill realtime student (needs a teacher checkpoint)
python train_student.py --teacher outputs_detector/best_detector.pt \
    --img_size 320 --epochs 40 --export --export_int8

# YOLO-format labels + training
python tools/coco_to_yolo.py --data_dir dataset --out_dir dataset_yolo
yolo detect train data=dataset_yolo/dataset.yaml model=yolo26n.pt \
    epochs=100 imgsz=640 batch=32

# Evaluate / single-image inference
python evaluate.py --checkpoint outputs/best_model.pt --data_dir dataset
python infer.py --checkpoint outputs/best_model.pt --image path.jpg

# Remote training over SSH (see project notes)
omp ssh add gpu-box --host <ip> --user <user>
```

`train_all.py` mirrors all console output to `train_all.log` and prints a
YOLO-style per-epoch summary (see `Progress UI Specification.md`).

Key config (`config.py`): `img_size=504` (classifier, from kNN probe),
`bbox_margin=0.15`, `lora_r=16`, `use_cahm=True`, `patch_paste_prob=0.4`,
`tip_zoom_prob=0.35`. `erase_neighbors` defaults to **False** — measured
worse than off in every ablation round (R1 0.9215 / R2 0.8848 / R3 0.9319).

## Measured results

| Run | data | val acc | note |
|---|---|---|---|
| Colab baseline (CAHM) | old single-instrument | 0.9590 | see TUNING doc |
| R4 | dataset_v2, no mix | **0.9595** | best; 25/25 on held-out test crops |

**Detector (DINOv2-S seg head, instance-F1 on real photos)**

| Run | F1 | note |
|---|---|---|
| teacher (Colab) | 0.987 | full-size model |
| student 320px (distilled) | 0.661 | ~20 ms/frame CPU |
| student 448px (distilled) | 0.6805 | ~38 ms/frame CPU |

**YOLO26n baseline**: mAP50 0.766 (prior run, 100 epochs) — fast but
no masks.

**CPU latency** (ONNX Runtime, 3 threads, desktop CPU — Pi 5 is slower):

| Model | ms/frame |
|---|---|
| student fp32 @320 | ~20 |
| student fp32 @448 | ~38 |
| yolo26n @640 | ~55 |
| DINOv2 detector @560 | ~1190 |

**Mix-dataset finding**: adding the 20 `WIN_*` multi-instrument photos
dropped classifier accuracy (R3 0.9319 on mix data vs R4 0.9595 without).
Neighbor-erased crops did not help. See `TUNING_EXPLANATION_TH.md`
(classifier tuning story, Thai) and `ALGORITHM.md` (full algorithm).

## Pi deployment (`pi_final_v7/`)

Raspberry Pi 5, ONNX Runtime CPU only:

```
pi_final_v7/
  app.py              # DINO detector + classifier (recommended main path)
  app_student.py      # distilled student detector (fastest, masks kept)
  app_dinoyolo.py     # YOLO26n boxes + DINOv2 labels
  app_yolo.py         # YOLO boxes only
  detector_onnx.py / dino_classifier_onnx.py / yolo_detector_onnx.py
  det_postprocess.py  # numpy/cv2 connected-components + NMS (shared)
  onnx_export/        # models + *_meta.json (models are git-ignored;
                      # copy from a training machine)
```

```bash
pip install flask flask-cors gunicorn opencv-python-headless numpy onnxruntime
gunicorn --workers 1 --threads 8 --worker-class gthread --timeout 0 \
    --bind 0.0.0.0:8000 app:app
# http://<pi-ip>:8000/video_feed?token=<API_KEY>
```

`--workers 1` always (one camera). See `pi_final_v7/README.txt`.

Notebooks (`DentalInstrument_*.ipynb`, git-ignored, regenerate with
`python tools/make_notebook.py`) reproduce the Colab flow.

## License

MIT
