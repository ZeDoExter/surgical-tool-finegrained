# Algorithm — Surgical-Tool Fine-grained Detection + Classification

> Short, runnable — baseline first, then why each kept add-on is added

---

## 0) Overview — two-stage pipeline, two training runs

```
                        ┌──────────────── STAGE 1: DETECTOR ────────────────┐
camera frame ──► DINOv2-S + SegDecoder ──► (1+C)-channel logits
                (train_detector.py)              │ softmax
                                                ▼
                        instances: {bbox, label, mask, score,
                                    length_px (minAreaRect), tip_crops}
                                                │
                        ┌──────────────── STAGE 2: CLASSIFIER ────────────┐
                crop + mask + length_cm ──► DINOv2-S + ArcFace ──► class
                (train.py)                 + tip TTA + length prior
```

**Two training runs, in order:**
1. `python train_detector.py` → `outputs_detector/best_detector.pt` (bbox + mask + label, YOLO-free)
2. `python train.py` → `outputs/best_model.pt` (fine-grained class per instance)

**Input:** frame of instruments on green cloth → **Output:** per-instance bbox + class + confidence (+ real length in cm after calibration)

**Challenges:** shadows on cloth make bounding hard; **hard pairs share the same length** — Needle_Holder↔Artery_Forceps and Forceps 23↔150 differ ONLY at the tip (curved vs straight); Root_Elevators 15.5 cm vs Root_Tip_Elevator_Straight 14.5 cm.

---

## 1) Stage 1 — Detector (no YOLO)

### 1.1 Model (`det_model.py`)
```
image (3,560,560) → DINOv2 ViT-S/14 → patch tokens (B,1600,384)
                                        + hidden_states[6],[9] (mid feats)
  → SegDecoder: fuse → Conv → PixelShuffle×2 → Conv → PixelShuffle×2 → head
  → logits (B, 1+14, 560, 560)   # ch0 = fg, ch1..14 = class
~1.7M trainable (LoRA r=16 on query/value + decoder); ONNX-safe ops only
```

### 1.2 Training data (`det_dataset.py`) — synthesized multi-tool scenes
```
COCO export has 1 tool/image (441 photos) — a detector needs multi-object
scenes. Generated ON THE FLY every epoch (infinite variety):
  - background: procedural green cloth (HSV noise + weave + vignette)
                or real photo with the instrument erased
  - paste 2-5 instrument foregrounds (COCO polygons, random rotate/scale/flip,
    overlap ≤20%)
  + shadows (simulate_shadow) + photometric aug (same as classifier)
Also mixed in: real photos (incl. patch-paste composites from
augment_dataset.py num_aug=8 → ~2600 extra scenes on disk).
Loader accepts real multi-annotation photos — a future "mix" dataset drops
in without code changes.
```

### 1.3 Loss (`train_detector.py`)
```
targets: (1+C, H, W) full-res — ch0 fg, ch1..C one-hot class
loss = CE(logits, bg|class) with bg_weight=0.25   # ~90% patches are bg
      + Dice(sigmoid(logits[0]), fg)               # handles small FG fraction
val (real photos): full post-process → greedy mask-IoU≥0.3 match → P/R/F1
```

### 1.4 Post-processing (`det_postprocess.py`, numpy/cv2 only — same file on Pi)
```
logits → softmax over (bg + classes) → fg_prob = 1 - p_bg
fg_prob > 0.5 → morph open → connected components (≥80 px)
per component: label = argmax mean class-prob, score = min(mean cls, mean fg)
mask-IoU NMS (0.4) → instances
per instance: bbox, mask, length_px = minAreaRect long side
              (accurate for diagonal tools — bbox max(w,h) inflates up to √2)
tip_crops = both ends along mask major axis (for Stage 2 tip TTA)
```

**Result (v2 @560):** val F1=0.987, P=0.974, R=1.000 at epoch 6 (early-stop 18). Caveats: val = single-tool photos, patch pool includes valid instruments → optimistic; verify on live camera + overlays.

---

## 2) Stage 2 — Classifier (per-instance crop)

### 2.1 Data (same as v2, + tip-zoom)
```
1. Parse COCO → records; label = sorted class index
2. length = minAreaRect(polygon mask) long side, normalized (train mean/std)
3. Crop bbox + 0.15 margin
4. NEW tip-zoom (p=0.35): replace the crop with a zoomed TIP view along the
   mask major axis (both ends stacked) — forces the model to learn the ONLY
   signal that separates same-length pairs (curved vs straight jaws)
5. Same photometric/shadow/flip aug; patch-paste composites (p=0.4)
```

### 2.2 Model + Loss (unchanged from v2)
```
AttentionPooling over patch tokens → concat(emb, length) → MLP → 384-d
ArcFace m=28.6°, s=64; AdamW (head 3e-4, LoRA 1e-4), warmup→cosine, AMP
```

### 2.3 Kept add-ons
* **CAHM** (`use_cahm=True`, default) — confusion-aware sample weighting (EMA of pair difficulty, α=2.0, after epoch 10). Ablation v1: 0.9467 (8 Needle↔Artery errors) → **0.9590 (3)**.
* **Tip TTA (inference)** — extra forward on tip crops; tip logits get 2.5× weight when top-2 lands in a same-length pair (Needle↔Artery, 23↔150).
* **Length prior (inference)** — Gaussian prior over classes from measured cm (σ=1.2) using REAL_LENGTH_CM; separates Root_Elevators 15.5 vs Straight 14.5 after `calibrate.py`. Length can NEVER separate the same-length pairs — tip must.
* LGMS / SEF tried and removed (see git history).

**Result (v2 @560, tip-zoom on):** val_acc=1.0 (epoch 15).

---

## 3) Deployment (Raspberry Pi 5, realtime)

```
PC/Colab: export_detector_onnx.py + export_to_onnx.py → onnx_export/
One-time:  calibrate.py (ruler photo on the rig → cm/px → meta json)
Pi:        pi_final_v3/ — ONNX Runtime, 3 threads (camera/detect/classify),
           IoU tracker (no ByteTrack/ultralytics/YOLO), tip TTA + length prior
           production serve:
           gunicorn --workers 1 --threads 8 --worker-class gthread \
             --timeout 0 --bind 0.0.0.0:8000 app:app
           # workers MUST be 1 (one camera), no --preload (threads die on fork)
```

---

## 4) Default Config (from experiments)

```
img_size=560 (40×14) — kNN probe 0.878 (v2 full-res); must be divisible by 14
bbox_margin=0.15, LoRA r=16, ArcFace m=28.6° s=64, CAHM on
tip_zoom_prob=0.35, tip_zoom_size=0.42
detector: batch 16 (T4) / 4 (1650), CE(bg 0.25)+Dice, instance-F1 selection,
          synth 2-5 objects, overlap ≤0.20, mask_thr 0.5, NMS IoU 0.4
```

---

## 5) Run (Colab notebook is local-only, not on GitHub)

```
Notebook: DentalInstrument_DINOv2_DetectClassify.ipynb (kept local — run
tools/make_notebook.py to regenerate from the .py files)
  cell 3   sanity check + patch-paste preview
  cell 4   generate patch-paste scenes (num_aug=8, ~2600 images, idempotent .bak)
  cell 5   kNN probe (~0.878 @560 on v2)
  cell 6   TRAIN #2: classifier (tip-zoom)        → best_model.pt
  cell 8   TRAIN #1: detector (synth scenes)       → best_detector.pt
  cell 9   detector eval (P/R/F1 + overlay gallery)
  cell 10  export ONNX (both) → onnx_export/
```

Resume without retraining the classifier: unzip an old `outputs.zip`, set
`best_ckpt`, skip cells 5-7. Re-running cell 4 is safe (seed 42, .bak guard).

See experimental results in `README.md`
