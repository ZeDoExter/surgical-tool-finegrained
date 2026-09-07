# -*- coding: utf-8 -*-
"""
pi_final_v3/app_dinoyolo.py — YOLO26n boxes (fast) + DINOv2 labels (accurate)

Best of both, by design:
  YOLO  : box+coarse label in ~50ms every frame — bbox quality is excellent
  DINOv2: fine-grained 14-class identity per track (ArcFace + tip TTA) —
          confirms/corrects YOLO's coarse guess; cached per track
  Dupe fix: class-agnostic second NMS inside yolo_detector_onnx (the
          in-graph NMS still emits same-spot boxes; tools overlap <=20%
          by design so neighbors are never suppressed by mistake)

    pip install onnxruntime opencv-python flask flask-cors numpy
    python app_dinoyolo.py
"""
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from yolo_detector_ncnn import YoloDetectorNCNN as YoloDetectorONNX
from dino_classifier_onnx import DinoClassifierONNX

API_KEY = "e5bdb16ce0552c091383244b3c814ffe51bd15fa73e0bc6f4dc7b09afe00a6a67d2bcc02d3ec3accceec4357b05d275a4cc5e98667b47cae0c154fc55c864f13"
APP_VERSION = "2026-09-07-v9pkg7"   # bump on every deploy; shown at boot and at
                                # / so a field report can always name its build
CLASSIFY_REFRESH_SEC = 5.0
DETECT_DIR = "onnx_export"
CLASSIFY_DIR = "onnx_export"
# MODEL SELECT: YOLO_MODEL=nomix (default, 20-class nomix weights).
# "mix" resolves only if its files are present (not shipped in final) —
# anything missing falls back to nomix so boot never crashes.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "nomix").strip().lower()
YOLO_MODEL_METAS = {"nomix": "yolo_ncnn_nomix.json", "mix": "yolo_ncnn_mix.json"}
YOLO_META_FILE = YOLO_MODEL_METAS.get(YOLO_MODEL, YOLO_MODEL_METAS["nomix"])
if (YOLO_MODEL not in YOLO_MODEL_METAS
        or not os.path.exists(os.path.join(DETECT_DIR, YOLO_META_FILE))):
    YOLO_MODEL, YOLO_META_FILE = "nomix", YOLO_MODEL_METAS["nomix"]
STICKY_CONFIRMED_SEC = 60.0     # confident labels stick for a minute — tools
                                # don't change identity on a table
CLASSIFY_MAX_PER_PASS = 2      # budget per loop pass — never starve YOLO
SHOW_CONF_MIN = 0.60           # draw a box only if its displayed confidence
                                # exceeds this (kills visual clutter; tracking
                                # and classify still run underneath)
IOU_MATCH = 0.3

# CASCADE: YOLO's own class+score is instant and already good. When it is
# confident AND the class is not one of the confusable same-shape pairs,
# accept it as the final label right away — the ViT only runs for the hard
# pairs and low-confidence cases (this is what makes labels feel instant).
YOLO_TRUST_SCORE = 0.80
VIT_ONLY_CLASSES = {           # same-shape/silhouette pairs the ViT must
    "Needle_Holder", "Artery_Forceps",
    "Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150",
    "Root_Elevators", "Root_Tip_Elevator_Straight",
    # NOTE: Cotton_Piler / Root_Tip_Pick / Dental_Mirror / Triple_Syringe /
    # Scalpel_Handle / Root_Tip_Elevator_LR / Cartridge_Syringe would qualify,
    # but field 2026-09-07 showed YOLO beats ViT on those 7 Pi-side — they
    # bypass ViT via YOLO_FINAL_CLASSES below instead.
}
# YOLO-FINAL 7 (field 2026-09-07: YOLO beats ViT on these Pi-side) —
# their YOLO label is final, the ViT never judges them. All other classes
# flow through the normal cascade/ViT pipeline below.
YOLO_FINAL_CLASSES = {
    "Cotton_Piler", "Root_Tip_Pick", "Dental_Mirror", "Triple_Syringe",
    "Scalpel_Handle", "Root_Tip_Elevator_LR", "Cartridge_Syringe",
}
YOLO_FINAL_MIN_SCORE = 0.50  # unknown-reject: below this the track stays "..."

# TIP-ZOOM PAIRS: same-silhouette class pairs where the tip shape is the
# true signal. When ViT's top-2 is one of these pairs and it isn't already
# sure, force tip-crop TTA (2 extra small forwards on the box ends) even if
# the generic low-conf gate wouldn't fire.
TIP_ZOOM_PAIRS = {
    frozenset(("Needle_Holder", "Artery_Forceps")),
    frozenset(("Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150")),
    frozenset(("Cotton_Piler", "Root_Tip_Elevator_LR")),
    frozenset(("Cotton_Piler", "Root_Elevators")),
    frozenset(("Root_Tip_Pick", "Triple_Syringe")),
    frozenset(("Root_Elevators", "Root_Tip_Elevator_Straight")),
}
TIP_PAIR_CONF = 0.85            # run pair-TTA below this confidence
# SKIN GATE: hands are the #1 out-of-domain false positive (model never saw
# a hand in training). Metal is low-saturation, skin is not — drop boxes
# whose crop is mostly skin-colored before they reach the tracker.
SKIN_REJECT = True
SKIN_FRac = 0.35               # measured: real tool boxes max 0.092 (p95 0.013,
                               # n=93 test anns), full-hand box = 1.00 — 0.35
                               # has ~4x margin, warm-lit metal stays below it

# MIN_HITS: a track must be seen in this many consecutive frames before the
# ViT spends ~1s on it. Kills the ViT storm when tools move (flicker boxes
# die young) without delaying stable tools noticeably.
MIN_HITS_FOR_CLS = 2
CLASS_COLORS_BGR = [
    (0, 0, 180),        # Artery_Forceps            dark red
    (20, 140, 180),     # Cartridge_Syringe         amber
    (180, 60, 0),       # Cotton_Piler              navy
    (160, 0, 160),      # Dental_Mirror             dark magenta
    (130, 130, 0),      # Forceps_23                dark teal
    (200, 70, 30),      # Forceps_150               dark blue
    (0, 110, 220),      # Needle_Holder             orange
    (120, 60, 200),     # Root_Elevators            dark pink
    (90, 90, 90),       # Root_Tip_Elevator_LR     dark gray
    (40, 20, 130),      # Root_Tip_Elevator_Straight maroon
    (30, 110, 110),     # Root_Tip_Pick             olive
    (60, 140, 0),       # Scalpel_Handle            dark green
    (180, 30, 110),     # Suture_Scissors           dark violet
    (130, 90, 70),      # Triple_Syringe            dark slate
]
UNKNOWN_COLOR_BGR = (0, 0, 180)
DETECT_THREADS = 3
CLASSIFY_THREADS = 2

app = Flask(__name__)
CORS(app)

print(f"[yolo] loading {YOLO_META_FILE} ...", flush=True)
detector = YoloDetectorONNX(DETECT_DIR, num_threads=DETECT_THREADS,
                            meta_file=YOLO_META_FILE)
print(f"[yolo] classes {len(detector.classes)} backend={detector.backend} "
      f"warmup={detector.last_ms:.0f}ms nms_iou={detector.nms_iou}", flush=True)
print("[classifier] loading ONNX ...", flush=True)
classifier = DinoClassifierONNX(CLASSIFY_DIR, num_threads=CLASSIFY_THREADS)
print(f"[classifier] loaded backend={classifier.backend}", flush=True)
print(f"[app] version={APP_VERSION} model={YOLO_MODEL}({YOLO_META_FILE}) "
      f"trust={YOLO_TRUST_SCORE} showmin={SHOW_CONF_MIN}", flush=True)

def _skin_frac(bgr_crop: np.ndarray) -> float:
    """Fraction of skin-colored pixels (HSV H 0-25, S>50, V>60).
    Metal tools are low-saturation; green cloth is H~60-90; shadows are
    dark. Hands are the only thing scoring high here."""
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 50, 60]), np.array([25, 255, 255]))
    return float((m > 0).mean())



def class_color(cls_name: str):
    try:
        idx = detector.classes.index(cls_name)
        return CLASS_COLORS_BGR[idx] if idx < len(CLASS_COLORS_BGR) else UNKNOWN_COLOR_BGR
    except (ValueError, AttributeError):
        return UNKNOWN_COLOR_BGR


class IoUTracker:
    """Tiny greedy IoU tracker (tools barely move on the cloth).

    Persists missed tracks up to max_age passes: a tool YOLO flickers on for
    a frame stays drawn (and keeps its label) instead of blinking out — the
    missing bbox is reused so the box does not jump."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 15):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # tid -> {bbox, age, cls, score, inst}

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(aa + bb - inter, 1e-6)

    def update(self, instances):
        unused = set(self.tracks)
        for inst in instances:
            box = inst["bbox_frame"]
            best_tid, best_iou = None, 0.0
            for tid in unused:
                iou = self._iou(box, self.tracks[tid]["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None and best_iou >= self.iou_thr:
                unused.discard(best_tid)
                hits = self.tracks[best_tid].get("hits", 0) + 1
                self.tracks[best_tid] = {"bbox": box, "age": 0,
                                         "cls": inst["class_name"],
                                         "hits": hits,
                                         "inst": inst}
                inst["track_id"] = best_tid
                inst["track_hits"] = hits
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "age": 0,
                                    "cls": inst["class_name"],
                                    "hits": 1,
                                    "inst": inst}
                inst["track_id"] = tid
                inst["track_hits"] = 1
        # missed tracks: age and keep returning their last instance so the
        # drawn frame never blinks when YOLO drops a frame
        for tid in unused:
            self.tracks[tid]["age"] += 1
        out = []
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if t["age"] > self.max_age:
                del self.tracks[tid]
                continue
            inst = dict(t["inst"])
            inst["track_id"] = tid
            if t["age"] > 0:
                inst["missed"] = t["age"]   # age>0 marks a stale box
            out.append(inst)
        return out

tracker = IoUTracker(iou_thr=IOU_MATCH)
latest_detections = {}
last_detection_time = 0
output_frame = None
latest_raw_frame = None
last_instances = []
last_detect_frame = None  # exact BGR frame detect_loop ran on (frame-synced crops)
fine_grained_cache = {}
frame_lock = threading.Lock()
detect_lock = threading.Lock()
raw_lock = threading.Lock()
cache_lock = threading.Lock()


def detect_loop():
    global last_instances, latest_detections, last_detection_time, last_detect_frame
    while True:
        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            # tip crops are cut here once per frame — classify_loop reuses them
            # when a track needs (re)labeling (saves the crop cost in that thread)
            insts = detector.detect(frame, want_tip_crops=True)
            if SKIN_REJECT:
                kept = []
                H, W = frame.shape[:2]
                for inst in insts:
                    x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, W), min(y2, H)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        kept.append(inst)
                        continue
                    if _skin_frac(frame[y1:y2, x1:x2]) < SKIN_FRac:
                        kept.append(inst)
                insts = kept
            insts = tracker.update(insts)
        except Exception as e:
            print(f"[yolo] {e}", flush=True)
            time.sleep(0.05)
            continue
        counts = {}
        for inst in insts:
            n = inst["class_name"]
            counts[n] = counts.get(n, 0) + 1
        with detect_lock:
            last_instances = insts
            latest_detections = counts
            last_detection_time = time.time()
            last_detect_frame = frame  # boxes below index into THIS frame


def classify_loop():
    while True:
        frame = None
        insts = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        with detect_lock:
            insts = list(last_instances)
            if last_detect_frame is not None:
                frame = last_detect_frame.copy()  # crop the frame boxes came from,
                                                  # not a newer one (race = shifted crops)
        if frame is None or not insts:
            time.sleep(0.2)
            continue
        now = time.time()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        done_this_pass = 0
        for inst in insts:
            if done_this_pass >= CLASSIFY_MAX_PER_PASS:
                break   # leave CPU for YOLO — remaining tracks wait for next pass
            tid = inst.get("track_id")
            if tid is None:
                continue
            if inst.get("missed"):
                continue  # replayed box from a YOLO-missed frame — the crop
                          # coords are stale; never burn a ViT call on ghosts
            with cache_lock:
                cached = fine_grained_cache.get(int(tid))
            if cached is not None:
                # STICKY: confidently answered recently -> don't re-ask at all
                if cached.get("confirmed") and (now - cached["ts"]) < STICKY_CONFIRMED_SEC:
                    continue
                # low-confidence answers retry with BACKOFF, not every 5s
                # forever: each failed retry doubles the wait (5/10/20/40s).
                # Without this, a few unsure tracks keep the ViT pegged and
                # the whole Pi stutters.
                tries = int(cached.get("tries", 0))
                wait = CLASSIFY_REFRESH_SEC * (2 ** min(tries, 3))
                if (now - cached["ts"]) < wait:
                    continue
            # EDGE HOLD: box touching the frame border may be a PARTIAL view
            # (tool extends out of frame, or YOLO caught only part of an
            # upright tool). Crop+length then look like a *short* tool and the
            # classifier answers a short class at 1.00 (proven locally:
            # half-crop + short length -> wrong short class, deterministically).
            # Hold the "..." until a full view arrives — never lock a label.
            x1e, y1e, x2e, y2e = [int(v) for v in inst["bbox_frame"]]
            if x1e <= 2 or y1e <= 2 or x2e >= w - 2 or y2e >= h - 2:
                continue
            if inst["class_name"] in YOLO_FINAL_CLASSES:
                # YOLO is the judge for these 4 — publish directly, never ViT.
                if inst["score"] >= YOLO_FINAL_MIN_SCORE:
                    with cache_lock:
                        fine_grained_cache[int(tid)] = {
                            "class": inst["class_name"],
                            "confidence": inst["score"],
                            "length_used": None,
                            "ts": now,
                            "confirmed": True,
                            "via": "yolo",
                        }
                continue
            # CASCADE fast path: confident YOLO class that is not one of the
            # same-shape confusable classes -> accept instantly, skip the ViT
            # (a ViT forward is ~900ms on the Pi; this makes the label appear
            # in the same frame as the box)
            if (inst["score"] >= YOLO_TRUST_SCORE
                    and inst["class_name"] not in VIT_ONLY_CLASSES):
                with cache_lock:
                    fine_grained_cache[int(tid)] = {
                        "class": inst["class_name"],
                        "confidence": inst["score"],
                        "length_used": None,
                        "ts": now,
                        "confirmed": True,
                        "via": "yolo",
                    }
                continue
            # ViT is expensive (~1s on the Pi): only young-track gate lives
            # here, AFTER the free cascade above. Flicker boxes die before
            # costing a forward; stable tools already passed via cascade or
            # reach this point on their 2nd sighting.
            if inst.get("track_hits", 99) < MIN_HITS_FOR_CLS:
                continue
            x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = rgb_frame[y1:y2, x1:x2]
            try:
                res = classifier.classify(
                    crop, x2 - x1, y2 - y1,
                    length_px=inst.get("length_px_frame"),
                    tip_crops=None,
                    length_cm=inst.get("length_cm"),
                )
                # unsure first pass -> tip TTA from the pre-cut box-end crops
                # (YOLO has no mask; box ends cover the jaws for most poses).
                # Known confusable pairs get tip TTA at a higher bar too:
                top2 = set()
                if res.get("top3") and len(res["top3"]) > 1:
                    top2 = {res["top3"][0]["class"], res["top3"][1]["class"]}
                pair_hit = (len(top2) == 2 and frozenset(top2) in TIP_ZOOM_PAIRS)
                if (inst.get("tip_crops")
                        and ((res["confidence"] < 0.55 and len(top2) == 2)
                             or (pair_hit and res["confidence"] < TIP_PAIR_CONF))):
                    res = classifier.classify(
                        crop, x2 - x1, y2 - y1,
                        length_px=inst.get("length_px_frame"),
                        tip_crops=inst.get("tip_crops"),
                        length_cm=inst.get("length_cm"),
                        force_tip_tta=True,
                    )
            except Exception as e:
                print(f"[classifier] {e}", flush=True)
                continue
            done_this_pass += 1
            confirmed = bool(res["confidence"] >= 0.70
                             and not res.get("tip_tta_used"))
            with cache_lock:
                prev = fine_grained_cache.get(int(tid), {})
                fine_grained_cache[int(tid)] = {
                    "class": res["class"], "confidence": res["confidence"],
                    "length_used": res["length_used"], "ts": now,
                    # confirmed = confident answer without needing tip TTA
                    "confirmed": confirmed,
                    # unconfirmed answers remember how many times ViT already
                    # tried, so the retry gate backs off instead of hammering
                    "tries": 0 if confirmed else int(prev.get("tries", 0)) + 1,
                    "via": "arcface",
                }
        time.sleep(0.05)


def camera_loop():
    global output_frame, latest_raw_frame
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fps = 0
    fps_counter = 0
    fps_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with raw_lock:
            latest_raw_frame = frame.copy()
        with detect_lock:
            insts = list(last_instances)
        annotated = frame.copy()
        for inst in insts:
            tid = inst.get("track_id")
            fine = None
            if tid is not None:
                with cache_lock:
                    fine = fine_grained_cache.get(int(tid))
            x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
            if fine is not None:
                name, conf = fine["class"], fine["confidence"]
            else:
                name, conf = inst["class_name"], inst["score"]
            if conf <= SHOW_CONF_MIN:
                continue  # low-confidence box: keep tracking/classifying it,
                          # just don't draw it until something confirms it
            color = class_color(name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            if tid is not None:
                label += f" #{tid}"
            if fine is None:
                label += " ..."
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if "length_cm" in inst:
                cv2.putText(annotated, f"{inst['length_cm']:.1f}cm", (x1 + 2, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        cv2.putText(annotated, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        with frame_lock:
            output_frame = annotated.copy()


def check_key():
    return request.args.get("token") == API_KEY


@app.route("/video_feed")
def video_feed():
    if not check_key():
        return {"error": "unauthorized"}, 401

    def generate():
        while True:
            frame_copy = None
            with frame_lock:
                if output_frame is not None:
                    frame_copy = output_frame.copy()
            if frame_copy is not None:
                ret, buf = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/detects")
def detects():
    if not check_key():
        return {"error": "unauthorized"}, 401
    # wait-for-cls: a tool is sent ONLY after the ViT (or cascade) has
    # actually answered its track. New tools appear ~1s late but never blink
    # between "found" and "labeled", and YOLO's per-frame class jitter never
    # reaches the client.
    with detect_lock:
        insts = list(last_instances)
    with cache_lock:
        labeled = {}
        for inst in insts:
            tid = inst.get("track_id")
            if tid is None:
                continue
            hit = fine_grained_cache.get(int(tid))
            if hit is None:
                continue  # ViT hasn't answered this track yet — hold it back
            labeled[int(tid)] = hit
        # evict labels of dead tracks so stale names never linger in JSON
        live = {int(i["track_id"]) for i in insts if i.get("track_id") is not None}
        for tid in [t for t in fine_grained_cache if t not in live]:
            del fine_grained_cache[tid]
    counts = {}
    for _tid, hit in labeled.items():
        n = hit["class"]
        counts[n] = counts.get(n, 0) + 1
    fine = {str(tid): v["class"] for tid, v in labeled.items()}
    # detail tells exactly which path answered each track (yolo cascade vs
    # arcface ViT) — the field-debug field for "who mislabeled this?"
    detail = {str(tid): {"class": v["class"],
                          "confidence": round(float(v["confidence"]), 3),
                          "via": v.get("via", "?")}
              for tid, v in labeled.items()}
    return jsonify({"data": counts, "fine_grained": fine,
                    "detail": detail, "version": APP_VERSION, "model": YOLO_MODEL,
                    "timestamp": last_detection_time})


@app.route("/")
def index():
    return (f"DENIS YOLO26n boxes + DINOv2 labels Server Running "
            f"{APP_VERSION} model={YOLO_MODEL}")


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()
threading.Thread(target=classify_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
