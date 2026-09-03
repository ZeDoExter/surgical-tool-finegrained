# -*- coding: utf-8 -*-
"""
pi_final_v3/app.py — realtime DINOv2 detector + fine-grained classifier on Pi 5

NO YOLO / ultralytics. Detector is DINOv2-seg ONNX; classifier is the existing
DINOv2-ArcFace ONNX with tip TTA + mask length.

    pip install onnxruntime opencv-python flask flask-cors numpy
    python app.py
"""
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from detector_onnx import DinoDetectorONNX
from dino_classifier_onnx import DinoClassifierONNX

API_KEY = "e5bdb16ce0552c091383244b3c814ffe51bd15fa73e0bc6f4dc7b09afe00a6a67d2bcc02d3ec3accceec4357b05d275a4cc5e98667b47cae0c154fc55c864f13"
DETECT_DIR = "onnx_export"
CLASSIFY_DIR = "onnx_export"
CLASSIFY_REFRESH_SEC = 5.0
IOU_MATCH = 0.3

# CPU budget on Pi 5 (4 cores): detector gets the most (it gates the visual
# latency), classifier runs on fewer threads but only when a new track needs
# labeling. Camera/annotate threads are I/O-bound and need almost no CPU.
DETECT_THREADS = 3
CLASSIFY_THREADS = 2

# one fixed color per class (BGR) — much easier to read than all-green boxes
CLASS_COLORS_BGR = [
    (60, 76, 255),    # Artery_Forceps          red
    (0, 200, 255),    # Cartridge_Syringe       yellow-orange
    (255, 80, 0),    # Cotton_Piler            blue
    (200, 0, 255),    # Dental_Mirror           magenta
    (0, 255, 255),    # Forceps_23              cyan-yellow
    (255, 255, 0),    # Forceps_150             light blue
    (0, 140, 255),    # Needle_Holder           orange
    (180, 105, 255),  # Root_Elevators          pink
    (100, 100, 100),  # Root_Tip_Elevator_LR    gray
    (50, 50, 128),    # Root_Tip_Elevator_Straight maroon
    (128, 128, 0),    # Root_Tip_Pick           olive
    (0, 255, 0),      # Scapel_Handle           green
    (255, 0, 140),    # Suture_Scissors         violet
    (30, 30, 30),    # Triple_Syringe          dark
]
UNKNOWN_COLOR_BGR = (0, 255, 0)

def class_color(cls_name: str):
    try:
        idx = detector.classes.index(cls_name)
        c = CLASS_COLORS_BGR[idx]
        return c if c != (30, 30, 30) else UNKNOWN_COLOR_BGR
    except (ValueError, AttributeError):
        return UNKNOWN_COLOR_BGR

app = Flask(__name__)
CORS(app)

print("[detector] loading ONNX ...", flush=True)
detector = DinoDetectorONNX(DETECT_DIR, num_threads=DETECT_THREADS)
print(f"[detector] classes {detector.classes} backend={detector.backend} "
      f"warmup={detector.last_ms:.0f}ms", flush=True)
print("[classifier] loading ONNX ...", flush=True)
classifier = DinoClassifierONNX(CLASSIFY_DIR, num_threads=CLASSIFY_THREADS)
print(f"[classifier] loaded backend={classifier.backend}", flush=True)


class IoUTracker:
    """Tiny greedy IoU tracker (replaces ByteTrack — tools barely move on the cloth)."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 15):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # tid -> {bbox, age, cls}

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
                self.tracks[best_tid] = {"bbox": box, "age": 0, "cls": inst["class_name"]}
                inst["track_id"] = best_tid
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "age": 0, "cls": inst["class_name"]}
                inst["track_id"] = tid
        for tid in unused:
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return instances


tracker = IoUTracker(iou_thr=IOU_MATCH)
latest_detections = {}
last_detection_time = 0
output_frame = None
latest_raw_frame = None
last_instances = []
fine_grained_cache = {}
frame_lock = threading.Lock()
detect_lock = threading.Lock()
raw_lock = threading.Lock()
cache_lock = threading.Lock()


def detect_loop():
    global last_instances, latest_detections, last_detection_time
    while True:
        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            # realtime path: NO tip crops here (crop extraction costs ~2-4ms
            # per instance) — the classifier thread cuts tips itself only for
            # tracks it actually needs to (re)label
            insts = detector.detect(frame, want_tip_crops=False)
            insts = tracker.update(insts)
        except Exception as e:
            print(f"[detector] {e}", flush=True)
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


def classify_loop():
    import det_postprocess as pp
    while True:
        frame = None
        insts = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        with detect_lock:
            insts = list(last_instances)
        if frame is None or not insts:
            time.sleep(0.2)
            continue
        now = time.time()
        h, w = frame.shape[:2]
        # resize frame once for tip-crop coordinate space (same as detector)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_model = cv2.resize(rgb_frame, (detector.img_size, detector.img_size))
        sx_m = detector.img_size / w
        sy_m = detector.img_size / h
        for inst in insts:
            tid = inst.get("track_id")
            if tid is None:
                continue
            with cache_lock:
                cached = fine_grained_cache.get(int(tid))
            if cached is not None and (now - cached["ts"]) < CLASSIFY_REFRESH_SEC:
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
                    tip_crops=inst.get("tip_crops"),   # None in realtime path
                    length_cm=inst.get("length_cm"),
                )
                # unsure first pass (small gap on a same-length pair) -> run
                # tip TTA once for this track with freshly-cut tips
                if (res["confidence"] < 0.55 and inst.get("mask") is not None
                        and res.get("top3") and len(res["top3"]) > 1):
                    m = inst["mask"]
                    if m.shape[:2] != rgb_model.shape[:2]:
                        m = cv2.resize(m, (detector.img_size, detector.img_size),
                                       interpolation=cv2.INTER_NEAREST)
                    tips = pp.extract_tip_crops(rgb_model, m)
                    if tips:
                        res = classifier.classify(
                            crop, x2 - x1, y2 - y1,
                            length_px=inst.get("length_px_frame"),
                            tip_crops=tips,
                            length_cm=inst.get("length_cm"),
                            force_tip_tta=True,
                        )
            except Exception as e:
                print(f"[classifier] {e}", flush=True)
                continue
            with cache_lock:
                fine_grained_cache[int(tid)] = {
                    "class": res["class"], "confidence": res["confidence"],
                    "length_used": res["length_used"], "ts": now,
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
        det_ms = getattr(detector, "last_ms", 0)
        hud = f"FPS:{fps} det:{det_ms:.0f}ms[{detector.backend}]"
        cv2.putText(annotated, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # color legend (top-right, compact) — drawn once per frame, tiny cost
        lx = annotated.shape[1] - 8
        for cidx in range(len(detector.classes)):
            cname = detector.classes[cidx]
            color = CLASS_COLORS_BGR[cidx] if cidx < len(CLASS_COLORS_BGR) else UNKNOWN_COLOR_BGR
            if color == (30, 30, 30):
                color = UNKNOWN_COLOR_BGR
            (tw, th), _ = cv2.getTextSize(cname, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(annotated, (lx - tw - 14, 10 + cidx * 16),
                          (lx - tw - 4, 20 + cidx * 16), color, -1)
            cv2.putText(annotated, cname, (lx - tw, 20 + cidx * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
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
    with cache_lock:
        fine = {tid: v["class"] for tid, v in fine_grained_cache.items()}
    return jsonify({"data": latest_detections, "fine_grained": fine, "timestamp": last_detection_time})


@app.route("/")
def index():
    return "DENIS DINOv2 Detector + Fine-Grained Server Running v3.0 (no YOLO)"


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()
threading.Thread(target=classify_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
