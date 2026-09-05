# -*- coding: utf-8 -*-
"""
pi_final_v3/app_yolo.py — YOLO26n ONLY on Raspberry Pi 5 (no DINOv2 at all)

YOLO gives boxes + 14-class labels straight from one forward (~50ms).
No classifier, no second model, no per-track labeling thread.

    pip install onnxruntime opencv-python flask flask-cors numpy
    python app_yolo.py
"""
import time
import threading

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from yolo_detector_onnx import YoloDetectorONNX

API_KEY = "e5bdb16ce0552c091383244b3c814ffe51bd15fa73e0bc6f4dc7b09afe00a6a67d2bcc02d3ec3accceec4357b05d275a4cc5e98667b47cae0c154fc55c864f13"
DETECT_DIR = "onnx_export"
IOU_MATCH = 0.3

# YOLO forward is cheap; 2 threads is plenty, leaving cores for camera/server
DETECT_THREADS = 2

# same dark palette as app.py (white text readable on all)
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
    (60, 140, 0),       # Scapel_Handle             dark green
    (180, 30, 110),     # Suture_Scissors           dark violet
    (130, 90, 70),      # Triple_Syringe            dark slate
]
UNKNOWN_COLOR_BGR = (0, 0, 180)

app = Flask(__name__)
CORS(app)

print("[yolo] loading ONNX ...", flush=True)
detector = YoloDetectorONNX(DETECT_DIR, num_threads=DETECT_THREADS)
print(f"[yolo] classes {detector.classes} backend={detector.backend} "
      f"warmup={detector.last_ms:.0f}ms", flush=True)


def class_color(cls_name: str):
    try:
        idx = detector.classes.index(cls_name)
        return CLASS_COLORS_BGR[idx] if idx < len(CLASS_COLORS_BGR) else UNKNOWN_COLOR_BGR
    except (ValueError, AttributeError):
        return UNKNOWN_COLOR_BGR


class IoUTracker:
    """Tiny greedy IoU tracker (tools barely move on the cloth)."""

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
frame_lock = threading.Lock()
detect_lock = threading.Lock()
raw_lock = threading.Lock()


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
            insts = detector.detect(frame, want_tip_crops=False)
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
            name, conf = inst["class_name"], inst["score"]
            color = class_color(name)
            x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            if tid is not None:
                label += f" #{tid}"
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
    with detect_lock:
        tracks = {int(inst["track_id"]): inst["class_name"]
                  for inst in last_instances if inst.get("track_id") is not None}
    return jsonify({"data": latest_detections, "tracks": tracks, "timestamp": last_detection_time})


@app.route("/")
def index():
    return "DENIS YOLO26n Server Running v3.1 (YOLO only, no DINO)"


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
