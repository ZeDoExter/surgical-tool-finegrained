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

app = Flask(__name__)
CORS(app)

print("[detector] loading ONNX ...", flush=True)
detector = DinoDetectorONNX(DETECT_DIR, num_threads=4)
print("[detector] classes", detector.classes, flush=True)
print("[classifier] loading ONNX ...", flush=True)
classifier = DinoClassifierONNX(CLASSIFY_DIR, num_threads=4)
print("[classifier] loaded", classifier.classes, flush=True)


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
            insts = detector.detect(frame, want_tip_crops=True)
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
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            try:
                res = classifier.classify(
                    crop, x2 - x1, y2 - y1,
                    length_px=inst.get("length_px_frame"),
                    tip_crops=inst.get("tip_crops"),
                    length_cm=inst.get("length_cm"),
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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if fine is not None:
                label = f"{fine['class']} {fine['confidence']:.2f} #{tid}"
            else:
                label = f"{inst['class_name']} {inst['score']:.2f} #{tid} (classifying...)"
            cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        cv2.putText(annotated, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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
