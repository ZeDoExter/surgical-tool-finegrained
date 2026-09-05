# -*- coding: utf-8 -*-
"""
yolo_detector_onnx.py — YOLO26n detector on Raspberry Pi 5 via ONNX Runtime

Pure onnxruntime + numpy + cv2. NO ultralytics needed on the Pi.
The exported graph already contains NMS: output0 = (1, 300, 6) rows of
[x1, y1, x2, y2, conf, class] in 640-letterboxed coords.

Returns the SAME instance-dict contract as DinoDetectorONNX so app.py code
(tracker, classifier, drawing) works unchanged:
  bbox, bbox_frame, label, class_name, score, length_px_frame,
  length_cm (if calibrated), tip_crops (cut from box ends — YOLO has no mask)

Files needed in onnx_dir: yolo26n.onnx, yolo_meta.json
"""
import json
import os
import time
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort


def _letterbox(im: np.ndarray, size: int = 640):
    """Ultralytics-style letterbox (aspect kept, gray pad). Returns (padded, scale, pad_x, pad_y)."""
    h, w = im.shape[:2]
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((size, size, 3), 114, np.uint8)
    out[(size - nh) // 2:(size - nh) // 2 + nh,
        (size - nw) // 2:(size - nw) // 2 + nw] = cv2.resize(im, (nw, nh))
    return out, s, (size - nw) // 2, (size - nh) // 2


def _box_tip_crops(rgb: np.ndarray, bbox, tip_frac: float = 0.45) -> List[np.ndarray]:
    """Tip crops from box ends along its long side (no-mask fallback)."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return []
    H, W = rgb.shape[:2]
    if w >= h:  # horizontal tool — take left/right ends
        side = max(int(h * 1.1), 24)
        cy = (y1 + y2) / 2
        boxes = [
            (x1, cy - side / 2, x1 + w * tip_frac, cy + side / 2),
            (x2 - w * tip_frac, cy - side / 2, x2, cy + side / 2),
        ]
    else:       # vertical tool — take top/bottom ends
        side = max(int(w * 1.1), 24)
        cx = (x1 + x2) / 2
        boxes = [
            (cx - side / 2, y1, cx + side / 2, y1 + h * tip_frac),
            (cx - side / 2, y2 - h * tip_frac, cx + side / 2, y2),
        ]
    crops = []
    for (a, b, c, d) in boxes:
        a, b = max(int(a), 0), max(int(b), 0)
        c, d = min(int(c), W), min(int(d), H)
        if c - a >= 10 and d - b >= 10:
            crops.append(rgb[b:d, a:c])
    return crops


def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def _nms_agnostic(insts: List[dict], thr: float) -> List[dict]:
    """Class-agnostic NMS: keep highest score, drop overlaps > thr."""
    insts = sorted(insts, key=lambda d: -d["score"])
    keep: List[dict] = []
    for inst in insts:
        if any(_iou_xyxy(inst["bbox_frame"], k["bbox_frame"]) > thr for k in keep):
            continue
        keep.append(inst)
    return keep


class YoloDetectorONNX:
    def __init__(self, onnx_dir: str = "onnx_export", num_threads: Optional[int] = 2,
                 model_file: str = "yolo26n.onnx"):
        with open(os.path.join(onnx_dir, "yolo_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.classes: List[str] = self.meta["classes"]
        self.img_size: int = self.meta.get("img_size", 640)
        self.conf_threshold: float = self.meta.get("conf_threshold", 0.25)
        # second-pass NMS (the in-graph NMS still lets same-spot duplicates
        # through — e.g. one tool answered twice, or two class boxes on the
        # same tool). Class-AGNOSTIC: any lower-conf box overlapping a kept
        # box more than nms_iou is suppressed. Tools on the cloth overlap
        # <=20% by design, so neighbors are never collaterally removed.
        self.nms_iou: float = self.meta.get("nms_iou", 0.45)
        self.calibration_ratio = self.meta.get("calibration_ratio")
        self.model_file = self.meta.get("model_file", model_file)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            os.path.join(onnx_dir, self.model_file),
            sess_options=so, providers=["CPUExecutionProvider"])
        self.backend = "yolo-fp32"
        self.last_ms = 0.0

        # warmup so the first real frame isn't slow
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.detect(dummy, want_tip_crops=False)
        self.last_ms = 0.0

    def detect(self, frame_bgr: np.ndarray, want_tip_crops: bool = True) -> List[dict]:
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb, s, px, py = _letterbox(rgb, self.img_size)
        x = (lb[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        (rows,) = self.session.run(["output0"], {"images": x})

        insts: List[dict] = []
        for row in rows[0]:
            x1, y1, x2, y2, conf, cls = (float(v) for v in row)
            if conf < self.conf_threshold:
                continue
            label = int(cls)
            if not (0 <= label < len(self.classes)):
                continue
            # un-letterbox to frame coords
            fx1, fy1 = (x1 - px) / s, (y1 - py) / s
            fx2, fy2 = (x2 - px) / s, (y2 - py) / s
            fx1, fy1 = max(fx1, 0), max(fy1, 0)
            fx2, fy2 = min(fx2, W), min(fy2, H)
            bw, bh = fx2 - fx1, fy2 - fy1
            if bw < 5 or bh < 5:
                continue
            length = float(max(bw, bh))  # bbox approx (diagonal tools overestimate — see note)
            inst = {
                "bbox": [fx1, fy1, fx2, fy2],
                "bbox_frame": [fx1, fy1, fx2, fy2],
                "label": label,
                "class_name": self.classes[label],
                "score": float(conf),
                "mask": None,              # YOLO has no mask
                "length_px": length,
                "length_px_frame": length,
                "angle_deg": 0.0,
            }
            if self.calibration_ratio:
                inst["length_cm"] = length * self.calibration_ratio
            if want_tip_crops:
                inst["tip_crops"] = _box_tip_crops(rgb, [fx1, fy1, fx2, fy2])
            insts.append(inst)

        # second-pass class-agnostic NMS (kills same-spot duplicates the
        # in-graph NMS missed — see __init__ note)
        insts = _nms_agnostic(insts, self.nms_iou)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return insts
