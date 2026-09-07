# -*- coding: utf-8 -*-
"""
yolo_detector_ncnn.py — YOLO26n detector on Raspberry Pi 5 via NCNN.

Pure ncnn + numpy + cv2. NO ultralytics, NO onnxruntime needed on the Pi
for the detector. The exported graph has NO in-graph NMS (end2end: false),
so multiclass NMS runs here in numpy/cv2 (cheap: a few hundred boxes).

Trained-with-Head model support: the v9 weights were trained with 20
classes (14 tools + 6 *_Head tip crops). Head-class rows are SKIPPED and
their boxes never emitted — the Pi contract stays at our 14 classes.

Returns the SAME instance-dict contract as YoloDetectorONNX:
  bbox, bbox_frame, label, class_name, score, length_px_frame,
  length_cm (if calibrated), tip_crops (cut from box ends — YOLO has no mask)

Files needed in onnx_dir: yolo_ncnn_meta.json + <model_dir>/model.ncnn.{param,bin}
"""
import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import ncnn
    _NCNN_OK = True
except ImportError:  # pragma: no cover
    _NCNN_OK = False


def _letterbox(im: np.ndarray, size: int = 832):
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class YoloDetectorNCNN:
    def __init__(self, onnx_dir: str = "onnx_export", num_threads: Optional[int] = 4,
                 meta_file: str = "yolo_ncnn_meta.json"):
        if not _NCNN_OK:
            raise RuntimeError("ncnn package not installed (pip install ncnn)")
        with open(os.path.join(onnx_dir, meta_file), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.classes: List[str] = self.meta["classes"]
        self.img_size: int = self.meta.get("img_size", 832)
        self.conf_threshold: float = self.meta.get("conf_threshold", 0.25)
        self.nms_iou: float = self.meta.get("nms_iou", 0.45)
        self.calibration_ratio = self.meta.get("calibration_ratio")
        # v9-class-index -> our 14-class index; Head classes absent = skipped
        self.v9_to_ours: Dict[int, int] = {
            int(k): int(v) for k, v in self.meta["v9_to_ours"].items()}
        self.model_dir = os.path.join(onnx_dir, self.meta["model_dir"])

        self.net = ncnn.Net()
        if num_threads:
            self.net.opt.num_threads = num_threads
        self.net.load_param(os.path.join(self.model_dir, "model.ncnn.param"))
        self.net.load_model(os.path.join(self.model_dir, "model.ncnn.bin"))
        self.backend = f"ncnn-{self.img_size}"
        self.last_ms = 0.0
        # v9-class-index -> head target name; _Head rows whose box center falls
        # inside a main box override that box's label (the non-head strategy:
        # tips disambiguate same-shape tools). Empty map = Head rows skipped.
        self.head_to_main: Dict[int, str] = {
            int(k): str(v) for k, v in self.meta.get("head_to_main", {}).items()}
        self.use_head_override: bool = bool(self.meta.get("use_head_override",
                                                           bool(self.head_to_main)))

        # warmup so the first real frame isn't slow
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.detect(dummy, want_tip_crops=False)
        self.last_ms = 0.0

    def _forward(self, lb_rgb: np.ndarray) -> np.ndarray:
        mat = ncnn.Mat.from_pixels(lb_rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                   self.img_size, self.img_size)
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        with self.net.create_extractor() as ex:
            ex.input("in0", mat)
            _, out0 = ex.extract("out0")
        a = np.array(out0)
        if a.ndim == 3:
            a = a[0]
        return np.ascontiguousarray(a, dtype=np.float32)  # (4+nc, N)

    def detect(self, frame_bgr: np.ndarray, want_tip_crops: bool = True) -> List[dict]:
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb, s, px, py = _letterbox(rgb, self.img_size)
        pred = self._forward(lb)  # (4+20, N): cx,cy,w,h + class scores
        box, cls = pred[:4], pred[4:]
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = _sigmoid(cls)  # raw logits (export-dependent)

        # per-class NMS on the small set that passes the threshold
        cand = np.argwhere(cls >= self.conf_threshold)
        insts: List[dict] = []
        for c in np.unique(cand[:, 0]) if len(cand) else []:
            ours = self.v9_to_ours.get(int(c))
            if ours is None:
                continue  # *_Head class row — never emitted
            rows = cand[cand[:, 0] == c][:, 1]
            bx = box[:, rows]  # (4, M)
            x1 = bx[0] - bx[2] / 2
            y1 = bx[1] - bx[3] / 2
            bboxes = [[float(x1[i]), float(y1[i]),
                       float(bx[2][i]), float(bx[3][i])] for i in range(len(rows))]
            scores = [float(cls[c, n]) for n in rows]
            keep = cv2.dnn.NMSBoxes(bboxes, scores,
                                    self.conf_threshold, self.nms_iou)
            for k in np.array(keep).ravel():
                bw, bh = bboxes[k][2], bboxes[k][3]
                fx1, fy1 = (bboxes[k][0] - px) / s, (bboxes[k][1] - py) / s
                bw, bh = bw / s, bh / s
                fx1, fy1 = max(fx1, 0), max(fy1, 0)
                fx2, fy2 = min(fx1 + bw, W), min(fy1 + bh, H)
                if fx2 - fx1 < 5 or fy2 - fy1 < 5:
                    continue
                length = float(max(fx2 - fx1, fy2 - fy1))
                inst = {
                    "bbox": [fx1, fy1, fx2, fy2],
                    "bbox_frame": [fx1, fy1, fx2, fy2],
                    "label": ours,
                    "class_name": self.classes[ours],
                    "score": scores[k],
                    "mask": None,
                    "length_px": length,
                    "length_px_frame": length,
                    "angle_deg": 0.0,
                }
                if self.calibration_ratio:
                    inst["length_cm"] = length * self.calibration_ratio
                if want_tip_crops:
                    inst["tip_crops"] = _box_tip_crops(rgb, [fx1, fy1, fx2, fy2])
                insts.append(inst)

        if self.use_head_override and self.head_to_main:
            # non-head strategy: collect _Head boxes, then let any head whose
            # center falls inside a main box override that box's label
            head_boxes = []
            for c, target in self.head_to_main.items():
                rows = cand[cand[:, 0] == c][:, 1] if len(cand) else []
                for n in rows:
                    if float(cls[c, n]) < self.conf_threshold:
                        continue
                    cx, cy, bw, bh = (float(v) for v in box[:, n])
                    hx1, hy1 = (cx - bw / 2 - px) / s, (cy - bh / 2 - py) / s
                    hx2, hy2 = (cx + bw / 2 - px) / s, (cy + bh / 2 - py) / s
                    head_boxes.append((target, (hx1 + hx2) / 2, (hy1 + hy2) / 2))
            for inst in insts:
                fx1, fy1, fx2, fy2 = inst["bbox_frame"]
                for target, cx, cy in head_boxes:
                    if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                        if target in self.classes:
                            inst["label"] = self.classes.index(target)
                            inst["class_name"] = target
                            inst["via"] = "head-override"
                        break

        # second-pass class-agnostic NMS (same-spot duplicates across classes)
        insts = _nms_agnostic(insts, self.nms_iou)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return insts
