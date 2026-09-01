# -*- coding: utf-8 -*-
"""
detector_onnx.py — DINOv2 detector (YOLO-free) on Raspberry Pi 5 via ONNX Runtime

Same post-processing as det_postprocess.py (numpy/cv2 only). One 560×560
forward per frame → mask + label per instance → bbox + mask length (minAreaRect).

Files needed in onnx_dir:
    detector_dino.onnx, detector_meta.json
"""
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# keep det_postprocess import-free of torch — reuse it directly
import det_postprocess as pp


class DinoDetectorONNX:
    def __init__(self, onnx_dir: str = "onnx_export", num_threads: Optional[int] = None):
        with open(os.path.join(onnx_dir, "detector_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.classes: List[str] = self.meta["classes"]
        self.img_size: int = self.meta["img_size"]
        self.mask_threshold: float = self.meta.get("mask_threshold", 0.5)
        self.min_instance_area: int = self.meta.get("min_instance_area", 80)
        self.nms_iou: float = self.meta.get("nms_iou", 0.4)
        self.conf_min_score: float = self.meta.get("conf_min_score", 0.35)
        self.calibration_ratio = self.meta.get("calibration_ratio")
        self.real_length_cm = self.meta.get("real_length_cm", {})

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            os.path.join(onnx_dir, "detector_dino.onnx"),
            sess_options=so, providers=["CPUExecutionProvider"])

    def detect(self, frame_bgr: np.ndarray, want_tip_crops: bool = True) -> List[dict]:
        """
        frame_bgr: camera frame (any size; internally resized to 560)
        → list of instances (bbox in FRAME coords, mask at 560 scale,
          length_px at frame scale, optional tip_crops for the classifier).
        """
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        px = pp.preprocess_frame(rgb, self.img_size, rgb_input=True)
        (logits,) = self.session.run(["logits"], {"pixel_values": px})
        logits = logits[0]  # (1+C, 560, 560)
        sx, sy = W / logits.shape[2], H / logits.shape[1]

        rgb560 = cv2.resize(rgb, (self.img_size, self.img_size))
        insts = pp.instances_from_logits(
            logits, self.classes,
            frame_rgb=rgb560,           # tip crops at 560-space (enough detail)
            mask_threshold=self.mask_threshold,
            min_instance_area=self.min_instance_area,
            nms_iou=self.nms_iou,
            conf_min_score=self.conf_min_score,
            want_tip_crops=want_tip_crops,
        )
        # frame-space boxes + length in frame px → cm via calibration
        for inst in insts:
            x1, y1, x2, y2 = inst["bbox"]
            inst["bbox_frame"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            inst["length_px_frame"] = inst["length_px"] * max(sx, sy)
            if self.calibration_ratio:
                inst["length_cm"] = inst["length_px_frame"] * self.calibration_ratio
            # tip crops are at 560-space — fine for the classifier input
        return insts

    def draw(self, frame_bgr: np.ndarray, instances: List[dict],
             fine: Optional[dict] = None) -> np.ndarray:
        canvas = frame_bgr.copy()
        for i, inst in enumerate(instances):
            x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = inst["class_name"]
            conf = inst["score"]
            if fine and i in fine:
                name = fine[i].get("class", name)
                conf = fine[i].get("confidence", conf)
            txt = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(canvas, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1, cv2.LINE_AA)
            if "length_cm" in inst:
                cv2.putText(canvas, f"{inst['length_cm']:.1f}cm", (x1 + 2, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return canvas
