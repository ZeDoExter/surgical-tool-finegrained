# -*- coding: utf-8 -*-
"""
detector_onnx.py — DINOv2 detector (YOLO-free) on Raspberry Pi 5 via ONNX Runtime

v3.1 realtime optimizations:
  - auto-prefers INT8 model if present (detector_dino_int8.onnx) — 2-3x faster
    on CPU with negligible accuracy loss for this task
  - thread budget: default 3 threads (leave 1 core for camera/annotate/classifier)
  - reports last inference time (ms) for the HUD
"""
import json
import os
import time
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

import det_postprocess as pp


class DinoDetectorONNX:
    def __init__(self, onnx_dir: str = "onnx_export", num_threads: Optional[int] = 3,
                 prefer_int8: bool = True, model_file: Optional[str] = None):
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
        # patch-token fast path (DINOv2 exports only; the tiny CNN student
        # has no tokens — fast labels then come from its own coarse output)
        self.has_tokens: bool = bool(self.meta.get("has_patch_tokens", False))
        self.token_grid: int = int(self.meta.get("token_grid", self.img_size // 14))
        self.patch_dim: int = int(self.meta.get("patch_dim", 384))
        self.last_tokens = None  # (patch_dim, G, G) of the latest frame

        # fp32 first — INT8 quantization broke this ViT graph in tests
        # (kept selectable only if you quantized + verified one yourself)
        if model_file is None:
            # v7: the full DINOv2 detector is the primary (app.py main path,
            # best quality). The distilled student CNN stays as fallback for
            # realtime-priority builds — pass model_file to pick explicitly.
            candidates = ["detector_dino.onnx",
                          "detector_dino_student.onnx",
                          "detector_dino_int8.onnx"]
        else:
            candidates = [model_file]
        onnx_name = next((n for n in candidates
                          if os.path.exists(os.path.join(onnx_dir, n))), candidates[-1])
        self.model_file = onnx_name
        self.backend = "fp32" if "int8" not in onnx_name else "int8"
        self.is_student = "student" in onnx_name

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            os.path.join(onnx_dir, onnx_name),
            sess_options=so, providers=["CPUExecutionProvider"])

        # warmup: first inference allocates memory arenas + spins the thread
        # pool — do it at startup, not on the first real frame
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.detect(dummy, want_tip_crops=False)
        self.last_ms = 0.0

    def detect(self, frame_bgr: np.ndarray, want_tip_crops: bool = True) -> List[dict]:
        """
        frame_bgr: camera frame (any size; internally resized to img_size)
        -> instances (bbox in FRAME coords, mask at model scale, length, tips).
        """
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # resize ONCE, reuse for both the model input and tip-crop space
        rgb_model = cv2.resize(rgb, (self.img_size, self.img_size))
        px = (rgb_model.astype(np.float32) / 255.0)
        px = (px - pp.IMAGENET_MEAN) / pp.IMAGENET_STD
        px = np.ascontiguousarray(px.transpose(2, 0, 1)[None], dtype=np.float32)
        want_toks = self.has_tokens and not self.is_student
        out_names = ["logits", "patch_tokens"] if want_toks else ["logits"]
        outs = self.session.run(out_names, {"pixel_values": px})
        logits = outs[0][0]  # (1+C, h, w)
        sx, sy = W / logits.shape[2], H / logits.shape[1]

        # zero-cost backbone tokens for the prototype fast path (DINOv2 only)
        self.last_tokens = None
        if want_toks and len(outs) > 1:
            self.last_tokens = outs[1][0].astype(np.float32)  # (384, G, G)

        insts = pp.instances_from_logits(
            logits, self.classes,
            frame_rgb=rgb_model,
            mask_threshold=self.mask_threshold,
            min_instance_area=self.min_instance_area,
            nms_iou=self.nms_iou,
            conf_min_score=self.conf_min_score,
            want_tip_crops=want_tip_crops,
        )
        for inst in insts:
            x1, y1, x2, y2 = inst["bbox"]
            inst["bbox_frame"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            inst["length_px_frame"] = inst["length_px"] * max(sx, sy)
            if self.calibration_ratio:
                inst["length_cm"] = inst["length_px_frame"] * self.calibration_ratio
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return insts

    def pool_tokens(self, mask: np.ndarray, tokens: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Mean-pool backbone patch tokens over an instance mask.

        mask: (Hm, Wm) at model scale -> downsampled to the token grid,
        cells with any FG counted. tokens: (D, G, G) or None (= latest).
        Returns L2-normalized (patch_dim,) or None.
        Runtime cost is ~1ms — this is the "free" embedding the prototype
        fast path classifies with (no 2nd ViT forward).
        """
        toks = tokens if tokens is not None else self.last_tokens
        if toks is None:
            return None
        D, G, _ = toks.shape
        small = cv2.resize((mask > 0).astype(np.uint8), (G, G),
                           interpolation=cv2.INTER_NEAREST) > 0
        if int(small.sum()) < 2:
            return None
        v = toks[:, small].mean(axis=1).astype(np.float64)
        n = float(np.linalg.norm(v))
        return (v / (n + 1e-8)).astype(np.float32)

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
