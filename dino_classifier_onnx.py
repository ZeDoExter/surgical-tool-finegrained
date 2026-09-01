# -*- coding: utf-8 -*-
"""
dino_classifier_onnx.py — รัน DINOv2 fine-grained classifier บน Raspberry Pi ด้วย ONNX Runtime
ไม่ต้องใช้ torch / transformers / peft เลย เบากว่าเดิมมากสำหรับ edge device

ก่อนใช้ไฟล์นี้ ต้อง:
1) export_to_onnx.py บนเครื่อง PC ก่อน ได้ 3 ไฟล์: surgical_dino_fusion.onnx,
   arcface_W.npy, classifier_meta.json
2) เอา 3 ไฟล์นั้นมาวางในโฟลเดอร์เดียวกับไฟล์นี้ (หรือแก้ ONNX_DIR ด้านล่าง)
3) pip install onnxruntime opencv-python numpy   (บน Pi ไม่ต้องลง torch เลย)
"""
import json
import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

ONNX_DIR = os.path.dirname(os.path.abspath(__file__))  # แก้ path ตรงนี้ถ้าไฟล์ไม่ได้อยู่โฟลเดอร์เดียวกัน

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoClassifierONNX:
    def __init__(self, onnx_dir: str = ONNX_DIR, num_threads: Optional[int] = None):
        with open(os.path.join(onnx_dir, "classifier_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.classes = self.meta["classes"]
        self.length_mean = self.meta["length_mean"]
        self.length_std = self.meta["length_std"]
        self.scale = self.meta["scale"]
        self.img_size = self.meta["img_size"]
        self.calibration_ratio = self.meta.get("calibration_ratio")

        W = np.load(os.path.join(onnx_dir, "arcface_W.npy"))  # (embed_dim, num_classes)
        # normalize ล่วงหน้าครั้งเดียว ตอน inference จะได้ทำแค่ dot product
        self.W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            so.intra_op_num_threads = num_threads  # จำนวน core ของ Pi เช่น 4

        onnx_path = os.path.join(onnx_dir, "surgical_dino_fusion.onnx")
        self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                             providers=["CPUExecutionProvider"])

    def _preprocess(self, rgb_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(rgb_crop, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)[None]  # (1,3,H,W)
        return np.ascontiguousarray(img, dtype=np.float32)

    def classify(self, rgb_crop: np.ndarray, box_w: int, box_h: int) -> dict:
        """
        rgb_crop: ภาพ crop RGB (uint8) ของเครื่องมือ 1 ชิ้น
        box_w, box_h: ขนาดของ bbox เป็นพิกเซล (ใช้แทน length จาก mask จริง — ดูคำเตือน
                      เรื่องความแม่นยำในไฟล์ teststream2_with_classifier.py)
        """
        length_px = max(box_w, box_h)
        if self.calibration_ratio:
            length_px = length_px * self.calibration_ratio
        length_norm = np.array([(length_px - self.length_mean) / self.length_std], dtype=np.float32)

        pixel_values = self._preprocess(rgb_crop)

        (embedding,) = self.session.run(
            ["embedding"],
            {"pixel_values": pixel_values, "length_feat": length_norm},
        )
        emb = embedding[0]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        cos = emb_norm @ self.W_norm  # (num_classes,) cosine similarity กับแต่ละคลาส
        logits = cos * self.scale
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        order = probs.argsort()[::-1]
        top_idx = int(order[0])
        top3 = [{"class": self.classes[int(i)], "prob": float(probs[i])} for i in order[:3]]

        return {
            "class": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "top3": top3,
            "length_used": float(length_px),
        }
