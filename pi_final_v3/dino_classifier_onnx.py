# -*- coding: utf-8 -*-
"""
dino_classifier_onnx.py — DINOv2 fine-grained classifier on Pi 5 (ONNX Runtime)

v3 additions vs pi_final/:
  - length from detector MASK (minAreaRect) instead of max(bbox w,h)
  - tip TTA: extra forward on both instrument-end crops when top-2 is a
    same-length pair (Needle_Holder↔Artery_Forceps, 23↔150)
  - length prior in cm (after calibrate.py) for Root_Elevators 15.5 vs
    Root_Tip_Elevator_Straight 14.5
"""
import json
import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

ONNX_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TIP_PAIRS = [
    {"Needle_Holder", "Artery_Forceps"},
    {"Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150"},
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _length_prior(length_cm: Optional[float], classes: List[str],
                  real_length_cm: dict, sigma_cm: float = 1.2) -> np.ndarray:
    K = len(classes)
    prior = np.ones(K, dtype=np.float64)
    if length_cm is None or not real_length_cm:
        return prior
    known = {c: real_length_cm[c] for c in classes if c in real_length_cm and real_length_cm[c] is not None}
    if not known:
        return prior
    for c, L in known.items():
        prior[classes.index(c)] = np.exp(-((length_cm - L) ** 2) / (2 * sigma_cm ** 2))
    neutral = max(prior[classes.index(c)] for c in known)
    for i, c in enumerate(classes):
        if c not in known:
            prior[i] = neutral
    return prior / prior.sum()


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
        self.real_length_cm = self.meta.get("real_length_cm") or {}

        W = np.load(os.path.join(onnx_dir, "arcface_W.npy"))
        self.W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            os.path.join(onnx_dir, "surgical_dino_fusion.onnx"),
            sess_options=so, providers=["CPUExecutionProvider"])

    def _preprocess(self, rgb_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(rgb_crop, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32)

    def _embed(self, rgb_crop: np.ndarray, length_norm: np.ndarray) -> np.ndarray:
        (embedding,) = self.session.run(
            ["embedding"],
            {"pixel_values": self._preprocess(rgb_crop), "length_feat": length_norm},
        )
        emb = embedding[0]
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _probs_from_emb(self, emb_norm: np.ndarray) -> np.ndarray:
        logits = (emb_norm @ self.W_norm) * self.scale
        return _softmax(logits)

    def classify(self, rgb_crop: np.ndarray, box_w: int, box_h: int,
                 length_px: Optional[float] = None,
                 tip_crops: Optional[List[np.ndarray]] = None,
                 length_cm: Optional[float] = None) -> dict:
        """
        rgb_crop: RGB crop of one instrument
        box_w, box_h: fallback length if no mask length is given
        length_px: minAreaRect long side from the detector mask (preferred)
        tip_crops: both-end RGB crops from det_postprocess.extract_tip_crops
        length_cm: physical length if calibration_ratio is set
        """
        L = float(length_px) if length_px is not None else float(max(box_w, box_h))
        if self.calibration_ratio and length_cm is None:
            length_cm = L * self.calibration_ratio
            L_feat = length_cm
        else:
            L_feat = L
        length_norm = np.array([(L_feat - self.length_mean) / self.length_std], dtype=np.float32)

        emb = self._embed(rgb_crop, length_norm)
        probs = self._probs_from_emb(emb)

        if tip_crops:
            tip_ps = []
            for crop in tip_crops:
                if crop is None or crop.size < 16:
                    continue
                try:
                    tip_ps.append(self._probs_from_emb(self._embed(crop, length_norm)))
                except Exception:
                    continue
            if tip_ps:
                probs_tip = np.mean(tip_ps, axis=0)
                top2 = {self.classes[int(i)] for i in np.argsort(probs)[::-1][:2]}
                w = 2.5 if any(top2 == p for p in TIP_PAIRS) else 1.0
                blended = probs * (probs_tip ** w)
                probs = blended / blended.sum()

        if length_cm is not None and self.real_length_cm:
            prior = _length_prior(length_cm, self.classes, self.real_length_cm)
            probs = probs * prior
            probs = probs / probs.sum()

        order = probs.argsort()[::-1]
        top_idx = int(order[0])
        top3 = [{"class": self.classes[int(i)], "prob": float(probs[i])} for i in order[:3]]
        return {
            "class": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "top3": top3,
            "length_used": float(L_feat),
        }
