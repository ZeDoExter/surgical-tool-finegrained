# -*- coding: utf-8 -*-
"""
fast_classifier.py — zero-cost labels from backbone tokens + class prototypes

Principle: DINOv2 patch features are linearly separable (linear-probe
paradigm). class_prototypes.npy holds one 384-d centroid per class, built
offline by build_prototypes.py. For each detected instance:

    emb   = mean(pool(tokens, instance_mask))   (~1ms, no new forward)
    probs = softmax(cos(emb, P) / tau) * length_prior
    if max(probs) low OR top-2 gap small OR top-2 in a same-length pair:
        -> return need_refine=True (caller runs the slow ArcFace classifier)
    else:
        -> instant label (cascade fast path)

No torch/onnxruntime needed here — pure numpy.
"""
import json
import os
from typing import List, Optional

import numpy as np

TIP_PAIRS = [
    {"Needle_Holder", "Artery_Forceps"},
    {"Mandibular_Universal_Forceps_23", "Maxillary_Universal_Forceps_150"},
]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _length_prior(length_cm: Optional[float], classes: List[str],
                  real_length_cm: dict, sigma_cm: float = 1.2) -> np.ndarray:
    K = len(classes)
    prior = np.ones(K, dtype=np.float64)
    if length_cm is None or not real_length_cm:
        return prior
    known = {c: real_length_cm[c] for c in classes
             if c in real_length_cm and real_length_cm[c] is not None}
    if not known:
        return prior
    for c, L in known.items():
        prior[classes.index(c)] = np.exp(-((length_cm - L) ** 2) / (2 * sigma_cm ** 2))
    neutral = max(prior[classes.index(c)] for c in known)
    for i, c in enumerate(classes):
        if c not in known:
            prior[i] = neutral
    return prior / prior.sum()


class FastClassifier:
    def __init__(self, onnx_dir: str = "onnx_export", tau: float = 0.07,
                 conf_threshold: float = 0.55, gap_threshold: float = 0.05):
        with open(os.path.join(onnx_dir, "detector_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes: List[str] = meta["classes"]
        proto_file = meta.get("prototypes", "class_prototypes.npy")
        self.P = np.load(os.path.join(onnx_dir, proto_file)).astype(np.float64)
        self.P = self.P / (np.linalg.norm(self.P, axis=0, keepdims=True) + 1e-8)
        self.real_length_cm = meta.get("real_length_cm", {})
        self.tau = tau
        self.conf_threshold = conf_threshold
        self.gap_threshold = gap_threshold
        self.available = self.P.shape == (self.P.shape[0], len(self.classes))

    def predict(self, emb: np.ndarray, length_cm: Optional[float] = None) -> dict:
        """emb: L2-normalized pooled embedding. Returns class/conf/need_refine."""
        cos = emb.astype(np.float64) @ self.P            # (C,)
        probs = _softmax(cos / self.tau)
        if length_cm is not None and self.real_length_cm:
            prior = _length_prior(length_cm, self.classes, self.real_length_cm)
            probs = probs * prior
            probs = probs / probs.sum()
        order = probs.argsort()[::-1]
        top1, top2 = int(order[0]), int(order[1])
        conf = float(probs[top1])
        gap = float(probs[top1] - probs[top2])
        pair = {self.classes[top1], self.classes[top2]}
        need_refine = (
            conf < self.conf_threshold
            or gap < self.gap_threshold
            or any(pair == p for p in TIP_PAIRS)
        )
        return {
            "class": self.classes[top1],
            "confidence": conf,
            "top3": [{"class": self.classes[int(i)], "prob": float(probs[i])}
                     for i in order[:3]],
            "need_refine": bool(need_refine),
            "via": "fast",
        }
