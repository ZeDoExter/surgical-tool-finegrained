# -*- coding: utf-8 -*-
"""
det_postprocess.py — detector output → YOLO-style instances (pure numpy/cv2)

Runs identically on PC (torch) and Raspberry Pi (ONNX Runtime):
input  = (1+C, H, W) logits (ch0 = fg, ch1.. = classes)
output = list of instances:
    {bbox(x1,y1,x2,y2 float), label int, class_name str, score float,
     mask (H,W) uint8, length_px float (minAreaRect long side, NOT max(w,h)
     of an axis-aligned bbox — diagonal tools inflate bbox by up to √2),
     angle_deg float, tip_crops [np.ndarray, np.ndarray] (RGB, both ends
     along the major axis — feed to the fine-grained classifier)}

No torch dependency here → the Pi copy is byte-identical.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

# (PIL/torch IMAGENET stats duplicated here to keep this module dependency-free)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_frame(bgr_or_rgb: np.ndarray, img_size: int, rgb_input: bool = False) -> np.ndarray:
    """Resize + normalize a camera frame → (1,3,H,W) float32 for the detector."""
    img = bgr_or_rgb if rgb_input else cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img, dtype=np.float32)


def _mask_nms(instances: List[dict], iou_thr: float) -> List[dict]:
    """Mask-IoU NMS among same-class instances (keeps highest score)."""
    if len(instances) <= 1:
        return instances
    instances = sorted(instances, key=lambda d: -d["score"])
    keep: List[dict] = []
    for inst in instances:
        suppressed = False
        for k in keep:
            if k["label"] != inst["label"]:
                continue
            inter = np.logical_and(inst["mask"] > 0, k["mask"] > 0).sum()
            union = np.logical_or(inst["mask"] > 0, k["mask"] > 0).sum()
            if union > 0 and inter / union > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(inst)
    return keep


def extract_tip_crops(rgb: np.ndarray, mask: np.ndarray,
                      tip_frac: float = 0.45) -> List[np.ndarray]:
    """
    Both instrument tips along the mask major axis, each padded to a square.
    These crops carry the curved-vs-straight-jaw signal that separates
    Needle_Holder↔Artery_Forceps (and 23↔150) — same length, different tips.
    """
    h, w = rgb.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
    length = max(rw, rh)
    if length < 10:
        return []
    if rw >= rh:
        dirv = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    else:
        dirv = np.array([-np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    n = np.linalg.norm(dirv)
    if n < 1e-8:
        return []
    dirv = dirv / n
    half = length * 0.5
    ends = [np.array([cx, cy]) - dirv * half, np.array([cx, cy]) + dirv * half]
    side = max(int(length * tip_frac), 24)
    crops = []
    for pt in ends:
        x0 = int(np.clip(pt[0] - side / 2, 0, max(w - side, 0)))
        y0 = int(np.clip(pt[1] - side / 2, 0, max(h - side, 0)))
        x1, y1 = min(x0 + side, w), min(y0 + side, h)
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        crops.append(rgb[y0:y1, x0:x1])
    return crops


def instances_from_logits(logits: np.ndarray, classes: List[str],
                          frame_rgb: Optional[np.ndarray] = None,
                          mask_threshold: float = 0.5,
                          min_instance_area: int = 80,
                          nms_iou: float = 0.40,
                          conf_min_score: float = 0.35,
                          scale_to_frame: Optional[Tuple[float, float]] = None,
                          want_tip_crops: bool = True,
                          tip_frac: float = 0.45) -> List[dict]:
    """
    logits: (1+C, h, w) — ch0 fg logits, ch1.. class logits (already sigmoid-able)
    classes: detector class names, len = C (label i ↔ classes[i])
    frame_rgb: original-resolution RGB frame (for tip crops + coordinate scaling)
    scale_to_frame: (sx, sy) map model-space → frame-space (model 560, frame W,H)
    """
    C = len(classes)
    assert logits.shape[0] == C + 1, f"expected {C+1} channels, got {logits.shape[0]}"
    x = logits.astype(np.float32)
    x = x - x.max(axis=0, keepdims=True)
    probs = np.exp(x)
    probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)
    fg_prob = 1.0 - probs[0]
    cls_prob = probs[1:]

    fg = (fg_prob > mask_threshold).astype(np.uint8)
    # remove tiny specks
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_comp, labels_im = cv2.connectedComponents(fg)
    if n_comp <= 1:
        return []

    H, W = logits.shape[1], logits.shape[2]
    out: List[dict] = []
    for ci in range(1, n_comp):
        comp = (labels_im == ci).astype(np.uint8)
        area = int(comp.sum())
        if area < min_instance_area:
            continue
        # class scores = mean class prob over component
        mean_probs = cls_prob[:, comp > 0].mean(axis=1)        # (C,)
        label = int(mean_probs.argmax())
        score = float(mean_probs[label])
        fg_mean = float(fg_prob[comp > 0].mean())
        score = min(score, fg_mean) if fg_mean > 0 else score  # conservative: penalize weak fg
        if score < conf_min_score:
            continue
        mask = comp * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        (rcx, rcy), (rw, rh), ang = rect
        length = float(max(rw, rh))
        ys, xs = np.where(comp > 0)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        inst = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "label": label,
            "class_name": classes[label],
            "score": score,
            "mask": mask,
            "length_px": length,
            "angle_deg": float(ang),
        }
        if want_tip_crops and frame_rgb is not None:
            inst["tip_crops"] = extract_tip_crops(frame_rgb, mask, tip_frac)
        if scale_to_frame is not None:
            sx, sy = scale_to_frame
            x1f, y1f = x1 * sx, y1 * sy
            x2f, y2f = x2 * sx, y2 * sy
            # scale mask cheaply: scale bbox + rescale mask for length measurement
            inst["bbox_frame"] = [float(x1f), float(y1f), float(x2f), float(y2f)]
            inst["length_px_frame"] = length * max(sx, sy)
        out.append(inst)

    out = _mask_nms(out, nms_iou)
    return out


def draw_instances(frame_bgr: np.ndarray, instances: List[dict],
                   use_frame_coords: bool = True, line: int = 2) -> np.ndarray:
    """Draw YOLO-style boxes + labels onto a BGR frame copy."""
    canvas = frame_bgr.copy()
    for inst in instances:
        if use_frame_coords and "bbox_frame" in inst:
            x1, y1, x2, y2 = [int(v) for v in inst["bbox_frame"]]
        else:
            x1, y1, x2, y2 = [int(v) for v in inst["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), line)
        txt = f"{inst['class_name']} {inst['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if "length_px_frame" in inst:
            cv2.putText(canvas, f"L={inst['length_px_frame']:.0f}px", (x1 + 2, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return canvas
