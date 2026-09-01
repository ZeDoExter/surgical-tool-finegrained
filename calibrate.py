# -*- coding: utf-8 -*-
"""
calibrate.py — cm-per-pixel calibration for the fixed camera rig

One-time setup: photograph a ruler (or any object of known length, e.g. a
15.5 cm Root Elevator) on the green cloth at the SAME height/distance the
instruments are placed. Click the two ends of the reference object in the
window (or pass --auto with a mask of the object) → cm/pixel ratio.

The ratio converts detector mask-length (px) → real cm, which:
  - separates Root_Elevators (15.5cm) vs Root_Tip_Elevator_Straight (14.5cm)
  - feeds the classifier's length prior with TRUE physical lengths
(Needle_Holder↔Artery_Forceps / 23↔150 share the same length — length can
NEVER separate those; tip shape must — see tip crops.)

Usage:
    python calibrate.py --image ruler.jpg --known_cm 15.5          # click 2 points
    python calibrate.py --image ruler.jpg --known_cm 15.5 --auto   # object mask

Writes calibration_ratio into:
    onnx_export/detector_meta.json  (detector)
    onnx_export/classifier_meta.json (classifier, if present)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def interactive_two_clicks(image_path: str) -> float:
    """Open a window, click both ends of the reference object → pixel distance."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {image_path}")
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.line(img, pts[-2], pts[-1], (0, 0, 255), 2)
            cv2.imshow("calibrate", img)

    cv2.imshow("calibrate", img)
    cv2.setMouseCallback("calibrate", on_mouse)
    print("Click BOTH ENDS of the reference object, then press any key.")
    while len(pts) < 2 and cv2.waitKey(50) < 0:
        pass
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    if len(pts) < 2:
        raise RuntimeError("need exactly 2 clicks")
    (x1, y1), (x2, y2) = pts[:2]
    return float(np.hypot(x2 - x1, y2 - y1))


def auto_from_mask(image_path: str) -> float:
    """Largest bright/foreground contour long side (minAreaRect) in px."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # metal instrument on green cloth → non-green + bright
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    non_green = cv2.inRange(hsv, (0, 0, 60), (179, 70, 255))
    non_green = cv2.morphologyEx(non_green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(non_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("no foreground object found (try manual clicks)")
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), a = cv2.minAreaRect(c)
    return float(max(w, h))


def update_meta(meta_path: str, ratio: float) -> bool:
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["calibration_ratio"] = ratio
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[ok] calibration_ratio={ratio:.6f} cm/px → {meta_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="photo of the reference object on the rig")
    ap.add_argument("--known_cm", type=float, required=True)
    ap.add_argument("--auto", action="store_true", help="auto-measure via mask (else click 2 ends)")
    ap.add_argument("--detector_meta", default="pi_final_v3/onnx_export/detector_meta.json")
    ap.add_argument("--classifier_meta", default="pi_final_v3/onnx_export/classifier_meta.json")
    args = ap.parse_args()

    px = auto_from_mask(args.image) if args.auto else interactive_two_clicks(args.image)
    ratio = args.known_cm / px
    print(f"reference length: {px:.1f}px = {args.known_cm}cm → {ratio:.6f} cm/px")

    ok = update_meta(args.detector_meta, ratio)
    ok2 = update_meta(args.classifier_meta, ratio)
    if not ok and not ok2:
        print("no meta files found — pass --detector_meta/--classifier_meta paths")


if __name__ == "__main__":
    main()
