# -*- coding: utf-8 -*-
"""
tools/evaluate_ablation.py — วัดและเปรียบเทียบผล 6 configs สำหรับ Phase 3

ใช้บน Colab เป็นเซลล์เดียว:
    from tools.evaluate_ablation import compare_checkpoints
    table = compare_checkpoints({
        "baseline": "outputs/best_model.pt",
        "CAHM":     "outputs/best_model_cahm.pt",
        "LGMS":     "outputs/best_model_lgms.pt",
        "SEF":      "outputs/best_model_sef.pt",
        "CAHM+LGMS":"outputs/best_model_cahm_lgms.pt",
        "ALL":      "outputs/best_model_all.pt",
    })

CLI:
    python tools/evaluate_ablation.py --data_dir dataset --ckpts baseline:outputs/best_model.pt,CAHM:outputs/best_model_cahm.pt

คืนตาราง markdown พร้อม Val Acc / Balanced Acc / Needle_Holder↔Artery_Forceps error
"""
import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np


def _find_pair_indices(class_names: List[str]):
    """หา index ของคู่ Needle_Holder ↔ Artery_Forceps แบบ tolerant (ชื่ออาจมี prefix)"""
    def find(key: str) -> Optional[int]:
        for i, n in enumerate(class_names):
            if key.lower() in n.lower():
                return i
        return None
    # ลองหลาย key เผื่อชื่อต่างกันเล็กน้อย
    needle = find("needle_holder") or find("needle")
    artery = find("artery_forceps") or find("artery")
    return needle, artery


def evaluate_single(ckpt_path: str, data_dir: Optional[str] = None) -> dict:
    """
    ประเมิน 1 checkpoint → dict
      accuracy, balanced_acc, cm, needle_error, report
    """
    from evaluate import evaluate_checkpoint
    # evaluate_checkpoint จะสร้าง confusion matrix + report ให้เสร็จ
    # show_plot=False เพื่อไม่เปิดหน้าต่างบน Colab
    res = evaluate_checkpoint(ckpt_path, data_dir=data_dir, show_plot=False, save_dir=None)
    # res มี: accuracy, balanced_accuracy, confusion_matrix, report, class_names
    # เติม needle↔artery error
    class_names: List[str] = res["class_names"] if "class_names" in res else res.get("classes", [])
    # fallback: ดูจาก res โดยตรง
    if not class_names:
        # evaluate_checkpoint คืน class_names ในบางเวอร์ชันเป็น "class_names"
        from evaluate import load_bundle
        bundle = load_bundle(ckpt_path)
        class_names = bundle["classes"]
    cm = res["confusion_matrix"]
    needle_idx, artery_idx = _find_pair_indices(class_names)
    needle_error = None
    needle_detail = ""
    if needle_idx is not None and artery_idx is not None:
        # นับทั้งสองทิศทาง
        a2n = int(cm[artery_idx, needle_idx]) if cm.shape[0] > max(needle_idx, artery_idx) else 0
        n2a = int(cm[needle_idx, artery_idx]) if cm.shape[0] > max(needle_idx, artery_idx) else 0
        needle_error = a2n + n2a
        needle_detail = f"{class_names[needle_idx]}↔{class_names[artery_idx]}: {n2a}+{a2n}={needle_error}"
    else:
        needle_detail = "คู่ Needle↔Artery ไม่พบ (ชื่อ class ไม่ตรง pattern)"

    return {
        "ckpt": ckpt_path,
        "class_names": class_names,
        "accuracy": float(res["accuracy"]),
        "balanced_acc": float(res.get("balanced_accuracy", res.get("balanced_acc", 0.0))),
        "cm": cm,
        "needle_error": needle_error,
        "needle_detail": needle_detail,
        "report": res.get("report", {}),
        "raw": res,
    }


def compare_checkpoints(ckpt_map: Dict[str, str], data_dir: Optional[str] = None,
                        save_csv: Optional[str] = None) -> List[dict]:
    """
    เปรียบเทียบหลาย checkpoint → พิมพ์ตาราง markdown + คืน list ผล

    ckpt_map: {"baseline": "outputs/best_model.pt", ...}
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: F401 (used inside evaluate_single)
    results = []
    print("\n| Config | Val Acc | Balanced Acc | Needle↔Artery error | ckpt |")
    print("|---|---|---|---|---|")
    for name, path in ckpt_map.items():
        if not os.path.exists(path):
            print(f"| {name} | — | — | checkpoint ไม่พบ: {path} | {path} |")
            results.append({"name": name, "path": path, "found": False})
            continue
        try:
            r = evaluate_single(path, data_dir=data_dir)
            acc = r["accuracy"]
            bacc = r["balanced_acc"]
            needle = r["needle_detail"]
            print(f"| {name} | {acc:.4f} | {bacc:.4f} | {needle} | {path} |")
            results.append({"name": name, **r, "found": True})
        except Exception as e:
            print(f"| {name} | error | error | {e} | {path} |")
            results.append({"name": name, "path": path, "found": False, "error": str(e)})

    # สรุปว่า “คุ้มเก็บไว้” ตามเกณฑ์ plan Phase 3
    print("\n**เกณฑ์ตัดสิน (ตาม plan):** 1) ดีขึ้นจาก k-fold เฉลี่ย 2) ลด Needle↔Artery ได้จริง 3) ถ้าแย่ลงให้ตัดออก")
    if save_csv:
        try:
            import csv
            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["config", "val_acc", "balanced_acc", "needle_error", "ckpt"])
                for r in results:
                    if r.get("found"):
                        w.writerow([r["name"], f'{r["accuracy"]:.4f}', f'{r["balanced_acc"]:.4f}', r.get("needle_error", ""), r["ckpt"]])
                    else:
                        w.writerow([r["name"], "", "", "", r["path"]])
            print(f"[saved] {save_csv}")
        except Exception as e:
            print(f"[csv] skip: {e}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="เปรียบเทียบ ablation checkpoints (Phase 3)")
    ap.add_argument("--data_dir", default="dataset", help="โฟลเดอร์ dataset (มี train/valid/_annotations.coco.json)")
    ap.add_argument("--ckpts", required=True,
                    help='เช่น "baseline:outputs/best_model.pt,CAHM:outputs/best_model_cahm.pt,LGMS:outputs/best_model_lgms.pt"')
    ap.add_argument("--pattern", default=None, help="หรือใช้ glob pattern เช่น 'outputs/best_model*.pt' (ชื่อ config จะเป็นชื่อไฟล์)")
    ap.add_argument("--save_csv", default=None, help="บันทึกผลเป็น CSV")
    args = ap.parse_args(argv)

    ckpt_map: Dict[str, str] = {}
    if args.pattern:
        for p in glob.glob(args.pattern):
            name = os.path.splitext(os.path.basename(p))[0]
            ckpt_map[name] = p
    if args.ckpts:
        for pair in args.ckpts.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                k, v = pair.split(":", 1)
                ckpt_map[k.strip()] = v.strip()
            else:
                ckpt_map[os.path.basename(pair)] = pair

    if not ckpt_map:
        ap.error("ไม่พบ checkpoint — ใส่ --ckpts หรือ --pattern")

    compare_checkpoints(ckpt_map, data_dir=args.data_dir, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
