# -*- coding: utf-8 -*-
"""
train_all.py — one command, full pipeline (detector + classifier)

    python train_all.py --data_dir dataset --num_workers 2
    python train_all.py --data_dir dataset --data_dirs dataset_extra1 dataset_extra2
    python train_all.py --data_dir dataset --offline_aug 4   # +offline patch-paste scenes on disk

Runs sequentially on ONE GPU (both models share the GPU; running them as
separate parallel processes would fight over the same SMs and VRAM with no
speed gain). Progress bars (tqdm) are shown by each stage like YOLO.

Stages:
  0) (optional) offline patch-paste scenes per dataset folder (--offline_aug N)
  1) detector   -> outputs_detector/best_detector.pt   (YOLO-free, masks+bbox+label)
  2) classifier -> outputs/best_model.pt                (fine-grained 14 classes)
  3) optional ONNX export for Raspberry Pi (--export)
"""
import argparse
import os
import sys
import time


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class _Tee:
    """Mirror stdout+stderr to train_all.log (survives crashes; inspect after interrupt)."""
    def __init__(self, path: str, stream):
        self.path = path
        self.stream = stream
        # tqdm needs .encoding to emit the unicode bar (━); without it tqdm
        # falls back to ASCII '#'. Passthrough from the wrapped stream.
        self.encoding = getattr(stream, "encoding", None) or "utf-8"
        self.fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)

    def write(self, s):
        try:
            self.stream.write(s)
        except UnicodeEncodeError:
            # piped console with a legacy codepage (cp874) can't render '━' —
            # degrade to '?' there instead of crashing training mid-epoch
            self.stream.write(s.encode(self.encoding, "replace")
                               .decode(self.encoding, "replace"))
        try:
            self.fh.write(s)
        except Exception:
            pass
        return len(s)

    def writelines(self, lines):
        for s in lines:
            self.write(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.fh.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return False


def _maybe_offline_augment(data_dirs, num_aug: int) -> None:
    """Run augment_dataset.py per folder (idempotent: skips when .bak exists)."""
    if num_aug is None or num_aug <= 0:
        return
    print("\n" + "=" * 70, flush=True)
    print(f"STAGE 0 — OFFLINE PATCH-PASTE x{num_aug} ({len(data_dirs)} folder(s))", flush=True)
    print("=" * 70, flush=True)
    from augment_dataset import augment_dataset
    for d in data_dirs:
        bak = os.path.join(d, "train", "_annotations.coco.json.bak")
        if os.path.exists(bak):
            print(f"  [skip] {d} already augmented (.bak exists)", flush=True)
            continue
        print(f"  [augment] {d} ...", flush=True)
        augment_dataset(d, num_aug=num_aug, max_pastes=4, max_overlap=0.20, seed=42)
    print("  done.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train detector + classifier in one go")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--data_dirs", nargs="*", default=[],
                    help="extra dataset folders merged into train (+ patch pool)")
    ap.add_argument("--offline_aug", type=int, default=0,
                    help="offline patch-paste copies per train photo, per folder "
                         "(0=off; idempotent via .bak guard)")
    ap.add_argument("--epochs_det", type=int, default=60)
    ap.add_argument("--epochs_cls", type=int, default=50)
    ap.add_argument("--batch_det", type=int, default=16)
    ap.add_argument("--batch_cls", type=int, default=32)
    ap.add_argument("--img_size_det", type=int, default=448,
                    help="detector input (448 default = ~1.7x faster on Pi; "
                         "must be divisible by 14)")
    ap.add_argument("--img_size_cls", type=int, default=504,
                    help="classifier input (from kNN probe: 504 beat 518/560)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="default: 2 on Linux/WSL/Colab, 0 on Windows "
                         "(spawn-in-notebook hangs there)")
    ap.add_argument("--finetune_mode", choices=["lora", "partial", "frozen"], default="lora")
    ap.add_argument("--skip_detector", action="store_true", help="train classifier only")
    ap.add_argument("--skip_classifier", action="store_true", help="train detector only")
    ap.add_argument("--export", action="store_true", help="export ONNX after training")
    ap.add_argument("--det_tokens", action="store_true",
                    help="export detector with patch tokens + build class "
                         "prototypes (zero-cost fast path on the Pi)")
    ap.add_argument("--log_file", default="train_all.log",
                    help="mirror all console output here (inspect after interrupt/crash)")
    args = ap.parse_args()

    sys.stdout = _Tee(args.log_file, sys.stdout)
    sys.stderr = _Tee(args.log_file, sys.stderr)
    print(f"[{_ts()}] [boot] train_all starting (pid {os.getpid()}) "
          f"-> logging to {args.log_file}", flush=True)

    if args.num_workers is None:
        import platform
        args.num_workers = 0 if platform.system() == "Windows" else 2
    print(f"[{_ts()}] workers={args.num_workers} "
          f"det_batch={args.batch_det} cls_batch={args.batch_cls}", flush=True)
    t0 = time.time()
    data_dirs = [args.data_dir] + list(args.data_dirs or [])
    _maybe_offline_augment(data_dirs, args.offline_aug)

    det_ckpt = None
    if not args.skip_detector:
        print("\n" + "=" * 70, flush=True)
        print(f"[{_ts()}] STAGE 1/2 — DETECTOR (DINOv2 + seg head, no YOLO, {args.img_size_det}px)", flush=True)
        print("=" * 70, flush=True)
        from config import DetectorConfig
        from train_detector import run_training as run_detector
        dcfg = DetectorConfig(
            data_dir=args.data_dir, extra_data_dirs=args.data_dirs or None,
            img_size=args.img_size_det,
            batch_size=args.batch_det, epochs=args.epochs_det,
            finetune_mode=args.finetune_mode, num_workers=args.num_workers,
        )
        det_ckpt = run_detector(dcfg)

    cls_ckpt = None
    if not args.skip_classifier:
        print("\n" + "=" * 70, flush=True)
        print(f"[{_ts()}] STAGE 2/2 — CLASSIFIER (DINOv2 + length fusion + ArcFace + tip-zoom, {args.img_size_cls}px)", flush=True)
        print("=" * 70, flush=True)
        from config import TrainConfig
        from train import run_training
        cfg = TrainConfig(
            data_dir=args.data_dir, extra_data_dirs=args.data_dirs or None,
            img_size=args.img_size_cls,
            batch_size=args.batch_cls, epochs=args.epochs_cls,
            finetune_mode=args.finetune_mode, num_workers=args.num_workers,
            use_cahm=True, patch_paste_prob=0.4, tip_zoom_prob=0.35,
        )
        cls_ckpt = run_training(cfg)

    if args.export:
        print("\n" + "=" * 70, flush=True)
        print(f"[{_ts()}] EXPORT — ONNX (for Raspberry Pi)", flush=True)
        print("=" * 70, flush=True)
        import export_detector_onnx as _ed
        import export_to_onnx as _ec
        if det_ckpt:
            _ed.export_all(det_ckpt, "pi_final_v3/onnx_export",
                           with_tokens=args.det_tokens)
            if args.det_tokens:
                import build_prototypes as _bp
                _bp.main(["--ckpt", det_ckpt, "--data_dir", args.data_dir,
                          "--out_dir", "pi_final_v3/onnx_export"])
        if cls_ckpt:
            _ec.export_all(cls_ckpt, "pi_final_v3/onnx_export")

    print("\n" + "=" * 70, flush=True)
    print(f"[{_ts()}] ALL DONE in {_fmt(time.time() - t0)} (full log: {args.log_file})", flush=True)
    if det_ckpt:
        print(f"  detector   : {det_ckpt}", flush=True)
    if cls_ckpt:
        print(f"  classifier : {cls_ckpt}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
