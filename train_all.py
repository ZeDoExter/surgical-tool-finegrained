# -*- coding: utf-8 -*-
"""
train_all.py — one command, full pipeline (detector + classifier)

    python train_all.py --data_dir dataset --num_workers 2

Runs sequentially on ONE GPU (both models share the GPU; running them as
separate parallel processes would fight over the same SMs and VRAM with no
speed gain). Progress bars (tqdm) are shown by each stage like YOLO.

Stages:
  1) detector   -> outputs_detector/best_detector.pt   (YOLO-free, masks+bbox+label)
  2) classifier -> outputs/best_model.pt                (fine-grained 14 classes)
  3) optional ONNX export for Raspberry Pi (--export)
"""
import argparse
import time


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train detector + classifier in one go")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs_det", type=int, default=60)
    ap.add_argument("--epochs_cls", type=int, default=50)
    ap.add_argument("--batch_det", type=int, default=16)
    ap.add_argument("--batch_cls", type=int, default=32)
    ap.add_argument("--img_size_det", type=int, default=448,
                    help="detector input (448 default = ~1.7x faster on Pi; "
                         "must be divisible by 14)")
    ap.add_argument("--img_size_cls", type=int, default=560,
                    help="classifier input (keep 560 — tip detail matters)")
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
    args = ap.parse_args()

    if args.num_workers is None:
        import platform
        args.num_workers = 0 if platform.system() == "Windows" else 2
    t0 = time.time()

    det_ckpt = None
    if not args.skip_detector:
        print("\n" + "=" * 70)
        print(f"STAGE 1/2 — DETECTOR (DINOv2 + seg head, no YOLO, {args.img_size_det}px)")
        print("=" * 70)
        from config import DetectorConfig
        from train_detector import run_training as run_detector
        dcfg = DetectorConfig(
            data_dir=args.data_dir, img_size=args.img_size_det,
            batch_size=args.batch_det, epochs=args.epochs_det,
            finetune_mode=args.finetune_mode, num_workers=args.num_workers,
        )
        det_ckpt = run_detector(dcfg)

    cls_ckpt = None
    if not args.skip_classifier:
        print("\n" + "=" * 70)
        print(f"STAGE 2/2 — CLASSIFIER (DINOv2 + length fusion + ArcFace + tip-zoom, {args.img_size_cls}px)")
        print("=" * 70)
        from config import TrainConfig
        from train import run_training
        cfg = TrainConfig(
            data_dir=args.data_dir, img_size=args.img_size_cls,
            batch_size=args.batch_cls, epochs=args.epochs_cls,
            finetune_mode=args.finetune_mode, num_workers=args.num_workers,
            use_cahm=True, patch_paste_prob=0.4, tip_zoom_prob=0.35,
        )
        cls_ckpt = run_training(cfg)

    if args.export:
        print("\n" + "=" * 70)
        print("EXPORT — ONNX (for Raspberry Pi)")
        print("=" * 70)
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

    print("\n" + "=" * 70)
    print(f"ALL DONE in {_fmt(time.time() - t0)}")
    if det_ckpt:
        print(f"  detector   : {det_ckpt}")
    if cls_ckpt:
        print(f"  classifier : {cls_ckpt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
