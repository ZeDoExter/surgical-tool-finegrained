#!/usr/bin/env bash
# ============================================================
#  setup_env_wsl_offline.sh — install the training env from LOCAL wheels
#  (no internet needed — built for the 2.4 Mbps machine)
#
#  Run INSIDE WSL (Ubuntu), in the repo root, with the wheels folder
#  next to the repo (this machine: Year4/wheels/linux):
#      bash setup_env_wsl_offline.sh
#
#  One command does everything: create venv -> install all 52 wheels ->
#  verify GPU -> print the train command.
# ============================================================
set -e
cd "$(dirname "$0")"

WHEELS="${1:-../wheels/linux}"
if [ ! -d "$WHEELS" ]; then
    echo "wheels folder not found at: $WHEELS"
    echo "usage: bash setup_env_wsl_offline.sh [path-to-wheels-linux]"
    exit 1
fi
N_WHEELS=$(ls "$WHEELS"/*.whl 2>/dev/null | wc -l)
echo "[wheels] $N_WHEELS files in $WHEELS"
if [ "$N_WHEELS" -lt 50 ]; then
    echo "!! expected ~52 wheels — copy the full folder from the flash drive"
    exit 1
fi

PY=python3.12
if ! $PY --version >/dev/null 2>&1; then
    # fall back to any python3 the WSL distro ships (wheels are cp312)
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
        echo "[python] using $PY ($($PY --version))"
    else
        echo "python3 not found — Ubuntu 22.04: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
else
    echo "[python] using $PY ($($PY --version 2>&1))"
fi
case "$($PY --version 2>&1)" in
    "Python 3.12"*) : ;;  # exact match for our wheels
    *) echo "!! wheels are cp312 — need Python 3.12 (got $($PY --version 2>&1))"; exit 1 ;;
esac

echo "[1/3] Creating .venv-rocm ..."
if [ ! -d .venv-rocm ]; then
    $PY -m venv .venv-rocm
fi
source .venv-rocm/bin/activate

echo "[2/3] Installing ALL wheels offline (~52 pkgs, torch is 4.5GB — takes a few minutes from flash drive) ..."
pip install --upgrade pip -q --no-index || true
pip install --no-index --find-links "$WHEELS" "$WHEELS"/torch-*.whl \
    "$WHEELS"/torchvision-*.whl "$WHEELS"/numpy-*.whl "$WHEELS"/onnxruntime-*.whl
pip install --no-index --find-links "$WHEELS" "$WHEELS"/*.whl

echo "[3/3] Verifying ..."
python - <<'EOF'
import torch, transformers, peft, albumentations, cv2, numpy, sklearn, onnxruntime
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("numpy       :", numpy.__version__)
print("GPU visible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda")
    print("matmul ok   :", bool(torch.isfinite(x @ x).all()))
else:
    print("(training would run on CPU — check AMD driver on Windows + rocm in WSL)")
EOF

cat <<'EOF'

Done. To train:
  source .venv-rocm/bin/activate
  python train_all.py --data_dir dataset --num_workers 2 --export

(notes: WSL = Linux, so num_workers 2 is fine; if you later need anything
extra from the internet: pip install <pkg> works normally alongside)
EOF
