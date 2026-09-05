#!/usr/bin/env bash
# ============================================================
#  setup_env_wsl.sh — training environment for AMD GPUs (ROCm via WSL2)
#
#  Run INSIDE WSL (Ubuntu 22.04/24.04), in the repo root:
#      bash setup_env_wsl.sh
#
#  Prereqs (one-time, outside/inside WSL as noted):
#    1. WSL2 with Ubuntu:      wsl --install -d Ubuntu-24.04   (from Windows)
#    2. AMD driver on WINDOWS (the host GPU driver is shared by WSL)
#    3. ROCm in WSL:           see https://rocm.docs.amd.com/projects/install-on-wsl/
#                               (amdgpu-install --usecase=wsl)
#    4. uv:                    curl -LsSf https://astral.sh/uv/install.sh | sh
#                               (uv will fetch Python 3.12 itself, no apt needed)
# ============================================================
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "[1/3] Creating .venv-rocm (Python 3.12 via uv) ..."
if [ ! -d .venv-rocm ]; then
    uv venv .venv-rocm --python 3.12
fi
source .venv-rocm/bin/activate

echo "[2/3] Installing pinned packages (ROCm torch) ..."
# --index-strategy unsafe-best-match: some pinned versions (e.g. tqdm) only
# exist on PyPI, not on the ROCm wheel index. uv defaults to "first index
# only" per package; this flag lets it fall through to PyPI for those.
# Safe here since every package below is version-pinned exactly.
#
# UV_HTTP_TIMEOUT: ROCm torch/sympy wheels are large; the default 30s
# timeout can trip on slower connections. Bump it and retry once on failure.
export UV_HTTP_TIMEOUT=300
uv pip install --index-strategy unsafe-best-match -r requirements-rocm.txt \
    || uv pip install --index-strategy unsafe-best-match -r requirements-rocm.txt

echo "[3/3] Verifying GPU ..."
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda-API available (ROCm):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(8, 8, device="cuda")
    print("matmul ok:", float((x @ x).sum()))
else:
    print("!! GPU not visible — check: 1) AMD driver on Windows host 2) rocm in WSL")
EOF

cat <<'EOF'

Done. To use:
  source .venv-rocm/bin/activate
  python train_all.py --data_dir dataset --num_workers 2 --export

(notes: WSL is Linux — num_workers 2 is fine here, unlike Windows)
EOF