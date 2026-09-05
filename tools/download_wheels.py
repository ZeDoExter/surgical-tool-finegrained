"""Download ALL wheels for the WSL/AMD target: Ubuntu 22.04 + Python 3.12.

Output: Year4/wheels/linux/  (cp312 manylinux wheels, everything in
requirements-rocm.txt + all transitive deps, since we use --no-deps)

Wheel-tag reality (verified by probing):
  - torch+rocm7.0 / torchvision / numpy 2.5.2 / onnxruntime 1.29.0 /
    pillow 12.3.0 : manylinux_2_28 ONLY  (Ubuntu 22.04 glibc 2.35 is fine)
  - everything else             : manylinux_2_17 set
  - scikit-learn 1.9.0 has NO linux wheel at all -> 1.7.2 (nearest real one)
"""
import os
import subprocess
import sys

ROOT = r"C:\Users\navap\Desktop\Nav\University\Year4"
PIP = os.path.join(ROOT, "DentalInstrument", ".venv", "Scripts", "pip.exe")
DEST = os.path.join(ROOT, "wheels", "linux")

PLAT_228 = ["--platform", "manylinux_2_28_x86_64", "--python-version", "312",
            "--implementation", "cp", "--abi", "cp312", "--only-binary=:all:"]
PLAT_217 = ["--platform", "manylinux_2_17_x86_64", "--python-version", "312",
            "--implementation", "cp", "--abi", "cp312", "--only-binary=:all:"]
ROCM_IDX = ["--extra-index-url", "https://download.pytorch.org/whl/rocm7.0"]

GROUPS = [
    ("torch stack (manylinux_2_28)", PLAT_228, ROCM_IDX,
     ["torch==2.10.0+rocm7.0", "torchvision==0.25.0+rocm7.0", "numpy==2.5.2",
      "onnxruntime==1.29.0", "pillow==12.3.0"]),
    ("core ML stack", PLAT_217, [],
     ["transformers==5.16.1", "peft==0.20.0", "pytorch-metric-learning==2.9.0",
      "albumentations==2.0.8", "opencv-python-headless==5.0.0.93",
      "tqdm==4.70.0", "seaborn==0.13.2", "matplotlib==3.11.1"]),
    ("sklearn fallback (1.9.0 has no linux wheel)", PLAT_217, [],
     ["scikit-learn==1.7.2"]),
    ("transitive deps", PLAT_217, [],
     ["accelerate", "safetensors", "huggingface-hub", "psutil",
      "scipy", "contourpy", "cycler", "fonttools", "kiwisolver",
      "packaging", "pandas", "pytz", "python-dateutil", "joblib",
      "threadpoolctl", "typing-extensions", "filelock", "sympy", "networkx",
      "jinja2", "MarkupSafe", "mpmath", "regex", "requests", "urllib3",
      "idna", "charset-normalizer", "certifi", "pyyaml", "click", "fsspec",
      "setproctitle", "qudida", "scikit-image", "imageio", "lazy-loader",
      "cloudpickle", "shapely"]),
]

os.makedirs(DEST, exist_ok=True)
failed = []
for name, plat, idx, reqs in GROUPS:
    print(f">> {name} ({len(reqs)} pkgs)")
    r = subprocess.run([PIP, "download", "-q", "--no-deps", "-d", DEST,
                        *idx, *plat, *reqs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        print("   FAILED:", tail[-1] if tail else "?")
        failed.append(name)
    else:
        print("   ok")

n = len(os.listdir(DEST))
gb = sum(os.path.getsize(os.path.join(DEST, x)) for x in os.listdir(DEST)) / 1e9
print(f"\nwheels/linux: {n} files, {gb:.2f} GB")
if failed:
    print("failed groups:", failed)
    sys.exit(1)
print("ALL DONE")
