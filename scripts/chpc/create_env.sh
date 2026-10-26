#!/usr/bin/env bash
# Build the CSDV environment with CUDA PyTorch on a CHPC login node.
#
# Usage:
#   scripts/chpc/create_env.sh [ENV_NAME]     # default: csdv-gpu
#
# PyTorch comes from the official cu126 pip wheels, which bundle the CUDA
# runtime, so no `module load cuda` is needed. cu126 covers every GPU in the
# general and freecycle partitions (Maxwell through Hopper: titanx, 1080ti,
# 2080ti, 3090, a100, l40s, h100). It does not cover Blackwell cards
# (rtxpr6000bl), so avoid granite-gpu-guest. Versions match the naip-chm pins.
#
# Crown segmentation comes from naipcrowns, a sibling clone of
# https://github.com/bullocke/naip-crowns (default: ../naip-crowns next to this
# repository, or set NAIPCROWNS_DIR). Both packages are installed with
# --no-deps --no-build-isolation, so pip never replaces conda's GDAL or PROJ.
#
# If the package cache fills your home quota, first run:
#   export CONDA_PKGS_DIRS=/scratch/general/vast/$USER/pkgs

set -euo pipefail

ENV_NAME="${1:-csdv-gpu}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
NAIPCROWNS_DIR="${NAIPCROWNS_DIR:-$REPO/../naip-crowns}"

if [[ ! -f "$NAIPCROWNS_DIR/pyproject.toml" ]]; then
    echo "ERROR: naipcrowns not found at $NAIPCROWNS_DIR. Clone it first:" \
        "git clone https://github.com/bullocke/naip-crowns.git $NAIPCROWNS_DIR" >&2
    exit 1
fi

if ! command -v mamba >/dev/null 2>&1; then
    echo "ERROR: mamba is not on PATH." >&2
    exit 1
fi
eval "$(mamba shell hook --shell bash)"

if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: env '$ENV_NAME' exists. Remove it first:" \
        "mamba env remove -n $ENV_NAME" >&2
    exit 1
fi

echo "== conda packages from environment.yml"
mamba env create -y -n "$ENV_NAME" -f "$REPO/environment.yml"

set +u
mamba activate "$ENV_NAME"
set -u

echo "== PyTorch (CUDA 12.6 wheels)"
python -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url "$TORCH_INDEX"

echo "== naipcrowns and csdv_core (editable, no dependency resolution)"
python -m pip install --no-deps --no-build-isolation -e "$NAIPCROWNS_DIR"
python -m pip install --no-deps --no-build-isolation -e "$REPO"

echo "== check"
python - <<'PY'
import torch, torchvision

print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
print(f"built for CUDA {torch.version.cuda}")
assert torch.version.cuda is not None, "CPU-only torch was installed"
PY
csdv --version
naipcrowns --version
echo "Done. GPU availability can only be checked on a GPU node (see docs/workflow_chpc.md)."
