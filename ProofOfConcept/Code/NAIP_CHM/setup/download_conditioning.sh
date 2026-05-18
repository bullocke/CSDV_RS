#!/usr/bin/env bash
# Download the 5 static conditioning rasters required by NAIP-CHM inference.
#
# Stores them in $NAIPCHM_COND_DIR (default: $NAIPCHM_REPO_DIR/data/conditioning_data).
# Skips files that already exist. Total size is a few GB.

set -euo pipefail

REPO_DIR="${NAIPCHM_REPO_DIR:-$HOME/code/naip-chm}"
COND_DIR="${NAIPCHM_COND_DIR:-$REPO_DIR/data/conditioning_data}"
ENV_NAME="${NAIPCHM_ENV_NAME:-naipchm}"

echo "[cond] repo  : $REPO_DIR"
echo "[cond] target: $COND_DIR"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "[cond] ERROR: NAIPCHM_REPO_DIR not found. Run clone_and_install.sh first." >&2
  exit 1
fi

mkdir -p "$COND_DIR"

# If target differs from repo default, point repo's expected dir at it via symlink.
DEFAULT_DIR="$REPO_DIR/data/conditioning_data"
if [[ "$COND_DIR" != "$DEFAULT_DIR" ]]; then
  mkdir -p "$(dirname "$DEFAULT_DIR")"
  if [[ -L "$DEFAULT_DIR" || -e "$DEFAULT_DIR" ]]; then
    rm -rf "$DEFAULT_DIR"
  fi
  ln -s "$COND_DIR" "$DEFAULT_DIR"
fi

cd "$REPO_DIR"
echo n | micromamba run -n "$ENV_NAME" python scripts/download_conditioning_data.py

REQUIRED=(elevation.tif climate_pca.tif soil_pca.tif nlcd.tif ecoregion.tif)
for f in "${REQUIRED[@]}"; do
  if [[ ! -s "$COND_DIR/$f" ]]; then
    echo "[cond] ERROR: missing $COND_DIR/$f" >&2
    exit 1
  fi
done

echo "[cond] All 5 conditioning rasters present in $COND_DIR"
echo "[cond] Export to your shell or SLURM script:"
echo "       export NAIPCHM_COND_DIR=$COND_DIR"
