#!/usr/bin/env bash
# Clone smorf-ntsg/naip-chm and install dependencies in a dedicated micromamba env.
#
# Idempotent: re-running updates the repo and reinstalls requirements.
#
# Environment variables:
#   NAIPCHM_REPO_DIR  Destination for the clone (default: $HOME/code/naip-chm)
#   NAIPCHM_ENV_NAME  Micromamba env name (default: naipchm)

set -euo pipefail

REPO_DIR="${NAIPCHM_REPO_DIR:-$HOME/code/naip-chm}"
ENV_NAME="${NAIPCHM_ENV_NAME:-naipchm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup] repo dir : $REPO_DIR"
echo "[setup] env name : $ENV_NAME"

mkdir -p "$(dirname "$REPO_DIR")"
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[setup] updating existing clone"
  git -C "$REPO_DIR" fetch --depth 1 origin main
  git -C "$REPO_DIR" reset --hard origin/main
else
  echo "[setup] cloning upstream repo"
  git clone --depth 1 https://github.com/smorf-ntsg/naip-chm.git "$REPO_DIR"
fi

# Create env if missing.
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] creating micromamba env from environment_chpc.yml"
  micromamba create -y -n "$ENV_NAME" -f "$HERE/environment_chpc.yml"
fi

echo "[setup] installing pip requirements from upstream repo"
micromamba run -n "$ENV_NAME" pip install -r "$REPO_DIR/requirements.txt"
micromamba run -n "$ENV_NAME" pip install rio-cogeo geemap earthengine-api click

cat <<EOF

[setup] Done.
  Add to ~/.bashrc or your job submission script:
    export NAIPCHM_REPO_DIR=$REPO_DIR
    export NAIPCHM_ENV_NAME=$ENV_NAME

  Next: bash $HERE/download_conditioning.sh
EOF
