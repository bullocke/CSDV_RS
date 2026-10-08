#!/bin/bash
#SBATCH --job-name=csdv_chm
#SBATCH --partition=notchpeak-gpu
#SBATCH --qos=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#
# NAIP-CHM inference, one array task per NAIP GeoTIFF, on node-local scratch.
# Normally submitted by scripts/slurm/submit_chm_crowns.sh, which builds the
# manifest and sets --array and --output.
#
# Manual use (after `source scripts/chpc/env.sh`):
#   sbatch --array=1-N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_chm_inference.sbatch MANIFEST.csv
#
# Manifest columns (header on line 1): site,year,naip_path,crowns_name
#
# Freecycle fallback when notchpeak-gpu is busy (preemptable, so requeue):
#   --partition=granite-gpu --qos=granite-gpu-freecycle --account=wangj --requeue
# Do not use granite-gpu-guest: its Blackwell cards need a newer CUDA build.
#
# Per task: skip if the CHM already exists, stage the NAIP tif and the
# conditioning rasters to /scratch/local, run inference, copy the CHM to
# $CSDV_DATA_ROOT/naip_chm/<site>/<year>/.

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=1-N $0 MANIFEST.csv}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

# Line 1 is the header, so task i reads line i+1.
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=',' read -r SITE YEAR NAIP_PATH _CROWNS_NAME <<<"$ROW"
: "${NAIP_PATH:?empty manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST}"
STEM="$(basename "$NAIP_PATH" .tif)"
CHM_DEST="$CSDV_DATA_ROOT/naip_chm/$SITE/$YEAR"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started=$(date -Is)"
echo "site=$SITE year=$YEAR naip=$NAIP_PATH"

if compgen -G "$CHM_DEST/${STEM}*chm*.tif" >/dev/null; then
    echo "CHM already in $CHM_DEST, skipping."
    exit 0
fi

activate_csdv
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

scratch_init
stage_in "$NAIP_PATH" "$NODE_SCRATCH/naip"
stage_in "$CSDV_CONDITIONING_DIR" "$NODE_SCRATCH/conditioning"
mkdir -p "$NODE_SCRATCH/chm"
register_stage_out "$NODE_SCRATCH/chm" "$CHM_DEST"

csdv chm-inference \
    --naip-quad "$NODE_SCRATCH/naip/$(basename "$NAIP_PATH")" \
    --output-dir "$NODE_SCRATCH/chm" \
    --conditioning-dir "$NODE_SCRATCH/conditioning"

ls -l "$NODE_SCRATCH/chm"
