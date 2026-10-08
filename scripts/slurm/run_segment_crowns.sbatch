#!/bin/bash
#SBATCH --job-name=csdv_crowns
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#
# Crown segmentation (watershed), one array task per NAIP GeoTIFF, on
# node-local scratch. Reads the same manifest as run_chm_inference.sbatch and
# is normally submitted by scripts/slurm/submit_chm_crowns.sh with
# --dependency=aftercorr, so task i starts when CHM task i succeeds.
#
# Manual use (after `source scripts/chpc/env.sh`):
#   sbatch --array=1-N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_segment_crowns.sbatch MANIFEST.csv
#
# Output: $CSDV_RESULTS_ROOT/crowns/<site>/<year>/<crowns_name> plus its
# .params.json sidecar. crowns_name is crowns.gpkg when the site-year has one
# NAIP image (what compute-metrics expects), else <naip stem>_crowns.gpkg.

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=1-N $0 MANIFEST.csv}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=',' read -r SITE YEAR NAIP_PATH CROWNS_NAME <<<"$ROW"
: "${CROWNS_NAME:?incomplete manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST}"
STEM="$(basename "$NAIP_PATH" .tif)"
CHM_DIR="$CSDV_DATA_ROOT/naip_chm/$SITE/$YEAR"
CROWNS_DEST="$CSDV_RESULTS_ROOT/crowns/$SITE/$YEAR"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started=$(date -Is)"
echo "site=$SITE year=$YEAR naip=$NAIP_PATH -> $CROWNS_NAME"

if [[ -e "$CROWNS_DEST/$CROWNS_NAME" ]]; then
    echo "Crowns already in $CROWNS_DEST, skipping."
    exit 0
fi

CHM_PATH=$(compgen -G "$CHM_DIR/${STEM}*chm*.tif" | head -n 1 || true)
if [[ -z "$CHM_PATH" ]]; then
    echo "ERROR: no CHM for $STEM in $CHM_DIR" >&2
    exit 1
fi

activate_csdv
scratch_init
stage_in "$CHM_PATH" "$NODE_SCRATCH/chm"
mkdir -p "$NODE_SCRATCH/crowns"
register_stage_out "$NODE_SCRATCH/crowns" "$CROWNS_DEST"

csdv segment-crowns \
    --chm "$NODE_SCRATCH/chm/$(basename "$CHM_PATH")" \
    --out-crowns "$NODE_SCRATCH/crowns/$CROWNS_NAME"

ls -l "$NODE_SCRATCH/crowns"
