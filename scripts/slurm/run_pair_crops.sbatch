#!/bin/bash
#SBATCH --job-name=csdv_pair_crops
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# 0.6 m NAIP crops of chosen boxes for close-up figures (`csdv pair crops`),
# CPU only. The first argument names a folder under $CSDV_EXPORT_DIR/pair_crops/
# that holds boxes.csv, copied there from the laptop. The crops are written
# beside it, as <unit>/<box>_<year>.tif.
#
# This script names no allocation. Pass yours at submission, from the option
# string in your scripts/chpc/env.sh (CSDV_SBATCH_CPU).
#
# Usage:
#   # Laptop: send the boxes
#   rsync -av results/vlm_labeling/2_scale_test/fire_check/boxes.csv \
#       "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/pair_crops/fire_check/"
#   # CHPC (after `source scripts/chpc/env.sh`)
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_pair_crops.sbatch fire_check
#   # Laptop, afterwards
#   scripts/chpc/pull_exports.sh pair_crops/fire_check
#
# Resources: a box of 1 km and three dates reads in a few seconds and is
# about 8 MB. 100 boxes take a few minutes.

set -euo pipefail

NAME="${1:?usage: sbatch $0 NAME (a folder under \$CSDV_EXPORT_DIR/pair_crops)}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

OUT="$CSDV_EXPORT_DIR/pair_crops/$NAME"
if [[ ! -f "$OUT/boxes.csv" ]]; then
    echo "ERROR: no $OUT/boxes.csv. Copy it from the laptop first." >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT started=$(date -Is)"
export OMP_NUM_THREADS=1 GDAL_CACHEMAX="${GDAL_CACHEMAX:-256}"

activate_csdv
csdv pair crops --boxes "$OUT/boxes.csv" --out-dir "$OUT" --data-root "$CSDV_DATA_ROOT"
du -sh "$OUT"
echo "finished=$(date -Is)"
