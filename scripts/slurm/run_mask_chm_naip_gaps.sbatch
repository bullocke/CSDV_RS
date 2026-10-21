#!/bin/bash
#SBATCH --job-name=csdv_mask_chm
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# Set NAIP gaps in an existing NAIP-CHM to nodata, in place, without
# re-downloading NAIP or re-running inference
# (scripts/mask_chm_naip_gaps.py). For CHMs predicted before
# `csdv chm-inference` masked gaps, e.g. NorthMaine 2023. The first two
# arguments are the site and year. Every further argument is passed to the
# script unchanged (--edge-m, --no-keep-original, --dry-run).
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_mask_chm_naip_gaps.sbatch NorthMaine 2023
#
# Steps: read naip/SITE/YEAR/*.tif and naip_chm/SITE/YEAR/*.tif straight from
# $CSDV_DATA_ROOT (one pass over each, so nothing is staged to scratch), write
# the masked CHM beside the original as <name>.tif.partial, keep the original
# as <name>.tif.unmasked and move the masked CHM into place. The job refuses
# to run when <name>.tif.unmasked already exists (CHM masked before).
#
# Resources: on WSL a synthetic CHM at the NorthMaine width (63,540 columns at
# 0.3 m, 9% of the 2.12 Gpx grid) took 10 s with a peak resident memory of
# 1.6 GB. Memory is set by the 1024-row strips, not the grid area, so the
# whole site needs about 2 GB and a few minutes, depending on storage speed.

set -euo pipefail

SITE="${1:?usage: sbatch $0 SITE YEAR [mask_chm_naip_gaps.py options...]}"
YEAR="${2:?usage: sbatch $0 SITE YEAR [mask_chm_naip_gaps.py options...]}"
shift 2

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "site=$SITE year=$YEAR args: $*"
echo "CSDV_DATA_ROOT=$CSDV_DATA_ROOT"

# Keep GDAL's block cache small: every block is read once.
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-256}"

activate_csdv
python "$CSDV_REPO/scripts/mask_chm_naip_gaps.py" \
    --site "$SITE" --year "$YEAR" "$@"

ls -l "$CSDV_DATA_ROOT/naip_chm/$SITE/$YEAR"
echo "done $(date -Is)"
