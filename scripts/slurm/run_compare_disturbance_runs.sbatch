#!/bin/bash
#SBATCH --job-name=csdv_compare
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
# Compare detect-disturbance runs of one site (area, overlap, event size) and
# draw chips with every variant's polygons of every interval over the same NAIP
# images
# (scripts/compare_disturbance_runs.py). NAIP windows are read from the site's
# mosaics under $CSDV_DATA_ROOT: `--naip-source local` is passed first, so a
# later --naip-source on the command line wins. Every argument is passed to the
# script unchanged. The first --run is the baseline (at most five runs).
#
# Usage (after `source scripts/chpc/env.sh`):
#   D=$CSDV_RESULTS_ROOT/disturbance
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_compare_disturbance_runs.sbatch \
#       --run D6=$D/LincolnNH --run D6s=$D/attempt_2/LincolnNH_D6s \
#       --run p09=$D/attempt_2/LincolnNH_p09 --run sig10=$D/attempt_2/LincolnNH_sig10 \
#       --run mmu1=$D/attempt_2/LincolnNH_mmu1 \
#       --out $D/attempt_2/comparison/LincolnNH --point S06=-71.52193,44.18478
#
# Writes into --out (created if missing; files from an earlier run are
# replaced): summary and overlap CSVs, agreement.tif, locations.gpkg,
# chips/*.png and report.md.
#
# Resources: on WSL the five NorthMaine runs (595 places) took 65 s and 1.05 GB
# for the statistics. Chips from local mosaics take about 1 s each (24 by
# default for a site like NorthMaine).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 --run LABEL=RUN_DIR [--run ...] --out DIR [options...]" >&2
    exit 2
fi

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "args: $*"
echo "CSDV_DATA_ROOT=$CSDV_DATA_ROOT"

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-256}"
export MPLBACKEND=Agg

activate_csdv
python "$CSDV_REPO/scripts/compare_disturbance_runs.py" --naip-source local "$@"

echo "done $(date -Is)"
