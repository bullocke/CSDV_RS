#!/bin/bash
#SBATCH --job-name=csdv_export
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Small download set of one grid run (`csdv grid export`), CPU only. Normally
# submitted by `csdv grid submit` as the last job of a unit's chain.
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_grid_export.sbatch UNIT RUN_TAG [csdv grid export options...]
#
# Example:
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_grid_export.sbatch q4407159 default --no-quicklooks
#
# Steps: write 8-bit COGs of the onset year and the highest probability, copy
# the run's vectors, tables, report, figures and chips to
# $CSDV_EXPORT_DIR/runs/RUN_TAG/UNIT/, write 3 m CHM and NAIP quicklooks of
# every year to $CSDV_EXPORT_DIR/quicklooks/UNIT/ (once per unit), and refresh
# status.csv. Inputs are read straight from the store; the outputs are small,
# so nothing is staged.
#
# Resources: the quicklooks read each 0.6 m mosaic once, averaged to 3 m, so
# memory stays well under 1 GB.

set -euo pipefail

UNIT="${1:?usage: sbatch $0 UNIT RUN_TAG [csdv grid export options...]}"
RUN_TAG="${2:?usage: sbatch $0 UNIT RUN_TAG [csdv grid export options...]}"
shift 2

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "unit=$UNIT run_tag=$RUN_TAG args: $*"

activate_csdv
csdv grid export --unit "$UNIT" --run-tag "$RUN_TAG" "$@"
csdv grid status --quiet || echo "WARNING: status.csv not refreshed" >&2

echo "done $(date -Is)"
