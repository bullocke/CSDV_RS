#!/bin/bash
#SBATCH --job-name=csdv_label_sheets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Draw the review sheets of a label run on CHPC, where the window stacks
# are (`csdv label sheets`). CPU only. This script names no allocation:
# pass $CSDV_SBATCH_CPU at submission. Further arguments go to
# `csdv label sheets` (--stage, --windows, --max-pairs).
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_sheets.sbatch LABEL_DIR --stage 1 --max-pairs 1
#
# The sheets go to LABEL_DIR/sheets/ (about 1 MB each) and are pulled with
# scripts/chpc/round2_pull.sh. Only windows that were rendered can be drawn,
# so select a stage that has been.

set -euo pipefail

LABEL_DIR="${1:?usage: sbatch $0 LABEL_DIR [csdv label sheets options...]}"
shift

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "label_dir=$LABEL_DIR args: $*"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg
activate_csdv

CONFIG=()
[[ -f "$LABEL_DIR/config.yaml" ]] && CONFIG=(--config "$LABEL_DIR/config.yaml")
csdv label sheets --label-dir "$LABEL_DIR" \
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG" \
    ${CONFIG[@]+"${CONFIG[@]}"} "$@"
du -sh "$LABEL_DIR/sheets"
echo "done $(date -Is)"
