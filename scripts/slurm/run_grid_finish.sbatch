#!/bin/bash
#SBATCH --job-name=csdv_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#
# Finish one unit of a grid queue on CPU: detection, the pair model's map
# (optional) and the export. Array task i reads line i + 1 of the row file.
# Submitted by `csdv grid queue` to run after every GPU worker has ended.
# This script names no allocation: pass $CSDV_SBATCH_CPU at submission.
#
# Usage:
#   sbatch $CSDV_SBATCH_CPU --array=1-N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_grid_finish.sbatch QUEUE_DIR ROWS.csv RUN_TAG \
#       [--min-years N] [--checkpoint best.pt] [--priority-intervals FILE] \
#       [--no-quicklooks] [-- detect-disturbance options...]
#
# Steps, each skipped when its output exists:
#   1. `csdv grid sync-unit`: the years that have a mosaic and a CHM. The
#      task stops here when the download or a CHM is still missing (give the
#      queue command again), when too few years are left, or when the
#      unit's priority interval lost a date.
#   2. Detection into $CSDV_RESULTS_ROOT/disturbance/<unit>/<tag>/, through
#      scripts/slurm/run_detect_disturbance.sbatch on those years.
#   3. With --checkpoint: `csdv pair predict --device cpu` into <run>/model/.
#      The mosaics are read where they are.
#   4. `csdv grid export`, which also copies the cell layers and the model
#      map that the label steps read.
#
# Resources: the request suits whole quads. A 4.8 km box needs a few minutes
# and a few GB for detection. The model map on CPU was not timed before the
# round-2 pilot: its log line gives the seconds.

set -euo pipefail

USAGE="usage: sbatch --array=1-N $0 QUEUE_DIR ROWS.csv RUN_TAG [options] [-- detect options]"
QUEUE_DIR="${1:?$USAGE}"
ROWS="${2:?$USAGE}"
RUN_TAG="${3:?$USAGE}"
shift 3

MIN_YEARS=2
CHECKPOINT=""
PRIORITY=()
EXPORT_ARGS=()
DETECT_ARGS=()
while (($#)); do
    case "$1" in
        --min-years) MIN_YEARS="${2:?}"; shift 2 ;;
        --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
        --priority-intervals) PRIORITY=(--priority-intervals "${2:?}"); shift 2 ;;
        --no-quicklooks) EXPORT_ARGS+=(--no-quicklooks); shift ;;
        --) shift; DETECT_ARGS=("$@"); break ;;
        *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
    esac
done

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

UNIT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROWS" | tr -d '\r')
: "${UNIT:?no row $SLURM_ARRAY_TASK_ID in $ROWS}"
NAME="$(basename "$QUEUE_DIR")"
RUN_DIR="$CSDV_RESULTS_ROOT/disturbance/$UNIT/$RUN_TAG"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started=$(date -Is)"
echo "unit=$UNIT run_tag=$RUN_TAG queue=$NAME checkpoint=${CHECKPOINT:-none}"

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-2048}"
export MPLBACKEND=Agg

activate_csdv

# 1. The years that are ready. Its message says what is missing.
YEARS=$(csdv grid sync-unit --name "$NAME" --unit "$UNIT" --min-years "$MIN_YEARS" \
    ${PRIORITY[@]+"${PRIORITY[@]}"})
echo "years: $YEARS"

# 2. Detection. The job script stages the CHMs to node scratch and copies
# the run back only when it succeeded.
if [[ -f "$RUN_DIR/summary.json" ]]; then
    echo "detection: run exists"
else
    if [[ -e "$RUN_DIR" ]]; then
        # A folder without summary.json is what a job killed while copying
        # leaves behind.
        mv "$RUN_DIR" "$RUN_DIR.incomplete.${SLURM_JOB_ID:-$$}"
    fi
    STARTED=$SECONDS
    bash "$CSDV_REPO/scripts/slurm/run_detect_disturbance.sbatch" "$UNIT" \
        --output-dir "$RUN_DIR" --years "$YEARS" ${DETECT_ARGS[@]+"${DETECT_ARGS[@]}"}
    echo "detection: $((SECONDS - STARTED)) s"
fi

# 3. The pair model's map, on CPU.
if [[ -n "$CHECKPOINT" ]]; then
    if [[ -f "$RUN_DIR/model/prob_$UNIT.tif" ]]; then
        echo "model map: exists"
    elif [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: no checkpoint at $CHECKPOINT" >&2
        exit 1
    else
        THREADS="${SLURM_CPUS_PER_TASK:-8}"
        TMP="$RUN_DIR/model.tmp.${SLURM_JOB_ID:-$$}"
        STARTED=$SECONDS
        OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" GDAL_CACHEMAX=512 \
            csdv pair predict --checkpoint "$CHECKPOINT" --unit "$UNIT" \
            --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" \
            --data-root "$CSDV_DATA_ROOT" --out-dir "$TMP" --device cpu --workers 2
        rm -rf "$RUN_DIR/model"
        mv "$TMP" "$RUN_DIR/model"
        echo "model map: $((SECONDS - STARTED)) s on $THREADS threads"
    fi
fi

# 4. Export.
csdv grid export --unit "$UNIT" --run-tag "$RUN_TAG" ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}

echo "done $(date -Is)"
