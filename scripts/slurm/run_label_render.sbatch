#!/bin/bash
#SBATCH --job-name=csdv_label_render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%A_%a.out
#
# Render the windows of a label run from the NAIP mosaics (`csdv label
# render`), CPU only. The first argument is the label run folder, which holds
# windows.csv, units.csv and candidates/ as pushed from the laptop by
# scripts/chpc/push_label_run.sh. Every further argument is passed to
# `csdv label render` unchanged (for example --kind polygon).
#
# As an array, task i renders the windows of the unit on line i + 1 of
# units.csv. Without --array the job renders every window of the run, which
# suits a small run such as the 40 example windows.
#
# This script names no allocation. Pass yours at submission, from the option
# string in your scripts/chpc/env.sh (CSDV_SBATCH_CPU).
#
# Usage (after `source scripts/chpc/env.sh`):
#   LABEL="$CSDV_EXPORT_DIR/label/scale_test"
#   N=$(($(wc -l < "$LABEL/units.csv") - 1))
#   # First one unit, and read its log before the rest:
#   sbatch $CSDV_SBATCH_CPU --array=1-1 --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_label_render.sbatch "$LABEL"
#   sbatch $CSDV_SBATCH_CPU --array=2-$N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_label_render.sbatch "$LABEL"
#
#   # After the verdicts are in and answers.jsonl is pushed: the polygon images
#   sbatch $CSDV_SBATCH_CPU --array=1-$N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_label_render.sbatch "$LABEL" --kind polygon
#
#   # A small run in one job
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_render.sbatch "$CSDV_EXPORT_DIR/label/examples"
#
# Steps: check that every mosaic the windows need is on disk (a compute node
# has no internet, so a missing mosaic fails the job and never becomes a
# download), read each window straight from $CSDV_DATA_ROOT/naip/<unit>/<year>/,
# write the stacks, the request images and the four-band chips to node
# scratch, copy them into the label run folder on success.
#
# Outputs under the label run folder: stacks/<window>/, requests/<window>/
# (JPEGs and requests.jsonl or polygons.jsonl), chips/<chip>_naip.tif and
# render_done/<unit>_<kind>.json. Pull requests/ and stacks/*/meta.json to
# the laptop. The chips stay on CHPC for training.
#
# A window on the edge of its unit is rendered with the ground beyond the
# unit's mosaic left black. A window that lacks a year inside the unit is
# skipped and named in the last line of `csdv label render`, and the job still
# succeeds. Look for "Skipped" in the log.
#
# Resources: a window of nine chips with eight dates reads in a few seconds
# and uses under 1 GB. A unit of 60 windows takes a few minutes.

set -euo pipefail

LABEL_DIR="${1:?usage: sbatch [--array=1-N] $0 LABEL_DIR [csdv label render options...]}"
shift

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"
KIND="verdict"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --kind) KIND="${ARGS[$((i + 1))]:?--kind needs a value}" ;;
        --kind=*) KIND="${ARGS[$i]#--kind=}" ;;
    esac
done

if [[ ! -f "$LABEL_DIR/windows.csv" ]]; then
    echo "ERROR: $LABEL_DIR has no windows.csv. Push the label run first." >&2
    exit 1
fi

UNIT_ARGS=()
UNIT="all"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    UNIT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LABEL_DIR/units.csv" | tr -d '\r')
    : "${UNIT:?no unit on line $((SLURM_ARRAY_TASK_ID + 1)) of $LABEL_DIR/units.csv}"
    UNIT_ARGS=(--unit "$UNIT")
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} unit=$UNIT kind=$KIND started=$(date -Is)"
echo "label_dir=$LABEL_DIR run_tag=$RUN_TAG args: ${ARGS[*]-}"

DONE="$LABEL_DIR/render_done/${UNIT}_${KIND}.json"
if [[ -e "$DONE" ]]; then
    echo "Already rendered ($DONE), skipping."
    exit 0
fi

# One core's worth of threads per library. GDAL's cache stays bounded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg

activate_csdv
scratch_init

WORK="$NODE_SCRATCH/label"
mkdir -p "$WORK"
cp "$LABEL_DIR/windows.csv" "$WORK/"
[[ -f "$LABEL_DIR/units.csv" ]] && cp "$LABEL_DIR/units.csv" "$WORK/"
[[ -f "$LABEL_DIR/answers.jsonl" ]] && cp "$LABEL_DIR/answers.jsonl" "$WORK/"
[[ -d "$LABEL_DIR/candidates" ]] && rsync -a "$LABEL_DIR/candidates/" "$WORK/candidates/"
register_stage_out "$WORK" "$LABEL_DIR"

WHERE=(--label-dir "$WORK"
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG"
    --naip-source local --data-root "$CSDV_DATA_ROOT")

csdv label render "${WHERE[@]}" ${UNIT_ARGS[@]+"${UNIT_ARGS[@]}"} --check

/usr/bin/time -v csdv label render "${WHERE[@]}" ${UNIT_ARGS[@]+"${UNIT_ARGS[@]}"} \
    --chips ${ARGS[@]+"${ARGS[@]}"}

mkdir -p "$WORK/render_done"
printf '{"unit": "%s", "kind": "%s", "job": "%s", "finished": "%s"}\n' \
    "$UNIT" "$KIND" "${SLURM_JOB_ID:-none}" "$(date -Is)" \
    >"$WORK/render_done/${UNIT}_${KIND}.json"
du -sh "$WORK"/* | sort -h | tail -8
