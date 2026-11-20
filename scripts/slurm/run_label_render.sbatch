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
# `csdv label render` unchanged (for example --kind polygon or --stage 1).
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
# write the stacks and the request images to node scratch, copy them into
# the label run folder on success.
#
# Outputs under the label run folder: stacks/<window>/ (one GeoTIFF per year,
# meta.json with the dates, cell_ndvi.tif), requests/<window>/ (JPEGs and
# requests.jsonl or polygons.jsonl) and render_done/<unit>_<kind>[_stage<S>].json.
# Only these three folders are copied back. The job reads answers.jsonl for
# the polygon pass and never writes it, so it can run while a verdict job
# (scripts/slurm/run_label_verdict.sbatch) appends to that file.
#
# The four-band chips are written by the label build
# (scripts/slurm/run_label_build.sbatch). Set CSDV_RENDER_CHIPS=1 to write
# them here as well, as the scale test did.
#
# The polygon pass reads the stacks of the verdict pass where they are and
# does not read the mosaics again.
#
# The date of a window comes from the .provenance.json beside each mosaic.
# The --check step counts mosaics without one, and the last lines of the
# render count the window-years by date rule. "mosaic_name" there means the
# unit's median date was used, which can be months off: stop and look.
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
STAGE=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --kind) KIND="${ARGS[$((i + 1))]:?--kind needs a value}" ;;
        --kind=*) KIND="${ARGS[$i]#--kind=}" ;;
        --stage) STAGE="${ARGS[$((i + 1))]:?--stage needs a value}" ;;
        --stage=*) STAGE="${ARGS[$i]#--stage=}" ;;
    esac
done
# A unit rendered for stage 1 still has windows to render in stage 2, so the
# marker of a finished task names its stage.
MARK="${KIND}${STAGE:+_stage${STAGE//,/-}}"

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

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} unit=$UNIT kind=$KIND stage=${STAGE:-all} started=$(date -Is)"
echo "label_dir=$LABEL_DIR run_tag=$RUN_TAG args: ${ARGS[*]-}"

DONE="$LABEL_DIR/render_done/${UNIT}_${MARK}.json"
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
# Small inputs that some runs have: the unit list, the candidates to judge
# per window, the location text of the units, the thresholds the run changes.
for extra in units.csv judged.csv locations.csv config.yaml; do
    [[ -f "$LABEL_DIR/$extra" ]] && cp "$LABEL_DIR/$extra" "$WORK/"
done
[[ -d "$LABEL_DIR/candidates" ]] && rsync -a "$LABEL_DIR/candidates/" "$WORK/candidates/"
if [[ "$KIND" == "polygon" ]]; then
    # Read-only here: the verdicts decide which candidates get a polygon.
    [[ -f "$LABEL_DIR/answers.jsonl" ]] && cp "$LABEL_DIR/answers.jsonl" "$WORK/"
    # The stacks of the verdict pass, read where they are.
    [[ -d "$LABEL_DIR/stacks" ]] && ln -s "$LABEL_DIR/stacks" "$WORK/stacks"
else
    register_stage_out "$WORK/stacks" "$LABEL_DIR/stacks"
fi
register_stage_out "$WORK/requests" "$LABEL_DIR/requests"
register_stage_out "$WORK/render_done" "$LABEL_DIR/render_done"
[[ "${CSDV_RENDER_CHIPS:-0}" == "1" ]] && register_stage_out "$WORK/chips" "$LABEL_DIR/chips"

WHERE=(--label-dir "$WORK"
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG")
[[ -f "$WORK/config.yaml" ]] && WHERE+=(--config "$WORK/config.yaml")
DATA=(--naip-source local --data-root "$CSDV_DATA_ROOT")
CHIPS=()
[[ "${CSDV_RENDER_CHIPS:-0}" == "1" ]] && CHIPS=(--chips)

csdv label render "${WHERE[@]}" "${DATA[@]}" ${UNIT_ARGS[@]+"${UNIT_ARGS[@]}"} --check

/usr/bin/time -v csdv label render "${WHERE[@]}" "${DATA[@]}" \
    ${UNIT_ARGS[@]+"${UNIT_ARGS[@]}"} ${CHIPS[@]+"${CHIPS[@]}"} ${ARGS[@]+"${ARGS[@]}"}

mkdir -p "$WORK/render_done"
printf '{"unit": "%s", "kind": "%s", "stage": "%s", "job": "%s", "finished": "%s"}\n' \
    "$UNIT" "$KIND" "${STAGE:-all}" "${SLURM_JOB_ID:-none}" "$(date -Is)" \
    >"$WORK/render_done/${UNIT}_${MARK}.json"
du -shL "$WORK"/requests "$WORK"/stacks 2>/dev/null | sort -h
