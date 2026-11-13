#!/bin/bash
#SBATCH --job-name=csdv_label_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#
# Build the labels of a label run beside its four-band chips (`csdv label
# build`), CPU only, one job for the whole run. The first argument is the
# label run folder on CHPC, which holds what `run_label_render.sbatch` wrote
# (stacks/, chips/<chip>_naip.tif) and what scripts/chpc/push_label_run.sh
# sent after the verdicts (answers.jsonl, and config.yaml when the run uses
# its own thresholds). Every further argument is passed to `csdv label build`
# unchanged.
#
# The laptop builds the same labels without imagery (`--no-imagery`) to look
# at them. This job builds them with the imagery, so that pixels without
# imagery are no_data, and writes the label rasters next to the imagery chips
# that training reads.
#
# This script names no allocation. Pass yours at submission, from the option
# string in your scripts/chpc/env.sh (CSDV_SBATCH_CPU).
#
# Usage (after `source scripts/chpc/env.sh`):
#   LABEL="$CSDV_EXPORT_DIR/label/scale_test"
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_build.sbatch "$LABEL"
#
# Steps: copy the small inputs to node scratch, read the stacks in place
# (through a link, they are tens of gigabytes), write chips/ and labels/ on
# scratch, copy them into the label run folder on success.
#
# Outputs under the label run folder: chips/<chip>_{change,agent,reason,aux}.tif,
# chips/chips.parquet and labels/ (labels.gpkg, intervals.csv, events_kept.csv,
# suspects.csv). The four-band chips are written again from the same stacks.
#
# `build` writes its tables for the windows it is given, so this job always
# builds the whole run. A window whose stack lacks a year is left out, as at
# render.
#
# Resources: about two seconds per window and under 4 GB.
set -euo pipefail

LABEL_DIR="${1:?usage: sbatch $0 LABEL_DIR [csdv label build options...]}"
shift

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"
ARGS=("$@")

for needed in windows.csv answers.jsonl; do
    if [[ ! -f "$LABEL_DIR/$needed" ]]; then
        echo "ERROR: $LABEL_DIR has no $needed. Push the label run first." >&2
        exit 1
    fi
done
if [[ ! -d "$LABEL_DIR/stacks" ]]; then
    echo "ERROR: $LABEL_DIR has no stacks/. Render the label run first." >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "label_dir=$LABEL_DIR run_tag=$RUN_TAG args: ${ARGS[*]-}"

# One core's worth of threads per library. GDAL's cache stays bounded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg

activate_csdv
scratch_init

WORK="$NODE_SCRATCH/label"
mkdir -p "$WORK"
cp "$LABEL_DIR/windows.csv" "$LABEL_DIR/answers.jsonl" "$WORK/"
[[ -f "$LABEL_DIR/units.csv" ]] && cp "$LABEL_DIR/units.csv" "$WORK/"
[[ -d "$LABEL_DIR/candidates" ]] && rsync -a "$LABEL_DIR/candidates/" "$WORK/candidates/"
# The stacks are read where they are. Only chips/ and labels/ are copied
# back, so the link itself never reaches the label run folder.
ln -s "$LABEL_DIR/stacks" "$WORK/stacks"
register_stage_out "$WORK/chips" "$LABEL_DIR/chips"
register_stage_out "$WORK/labels" "$LABEL_DIR/labels"

CONFIG=()
if [[ -f "$LABEL_DIR/config.yaml" ]]; then
    cp "$LABEL_DIR/config.yaml" "$WORK/"
    CONFIG=(--config "$WORK/config.yaml")
    echo "thresholds of this run:"
    cat "$WORK/config.yaml"
fi

/usr/bin/time -v csdv label build --label-dir "$WORK" \
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG" \
    ${CONFIG[@]+"${CONFIG[@]}"} ${ARGS[@]+"${ARGS[@]}"}

du -sh "$WORK/chips" "$WORK/labels"
