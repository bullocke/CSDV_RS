#!/usr/bin/env bash
# Copy the cell layers that `csdv label` needs from CHPC to this machine.
# Run locally, not on CHPC.
#
# The grid export leaves the float cell layers of each disturbance run on
# CHPC. The labelling rules need four of them, all on the 19.2 m cell grid:
#
#   detection/probability.tif   disturbance probability, one band per year
#   layers/forest.tif           forest mask, one band per year
#   layers/chm_cover.tif        share of the cell taller than the forest height
#   layers/chm_height.tif       mean canopy height
#
# They are a few megabytes per unit. They land in
# <local exports>/run_layers/<unit>/<run_tag>/, which has the shape of a run
# folder, so the same reader works here and on CHPC.
#
# Usage:
#   scripts/chpc/pull_label_layers.sh [RUN_TAG] [--dry-run]
#
#   RUN_TAG    Run tag to copy. Default: default.
#   --dry-run  List what would be copied.
#
# Settings (environment, e.g. in ~/.bashrc):
#   CSDV_CHPC_HOST     ssh target, e.g. u0123456@dtn07.chpc.utah.edu.
#   CSDV_CHPC_EXPORTS  $CSDV_EXPORT_DIR on CHPC (as for pull_exports.sh).
#   CSDV_CHPC_RESULTS  $CSDV_RESULTS_ROOT on CHPC. Default: the `results`
#                      folder beside CSDV_CHPC_EXPORTS.
#   CSDV_LOCAL_EXPORTS Local export folder. Default: <repo>/results/chpc_exports.
#
# The units are those already exported locally under runs/<RUN_TAG>/. One
# rsync call copies every unit, so there is one login. Nothing is deleted.

set -euo pipefail

RUN_TAG="default"
DRY=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=(--dry-run) ;;
        -h | --help) sed -n '2,32p' "$0"; exit 0 ;;
        *) RUN_TAG="$arg" ;;
    esac
done

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
RESULTS="${CSDV_CHPC_RESULTS:-${CSDV_CHPC_EXPORTS%/}}"
if [[ -z "${CSDV_CHPC_RESULTS:-}" ]]; then
    RESULTS="$(dirname "$RESULTS")/results"
fi
if [[ "$RESULTS" != /* ]]; then
    echo "The CHPC results folder must be an absolute path: $RESULTS" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"
RUNS="$LOCAL/runs/$RUN_TAG"
if [[ ! -d "$RUNS" ]]; then
    echo "No exported runs at $RUNS. Run scripts/chpc/pull_exports.sh first." >&2
    exit 1
fi

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
n_units=0
for unit_dir in "$RUNS"/*/; do
    unit="$(basename "$unit_dir")"
    n_units=$((n_units + 1))
    for layer in detection/probability.tif layers/forest.tif \
        layers/chm_cover.tif layers/chm_height.tif; do
        echo "$unit/$RUN_TAG/$layer" >>"$LIST"
    done
done

DEST="$LOCAL/run_layers/"
mkdir -p "$DEST"
echo "$CSDV_CHPC_HOST:$RESULTS/disturbance/ -> $DEST ($n_units units, tag $RUN_TAG)"
rsync -avh --partial --info=stats1,progress2 ${DRY[@]+"${DRY[@]}"} \
    --files-from="$LIST" "$CSDV_CHPC_HOST:$RESULTS/disturbance/" "$DEST"
