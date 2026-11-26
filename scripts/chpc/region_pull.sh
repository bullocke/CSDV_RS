#!/usr/bin/env bash
# Bring the small maps of a block of quads from CHPC to this machine. Run
# locally, after scripts/chpc/region.sh status NAME on CHPC shows every quad
# exported.
#
#   scripts/chpc/region_pull.sh [NAME ...] [--blocks FOLDER] [--dry-run]
#
# NAME is a batch file of config/grid (default: indiana_block indiana_ring).
# Per quad it copies the cell layers at 19.2 m (probability of the
# Disturbance Index, forest mask, canopy cover and height), the pair model's
# maps (probability and cause per interval), and the run's summary tables.
# About 25 MB per quad. No vector file, figure or quicklook is copied: with
# those a quad comes to about 300 MB.
#
# --blocks FOLDER copies the patch layer of a block (scripts/chpc/region.sh
# patches ... --page FOLDER): a few MB. With it and no NAME, nothing else is
# copied.
#
# Settings (environment, as for round2_pull.sh): CSDV_CHPC_HOST,
# CSDV_CHPC_EXPORTS, and optionally CSDV_LOCAL_EXPORTS.

set -euo pipefail

DRY=() NAMES=() BLOCKS=()
while (($#)); do
    case "$1" in
        --dry-run) DRY=(--dry-run); shift ;;
        --blocks) BLOCKS+=("${2:?--blocks needs a folder name}"); shift 2 ;;
        -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
        *) NAMES+=("$1"); shift ;;
    esac
done
((${#NAMES[@]} + ${#BLOCKS[@]})) || NAMES=(indiana_block indiana_ring)

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"
RSYNC=(rsync -avh --partial --prune-empty-dirs --info=stats1 ${DRY[@]+"${DRY[@]}"})

df -h /mnt/c 2>/dev/null | tail -1 || true
for name in ${NAMES[@]+"${NAMES[@]}"}; do
    batch="$REPO/config/grid/$name.csv"
    [[ -f "$batch" ]] || { echo "ERROR: no batch file $batch" >&2; exit 1; }
    mapfile -t UNITS < <(grep -v '^#' "$batch" | tail -n +2 | cut -d, -f5 | grep .)
    LAYERS=() RUNS=()
    for unit in "${UNITS[@]}"; do
        LAYERS+=(--include="$unit/***")
        RUNS+=(--include="$unit/" --include="$unit/*.json" --include="$unit/objects.csv"
            --include="$unit/report.md")
    done
    echo "== $name: ${#UNITS[@]} quads"
    mkdir -p "$LOCAL/run_layers" "$LOCAL/runs/default" "$LOCAL/checks/$name"
    "${RSYNC[@]}" "${LAYERS[@]}" --exclude='*' \
        "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/run_layers/" "$LOCAL/run_layers/"
    "${RSYNC[@]}" "${RUNS[@]}" --exclude='*' \
        "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/runs/default/" "$LOCAL/runs/default/"
    "${RSYNC[@]}" "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/checks/$name/" "$LOCAL/checks/$name/"
done
for folder in ${BLOCKS[@]+"${BLOCKS[@]}"}; do
    echo "== patches of $folder"
    mkdir -p "$LOCAL/blocks/$folder"
    "${RSYNC[@]}" "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/blocks/$folder/" "$LOCAL/blocks/$folder/"
done
