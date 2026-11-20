#!/usr/bin/env bash
# Bring the small results of a round-2 step from CHPC to this machine. Run
# locally. No request image, stack or four-band chip is copied.
#
# Usage:
#   scripts/chpc/round2_pull.sh pilot  [--dry-run]
#   scripts/chpc/round2_pull.sh grid   [--dry-run]
#   scripts/chpc/round2_pull.sh stage1 [--dry-run]
#   scripts/chpc/round2_pull.sh final  [--dry-run]
#
# pilot   The label folder round2_pilot (tables, answers, ledger, labels,
#         sheets), the exported runs, cell layers, model maps and quicklooks
#         of the pilot boxes, and the box-against-quad comparison.
# grid    The label folder round2 as far as it exists (candidates, the
#         sample, render markers and dates) and the queue's status table
#         (written by `scripts/chpc/round2.sh status` on CHPC).
# stage1  The same plus answers, ledger, labels and sheets.
# final   The same after stage 2, plus the label rasters of the chips and
#         the model runs of round 2 (pairmodel/r2_*).
#
# Settings (environment, as for pull_exports.sh): CSDV_CHPC_HOST,
# CSDV_CHPC_EXPORTS, and optionally CSDV_LOCAL_EXPORTS.
#
# Sizes: the label folder without images is tens of MB. Sheets are about
# 1 MB each. Check `df -h /mnt/c` before `final`.

set -euo pipefail

STEP="${1:?usage: $0 pilot|grid|stage1|final [--dry-run]}"
DRY=()
[[ "${2:-}" == "--dry-run" ]] && DRY=(--dry-run)

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"
RSYNC=(rsync -avh --partial --prune-empty-dirs --info=stats1 ${DRY[@]+"${DRY[@]}"})

# label NAME [extra include rules...]: the small files of a label run.
label() {
    local name="$1"
    shift
    local rules=(--include='*.csv' --include='config.yaml' --include='*.jsonl'
        --include='last_chain_job.txt'
        --include='candidates/***' --include='render_done/***' --include='labels/***'
        --include='stacks/' --include='stacks/*/' --include='stacks/*/meta.json'
        --include='stacks/*/cell_ndvi.tif' "$@" --exclude='*')
    mkdir -p "$LOCAL/label/$name"
    echo "== label/$name"
    "${RSYNC[@]}" "${rules[@]}" \
        "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/label/$name/" "$LOCAL/label/$name/"
}

# tree SUBPATH [rules...]: part of the export tree.
tree() {
    local sub="$1"
    shift
    mkdir -p "$LOCAL/$sub"
    echo "== $sub"
    "${RSYNC[@]}" "$@" "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/$sub/" "$LOCAL/$sub/"
}

# units_of NAME: include rules for the units of a batch file.
units_of() {
    python3 - "$REPO/config/grid/$1.csv" <<'PY'
import csv, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.lstrip().startswith("#")]
for row in csv.DictReader(rows):
    unit = row.get("cell") or ("p_" + row["label"] if row.get("buffer_km") else "")
    if unit:
        print(f"--include={unit.strip()}/***")
PY
}

case "$STEP" in
    pilot)
        label round2_pilot --include='sheets/***'
        mapfile -t UNITS < <(units_of round2_pilot)
        tree runs/default "${UNITS[@]}" --exclude='*'
        tree run_layers "${UNITS[@]}" --exclude='*'
        tree quicklooks "${UNITS[@]}" --exclude='*'
        tree checks/round2_pilot
        ;;
    grid)
        label round2
        tree checks/round2
        ;;
    stage1)
        label round2 --include='sheets/***'
        ;;
    final)
        df -h /mnt/c 2>/dev/null | tail -1 || true
        label round2 --include='sheets/***' --include='chips/' \
            --include='chips/chips.parquet' --include='chips/*_change.tif' \
            --include='chips/*_reason.tif' --include='chips/*_agent.tif' \
            --include='chips/*_aux.tif'
        tree pairmodel --include='r2_*/***' --exclude='*.pt' --exclude='*'
        ;;
    *)
        echo "usage: $0 pilot|grid|stage1|final [--dry-run]" >&2
        exit 2
        ;;
esac
