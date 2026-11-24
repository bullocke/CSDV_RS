#!/usr/bin/env bash
# Bring the small results of a round-2 step from CHPC to this machine. Run
# locally. No request image, stack or four-band chip is copied.
#
# Usage:
#   scripts/chpc/round2_pull.sh pilot  [--dry-run]
#   scripts/chpc/round2_pull.sh grid   [--dry-run]
#   scripts/chpc/round2_pull.sh stage1 [--dry-run]
#   scripts/chpc/round2_pull.sh final  [--dry-run]
#   scripts/chpc/round2_pull.sh sheets LIST [--dry-run]
#   scripts/chpc/round2_pull.sh chips  [--dry-run]
#   scripts/chpc/round2_pull.sh evaluate [--optional] [--dry-run]
#   scripts/chpc/round2_pull.sh blind  [--dry-run]
#
# pilot   The label folder round2_pilot (tables, answers, ledger, labels,
#         sheets), the exported runs, cell layers, model maps and quicklooks
#         of the pilot boxes, and the box-against-quad comparison.
# grid    The label folder round2 as far as it exists (candidates, the
#         sample, render markers and dates) and the queue's status table
#         (written by `scripts/chpc/round2.sh status` on CHPC).
#         After stage 1 it also brings answers, ledger and labels, without
#         the sheets.
# stage1  The same plus answers, ledger, labels and every sheet. An overview
#         sheet is about 6 MB and a pair sheet 1.5 MB, so the 1,600 windows
#         of stage 1 come to about 12 GB. Use `grid` and `sheets` for less.
# sheets  The sheets of the windows named in LIST, a text file with one
#         window ID a line (`#` starts a comment).
# final   The label folder after stage 2: tables, answers, ledger, labels
#         and the chip table (chips/chips.parquet). Tens of MB. No sheet and
#         no raster: the sheets of both stages come to about 25 GB, so take
#         the ones to read with `sheets LIST`.
# chips   The label rasters of the chips of round 2 (change, reason, agent,
#         aux), for galleries of training samples. Not needed for the
#         tables or the maps.
# evaluate  The model runs of config/pair/round2.csv that exist on CHPC, as
#         scripts/chpc/pull_pair_defoliation.sh --table does (maps, scores,
#         comparisons, no checkpoint), and the chip table and sites table of
#         the merged folders and of round2_norej.
# blind   The images of the blind check: the request images of the windows
#         of its candidates, the whole stacks of its quiet windows and the
#         label rasters of their chips, as listed by the draw (by default in
#         results/vlm_labeling/3_round2/blind_check). Under 1 GB.
#
# Settings (environment, as for pull_exports.sh): CSDV_CHPC_HOST,
# CSDV_CHPC_EXPORTS, and optionally CSDV_LOCAL_EXPORTS.
#
# Sizes: the label folder without images is tens of MB. Sheets are about
# 1 MB each. Check `df -h /mnt/c` before `final`.

set -euo pipefail

USAGE="usage: $0 pilot|grid|stage1|final|chips|evaluate|blind [--dry-run], or sheets LIST [--dry-run]"
STEP="${1:?$USAGE}"
shift
DRY=()
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then DRY=(--dry-run); else ARGS+=("$arg"); fi
done

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
        label round2 --include='chips/' --include='chips/chips.parquet'
        ;;
    chips)
        df -h /mnt/c 2>/dev/null | tail -1 || true
        label round2 --include='chips/' \
            --include='chips/chips.parquet' --include='chips/*_change.tif' \
            --include='chips/*_reason.tif' --include='chips/*_agent.tif' \
            --include='chips/*_aux.tif'
        ;;
    evaluate)
        OPTIONAL=()
        [[ " ${ARGS[*]-} " == *" --optional "* ]] && OPTIONAL=(--optional)
        "$REPO/scripts/chpc/pull_pair_defoliation.sh" --table "$REPO/config/pair/round2.csv" \
            ${OPTIONAL[@]+"${OPTIONAL[@]}"} ${DRY[@]+"${DRY[@]}"}
        label round2_norej --include='chips/' --include='chips/chips.parquet'
        for name in merged_r2 merged_r2_norej; do
            tree "label/$name" --include='chips/' --include='chips/chips.parquet' \
                --include='chips/sites.csv' --exclude='*'
        done
        ;;
    sheets)
        LIST="${ARGS[0]:?$USAGE}"
        mapfile -t WINDOWS < <(sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "$LIST")
        [[ ${#WINDOWS[@]} -gt 0 ]] || { echo "ERROR: no window ID in $LIST" >&2; exit 1; }
        RULES=(--include='sheets/')
        for window in "${WINDOWS[@]}"; do
            RULES+=("--include=sheets/${window}_*")
        done
        echo "Sheets of ${#WINDOWS[@]} windows"
        tree label/round2 "${RULES[@]}" --exclude='*'
        ;;
    blind)
        DIR="${ARGS[0]:-$REPO/results/vlm_labeling/3_round2/blind_check}"
        for list in windows_requests.txt windows_stacks.txt chips.txt; do
            [[ -f "$DIR/$list" ]] || { echo "ERROR: no $DIR/$list. Draw the sample first." >&2; exit 1; }
        done
        names() { sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "$1"; }
        RULES=(--include='requests/' --include='stacks/' --include='chips/')
        while read -r window; do
            RULES+=("--include=requests/$window/***")
        done < <(names "$DIR/windows_requests.txt")
        while read -r window; do
            RULES+=("--include=stacks/$window/***")
        done < <(names "$DIR/windows_stacks.txt")
        while read -r chip; do
            RULES+=("--include=chips/${chip}_change.tif")
        done < <(names "$DIR/chips.txt")
        echo "Blind check: requests of $(names "$DIR/windows_requests.txt" | wc -l) windows," \
            "stacks of $(names "$DIR/windows_stacks.txt" | wc -l)," \
            "label rasters of $(names "$DIR/chips.txt" | wc -l) chips"
        tree label/round2 "${RULES[@]}" --exclude='*'
        ;;
    *)
        echo "$USAGE" >&2
        exit 2
        ;;
esac
