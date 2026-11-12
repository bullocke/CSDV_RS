#!/usr/bin/env bash
# Copy the small files of a label run from this machine to CHPC, so that the
# windows can be rendered there from the NAIP mosaics. Run locally.
#
# Sent: windows.csv, units.csv, candidates/ and, when it exists, answers.jsonl
# (the polygon images are rendered from the verdicts). Nothing else leaves
# this machine: no API key, no ledger, no response cache.
#
# Usage:
#   scripts/chpc/push_label_run.sh LABEL_DIR NAME [--dry-run]
#
#   LABEL_DIR  Local folder of the label run.
#   NAME       Folder name on CHPC, under $CSDV_EXPORT_DIR/label/.
#
# Settings (environment, as for pull_exports.sh):
#   CSDV_CHPC_HOST     ssh target, e.g. u0123456@dtn07.chpc.utah.edu.
#   CSDV_CHPC_EXPORTS  $CSDV_EXPORT_DIR on CHPC.
#
# Afterwards, on CHPC, submit scripts/slurm/run_label_render.sbatch for
# $CSDV_EXPORT_DIR/label/NAME, then bring the request images back with
#   scripts/chpc/pull_exports.sh label/NAME/requests

set -euo pipefail

DRY=()
POS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=(--dry-run) ;;
        -h | --help) sed -n '2,22p' "$0"; exit 0 ;;
        *) POS+=("$arg") ;;
    esac
done
LABEL_DIR="${POS[0]:?usage: $0 LABEL_DIR NAME [--dry-run]}"
NAME="${POS[1]:?usage: $0 LABEL_DIR NAME [--dry-run]}"

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
if [[ ! -f "$LABEL_DIR/windows.csv" || ! -f "$LABEL_DIR/units.csv" ]]; then
    echo "$LABEL_DIR needs windows.csv and units.csv (csdv label sample, or csdv label units)." >&2
    exit 1
fi

DEST="$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/label/$NAME/"
echo "$LABEL_DIR -> $DEST"
rsync -avh --partial --mkpath ${DRY[@]+"${DRY[@]}"} \
    --include='windows.csv' --include='units.csv' --include='answers.jsonl' \
    --include='candidates/' --include='candidates/*.gpkg' --exclude='*' \
    "${LABEL_DIR%/}/" "$DEST"
