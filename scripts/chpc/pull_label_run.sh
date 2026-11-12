#!/usr/bin/env bash
# Copy what the laptop needs of a rendered label run from CHPC. Run locally,
# not on CHPC.
#
# `csdv label render` on CHPC writes, under <exports>/label/NAME/:
#
#   requests/<window>/     request JPEGs and requests.jsonl (polygons.jsonl)
#   stacks/<window>/       meta.json (dates), cell_ndvi.tif (a few kB) and one
#                          four-band GeoTIFF per year (tens of MB per window)
#   chips/                 four-band training chips (stay on CHPC)
#   render_done/           one marker per unit and kind
#
# The verdicts and the labels need the requests, and of each stack only
# meta.json and cell_ndvi.tif. The yearly GeoTIFFs are needed only for
# review sheets (`csdv label sheets`), so they are copied only for the
# windows named with --stacks.
#
# Usage:
#   scripts/chpc/pull_label_run.sh NAME [--stacks W0001,W0002 | --stacks-from FILE]
#                                       [--no-requests] [--dry-run]
#
#   NAME            Name of the label run on CHPC (as given to push_label_run.sh).
#   --stacks LIST   Comma-separated window IDs whose whole stacks are copied.
#   --stacks-from FILE  The same, one window ID per line.
#   --no-requests   Leave the request images on CHPC (dates and NDVI only).
#   --dry-run       List what would be copied.
#
# Settings (environment, e.g. in ~/.bashrc):
#   CSDV_CHPC_HOST     ssh target, e.g. u0123456@dtn07.chpc.utah.edu.
#   CSDV_CHPC_EXPORTS  $CSDV_EXPORT_DIR on CHPC, an absolute path.
#   CSDV_LOCAL_EXPORTS Local export folder. Default: <repo>/results/chpc_exports.
#
# Everything goes through one rsync call, so there is one login. rsync copies
# only new or changed files, so running it again after more units have
# rendered fetches just those. Nothing is deleted locally.

set -euo pipefail

NAME=""
DRY=()
WINDOWS=()
REQUESTS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=(--dry-run) ;;
        --no-requests) REQUESTS=0 ;;
        --stacks)
            IFS=',' read -r -a more <<<"${2:?--stacks needs a list of window IDs}"
            WINDOWS+=("${more[@]}")
            shift
            ;;
        --stacks-from)
            file="${2:?--stacks-from needs a file}"
            while IFS= read -r line; do
                line="${line//[$'\r\t ']/}"
                [[ -n "$line" ]] && WINDOWS+=("$line")
            done <"$file"
            shift
            ;;
        -h | --help) sed -n '2,35p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) NAME="$1" ;;
    esac
    shift
done
: "${NAME:?usage: $0 NAME [--stacks W1,W2 | --stacks-from FILE] [--no-requests] [--dry-run]}"

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
if [[ "$CSDV_CHPC_EXPORTS" != /* ]]; then
    echo "CSDV_CHPC_EXPORTS must be an absolute path: $CSDV_CHPC_EXPORTS" >&2
    exit 1
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"

# The first rule that matches a path decides, so the whole stacks of the
# named windows come before the rule that keeps only the small files.
FILTER=(--include='render_done/***')
if [[ "$REQUESTS" -eq 1 ]]; then
    FILTER+=(--include='requests/***')
fi
for window in ${WINDOWS[@]+"${WINDOWS[@]}"}; do
    FILTER+=(--include="stacks/$window/***")
done
FILTER+=(--include='stacks/' --include='stacks/*/'
    --include='stacks/*/meta.json' --include='stacks/*/cell_ndvi.tif'
    --exclude='*')

SRC="$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/label/$NAME/"
DEST="$LOCAL/label/$NAME/"
mkdir -p "$DEST"
echo "$SRC -> $DEST"
echo "whole stacks of ${#WINDOWS[@]} windows, requests: $REQUESTS"
rsync -avh --partial --prune-empty-dirs --info=stats1,progress2 \
    ${DRY[@]+"${DRY[@]}"} "${FILTER[@]}" "$SRC" "$DEST"
