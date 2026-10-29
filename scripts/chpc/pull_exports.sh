#!/usr/bin/env bash
# Copy the grid exports from CHPC to this machine. Run locally, not on CHPC.
#
# Usage:
#   scripts/chpc/pull_exports.sh [SUBPATH] [--dry-run]
#
#   SUBPATH    Part of the export tree to copy, e.g. runs/default or
#              quicklooks/q4407159. Default: everything.
#   --dry-run  List what would be copied.
#
# Settings (environment, e.g. in ~/.bashrc):
#   CSDV_CHPC_HOST     ssh target, e.g. u0123456@dtn07.chpc.utah.edu (a data
#                      transfer node is faster than a login node).
#   CSDV_CHPC_EXPORTS  $CSDV_EXPORT_DIR on CHPC, e.g.
#                      /uufs/chpc.utah.edu/common/home/<group>/data/CSDV/exports
#   CSDV_LOCAL_EXPORTS Local destination. Default: <repo>/results/chpc_exports.
#
# rsync copies only new or changed files, so running it again after more units
# finish fetches just those. Nothing is deleted locally.

set -euo pipefail

SUBPATH=""
DRY=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=(--dry-run) ;;
        -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
        *) SUBPATH="${arg#/}" ;;
    esac
done

: "${CSDV_CHPC_HOST:?set CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?set CSDV_CHPC_EXPORTS to \$CSDV_EXPORT_DIR on CHPC}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"

SRC="$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/${SUBPATH:+$SUBPATH/}"
DEST="$LOCAL/${SUBPATH:+$SUBPATH/}"
mkdir -p "$DEST"
echo "$SRC -> $DEST"
rsync -avh --partial --info=stats1,progress2 ${DRY[@]+"${DRY[@]}"} "$SRC" "$DEST"
