#!/usr/bin/env bash
# Copy the site maps of the pair model from CHPC to this machine, in one
# connection. Run locally, not on CHPC, after the jobs of
# scripts/chpc/submit_pair_sites.sh have finished.
#
# Usage:
#   scripts/chpc/pull_pair_sites.sh [--mode look|holdout] [--dry-run]
#
#   --mode     Which runs to copy: look_<unit> (default) or holdout_<unit>.
#   --dry-run  List what would be copied.
#
# About 20 MB per unit (the maps, the comparison and its close-ups, no
# checkpoint). The backdrops of the figures are the 3 m quicklooks of each
# unit: scripts/chpc/pull_exports.sh quicklooks/<unit> fetches one that is
# missing.
#
# Settings (environment): CSDV_CHPC_HOST, CSDV_CHPC_EXPORTS and
# CSDV_LOCAL_EXPORTS, as for scripts/chpc/pull_exports.sh.

set -euo pipefail

MODE=look
DRY=()
while (($#)); do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --dry-run) DRY=(--dry-run); shift ;;
        -h | --help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
if [[ "$MODE" != "look" && "$MODE" != "holdout" ]]; then
    echo "--mode must be look or holdout, not $MODE" >&2
    exit 1
fi

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
if [[ "$CSDV_CHPC_EXPORTS" != /* ]]; then
    echo "CSDV_CHPC_EXPORTS must be an absolute path: $CSDV_CHPC_EXPORTS" >&2
    exit 1
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"

SRC="$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/pairmodel/"
DEST="$LOCAL/pairmodel/"
mkdir -p "$DEST"
echo "$SRC${MODE}_q* -> $DEST"
rsync -avh --partial --info=stats1,progress2 ${DRY[@]+"${DRY[@]}"} \
    --include="${MODE}_q*/" --include="${MODE}_q*/**" --exclude='*' "$SRC" "$DEST"
ls -d "$DEST${MODE}"_q* 2>/dev/null | wc -l | xargs -I{} echo "{} site runs in $DEST"
