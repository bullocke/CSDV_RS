#!/usr/bin/env bash
# Copy the runs of the defoliation test of the pair model from CHPC to this
# machine, in one connection. Run locally, not on CHPC, after the jobs of
# scripts/chpc/submit_pair_defoliation.sh have finished.
#
# Usage:
#   scripts/chpc/pull_pair_defoliation.sh [--optional] [--table FILE] [--dry-run]
#
#   --optional    Also copy the runs with batch "optional".
#   --table FILE  Run table. Default config/pair/defoliation.csv.
#   --dry-run     List what would be copied.
#
# Every run of the table that exists on CHPC is copied, with the smoke run:
# about 20 to 30 MB each (the maps, the comparison and its close-ups, no
# checkpoint). Runs that are not there yet are left out without an error, so
# the script can be run again as jobs finish.
#
# Settings (environment): CSDV_CHPC_HOST, CSDV_CHPC_EXPORTS and
# CSDV_LOCAL_EXPORTS, as for scripts/chpc/pull_exports.sh.

set -euo pipefail

OPTIONAL=0
TABLE=""
DRY=()
while (($#)); do
    case "$1" in
        --optional) OPTIONAL=1; shift ;;
        --table) TABLE="$2"; shift 2 ;;
        --dry-run) DRY=(--dry-run); shift ;;
        -h | --help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${CSDV_CHPC_HOST:?export CSDV_CHPC_HOST, e.g. u0123456@dtn07.chpc.utah.edu}"
: "${CSDV_CHPC_EXPORTS:?export CSDV_CHPC_EXPORTS, the absolute \$CSDV_EXPORT_DIR on CHPC}"
if [[ "$CSDV_CHPC_EXPORTS" != /* ]]; then
    echo "CSDV_CHPC_EXPORTS must be an absolute path: $CSDV_CHPC_EXPORTS" >&2
    exit 1
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${CSDV_LOCAL_EXPORTS:-$REPO/results/chpc_exports}"
TABLE="${TABLE:-$REPO/config/pair/defoliation.csv}"

INCLUDE=()
count=0
while IFS=, read -r run kind labels dates unit seed model batch notes; do
    [[ -z "$run" || "$run" == \#* || "$run" == "run" ]] && continue
    [[ "$batch" == "yes" ]] || { ((OPTIONAL)) && [[ "$batch" == "optional" ]]; } || continue
    INCLUDE+=(--include="$run/" --include="$run/**"
        --include="${run}_smoke/" --include="${run}_smoke/**")
    count=$((count + 1))
done <"$TABLE"
if ((count == 0)); then
    echo "No run to copy in $TABLE" >&2
    exit 1
fi

SRC="$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/pairmodel/"
DEST="$LOCAL/pairmodel/"
mkdir -p "$DEST"
echo "$count runs of $TABLE: $SRC -> $DEST"
rsync -avh --partial --info=stats1,progress2 ${DRY[@]+"${DRY[@]}"} \
    "${INCLUDE[@]}" --exclude='*' "$SRC" "$DEST"
