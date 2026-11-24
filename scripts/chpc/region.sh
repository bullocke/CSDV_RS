#!/usr/bin/env bash
# A block of adjoining quads mapped wall to wall: the rehearsal of a regional
# map. Run on CHPC, after `source scripts/chpc/env.sh` and `activate_csdv`.
#
#   scripts/chpc/region.sh block [NAME] [--dry-run]
#       Send the quads of config/grid/NAME.csv through the grid queue
#       (downloads, height models on the GPU pools, detection, the pair
#       model's map, export). Give the same command again to submit what is
#       left. NAME defaults to indiana_block.
#   scripts/chpc/region.sh status [NAME]
#       Quads that are not exported yet, your jobs, and the disk use of the
#       block, per quad.
#
# The block step prints the free space of the data root first, with the size
# of a finished quad, so that the storage of the block can be judged before
# anything is submitted. A quad of 7.5 minutes holds four quarter quads a
# year, and its mosaics and height models stay on disk.
#
# Settings, all in scripts/chpc/env.sh: CSDV_SBATCH_CPU and CSDV_GPU_POOLS,
# as for scripts/chpc/round2.sh. CSDV_REGION_CHECKPOINT names the pair model
# that maps every quad (default: reserved_r2_s0, the model of test B of round
# 2 with seed 0). CSDV_REGION_QUEUE_ARGS replaces the queue options, which
# default to "--worker-mem 48G --worker-hours 6": the height model of a whole
# quad needs 21 to 34 GB, more than the 24 GB the queue gives a small box.
#
# The grid queue was written for many small boxes, and whole quads go through
# it here for the first time. A quad that stops is named by `status`, with
# the log to read.

set -euo pipefail

STEP="${1:-}"
[[ -n "$STEP" ]] && shift
NAME="indiana_block" DRY=()
while (($#)); do
    case "$1" in
        --dry-run) DRY=(--dry-run); shift ;;
        -h | --help) sed -n '2,29p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) NAME="$1"; shift ;;
    esac
done

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_CPU:?CSDV_SBATCH_CPU is not set in scripts/chpc/env.sh}"
cd "$CSDV_REPO"
mkdir -p "$CSDV_LOG_DIR"
BATCH="config/grid/$NAME.csv"
[[ -f "$BATCH" ]] || { echo "ERROR: no batch file $BATCH" >&2; exit 1; }
CKPT="${CSDV_REGION_CHECKPOINT:-$CSDV_RESULTS_ROOT/pairmodel/reserved_r2_s0/best.pt}"
read -r -a QUEUE_ARGS <<<"${CSDV_REGION_QUEUE_ARGS---worker-mem 48G --worker-hours 6}"

# units: the unit IDs of the batch, one a line.
units() {
    python - "$BATCH" <<'PY'
import sys
from pathlib import Path

from csdv_core.grid.units import read_batch

for unit in read_batch(Path(sys.argv[1])):
    print(unit.unit_id)
PY
}

case "$STEP" in
    block)
        [[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
        echo "== Free space of the data root"
        df -h "$CSDV_DATA_ROOT" | tail -n +1
        # A finished quad of the same state shows what one quad takes.
        for kind in naip naip_chm; do
            for done_unit in "$CSDV_DATA_ROOT/$kind"/q3908654; do
                [[ -d "$done_unit" ]] && du -sh "$done_unit" 2>/dev/null || true
            done
        done
        echo "The block holds $(units | wc -l) quads. Each takes about what" \
            "q3908654 takes above (mosaics and height models)."
        echo
        csdv grid queue --batch "$BATCH" --name "$NAME" --checkpoint "$CKPT" \
            ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} ${DRY[@]+"${DRY[@]}"}
        cat <<MSG

Block $NAME $( ((${#DRY[@]})) && echo "would be" || echo "is") submitted. No paid call is in it. In the morning:
  scripts/chpc/region.sh status $NAME    # quads exported, disk use per quad
  scripts/chpc/region.sh block $NAME     # again, if anything is left
MSG
        ;;
    status)
        echo "== Quads of $NAME that are not exported yet"
        csdv grid queue-status --name "$NAME" --problems || true
        mkdir -p "$CSDV_EXPORT_DIR/checks/$NAME"
        csdv grid queue-status --name "$NAME" \
            >"$CSDV_EXPORT_DIR/checks/$NAME/queue_status.txt" 2>&1 || true
        echo
        echo "== Your jobs in the queue: $(squeue -h -u "$USER" | wc -l)"
        squeue -h -u "$USER" -o "%.12i %.24j %.9T %.10M %R" | sort -k2 |
            uniq -c -f1 | head -20 || true
        echo
        echo "== Disk use per quad (mosaics, height models)"
        {
            echo "unit naip naip_chm"
            for unit in $(units); do
                a="$(du -sh "$CSDV_DATA_ROOT/naip/$unit" 2>/dev/null | cut -f1)"
                b="$(du -sh "$CSDV_DATA_ROOT/naip_chm/$unit" 2>/dev/null | cut -f1)"
                echo "$unit ${a:--} ${b:--}"
            done
        } | tee "$CSDV_EXPORT_DIR/checks/$NAME/disk_use.txt" | column -t
        echo
        echo "== Run times of the finish tasks (detection, map, export)"
        sacct -u "$USER" --name="csdv_fin_$NAME" -S "$(date -d '3 days ago' +%F)" -X -n \
            -o "JobID%18,State%12,Elapsed%10,MaxRSS%10" 2>/dev/null | tail -n 12 || true
        ;;
    *)
        echo "usage: $0 block|status [NAME] [--dry-run]" >&2
        exit 2
        ;;
esac
