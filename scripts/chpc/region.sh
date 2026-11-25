#!/usr/bin/env bash
# Blocks of adjoining quads mapped wall to wall: a regional map. Run on CHPC,
# after `source scripts/chpc/env.sh` and `activate_csdv`.
#
#   scripts/chpc/region.sh block  NAME... [--dry-run]
#       Send the quads of config/grid/NAME.csv through the grid queue
#       (downloads, height models on the GPU pools, detection, the pair
#       model's map, export). Give the same command again to submit what is
#       left. A NAME without a file of its own stands for its parts
#       NAME_1.csv, NAME_2.csv, ... (scripts/grid/block_csv.py writes them):
#       each part is a queue of its own, so that its quads are finished as
#       soon as its own height models are done.
#   scripts/chpc/region.sh remap  NAME... [--checkpoint RUN] [--dry-run]
#       Map the finished quads again with another pair model and export
#       them again. Downloads, height models and detection are kept. RUN is
#       a training run under $CSDV_RESULTS_ROOT/pairmodel or a path to a
#       best.pt (default: CSDV_REGION_CHECKPOINT). The old map of a quad
#       stays in its run folder as model.prev.<job>.
#   scripts/chpc/region.sh page   NAME... [--page FOLDER] [--title TEXT] [--dry-run]
#       One web page for the quads of all the names: NAIP per year and the
#       model's loss and cause per interval, as map tiles (`csdv grid
#       webmap`, one CPU job). It goes to CSDV_REGION_PAGE_DIR/FOLDER (the
#       first NAME when --page is not given),
#       by default under $CSDV_PUBLIC_HTML/csdv_blocks. The job stops when
#       the page is projected above CSDV_REGION_PAGE_GB (default 6).
#   scripts/chpc/region.sh status NAME...
#       Quads that are not exported yet, your jobs, the disk use and the
#       years of every quad, and the model that mapped it.
#
# NAME defaults to indiana_block.
#
# The block step prints the free space first. A quad of 7.5 minutes takes
# about 12 GB (mosaics and height models of all years), which stay on disk
# because a later remap reads the mosaics.
#
# Settings, all in scripts/chpc/env.sh: CSDV_SBATCH_CPU and CSDV_GPU_POOLS,
# as for scripts/chpc/round2.sh.
#   CSDV_REGION_CHECKPOINT  the pair model that maps every quad (default:
#       reserved_r2_s0, the model of test B of round 2 with seed 0).
#   CSDV_REGION_STORE  a folder on a larger volume. When set, the mosaics and
#       height models of quads that are not on disk yet are kept there, and
#       $CSDV_DATA_ROOT/naip/<unit> and naip_chm/<unit> are links to them.
#       Every other path stays as it is.
#   CSDV_REGION_QUEUE_ARGS  replaces the queue options, which default to
#       "--worker-mem 48G --worker-hours 8": the height model of a whole
#       quad needs 21 to 34 GB, and a part of 40 quads about 26 GPU hours.
#
# A quad that stops is named by `status`, with the log to read.

set -euo pipefail

STEP="${1:-}"
[[ -n "$STEP" ]] && shift
NAMES=() DRY=() CKPT_ARG="" TITLE="" PAGE=""
while (($#)); do
    case "$1" in
        --dry-run) DRY=(--dry-run); shift ;;
        --checkpoint) CKPT_ARG="${2:?--checkpoint needs a run or a path}"; shift 2 ;;
        --page) PAGE="${2:?--page needs a folder name}"; shift 2 ;;
        --title) TITLE="${2:?--title needs a text}"; shift 2 ;;
        -h | --help) sed -n '2,48p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) NAMES+=("$1"); shift ;;
    esac
done
((${#NAMES[@]})) || NAMES=(indiana_block)

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_CPU:?CSDV_SBATCH_CPU is not set in scripts/chpc/env.sh}"
cd "$CSDV_REPO"
mkdir -p "$CSDV_LOG_DIR"

# A name with a batch file is itself. A name without one is its parts.
BATCHES=()
for name in "${NAMES[@]}"; do
    if [[ -f "config/grid/$name.csv" ]]; then
        BATCHES+=("$name")
    else
        mapfile -t PARTS < <(ls config/grid/"$name"_[0-9]*.csv 2>/dev/null | sort -V)
        ((${#PARTS[@]})) || { echo "ERROR: no batch file config/grid/$name.csv" >&2; exit 1; }
        for part in "${PARTS[@]}"; do BATCHES+=("$(basename "$part" .csv)"); done
    fi
done

CKPT="${CKPT_ARG:-${CSDV_REGION_CHECKPOINT:-reserved_r2_s0}}"
[[ "$CKPT" == */* ]] || CKPT="$CSDV_RESULTS_ROOT/pairmodel/$CKPT/best.pt"
read -r -a QUEUE_ARGS <<<"${CSDV_REGION_QUEUE_ARGS---worker-mem 48G --worker-hours 8}"

# units NAME: the unit IDs of a batch, one a line.
units() {
    python - "config/grid/$1.csv" <<'PY'
import sys
from pathlib import Path

from csdv_core.grid.units import read_batch

for unit in read_batch(Path(sys.argv[1])):
    print(unit.unit_id)
PY
}

# link_store NAME: keep the imagery of new quads on the larger volume.
link_store() {
    local unit kind n=0
    [[ -n "${CSDV_REGION_STORE:-}" ]] || return 0
    for unit in $(units "$1"); do
        for kind in naip naip_chm; do
            [[ -e "$CSDV_DATA_ROOT/$kind/$unit" ]] && continue
            n=$((n + 1))
            ((${#DRY[@]})) && continue
            mkdir -p "$CSDV_REGION_STORE/$kind/$unit" "$CSDV_DATA_ROOT/$kind"
            ln -s "$CSDV_REGION_STORE/$kind/$unit" "$CSDV_DATA_ROOT/$kind/$unit"
        done
    done
    echo "$1: $n folders $( ((${#DRY[@]})) && echo "would be" || echo "are") linked to $CSDV_REGION_STORE"
}

case "$STEP" in
    block)
        [[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
        echo "== Free space"
        df -h "$CSDV_DATA_ROOT" ${CSDV_REGION_STORE:+"$CSDV_REGION_STORE"} | tail -n +1
        TOTAL=0
        for name in "${BATCHES[@]}"; do TOTAL=$((TOTAL + $(units "$name" | wc -l))); done
        echo "${#BATCHES[@]} batches, $TOTAL quads, about $((TOTAL * 12)) GB when all are new."
        echo "Model: $CKPT"
        echo
        for name in "${BATCHES[@]}"; do
            echo "== $name"
            link_store "$name"
            csdv grid queue --batch "config/grid/$name.csv" --name "$name" \
                --checkpoint "$CKPT" ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} \
                ${DRY[@]+"${DRY[@]}"}
        done
        cat <<MSG

$( ((${#DRY[@]})) && echo "Nothing was submitted (dry run)." || echo "Submitted.") No paid call is in it. In the morning:
  scripts/chpc/region.sh status ${NAMES[*]}    # quads exported, disk use, years
  scripts/chpc/region.sh block ${NAMES[*]}     # again, if anything is left
MSG
        ;;
    remap)
        [[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
        echo "Model: $CKPT"
        for name in "${BATCHES[@]}"; do
            QUEUE="$CSDV_GRID_DIR/queues/$name"
            ROWS="$QUEUE/remap_$(date +%Y%m%dT%H%M%S).csv"
            # Only quads with a finished detection: the rest belong to `block`.
            READY=()
            for unit in $(units "$name"); do
                [[ -f "$CSDV_RESULTS_ROOT/disturbance/$unit/default/summary.json" ]] &&
                    READY+=("$unit")
            done
            echo "== $name: ${#READY[@]} quads to map again"
            ((${#READY[@]})) || continue
            if ((${#DRY[@]})); then
                printf '  %s\n' "${READY[@]}" | head -5
                continue
            fi
            mkdir -p "$QUEUE"
            { echo unit; printf '%s\n' "${READY[@]}"; } >"$ROWS"
            # shellcheck disable=SC2086
            sbatch --parsable $CSDV_SBATCH_CPU --job-name="csdv_remap_$name" \
                --array="1-${#READY[@]}%20" --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
                scripts/slurm/run_grid_finish.sbatch "$QUEUE" "$ROWS" default \
                --checkpoint "$CKPT" --remap --no-quicklooks
        done
        ;;
    page)
        ROOT="${CSDV_REGION_PAGE_DIR:-${CSDV_PUBLIC_HTML:?set CSDV_PUBLIC_HTML or CSDV_REGION_PAGE_DIR in scripts/chpc/env.sh}/csdv_blocks}"
        PAGE="${PAGE:-${NAMES[0]}}"
        OUT="$ROOT/$PAGE"
        read -r -a CPU <<<"$CSDV_SBATCH_CPU"
        CMD=(sbatch --parsable "${CPU[@]}" --job-name="csdv_page_$PAGE"
            --cpus-per-task=8 --mem=32G --time=08:00:00
            --output="$CSDV_LOG_DIR/%x-%j.out"
            scripts/slurm/run_csdv_cpu.sbatch grid webmap --out "$OUT"
            --export-dir "$CSDV_EXPORT_DIR" --data-root "$CSDV_DATA_ROOT"
            --title "${TITLE:-CSDV disturbance map: $PAGE}" --workers 8
            --budget-gb "${CSDV_REGION_PAGE_GB:-6}"
            --tmp-dir "$CSDV_EXPORT_DIR/checks/blockmap_tmp/$PAGE")
        for name in "${BATCHES[@]}"; do CMD+=(--batch "config/grid/$name.csv"); done
        if ((${#DRY[@]})); then
            printf 'would submit:'; printf ' %q' "${CMD[@]}"; echo
        else
            mkdir -p "$OUT"
            echo "page job: $("${CMD[@]}")"
        fi
        echo "The page goes to $OUT."
        echo "The last lines of the log give its size:" \
            "tail -n 3 $CSDV_LOG_DIR/csdv_page_$PAGE-*.out"
        ;;
    status)
        for name in "${BATCHES[@]}"; do
            echo "== Quads of $name that are not exported yet"
            csdv grid queue-status --name "$name" --problems || true
            mkdir -p "$CSDV_EXPORT_DIR/checks/$name"
            csdv grid queue-status --name "$name" \
                >"$CSDV_EXPORT_DIR/checks/$name/queue_status.txt" 2>&1 || true
            echo
            echo "== Disk use (mosaics, height models), years and model per quad"
            {
                echo "unit naip naip_chm model years"
                for unit in $(units "$name"); do
                    a="$(du -shL "$CSDV_DATA_ROOT/naip/$unit" 2>/dev/null | cut -f1)"
                    b="$(du -shL "$CSDV_DATA_ROOT/naip_chm/$unit" 2>/dev/null | cut -f1)"
                    m="$(cat "$CSDV_RESULTS_ROOT/disturbance/$unit/default/model/checkpoint.txt" 2>/dev/null || true)"
                    y="$(ls "$CSDV_DATA_ROOT/naip_chm/$unit" 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
                    echo "$unit ${a:--} ${b:--} ${m:--} ${y:--}"
                done
            } | tee "$CSDV_EXPORT_DIR/checks/$name/disk_use.txt" | column -t
            echo
            echo "== Run times of the finish tasks of $name (detection, map, export)"
            sacct -u "$USER" --name="csdv_fin_$name,csdv_remap_$name" \
                -S "$(date -d '3 days ago' +%F)" -X -n \
                -o "JobID%18,JobName%24,State%12,Elapsed%10" 2>/dev/null | tail -n 12 || true
            echo
        done
        echo "== Your jobs in the queue: $(squeue -h -u "$USER" | wc -l)"
        squeue -h -u "$USER" -o "%.12i %.24j %.9T %.10M %R" | sort -k2 |
            uniq -c -f1 | head -20 || true
        ;;
    *)
        echo "usage: $0 block|remap|page|status NAME... [--checkpoint RUN] [--page FOLDER] [--title TEXT] [--dry-run]" >&2
        exit 2
        ;;
esac
