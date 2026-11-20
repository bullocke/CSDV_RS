#!/bin/bash
#SBATCH --job-name=csdv_chmq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#
# One worker of a grid queue: NAIP-CHM inference for one unit-year after
# another, on one GPU. Submitted by `csdv grid queue`, several at a time and
# on several GPU pools. This script names no allocation: the pool (account,
# partition, QOS, --gres=gpu:1) comes from $CSDV_GPU_POOLS at submission.
#
# Usage (after `source scripts/chpc/env.sh`), for a worker by hand:
#   sbatch <one pool of CSDV_GPU_POOLS> --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_chm_worker.sbatch "$CSDV_GRID_DIR/queues/NAME"
#
# What a worker does. It checks its GPU, stages the conditioning rasters to
# node scratch once, then looks through the queue's units for a year that
# has a NAIP mosaic and no CHM. Only units whose download has left its
# marker QUEUE_DIR/downloaded/<unit> are looked at: a download that is still
# running may be writing a mosaic. It claims that unit-year by creating the
# folder QUEUE_DIR/claims/<unit>_<year>, which only one job can do, runs the
# inference on scratch, copies the CHM to
# $CSDV_DATA_ROOT/naip_chm/<unit>/<year>/ and drops the claim. It goes on
# until nothing is left or its walltime is near (CSDV_WORKER_SECONDS, set at
# submission, less a quarter of an hour).
#
# There is no list of rows. What is left is read from disk on every pass,
# so a worker that starts late simply finds less to do, and one that finds
# nothing exits at once. Workers on several pools can therefore be queued at
# the same time, and whichever pool frees a GPU first does the work.
#
# Preemption and failures. A preempted worker is cancelled, not requeued.
# Its claim is dropped by the exit trap, or, after a hard kill, goes stale
# after 45 minutes and is taken over by another worker. A unit-year whose
# inference fails is noted in QUEUE_DIR/failed/<unit>_<year> and tried once
# more by another worker. After two failures it is left out, and the finish
# job runs the unit without that year. Rows left when every worker has ended are picked up by
# giving the `csdv grid queue` command again.
#
# The last worker to find nothing left cancels its siblings that are still
# pending, so that the finish array does not wait on a pool that stays busy.
# QUEUE_DIR/workers.txt names them, one `job [cluster]` per line.
#
# A worker on another Slurm cluster (a pool with --clusters=NAME) cannot
# depend on the download job, because job IDs belong to one cluster. It is
# submitted with CSDV_WORKER_WAIT=1 and starts at once. When it finds nothing
# to do while some unit still has no download marker, it looks again every
# minute. It gives up when no marker has appeared and no work was found for
# CSDV_WORKER_IDLE_S seconds (900), which is also what ends it when a
# download has failed. What it leaves is picked up by giving the queue
# command again. Without the variable a worker never waits.
#
# A faulty card fails the job before any work (seen on a guest node). Give
# the queue command again with --exclude-nodes <node>.

set -euo pipefail

QUEUE_DIR="${1:?usage: sbatch <gpu pool options> $0 QUEUE_DIR}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

UNITS="$QUEUE_DIR/units.csv"
CLAIMS="$QUEUE_DIR/claims"
FAILED="$QUEUE_DIR/failed"
STALE_S="${CSDV_CLAIM_STALE_S:-2700}"
BUDGET_S="${CSDV_WORKER_SECONDS:-14400}"
MARGIN_S="${CSDV_WORKER_MARGIN_S:-900}"
WAIT="${CSDV_WORKER_WAIT:-0}"
IDLE_S="${CSDV_WORKER_IDLE_S:-900}"
POLL_S="${CSDV_WORKER_POLL_S:-60}"
MAX_FAILURES=2

[[ -f "$UNITS" ]] || { echo "ERROR: no $UNITS" >&2; exit 1; }
mkdir -p "$CLAIMS" "$FAILED"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "queue=$QUEUE_DIR walltime=${BUDGET_S}s wait_for_downloads=$WAIT"

activate_csdv
nvidia-smi -L || echo "WARNING: nvidia-smi failed on $(hostname -s)" >&2
# A small convolution, not only an allocation: a card that this build of
# PyTorch does not support passes is_available() and fails at its first
# kernel. Failing here costs nothing. Failing later would count against the
# unit-years the worker had claimed.
if ! python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
x = torch.randn(2, 4, 64, 64, device='cuda'); \
y = torch.nn.functional.conv2d(x, torch.randn(8, 4, 3, 3, device='cuda')); \
assert torch.isfinite(y).all().item(), 'convolution gave no finite result'; \
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"; then
    echo "ERROR: no usable GPU on $(hostname -s). Give the queue command again" \
        "with --exclude-nodes $(hostname -s), or take its pool out of" \
        "CSDV_GPU_POOLS if the card is too old for this build of PyTorch." >&2
    exit 1
fi

scratch_init
MINE=""
release() {
    [[ -n "$MINE" ]] && rm -rf "$MINE"
    MINE=""
    return 0
}
# Drop the claim first, then let scratch.sh clean up with the job's own code.
trap 'rc=$?; release; (exit $rc); _scratch_exit' EXIT

# claim KEY: succeed when this job now owns QUEUE_DIR/claims/KEY.
claim() {
    local dir="$CLAIMS/$1"
    if ! mkdir "$dir" 2>/dev/null; then
        # Someone holds it. A claim older than STALE_S belongs to a job that
        # was killed without a chance to drop it. Moving it away is the step
        # only one job can win.
        local born
        born=$(stat -c %Y "$dir" 2>/dev/null || echo "")
        [[ -n "$born" ]] || return 1
        (($(date +%s) - born > STALE_S)) || return 1
        mv "$dir" "$dir.stale.${SLURM_JOB_ID:-$$}" 2>/dev/null || return 1
        rm -rf "$dir.stale.${SLURM_JOB_ID:-$$}"
        echo "took over a stale claim: $1"
        mkdir "$dir" 2>/dev/null || return 1
    fi
    echo "${SLURM_JOB_ID:-none} $(hostname -s) $(date -Is)" >"$dir/owner"
    MINE="$dir"
    return 0
}

failures() {
    local file="$FAILED/$1"
    [[ -f "$file" ]] && wc -l <"$file" || echo 0
}

# infer NAIP_TIF DEST: one CHM, made on scratch and copied out when whole.
infer() {
    local naip="$1" dest="$2"
    rm -rf "$NODE_SCRATCH/naip" "$NODE_SCRATCH/chm"
    mkdir -p "$NODE_SCRATCH/naip" "$NODE_SCRATCH/chm"
    rsync -a "$naip" "$NODE_SCRATCH/naip/"
    csdv chm-inference \
        --naip-quad "$NODE_SCRATCH/naip/$(basename "$naip")" \
        --output-dir "$NODE_SCRATCH/chm" \
        --conditioning-dir "$NODE_SCRATCH/conditioning"
    compgen -G "$NODE_SCRATCH/chm/*chm*.tif" >/dev/null
    mkdir -p "$dest"
    # The CHM itself goes last: its presence is what marks the year as done.
    rsync -a --exclude='*.tif' "$NODE_SCRATCH/chm/" "$dest/"
    rsync -a "$NODE_SCRATCH/chm/" "$dest/"
}

stage_in "$CSDV_CONDITIONING_DIR" "$NODE_SCRATCH/conditioning"

declare -A TRIED=()
N_DONE=0
N_FAILED=0
OUT_OF_TIME=0
GAVE_UP=0
N_MARKED_BEFORE=-1
LAST_NEWS=$SECONDS
while :; do
    FOUND=0
    N_MARKED=0
    N_UNMARKED=0
    while IFS=, read -r UNIT _; do
        [[ -z "$UNIT" || "$UNIT" == "unit" ]] && continue
        # No marker: the unit's download has not finished (or has failed).
        if [[ ! -e "$QUEUE_DIR/downloaded/$UNIT" ]]; then
            N_UNMARKED=$((N_UNMARKED + 1))
            continue
        fi
        N_MARKED=$((N_MARKED + 1))
        for YEAR_DIR in "$CSDV_DATA_ROOT/naip/$UNIT"/*/; do
            [[ -d "$YEAR_DIR" ]] || continue
            YEAR="$(basename "$YEAR_DIR")"
            mapfile -t TIFS < <(find "$YEAR_DIR" -maxdepth 1 -name '*.tif' | sort)
            ((${#TIFS[@]} == 1)) || continue
            NAIP="${TIFS[0]}"
            DEST="$CSDV_DATA_ROOT/naip_chm/$UNIT/$YEAR"
            compgen -G "$DEST/$(basename "$NAIP" .tif)*chm*.tif" >/dev/null && continue
            KEY="${UNIT}_${YEAR}"
            (($(failures "$KEY") >= MAX_FAILURES)) && continue
            # The second try belongs to another worker, on another card.
            [[ -n "${TRIED[$KEY]:-}" ]] && continue
            if ((SECONDS > BUDGET_S - MARGIN_S)); then
                OUT_OF_TIME=1
                break 2
            fi
            claim "$KEY" || continue
            FOUND=1
            STARTED=$SECONDS
            echo "--- $KEY  $(date -Is)"
            # In a subshell, so that a failed inference ends this unit-year
            # and not the worker. The subshell is not the condition of an
            # `if`: bash ignores `set -e` inside a command that is tested.
            set +e
            (
                set -e
                infer "$NAIP" "$DEST"
            )
            RC=$?
            set -e
            if ((RC == 0)); then
                N_DONE=$((N_DONE + 1))
                echo "$KEY done in $((SECONDS - STARTED)) s"
            else
                N_FAILED=$((N_FAILED + 1))
                TRIED[$KEY]=1
                echo "${SLURM_JOB_ID:-none} $(hostname -s) $(date -Is)" >>"$FAILED/$KEY"
                echo "WARNING: $KEY failed ($(failures "$KEY") of $MAX_FAILURES)" >&2
            fi
            release
        done
    done <"$UNITS"
    ((OUT_OF_TIME)) && break
    if ((FOUND)); then
        LAST_NEWS=$SECONDS
        continue
    fi
    # A pass that claimed nothing found every unit-year done, failed twice or
    # held by another worker. That ends the worker, unless it was told to
    # wait for downloads and some are still to come.
    ((WAIT == 1 && N_UNMARKED > 0)) || break
    if ((N_MARKED != N_MARKED_BEFORE)); then
        N_MARKED_BEFORE=$N_MARKED
        LAST_NEWS=$SECONDS
    fi
    if ((SECONDS - LAST_NEWS >= IDLE_S)); then
        GAVE_UP=1
        echo "no download finished and nothing to do for $((SECONDS - LAST_NEWS)) s:" \
            "$N_UNMARKED units have no marker in $QUEUE_DIR/downloaded"
        break
    fi
    if ((SECONDS + POLL_S > BUDGET_S - MARGIN_S)); then
        OUT_OF_TIME=1
        break
    fi
    sleep "$POLL_S"
done

echo "unit-years done: $N_DONE, failed: $N_FAILED, stopped for time: $OUT_OF_TIME," \
    "gave up waiting for downloads: $GAVE_UP"

# Nothing left and nobody else at work: siblings that are still waiting for
# a GPU have nothing to do. Cancelling them lets the finish array start.
# A worker that gave up waiting leaves its siblings alone: downloads may
# still arrive. Each line of workers.txt is `job [cluster]`. With a cluster
# the question goes to that cluster (-M), whose answer starts with a line
# `CLUSTER: name` even without a header.
if ((!OUT_OF_TIME && !GAVE_UP)) && [[ -z "$(ls -A "$CLAIMS" 2>/dev/null)" && -f "$QUEUE_DIR/workers.txt" ]]; then
    while read -r OTHER WHERE _; do
        [[ -z "$OTHER" ]] && continue
        # The same number on another cluster is another job.
        if [[ "$OTHER" == "${SLURM_JOB_ID:-}" ]] &&
            [[ -z "$WHERE" || -z "${SLURM_CLUSTER_NAME:-}" || "$WHERE" == "$SLURM_CLUSTER_NAME" ]]; then
            continue
        fi
        ON=()
        [[ -n "$WHERE" ]] && ON=(-M "$WHERE")
        STATE=$(squeue -h ${ON[@]+"${ON[@]}"} -j "$OTHER" -o %T 2>/dev/null |
            grep -v '^CLUSTER:' | tr -d '[:space:]' || true)
        if [[ "$STATE" == "PENDING" ]]; then
            scancel ${ON[@]+"${ON[@]}"} "$OTHER" 2>/dev/null &&
                echo "cancelled pending worker $OTHER${WHERE:+ on $WHERE}"
        fi
    done <"$QUEUE_DIR/workers.txt"
fi
exit 0
