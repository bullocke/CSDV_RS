#!/bin/bash
#SBATCH --job-name=csdv_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# The gate of a grid queue that has GPU workers on other Slurm clusters.
# Submitted by `csdv grid queue` when a pool of $CSDV_GPU_POOLS carries
# --clusters=NAME. This script names no allocation: it is submitted with
# $CSDV_SBATCH_CPU.
#
# Why it exists. Job IDs belong to one cluster, so the finish array cannot
# depend on a worker that runs elsewhere. It depends on this job instead,
# which does nothing but ask the other clusters' queues about those workers
# and ends when none of them is pending or running any more.
#
# Usage:
#   sbatch $CSDV_SBATCH_CPU --time=24:00:00 --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_queue_gate.sbatch WORKERS_FILE
#
#   WORKERS_FILE  One `job cluster` per line: the workers to wait for
#                 (QUEUE_DIR/gate_<time>.txt, written at submission).
#
# A worker counts as ended only on a clear answer: the queue returns no
# line for it or a final state, or says the job ID is not known. Any other
# failure of squeue (a controller that does not answer) means it is asked
# again. The gate therefore errs on the side of waiting.
#
# If the gate reaches its walltime (--gate-hours of `csdv grid queue`) the
# finish array starts all the same and refuses the units whose CHMs are
# still to be made. Give the queue command again then.

set -uo pipefail

WORKERS="${1:?usage: sbatch <cpu options> $0 WORKERS_FILE}"
POLL_S="${CSDV_GATE_POLL_S:-60}"

[[ -f "$WORKERS" ]] || { echo "ERROR: no $WORKERS" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "waiting for the workers of $WORKERS:"
cat "$WORKERS"

# ended JOB CLUSTER: succeed when the job is known to be over.
ended() {
    local out rc state
    out=$(squeue -h -M "$2" -j "$1" -o %T 2>&1)
    rc=$?
    if ((rc != 0)); then
        # A job that left the queue some minutes ago is no longer known.
        grep -qi 'invalid job id' <<<"$out"
        return
    fi
    while read -r state _; do
        case "$state" in
            "" | CLUSTER:*) ;;
            COMPLETED | CANCELLED | FAILED | TIMEOUT | NODE_FAIL | PREEMPTED | \
                BOOT_FAIL | DEADLINE | OUT_OF_MEMORY) ;;
            *) return 1 ;;
        esac
    done <<<"$out"
    return 0
}

PASS=0
while :; do
    LEFT=()
    while read -r JOB WHERE _; do
        [[ -z "$JOB" ]] && continue
        if [[ -z "$WHERE" ]]; then
            echo "WARNING: $JOB names no cluster and is not waited for" >&2
            continue
        fi
        ended "$JOB" "$WHERE" || LEFT+=("$JOB@$WHERE")
    done <"$WORKERS"
    ((${#LEFT[@]})) || break
    # A line now and then, so that the log shows what the gate waits for.
    ((PASS % 10 == 0)) && echo "$(date -Is) still in a queue: ${LEFT[*]}"
    PASS=$((PASS + 1))
    sleep "$POLL_S"
done

echo "no worker is left on another cluster. done $(date -Is)"
exit 0
