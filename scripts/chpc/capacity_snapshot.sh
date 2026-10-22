#!/usr/bin/env bash
# Record the CHPC capacity the national crown estimate can draw on.
#
# Run on a login node after `source scripts/chpc/env.sh`:
#   scripts/chpc/capacity_snapshot.sh > "$CSDV_STORE/logs/capacity_$(date +%F).txt"
#
# Prints allocations, partition sizes (nodes, cores and memory per node,
# walltime limit), current idle cores, and free space on the group store and
# VAST scratch. Nothing is changed.

set -uo pipefail

PARTITIONS="${PARTITIONS:-wangj-np,notchpeak,notchpeak-shared-short,kingspeak,lonepeak,granite,granite-freecycle,notchpeak-freecycle,notchpeak-dtn}"

echo "== date: $(date -Is)  host: $(hostname)  user: $USER"

echo
echo "== myallocation"
if command -v myallocation >/dev/null 2>&1; then
    myallocation
else
    echo "(myallocation not on PATH)"
fi

echo
echo "== partitions: nodes, CPUs/node, memory/node (MB), time limit"
sinfo -p "$PARTITIONS" -o "%-24P %6D %8c %10m %12l %20G" 2>/dev/null | sort -u

echo
echo "== cores allocated/idle/other/total by partition"
sinfo -p "$PARTITIONS" -o "%-24P %C" 2>/dev/null

echo
echo "== my running and pending jobs"
squeue -u "$USER" -o "%.12i %.20P %.20j %.8T %.10M %.6C %R" 2>/dev/null | head -40

echo
echo "== storage"
for d in "${CSDV_STORE:-}" "/scratch/general/vast/$USER" "/scratch/general/nfs1/$USER"; do
    [[ -n "$d" && -d "$d" ]] || continue
    df -h "$d" | tail -1 | awk -v d="$d" '{print d": size "$2", used "$3", avail "$4}'
done
if command -v mydu >/dev/null 2>&1; then
    echo "-- mydu"
    mydu 2>/dev/null | head -20
fi
