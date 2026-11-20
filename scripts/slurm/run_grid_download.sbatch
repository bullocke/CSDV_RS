#!/bin/bash
#SBATCH --job-name=csdv_qdl
#SBATCH --partition=notchpeak-dtn
#SBATCH --qos=notchpeak-dtn
#SBATCH --account=dtn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#
# NAIP download of one unit of a grid queue, on a data transfer node. Array
# task i reads line i + 1 of the row file (unit,west,south,east,north).
# Submitted by `csdv grid queue`. Further arguments go to
# `csdv download naip-auto` (--start-year, --min-coverage, --min-years).
#
# The task plans the unit's years itself from the two catalogues, so nothing
# is planned on the login node. With --min-years a year that a catalogue
# lists and cannot deliver is dropped when enough years arrive, in place of
# failing the unit. On success the task leaves QUEUE_DIR/downloaded/<unit>,
# which the finish job asks for. A unit that has the marker is skipped.
#
# A 4.8 km box holds about 85 MB per year in memory at 0.6 m and a few
# hundred at 0.3 m. The 32G request is for whole quads.

set -euo pipefail

QUEUE_DIR="${1:?usage: sbatch --array=1-N $0 QUEUE_DIR ROWS.csv [naip-auto options...]}"
ROWS="${2:?usage: sbatch --array=1-N $0 QUEUE_DIR ROWS.csv [naip-auto options...]}"
shift 2

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROWS" | tr -d '\r')
IFS=',' read -r UNIT WEST SOUTH EAST NORTH <<<"$ROW"
: "${NORTH:?no row $SLURM_ARRAY_TASK_ID in $ROWS}"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started=$(date -Is)"
echo "unit=$UNIT box=$WEST $SOUTH $EAST $NORTH args: $*"

MARK="$QUEUE_DIR/downloaded/$UNIT"
if [[ -e "$MARK" ]]; then
    echo "Already downloaded ($MARK), skipping."
    exit 0
fi

activate_csdv
csdv download naip-auto --site "$UNIT" --bbox "$WEST" "$SOUTH" "$EAST" "$NORTH" "$@"

mkdir -p "$QUEUE_DIR/downloaded"
ls "$CSDV_DATA_ROOT/naip/$UNIT" | tr '\n' ' ' >"$MARK"
echo "years on disk: $(cat "$MARK")"
echo "done $(date -Is)"
