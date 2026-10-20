#!/bin/bash
#SBATCH --job-name=csdv_dist_chips
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Inspection chips for a disturbance run (`csdv disturbance-chips`), CPU only.
# The first argument is the run folder written by detect-disturbance. Every
# further argument is passed to `csdv disturbance-chips` unchanged, except
# --output-dir, which sets where the chips folder is copied at the end.
#
# The command draws a stratified sample of detected objects (plus near misses
# removed by the minimum mapping unit) and writes one figure per unit with the
# NAIP RGB and CHM of every year, a contact sheet, index.md and samples.csv.
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_disturbance_chips.sbatch RUN_DIR [csdv options...]
#
# Examples:
#   # Default sample (24 detected objects, 4 near misses) into RUN_DIR/inspection
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_disturbance_chips.sbatch \
#       "$CSDV_RESULTS_ROOT/disturbance/LincolnNH"
#
#   # Larger sample with undisturbed forest cells, into a new folder
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_disturbance_chips.sbatch \
#       "$CSDV_RESULTS_ROOT/disturbance/LincolnNH" \
#       --n-detected 40 --n-no-detection 6 --seed 1 \
#       --output-dir "$CSDV_RESULTS_ROOT/disturbance/LincolnNH/inspection_seed1"
#
# Steps: refuse to start if the destination exists, read small NAIP and CHM
# windows straight from $CSDV_DATA_ROOT (naip/<site>/<year>/ and
# naip_chm/<site>/<year>/, the site comes from the run's run_params.json),
# write the chips to node scratch, copy them to the destination. Only windows
# around the sampled units are read, so the NAIP mosaics are not staged.
#
# Resources: about 2 s per unit and under 1 GB of memory on WSL for the
# ElkinsvilleNE run (six years, 30 units in 53 s).

set -euo pipefail

RUN_DIR="${1:?usage: sbatch $0 RUN_DIR [csdv disturbance-chips options...]}"
shift

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

# Pull --output-dir out of the arguments: it names the persistent destination,
# while csdv itself writes to scratch.
DEST=""
ARGS=()
while (($#)); do
    case "$1" in
        --output-dir)
            DEST="${2:?--output-dir needs a value}"
            shift 2
            ;;
        --output-dir=*)
            DEST="${1#--output-dir=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
DEST="${DEST:-$RUN_DIR/inspection}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "run_dir=$RUN_DIR dest=$DEST"
echo "args: ${ARGS[*]-}"

if [[ ! -f "$RUN_DIR/run_params.json" ]]; then
    echo "ERROR: $RUN_DIR is not a detect-disturbance run folder" >&2
    exit 1
fi
if [[ -e "$DEST" ]]; then
    echo "ERROR: output folder already exists: $DEST" >&2
    echo "Pass a new folder with --output-dir, or move the old one away." >&2
    exit 1
fi

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg

activate_csdv
scratch_init
# The output folder is not created here: disturbance-chips claims it and
# refuses a folder that already exists.
register_stage_out "$NODE_SCRATCH/inspection" "$DEST"

csdv disturbance-chips \
    --run-dir "$RUN_DIR" \
    --output-dir "$NODE_SCRATCH/inspection" \
    ${ARGS[@]+"${ARGS[@]}"}

ls -l "$NODE_SCRATCH/inspection"
