#!/bin/bash
#SBATCH --job-name=csdv_disturbance
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Forest disturbance detection for one site from its NAIP-CHM years
# (`csdv detect-disturbance`), CPU only, on node-local scratch. The first
# argument is the site. Every further argument is passed to
# `csdv detect-disturbance` unchanged, except --output-dir, which sets where
# the run folder is copied at the end.
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_detect_disturbance.sbatch SITE [csdv options...]
#
# Examples:
#   # Default run (best tested method) into $CSDV_RESULTS_ROOT/disturbance/SITE
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_detect_disturbance.sbatch ElkinsvilleNE
#
#   # NDVI input (also stages NAIP), single-pass window, rule D5
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_detect_disturbance.sbatch ElkinsvilleNE \
#       --input d_ndvi --method local --rule D5 \
#       --output-dir "$CSDV_RESULTS_ROOT/disturbance/ElkinsvilleNE_ndvi_D5"
#
#   # Second run with other parameters: a new output folder is required
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_detect_disturbance.sbatch ElkinsvilleNE \
#       --rule D6s --mmu-ha 1.0 \
#       --output-dir "$CSDV_RESULTS_ROOT/disturbance/ElkinsvilleNE_D6s_1ha"
#
# Freecycle fallback when wangj-np is busy (preemptable, so requeue):
#   sbatch --partition=notchpeak-freecycle --qos=notchpeak-freecycle \
#       --account=wangj --requeue --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_detect_disturbance.sbatch SITE
#
# Steps: refuse to start if the destination folder exists (the run never
# overwrites), stage naip_chm/SITE (and naip/SITE when the run reads NAIP:
# `d_ndvi` input or --naip-backdrop) to /scratch/local, run detection with
# CSDV_DATA_ROOT pointing at scratch, copy the run folder to the destination.
# A failed run's folder and log go to $CSDV_STORE/failed/<job id>/out/.
#
# Resources: on WSL the ElkinsvilleNE quarter quad (3,800 ha, six years,
# 0.6 m float32 CHMs) took 65 s wall time with a peak resident memory of
# 0.65 GB for the default run (layers 16 s, 0.6 m refinement 32 s, figures
# 8 s), and 62 s and 0.46 GB for the NDVI input. Memory grows with the site
# area (the cell layers of every year are held in memory and the CHM is read
# in strips). The 24 GB and 3 h requested here leave ample room for sites tens
# of times larger. Lower them for quarter-quad sites to start sooner.
#
# NAIP staging only looks at the command line. A --config YAML that sets
# inputs.input: d_ndvi or figures.naip_backdrop: true needs the matching
# option on the command line too (or NAIP will be missing on scratch).

set -euo pipefail

SITE="${1:?usage: sbatch $0 SITE [csdv detect-disturbance options...]}"
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
DEST="${DEST:-$CSDV_RESULTS_ROOT/disturbance/$SITE}"

# NAIP is read only for the NDVI input or the RGB backdrop.
NEED_NAIP=0
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
    case "$arg" in
        d_ndvi | --input=d_ndvi | --naip-backdrop) NEED_NAIP=1 ;;
    esac
done

CHM_SRC="$CSDV_DATA_ROOT/naip_chm/$SITE"
NAIP_SRC="$CSDV_DATA_ROOT/naip/$SITE"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "site=$SITE dest=$DEST need_naip=$NEED_NAIP"
echo "args: ${ARGS[*]-}"

if [[ -e "$DEST" ]]; then
    echo "ERROR: output folder already exists: $DEST" >&2
    echo "detect-disturbance never overwrites a run. Pass a new folder with" \
        "--output-dir, or move the old run away." >&2
    exit 1
fi
if [[ ! -d "$CHM_SRC" ]]; then
    echo "ERROR: no NAIP-CHM folder for $SITE: $CHM_SRC" >&2
    exit 1
fi
if ((NEED_NAIP)) && [[ ! -d "$NAIP_SRC" ]]; then
    echo "ERROR: this run reads NAIP, but $NAIP_SRC does not exist" >&2
    exit 1
fi

# Keep GDAL's block cache bounded (MB) and plot without a display.
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-2048}"
export MPLBACKEND=Agg

activate_csdv
scratch_init
stage_in "$CHM_SRC" "$NODE_SCRATCH/data/naip_chm/$SITE"
if ((NEED_NAIP)); then
    stage_in "$NAIP_SRC" "$NODE_SCRATCH/data/naip/$SITE"
fi
# The output folder is not created here: detect-disturbance claims it and
# refuses a folder that already exists.
register_stage_out "$NODE_SCRATCH/out" "$DEST"

CSDV_DATA_ROOT="$NODE_SCRATCH/data" csdv detect-disturbance \
    --site "$SITE" \
    --output-dir "$NODE_SCRATCH/out" \
    ${ARGS[@]+"${ARGS[@]}"}

ls -l "$NODE_SCRATCH/out"
