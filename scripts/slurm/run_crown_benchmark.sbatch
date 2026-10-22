#!/bin/bash
#SBATCH --job-name=csdv_crown_bench
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --account=wangj-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#
# National crown cost benchmark: one array task runs both crown engines over
# every 2048 px block of one full NAIP-CHM tile, on one core, and records
# per-block time, peak memory, crown counts and output bytes per crown. One
# core per task is deliberate. A production run would place one tile per core,
# so this is the unit the national cost model needs.
#
# Reads tiles fetched by run_naipchm_fetch_dtn.sbatch (compute nodes have no
# internet). The manifest is national_bench_manifest.csv from
# analysis/NAIP_National_Crown_AGB_Map/04_chpc_manifests.py: a header row, then
# tile_file,tile_id,scale_factor,stratum,cell. Array task i reads line i + 1.
#
# The env needs R with lidR. If csdv-gpu predates r-lidr in environment.yml:
#   mamba install -n csdv-gpu -c conda-forge r-base r-lidr r-terra r-sf
#
# Submit after `source scripts/chpc/env.sh`, with N = manifest rows minus the header:
#   FETCH=/scratch/general/vast/$USER/naip_chm_national
#   sbatch --array=1-N --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
#       scripts/slurm/run_crown_benchmark.sbatch \
#       "$CSDV_MANIFEST_DIR/national_bench_manifest.csv" "$FETCH/tiles"
# To time the general partition too, resubmit the same array with
#   -p notchpeak -A <general account> --qos notchpeak
# and a different OUT_TAG (third argument), e.g. notchpeak.
#
# Output: $CSDV_RESULTS_ROOT/national_crowns/benchmark/<OUT_TAG>/<tile>_*.

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=1-N $0 MANIFEST.csv TILE_DIR [OUT_TAG]}"
TILE_DIR="${2:?usage: sbatch --array=1-N $0 MANIFEST.csv TILE_DIR [OUT_TAG]}"
OUT_TAG="${3:-${SLURM_JOB_PARTITION:-local}}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=',' read -r TILE_FILE TILE_ID SCALE_FACTOR STRATUM _ <<<"$ROW"
: "${STRATUM:?incomplete manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST}"
DEST="$CSDV_RESULTS_ROOT/national_crowns/benchmark/$OUT_TAG"

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started=$(date -Is)"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
echo "tile=$TILE_ID stratum=$STRATUM"

if [[ -e "$DEST/${TILE_ID}_summary.json" ]]; then
    echo "Summary already in $DEST, skipping."
    exit 0
fi

# Single-threaded numerics, so one task really is one core.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=256

activate_csdv
Rscript -e 'suppressPackageStartupMessages(library(lidR)); cat("lidR", as.character(packageVersion("lidR")), "\n")'
scratch_init
stage_in "$TILE_DIR/$TILE_FILE" "$NODE_SCRATCH/tile"
mkdir -p "$NODE_SCRATCH/out"
register_stage_out "$NODE_SCRATCH/out" "$DEST"

/usr/bin/time -v csdv benchmark-crowns \
    --tile "$NODE_SCRATCH/tile/$TILE_FILE" \
    --tile-id "$TILE_ID" \
    --out-dir "$NODE_SCRATCH/out" \
    --engines watershed,lidr \
    --block-px 2048 \
    --scale-factor "$SCALE_FACTOR"

ls -l "$NODE_SCRATCH/out"
