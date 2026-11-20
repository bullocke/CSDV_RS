#!/bin/bash
#SBATCH --job-name=csdv_cmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Run one `csdv` command on a CPU node. For small steps that read data on
# CHPC and have no job script of their own, such as
# `csdv grid compare-units`. This script names no allocation: pass
# $CSDV_SBATCH_CPU at submission.
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_csdv_cpu.sbatch grid compare-units \
#       --unit p_pilot_wv --against q3708104 --out-dir "$CSDV_EXPORT_DIR/checks/round2_pilot"
#
# Inputs are read where they are and outputs written where the command says.
# No node scratch, so use it for steps whose reading is light.

set -euo pipefail

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-1024}"
export MPLBACKEND=Agg

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "csdv $*"
activate_csdv
csdv "$@"
echo "done $(date -Is)"
