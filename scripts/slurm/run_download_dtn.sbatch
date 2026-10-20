#!/bin/bash
#SBATCH --job-name=csdv_download
#SBATCH --partition=notchpeak-dtn
#SBATCH --qos=notchpeak-dtn
#SBATCH --account=dtn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Run a `csdv download ...` command on a CHPC data transfer node. Compute and
# GPU nodes have no outbound internet, so every download goes through here.
# Output is written straight to $CSDV_STORE (a transfer, not compute, so no
# node scratch).
#
# Submit from a login node after `source scripts/chpc/env.sh`:
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
#       naip-pc --site SantaClara --bbox -86.3125 39.0625 -86.25 39.125 \
#       --years 2016,2018,2020,2022
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
#       conditioning
#
# Arguments are passed to `csdv download` unchanged.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 SUBCOMMAND [ARGS...]   (e.g. naip-pc, conditioning)" >&2
    exit 2
fi

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
activate_csdv

echo "host=$(hostname) job=$SLURM_JOB_ID started=$(date -Is)"
echo "csdv download $*"
echo "CSDV_DATA_ROOT=$CSDV_DATA_ROOT"

csdv download "$@"

echo "done $(date -Is)"
