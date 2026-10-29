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
#       naip-auto --site NorthMaine --bbox -69.5656 45.6636 -69.3253 45.7501 \
#       --start-year 2010
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
#       conditioning
#
# `naip-auto` takes each year from the Planetary Computer when it has it and
# from the USDA image service otherwise (2024 and later), and skips years that
# cover less than 90% of the box. `naip-pc` and `naip-rest` force one source.
# `csdv grid submit` submits this script for grid units.
#
# Arguments are passed to `csdv download` unchanged.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 SUBCOMMAND [ARGS...]   (e.g. naip-auto, conditioning)" >&2
    exit 2
fi

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
activate_csdv

echo "host=$(hostname) job=$SLURM_JOB_ID started=$(date -Is)"
echo "csdv download $*"
echo "CSDV_DATA_ROOT=$CSDV_DATA_ROOT"

csdv download "$@"

echo "done $(date -Is)"
