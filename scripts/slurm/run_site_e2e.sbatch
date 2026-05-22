#!/bin/bash
#SBATCH --requeue                             
#SBATCH --account=wangj-np
#SBATCH --partition=wangj-np
#SBATCH --qos=wangj-np
#SBATCH --nodes=1                              
#SBATCH --ntasks-per-node=1                     
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                              
#SBATCH --time=6:00:00
#SBATCH --job-name=csdv_e2e
#SBATCH --output=results/logs/slurm/%x-%j.out
#SBATCH --error=results/logs/slurm/%x-%j.err

#
# Submit:
#   sbatch scripts/slurm/run_site_e2e.sbatch SCBI 2014,2018 50
#
# Required environment (set in your shell before sbatch, or in this file):
#   CSDV_DATA_ROOT, CSDV_RESULTS_ROOT, NAIPCHM_REPO_DIR
#
# Loads the project env (mamba) then dispatches to run_site_e2e.sh.

set -euo pipefail

# Move into the repo before doing anything else so relative paths work.
cd /uufs/chpc.utah.edu/common/home/dycelab/users/eric/code/CSDV/CSDV_RS

if [[ $# -lt 3 ]]; then
    echo "usage: sbatch $0 SITE YEARS_CSV WINDOW_M [extra args]" >&2
    exit 2
fi

mkdir -p results/logs/slurm

# Activate the project mamba env. SLURM batch shells do not source
# ~/.bashrc by default, so we source the shell hook ourselves before
# calling `mamba activate`.
#
# Disable `nounset` around activation: conda/mamba activation scripts
# (e.g. libpdal-core) reference variables that may be unset on first
# activation, and would otherwise trip `set -u`.
if ! command -v mamba >/dev/null 2>&1; then
    echo "ERROR: mamba not on PATH inside the batch job." >&2
    echo "Add 'module load miniconda3' or similar to this sbatch." >&2
    exit 1
fi
set +u
eval "$(mamba shell hook --shell bash)"
mamba activate csdv
set -u

# Forward the CSDV_* env vars set in ~/.bashrc into the job. SLURM
# --export=ALL is the default, but be explicit so a misconfigured
# submission environment fails loudly here instead of silently writing
# to the wrong path.
: "${CSDV_DATA_ROOT:?CSDV_DATA_ROOT not set}"
: "${CSDV_RESULTS_ROOT:?CSDV_RESULTS_ROOT not set}"
: "${NAIPCHM_REPO_DIR:?NAIPCHM_REPO_DIR not set (must point to a clone of smorf-ntsg/naip-chm)}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-N/A} started=$(date -Is)"
echo "args: $*"
echo "CSDV_DATA_ROOT=$CSDV_DATA_ROOT"
echo "CSDV_RESULTS_ROOT=$CSDV_RESULTS_ROOT"
echo "NAIPCHM_REPO_DIR=$NAIPCHM_REPO_DIR"
csdv --version

exec scripts/run_site_e2e.sh "$@"
