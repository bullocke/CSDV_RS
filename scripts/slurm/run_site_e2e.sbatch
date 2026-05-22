#!/bin/bash
#SBATCH --job-name=csdv_e2e
#SBATCH --account=dycelab
#SBATCH --partition=dycelab-shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/slurm/%x-%j.out
#SBATCH --error=results/logs/slurm/%x-%j.err
#
# Submit:
#   sbatch scripts/slurm/run_site_e2e.sbatch SCBI 2014,2018 50
#
# Required environment (set in your shell before sbatch, or in this file):
#   CSDV_DATA_ROOT, CSDV_RESULTS_ROOT, NAIPCHM_REPO_DIR
#
# Loads the project env (micromamba) then dispatches to run_site_e2e.sh.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: sbatch $0 SITE YEARS_CSV WINDOW_M [extra args]" >&2
    exit 2
fi

mkdir -p results/logs/slurm

# Activate the project micromamba env. Adjust MAMBA_ROOT_PREFIX if needed.
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate CSDV
fi

echo "host=$(hostname) job=$SLURM_JOB_ID started=$(date -Is)"
echo "args: $*"
csdv --version

exec scripts/run_site_e2e.sh "$@"
