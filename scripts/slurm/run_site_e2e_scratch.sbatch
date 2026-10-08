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
# Optional environment:
#   CSDV_CONDITIONING_DIR - location of the NAIP-CHM conditioning rasters.
#       Defaults to $CSDV_DATA_ROOT/chm_model/conditioning. The e2e script
#       auto-fetches them (~1.65 GB) if absent, but compute nodes have no
#       outbound internet, so an in-job download times out (Errno 110).
#       Pre-fetch them once on a CHPC data transfer node with
#       `csdv download conditioning` (see docs/workflow_chpc.md, step 1a).
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
echo "Original CSDV_DATA_ROOT=$CSDV_DATA_ROOT"
echo "Original CSDV_RESULTS_ROOT=$CSDV_RESULTS_ROOT"
echo "NAIPCHM_REPO_DIR=$NAIPCHM_REPO_DIR"
csdv --version

# =====================================================================
# 1. DEFINE LOCAL JOB SCRATCH DIRECTORIES
# =====================================================================
# CHPC automatically creates this unique directory during the job prolog
NODE_SCRATCH="/scratch/local/$USER/$SLURM_JOB_ID"

LOCAL_DATA_DIR="$NODE_SCRATCH/data"
LOCAL_RESULTS_DIR="$NODE_SCRATCH/results"

mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_RESULTS_DIR"

# =====================================================================
# 2. STAGE INPUT DATA TO LOCAL NODE (Fast I/O Reads)
# =====================================================================
echo "Staging input data from network storage to local node scratch..."
# Assuming you need your existing data root contents to run:
if [ -d "$CSDV_DATA_ROOT" ] && [ "$(ls -A "$CSDV_DATA_ROOT")" ]; then
    cp -r "$CSDV_DATA_ROOT"/* "$LOCAL_DATA_DIR"/
fi

# Save the original network results path so we know where to save back to later
NETWORK_RESULTS_ROOT="$CSDV_RESULTS_ROOT"

# =====================================================================
# 3. OVERRIDE ENVIRONMENT PATHS FOR LOCAL NODE EXECUTION
# =====================================================================
export CSDV_DATA_ROOT="$LOCAL_DATA_DIR"
export CSDV_RESULTS_ROOT="$LOCAL_RESULTS_DIR"

echo "Redirected CSDV_DATA_ROOT=$CSDV_DATA_ROOT"
echo "Redirected CSDV_RESULTS_ROOT=$CSDV_RESULTS_ROOT"

# =====================================================================
# 4. RUN WORKFLOW AND STAGE OUT RESULTS
# =====================================================================
# We remove 'exec' so that the shell script stays alive to copy files 
# back to network storage before the job terminates.
echo "Starting end-to-end processing pipeline..."
scripts/run_site_e2e.sh "$@"

echo "Pipeline finished. Staging results back to persistent network storage..."
mkdir -p "$NETWORK_RESULTS_ROOT"
cp -r "$LOCAL_RESULTS_DIR"/* "$NETWORK_RESULTS_ROOT"/

echo "Job pipeline and scratch synchronization complete: $(date -Is)"

