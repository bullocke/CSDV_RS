# shellcheck shell=bash
# CSDV settings for CHPC. Copy to scripts/chpc/env.sh (gitignored), edit the
# two paths marked EDIT, then `source scripts/chpc/env.sh` in every login
# session before running csdv or submitting jobs.
#
# Slurm exports the submitting shell's environment into each job, so the
# sbatch scripts find this file again through $CSDV_REPO.

# This repository, derived from the location of this file.
CSDV_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CSDV_REPO

# EDIT: persistent storage. Jobs copy inputs from here to node-local scratch
# and copy outputs back here. Do not point this at /scratch (it is purged).
export CSDV_STORE=/uufs/chpc.utah.edu/common/home/<group>/data/CSDV

# Clone of https://github.com/bullocke/naip-crowns (crown segmentation), read by
# scripts/chpc/create_env.sh.
export NAIPCROWNS_DIR="$CSDV_REPO/../naip-crowns"

# EDIT: clone of https://github.com/smorf-ntsg/naip-chm (code and weights).
export NAIPCHM_REPO_DIR="$CSDV_REPO/../naip-chm"

# EDIT: Slurm options of your CPU allocation, passed at submission by the job
# scripts that name no allocation themselves:
#   sbatch $CSDV_SBATCH_CPU scripts/slurm/run_label_render.sbatch ...
# Keep real names in env.sh only (it is gitignored). Never export
# SBATCH_ACCOUNT, SBATCH_PARTITION or SBATCH_QOS: they override the #SBATCH
# lines of every script, including the data transfer jobs.
export CSDV_SBATCH_CPU="--account=EDIT --partition=EDIT --qos=EDIT"

# EDIT: the same for one GPU, used by scripts/slurm/run_pair_first_look.sbatch:
#   sbatch $CSDV_SBATCH_GPU scripts/slurm/run_pair_first_look.sbatch ...
# Any card with 11 GB or more will do. Avoid granite-gpu-guest: its Blackwell
# cards are not supported by the cu126 build of PyTorch.
export CSDV_SBATCH_GPU="--account=EDIT --partition=EDIT --qos=EDIT --gres=gpu:1"

# EDIT: the GPU pools that `csdv grid queue` may send CHM workers to, each a
# string of sbatch options for one GPU, with a semicolon between pools. The
# workers of every pool are queued at once and whichever pool frees a GPU
# first does the work, so name every pool you may use: the general GPU
# partition, the group's preemptable route, guest nodes. Preempted workers
# are cancelled, which the queue is built for. Not granite-gpu-guest
# (Blackwell cards, see above). Every pool must be a partition of the cluster
# whose login node you submit from (`sinfo -o %P` lists them): kingspeak,
# lonepeak and granite are Slurm clusters of their own, and the queue's jobs
# cannot wait for a job of another cluster. `csdv grid queue` asks Slurm
# about every pool before it submits anything.
export CSDV_GPU_POOLS="--account=EDIT --partition=EDIT --qos=EDIT --gres=gpu:1;--account=EDIT --partition=EDIT --qos=EDIT --gres=gpu:1"

# EDIT: a private file with the API keys of the vision models, read only by
# scripts/slurm/run_label_verdict.sbatch. Two lines,
#   export GEMINI_API_KEY=...
#   export OpenRouter_API_KEY=...
# in a file outside the repository that you alone can read (chmod 600). Do
# not export the keys in this file or in your login shell: Slurm copies the
# login environment into every job.
export CSDV_KEYS_FILE="$HOME/.csdv_keys"

export CSDV_ENV=csdv-gpu
export CSDV_DATA_ROOT="$CSDV_STORE/data"
export CSDV_RESULTS_ROOT="$CSDV_STORE/results"
export CSDV_CACHE_ROOT="$CSDV_STORE/cache"
export CSDV_CONDITIONING_DIR="$CSDV_DATA_ROOT/chm_model/conditioning"
export CSDV_LOG_DIR="$CSDV_STORE/logs/slurm"
export CSDV_MANIFEST_DIR="$CSDV_STORE/manifests"
# Grid runs (`csdv grid`): unit records, submission ledger and status.csv, and
# the small download set pulled to a local machine by scripts/chpc/pull_exports.sh.
export CSDV_GRID_DIR="$CSDV_STORE/grid"
export CSDV_EXPORT_DIR="$CSDV_STORE/exports"

# Keep GDAL's block cache bounded (MB).
export GDAL_CACHEMAX=2048

# Activate the project env. Batch shells do not read ~/.bashrc, so load the
# mamba shell hook here. Activation scripts (e.g. libpdal-core) reference
# unset variables, so nounset is relaxed around the call.
activate_csdv() {
    if ! command -v mamba >/dev/null 2>&1; then
        echo "ERROR: mamba is not on PATH. Source env.sh from a shell where" \
            "mamba works, or add its bin directory to PATH here." >&2
        return 1
    fi
    local nounset=0
    [[ $- == *u* ]] && nounset=1
    set +u
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$CSDV_ENV"
    ((nounset)) && set -u
    return 0
}
