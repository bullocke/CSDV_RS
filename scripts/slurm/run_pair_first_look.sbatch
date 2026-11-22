#!/bin/bash
#SBATCH --job-name=csdv_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#
# First look at the pair model (`csdv pair`): train on the chips of a label
# run with one unit held out, map that unit for every pair of consecutive
# dates, and set the map beside the disturbance run and the held-out labels.
# One GPU, one job.
#
# This script names no allocation and no GPU. Pass both at submission, from
# the option string in your scripts/chpc/env.sh (CSDV_SBATCH_GPU).
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch $CSDV_SBATCH_GPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_pair_first_look.sbatch RUN_NAME [--smoke]
#
#   RUN_NAME   Name of the run. Checkpoints and results go to
#              $CSDV_RESULTS_ROOT/pairmodel/RUN_NAME, and everything except
#              the checkpoints is copied to $CSDV_EXPORT_DIR/pairmodel/RUN_NAME
#              for scripts/chpc/pull_exports.sh.
#   --smoke    A ten minute check of every step: two short epochs, six blocks
#              of the map, three close-ups. Writes to RUN_NAME_smoke, so the
#              real run never resumes from it.
#
# Settings (environment, all optional):
#   CSDV_PAIR_LABEL_DIR  Label run folder. Default $CSDV_EXPORT_DIR/label/scale_test.
#   CSDV_PAIR_HOLDOUT    Unit that is held out and mapped. Default q4412328.
#   CSDV_PAIR_VAL        Validation units, comma-separated (units split only).
#   CSDV_PAIR_SPLIT      "units" (default) validates on the units of
#                        CSDV_PAIR_VAL. "blocks" validates on blocks of ground
#                        inside every unit that is not held out, so every
#                        agent is trained on and validated on.
#   CSDV_PAIR_CACHE      Folder for decoded chips, read by the loaders. Default
#                        a folder on node scratch (about 30 GB for a full
#                        label run). "none" reads the chip rasters instead.
#   CSDV_PAIR_WEIGHTS    Encoder weights. Default
#                        $CSDV_DATA_ROOT/models/torchvision/resnet34-b627a593.pth.
#                        Compute nodes have no internet: fetch the file on a
#                        login node first (the command is printed when it is
#                        missing).
#   CSDV_PAIR_STAGES     Steps to run. Default "train predict compare". The
#                        step "score" scores the checkpoint on the label
#                        chips of the units with the role CSDV_PAIR_SCORE_ROLE
#                        (default test) in the chip folder's sites.csv, and
#                        maps nothing: "train score" for a test on many
#                        small places, "score" alone with the checkpoint of
#                        another run.
#   CSDV_PAIR_CHECKPOINT Checkpoint for the map and the comparison. Default
#                        best.pt of this run. With the checkpoint of another
#                        run and CSDV_PAIR_STAGES="predict compare", the job
#                        maps CSDV_PAIR_HOLDOUT with that model and trains
#                        nothing (scripts/chpc/submit_pair_sites.sh). The
#                        comparison is then a test only where that model did
#                        not train on the unit.
#   CSDV_PAIR_TRAIN_ARGS Further options for `csdv pair train`, for example
#                        "--dates 3 --seed 1".
#   CSDV_PAIR_PREDICT_ARGS  Further options for `csdv pair predict`, for
#                        example "--skip-pairs".
#   CSDV_LABEL_RUN_TAG   Run tag of the disturbance runs. Default "default".
#
# Before training, the label rasters must lie beside the four-band chips
# (scripts/slurm/run_label_build.sbatch). The job checks this first.
#
# A job that is stopped (time limit, preemption) leaves last.pt in the run
# folder. Submit the same command again and training goes on from there.
set -euo pipefail

RUN_NAME="${1:?usage: sbatch $0 RUN_NAME [--smoke]}"
SMOKE=0
if [[ "${2:-}" == "--smoke" ]]; then
    SMOKE=1
    RUN_NAME="${RUN_NAME}_smoke"
fi

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
source "$CSDV_REPO/scripts/chpc/scratch.sh"

LABEL_DIR="${CSDV_PAIR_LABEL_DIR:-$CSDV_EXPORT_DIR/label/scale_test}"
HOLDOUT="${CSDV_PAIR_HOLDOUT:-q4412328}"
VAL="${CSDV_PAIR_VAL:-q4709438,q3908654,q3508207}"
SPLIT_MODE="${CSDV_PAIR_SPLIT:-units}"
WEIGHTS="${CSDV_PAIR_WEIGHTS:-$CSDV_DATA_ROOT/models/torchvision/resnet34-b627a593.pth}"
STAGES="${CSDV_PAIR_STAGES:-train predict compare}"
RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"
RUN_DIR="$CSDV_RESULTS_ROOT/pairmodel/$RUN_NAME"
CHECKPOINT="${CSDV_PAIR_CHECKPOINT:-$RUN_DIR/best.pt}"
EXPORT="$CSDV_EXPORT_DIR/pairmodel/$RUN_NAME"
read -r -a TRAIN_ARGS <<<"${CSDV_PAIR_TRAIN_ARGS:-}"
read -r -a PREDICT_ARGS <<<"${CSDV_PAIR_PREDICT_ARGS:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "run=$RUN_DIR holdout=$HOLDOUT split=$SPLIT_MODE val=$VAL stages=$STAGES smoke=$SMOKE"
echo "checkpoint=$CHECKPOINT"
if [[ " $STAGES " == *" train "* && "$CHECKPOINT" != "$RUN_DIR/best.pt" ]]; then
    echo "ERROR: CSDV_PAIR_CHECKPOINT names another run, so this job must not train." \
        "Set CSDV_PAIR_STAGES=\"predict compare\" or \"score\"." >&2
    exit 1
fi
if [[ " $STAGES " != *" train "* && ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint at $CHECKPOINT" >&2
    exit 1
fi
if [[ "$SPLIT_MODE" != "units" && "$SPLIT_MODE" != "blocks" ]]; then
    echo "ERROR: CSDV_PAIR_SPLIT must be units or blocks, not $SPLIT_MODE" >&2
    exit 1
fi

if [[ ! -f "$LABEL_DIR/chips/chips.parquet" ]]; then
    echo "ERROR: no chips/chips.parquet in $LABEL_DIR. Build the labels first" \
        "(scripts/slurm/run_label_build.sbatch)." >&2
    exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: no encoder weights at $WEIGHTS. On a login node:" >&2
    echo "  wget -P $(dirname "$WEIGHTS") https://download.pytorch.org/models/$(basename "$WEIGHTS")" >&2
    exit 1
fi

# Loader workers are separate processes: one thread each, and a small GDAL
# cache each (the value in env.sh is per process).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=256
export MPLBACKEND=Agg

activate_csdv
nvidia-smi -L || echo "WARNING: nvidia-smi failed on $(hostname -s)" >&2
if ! python -c "import rasterio, torch; assert torch.cuda.is_available(), 'no CUDA device'; \
torch.zeros(1, device='cuda'); \
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"; then
    echo "ERROR: no usable GPU on $(hostname -s); resubmit with --exclude=$(hostname -s)" >&2
    exit 1
fi

scratch_init
mkdir -p "$RUN_DIR" "$EXPORT"
# Which model made this map, for the figures on the laptop.
printf '{"unit": "%s", "checkpoint": "%s", "model_run": "%s"}\n' \
    "$HOLDOUT" "$CHECKPOINT" "$(basename "$(dirname "$CHECKPOINT")")" >"$RUN_DIR/site.json"

# Copy what has been written so far, without the checkpoints, to the export
# folder. Called after every step, so a failure later still leaves the
# earlier results where the laptop can pull them.
export_results() {
    rsync -a --exclude='*.pt' --exclude='*.tmp' "$RUN_DIR"/ "$EXPORT"/
    [[ -n "${SLURM_JOB_ID:-}" && -f "$CSDV_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" ]] &&
        cp "$CSDV_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$EXPORT/job.log"
    return 0
}

WORKERS=$((${SLURM_CPUS_PER_TASK:-8} - 1))
CHIPS="$LABEL_DIR/chips"

if [[ " $STAGES " == *" train "* ]]; then
    # The chips are read thousands of times, so they go to node scratch.
    STARTED=$SECONDS
    stage_in "$LABEL_DIR/chips" "$NODE_SCRATCH/chips"
    CHIPS="$NODE_SCRATCH/chips"
    du -sh "$CHIPS"
    echo "chips staged in $((SECONDS - STARTED)) s"
    SPLIT=(--chips-dir "$CHIPS" --holdout-units "$HOLDOUT")
    if [[ "$SPLIT_MODE" == "blocks" ]]; then
        SPLIT+=(--split blocks)
    else
        SPLIT+=(--val-units "$VAL")
    fi
    csdv pair train "${SPLIT[@]}" --encoder-weights "$WEIGHTS" --dry-run
    # Decoded chips for the loaders, built by the training command itself.
    CACHE="${CSDV_PAIR_CACHE:-$NODE_SCRATCH/pair_cache}"
    CACHE_ARGS=()
    [[ "$CACHE" != "none" ]] && CACHE_ARGS=(--cache-dir "$CACHE")
    SIZE=()
    if ((SMOKE)); then
        SIZE=(--epochs 2 --samples-per-epoch 256 --max-val-samples 128)
    fi
    csdv pair train "${SPLIT[@]}" --encoder-weights "$WEIGHTS" \
        --out-dir "$RUN_DIR" --workers "$WORKERS" \
        ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
        ${SIZE[@]+"${SIZE[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
    [[ "$CACHE" != "none" ]] && du -sh "$CACHE"
    export_results
fi

WHERE=(--unit "$HOLDOUT" --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG")

if [[ " $STAGES " == *" predict "* ]]; then
    LIMIT=()
    ((SMOKE)) && LIMIT=(--limit-blocks 6)
    # The mosaics are read where they are, as the render job does.
    csdv pair predict --checkpoint "$CHECKPOINT" "${WHERE[@]}" \
        --data-root "$CSDV_DATA_ROOT" --out-dir "$RUN_DIR/map" \
        --workers "$WORKERS" ${LIMIT[@]+"${LIMIT[@]}"} \
        ${PREDICT_ARGS[@]+"${PREDICT_ARGS[@]}"}
    export_results
fi

if [[ " $STAGES " == *" compare "* ]]; then
    PANELS=12
    ((SMOKE)) && PANELS=2
    csdv pair compare --prob "$RUN_DIR/map/prob_$HOLDOUT.tif" --chips-dir "$CHIPS" \
        "${WHERE[@]}" --checkpoint "$CHECKPOINT" --data-root "$CSDV_DATA_ROOT" \
        --out-dir "$RUN_DIR/compare" --panels "$PANELS"
    export_results
fi

if [[ " $STAGES " == *" score "* ]]; then
    csdv pair score --checkpoint "$CHECKPOINT" --chips-dir "$CHIPS" \
        --role "${CSDV_PAIR_SCORE_ROLE:-test}" --out-dir "$RUN_DIR/score" \
        --workers "$WORKERS"
    export_results
fi

du -sh "$RUN_DIR" "$EXPORT"
echo "Pull to the laptop with: scripts/chpc/pull_exports.sh pairmodel/$RUN_NAME"
