#!/bin/bash
#SBATCH --job-name=csdv_audit_model_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# The fifth audit stratum: draw the patches that only the pair model flags,
# on units it never trained on, and render them like block 1 of the audit
# (`csdv audit model-only draw`, then `render`). CPU only.
#
# The draw reads, per unit, the map of pairmodel/holdout_<unit>/ under the
# export folder, else the map of first_look_blocks2 (Oregon), and the unit's
# candidates in the label run. It is written once: a second job finds
# patches.gpkg and only renders what is missing.
#
# This script names no allocation. Pass yours at submission, from the option
# string in your scripts/chpc/env.sh (CSDV_SBATCH_CPU).
#
# Usage (after `source scripts/chpc/env.sh`, when every holdout run is done):
#   # First one patch, and look at its log:
#   CSDV_MODEL_ONLY_LIMIT=1 sbatch $CSDV_SBATCH_CPU \
#       --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_audit_model_only.sbatch
#   # Then all of them:
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_audit_model_only.sbatch
#
# Outputs in $CSDV_EXPORT_DIR/audit/model_only/: patches.gpkg (the drawn
# patches, the key of block 3), population.csv, patches.txt, requests/<patch>/
# (one JPEG per date and requests.jsonl) and stacks/<patch>/meta.json. On the
# laptop: scripts/chpc/pull_exports.sh audit/model_only, then
# `csdv audit model-only add` and `csdv audit build`.
#
# Resources: one stack of about nine chips and seven dates per patch, a few
# seconds each to read. 33 patches take well under an hour.

set -euo pipefail

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

LABEL_DIR="${CSDV_PAIR_LABEL_DIR:-$CSDV_EXPORT_DIR/label/scale_test}"
OUT="${CSDV_MODEL_ONLY_DIR:-$CSDV_EXPORT_DIR/audit/model_only}"
RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"
LIMIT="${CSDV_MODEL_ONLY_LIMIT:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "label_dir=$LABEL_DIR out=$OUT limit=${LIMIT:-none}"

if [[ ! -d "$LABEL_DIR/candidates" ]]; then
    echo "ERROR: no candidates/ in $LABEL_DIR" >&2
    exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg

activate_csdv

if [[ ! -f "$OUT/patches.gpkg" ]]; then
    csdv audit model-only draw --label-dir "$LABEL_DIR" --out-dir "$OUT" \
        --pair-dir "$CSDV_EXPORT_DIR/pairmodel"
else
    echo "Patches already drawn ($OUT/patches.gpkg), rendering what is missing."
fi

LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
/usr/bin/time -v csdv audit model-only render --label-dir "$LABEL_DIR" --out-dir "$OUT" \
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG" \
    --naip-source local --data-root "$CSDV_DATA_ROOT" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}

du -sh "$OUT"
echo "finished=$(date -Is)"
