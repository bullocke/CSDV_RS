#!/bin/bash
#SBATCH --job-name=csdv_label_verdict
#SBATCH --partition=notchpeak-dtn
#SBATCH --qos=notchpeak-dtn
#SBATCH --account=dtn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#
# Send the rendered requests of a label run to the vision models (`csdv label
# verdict`). The calls need the internet, which compute nodes do not have, so
# the job runs on a data transfer node like the downloads. It is light: a few
# megabytes of images a minute and almost no CPU.
#
# The first argument is the label run folder. Every further argument is
# passed to `csdv label verdict` unchanged. --cap is required.
#
# Usage (after `source scripts/chpc/env.sh`):
#   LABEL="$CSDV_EXPORT_DIR/label/round2"
#   sbatch --dependency=singleton --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_verdict.sbatch "$LABEL" --stage 1 --cap 16
#   # The polygon calls, after `run_label_render.sbatch "$LABEL" --kind polygon`
#   sbatch --dependency=singleton --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_verdict.sbatch "$LABEL" --kind polygon --stage 1 --cap 16
#
# scripts/chpc/submit_label_chain.sh submits the whole chain.
#
# The keys. The job reads GEMINI_API_KEY and OpenRouter_API_KEY from the file
# named by CSDV_KEYS_FILE in scripts/chpc/env.sh: two `export NAME=value`
# lines, readable by you alone (chmod 600). The file is sourced inside the job
# only. Do not export the keys in your login shell: Slurm copies that
# environment into every job you submit. Keys are never printed.
#
# The cap is for the whole label run, not for this job: it counts every call
# in the run's ledger.jsonl. The job writes answers.jsonl, ledger.jsonl and
# responses/ straight into the label run folder, with no node scratch, so
# that a job killed part of the way loses nothing but the calls in flight.
# Run one verdict job per label run at a time (--dependency=singleton): two
# jobs would each count the spending on their own.
#
# Guards, all of which end the job with a code a chain can act on:
#   3  the expected cost would pass the cap (nothing sent), or the cap
#      stopped the run part of the way
#   4  more than a fifth of the calls failed (a wrong key stops it after
#      the first batch)
#   5  more than a tenth of the selected windows have no request file
# A second job with the same arguments sends only what is missing and costs
# nothing when the first finished.

set -euo pipefail

LABEL_DIR="${1:?usage: sbatch $0 LABEL_DIR --cap DOLLARS [csdv label verdict options...]}"
shift

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"

if [[ ! -f "$LABEL_DIR/windows.csv" ]]; then
    echo "ERROR: $LABEL_DIR has no windows.csv." >&2
    exit 1
fi
: "${CSDV_KEYS_FILE:?set CSDV_KEYS_FILE in scripts/chpc/env.sh (a private file with the two API keys)}"
if [[ ! -f "$CSDV_KEYS_FILE" ]]; then
    echo "ERROR: CSDV_KEYS_FILE names $CSDV_KEYS_FILE, which does not exist." >&2
    exit 1
fi
MODE=$(stat -c '%a' "$CSDV_KEYS_FILE")
if [[ "$MODE" != "600" && "$MODE" != "400" ]]; then
    echo "ERROR: $CSDV_KEYS_FILE has mode $MODE. Run: chmod 600 $CSDV_KEYS_FILE" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "label_dir=$LABEL_DIR args: $*"

activate_csdv

set +x
set -a
# shellcheck disable=SC1090
source "$CSDV_KEYS_FILE"
set +a
for key in GEMINI_API_KEY OpenRouter_API_KEY; do
    if [[ -z "${!key:-}" ]]; then
        echo "ERROR: $CSDV_KEYS_FILE does not set $key." >&2
        exit 1
    fi
done

CONFIG=()
[[ -f "$LABEL_DIR/config.yaml" ]] && CONFIG=(--config "$LABEL_DIR/config.yaml")

csdv label verdict --label-dir "$LABEL_DIR" \
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG" \
    ${CONFIG[@]+"${CONFIG[@]}"} \
    --refuse-over-cap --max-unrendered-share 0.1 "$@"

echo "done $(date -Is)"
