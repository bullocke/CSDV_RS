#!/usr/bin/env bash
# Map several units with the pair model: one job per unit of
# config/pair/sites.csv. Run on CHPC, after `source scripts/chpc/env.sh`.
#
# Usage:
#   scripts/chpc/submit_pair_sites.sh [--mode look|holdout] [--model RUN]
#       [--units q1,q2,...] [--sites FILE] [--dry-run]
#
#   --mode look     (default) Map every unit with the model of --model. No
#                   training, about 15 minutes of GPU per unit. The model
#                   trained on label chips inside these units (5 to 10% of
#                   the ground of a unit), so the maps show what the model
#                   does there and are not an independent test.
#                   Runs: look_<unit>.
#   --mode holdout  Train one model per unit with that unit held out (split
#                   by blocks), then map it. About 50 minutes of GPU per
#                   unit. An independent test of each unit. A type whose
#                   labels lie mostly in the held-out unit is then hardly
#                   trained. Runs: holdout_<unit>.
#   --model RUN     Model run of the look mode. Default first_look_blocks2.
#   --units LIST    Units to take, in place of the rows with batch "yes".
#   --sites FILE    Site list. Default config/pair/sites.csv.
#   --dry-run       Print the sbatch commands and submit nothing.
#
# A unit that the model run mapped itself (its held-out unit) is skipped in
# both modes, because that run is already its held-out map. A unit whose run
# already has a comparison in the export folder is skipped as well.
#
# Afterwards, on the laptop: scripts/chpc/pull_pair_sites.sh --mode MODE

set -euo pipefail

MODE=look
MODEL=first_look_blocks2
UNITS=""
SITES=""
DRY=0
while (($#)); do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --units) UNITS="$2"; shift 2 ;;
        --sites) SITES="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) sed -n '2,31p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
if [[ "$MODE" != "look" && "$MODE" != "holdout" ]]; then
    echo "--mode must be look or holdout, not $MODE" >&2
    exit 1
fi

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_GPU:?set CSDV_SBATCH_GPU in scripts/chpc/env.sh}"
SITES="${SITES:-$CSDV_REPO/config/pair/sites.csv}"
MODEL_DIR="$CSDV_RESULTS_ROOT/pairmodel/$MODEL"
MODEL_EXPORT="$CSDV_EXPORT_DIR/pairmodel/$MODEL"

if [[ -z "$UNITS" ]]; then
    # Rows "unit,yes,...". Comment lines and the header do not match.
    UNITS="$(grep -E '^q[0-9]+,yes,' "$SITES" | cut -d, -f1 | paste -sd, -)"
fi
if [[ -z "$UNITS" ]]; then
    echo "No unit to map in $SITES" >&2
    exit 1
fi
if [[ "$MODE" == "look" && ! -f "$MODEL_DIR/best.pt" ]]; then
    echo "No checkpoint $MODEL_DIR/best.pt" >&2
    exit 1
fi

read -r -a GPU <<<"$CSDV_SBATCH_GPU"
SCRIPT="$CSDV_REPO/scripts/slurm/run_pair_first_look.sbatch"
submitted=0
for unit in ${UNITS//,/ }; do
    run="${MODE}_${unit}"
    if [[ -f "$MODEL_EXPORT/map/prob_$unit.tif" ]]; then
        echo "$unit: held out and mapped by $MODEL itself, skipped"
        continue
    fi
    if [[ -f "$CSDV_EXPORT_DIR/pairmodel/$run/compare/compare.json" ]]; then
        echo "$unit: $run is done, skipped"
        continue
    fi
    if [[ "$MODE" == "look" ]]; then
        settings=(CSDV_PAIR_HOLDOUT="$unit" CSDV_PAIR_STAGES="predict compare"
            CSDV_PAIR_CHECKPOINT="$MODEL_DIR/best.pt")
        limit=00:45:00
    else
        settings=(CSDV_PAIR_HOLDOUT="$unit" CSDV_PAIR_SPLIT=blocks)
        limit=02:30:00
    fi
    command=(sbatch --parsable "${GPU[@]}" --time="$limit" --job-name="pair_$run"
        --output="$CSDV_LOG_DIR/%x-%j.out" "$SCRIPT" "$run")
    if ((DRY)); then
        printf '%q ' "${settings[@]}" "${command[@]}"
        echo
    else
        job="$(env "${settings[@]}" "${command[@]}")"
        echo "$unit: job $job, run $run"
    fi
    submitted=$((submitted + 1))
done
echo "$submitted jobs$( ((DRY)) && echo " would be") submitted. Watch with: squeue -u \$USER"
echo "Then, on the laptop: scripts/chpc/pull_pair_sites.sh --mode $MODE"
