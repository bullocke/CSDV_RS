#!/usr/bin/env bash
# Submit the runs of the defoliation test of the pair model: one job per row
# of config/pair/defoliation.csv. Run on CHPC, after `source scripts/chpc/env.sh`.
#
# The test compares four models on forest that was defoliated and recovered:
# the labels as they are or with recovered defoliation as background, and two
# dates or three. A "train" row trains a model without one unit and maps that
# unit. A "look" row maps a unit with a model of another row and trains
# nothing. Every map is also written for the pairs that skip a date
# (prob_skip_<unit>.tif).
#
# Other run tables use the same script (--table): config/pair/stable_forest.csv
# and config/pair/round2.csv. Two more kinds of row serve tests on many small
# places. A "score" row trains a model and scores it on the label chips of
# the units with the role "test" in the chip folder's sites.csv, without a
# map (`csdv pair score`). A "rescore" row scores the model of another row
# on the chips of its own label run.
#
# Usage:
#   scripts/chpc/submit_pair_defoliation.sh [--after JOBID] [--only RUN,RUN]
#       [--optional] [--no-smoke] [--table FILE] [--predict-args "ARGS"]
#       [--dry-run]
#
#   --after JOBID  Start nothing before this job has ended well, for example
#                  the label build of the variant (run_label_build.sbatch).
#   --only LIST    Take only these runs of the table.
#   --optional     Also take the rows with batch "optional".
#   --no-smoke     Skip the smoke job. By default the first three-date run is
#                  first run as a ten minute check (RUN_smoke), and every
#                  other job waits for it to end well.
#   --table FILE   Run table. Default config/pair/defoliation.csv.
#   --predict-args "ARGS"  Options of `csdv pair predict`. Default
#                  "--skip-pairs", which the defoliation test needs. Pass ""
#                  for maps of consecutive dates only.
#   --dry-run      Print the sbatch commands and submit nothing.
#
# The run table may end with a column train_args: further options for
# `csdv pair train` (no commas), for example "--run-draws stable_forest=0.15"
# in config/pair/stable_forest.csv. Tables without it work as before.
#
# A run that already has a comparison in the export folder is skipped. A
# "look" row waits for the job that trains its model when that job is
# submitted by the same call.
#
# About 50 minutes of GPU for a run with two dates, 75 for one with three,
# and 15 for a "look" row.
#
# Afterwards, on the laptop: scripts/chpc/pull_pair_defoliation.sh

set -euo pipefail

AFTER=""
ONLY=""
OPTIONAL=0
SMOKE=1
TABLE=""
PREDICT_ARGS="--skip-pairs"
DRY=0
while (($#)); do
    case "$1" in
        --predict-args) PREDICT_ARGS="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --optional) OPTIONAL=1; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        --table) TABLE="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) sed -n '2,48p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_GPU:?set CSDV_SBATCH_GPU in scripts/chpc/env.sh}"
TABLE="${TABLE:-$CSDV_REPO/config/pair/defoliation.csv}"
if [[ ! -f "$TABLE" ]]; then
    echo "No run table at $TABLE" >&2
    exit 1
fi

read -r -a GPU <<<"$CSDV_SBATCH_GPU"
SCRIPT="$CSDV_REPO/scripts/slurm/run_pair_first_look.sbatch"

wanted() {
    # Whether a run is taken: named by --only, or in the batch.
    local run="$1" batch="$2"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$run,"* ]]
        return
    fi
    [[ "$batch" == "yes" ]] || { ((OPTIONAL)) && [[ "$batch" == "optional" ]]; }
}

submit() {
    # submit DEPENDENCY SETTINGS... -- SBATCH ARGUMENTS...
    # Prints the job ID, or a placeholder in a dry run.
    local dependency="$1"
    shift
    local settings=()
    while [[ "$1" != "--" ]]; do
        settings+=("$1")
        shift
    done
    shift
    local command=(sbatch --parsable "${GPU[@]}")
    [[ -n "$dependency" ]] && command+=(--dependency="afterok:$dependency")
    command+=("$@")
    if ((DRY)); then
        printf '%q ' "${settings[@]}" "${command[@]}" >&2
        echo >&2
        # A stand-in for the job ID, so that the dependencies can be read.
        local arg
        for arg in "$@"; do
            if [[ "$arg" == --job-name=* ]]; then
                echo "<${arg#--job-name=}>"
            fi
        done
    else
        env "${settings[@]}" "${command[@]}"
    fi
}

join_ids() {
    # Job IDs joined with ":" for --dependency, empty ones left out.
    local out="" id
    for id in "$@"; do
        [[ -n "$id" ]] && out="${out:+$out:}$id"
    done
    echo "$out"
}

declare -A TRAIN_JOB=()
ROWS=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* || "$line" == run,* ]] && continue
    ROWS+=("$line")
done <"$TABLE"

# The labels of every row that is taken have to be there, unless a job that
# builds them is named with --after.
for line in "${ROWS[@]}"; do
    IFS=, read -r run kind labels dates unit seed model batch notes train_args <<<"$line"
    wanted "$run" "$batch" || continue
    label_dir="$CSDV_EXPORT_DIR/label/$labels"
    if [[ -z "$AFTER" && ! -f "$label_dir/chips/chips.parquet" ]]; then
        echo "No chips/chips.parquet in $label_dir (run $run). Build the labels" \
            "first, or name the build job with --after." >&2
        exit 1
    fi
done

SMOKE_JOB=""
if ((SMOKE)); then
    for line in "${ROWS[@]}"; do
        IFS=, read -r run kind labels dates unit seed model batch notes train_args <<<"$line"
        wanted "$run" "$batch" || continue
        [[ "$kind" == "train" && "$dates" == "3" ]] || continue
        SMOKE_JOB="$(submit "$AFTER" \
            CSDV_PAIR_LABEL_DIR="$CSDV_EXPORT_DIR/label/$labels" \
            CSDV_PAIR_HOLDOUT="$unit" CSDV_PAIR_SPLIT=blocks \
            CSDV_PAIR_TRAIN_ARGS="--dates 3 --seed $seed${train_args:+ $train_args}" \
            CSDV_PAIR_PREDICT_ARGS="--skip-pairs" -- \
            --time=00:45:00 --job-name="pair_${run}_smoke" \
            --output="$CSDV_LOG_DIR/%x-%j.out" "$SCRIPT" "$run" --smoke)"
        echo "smoke: job $SMOKE_JOB, run ${run}_smoke"
        break
    done
fi
FIRST="$(join_ids "$AFTER" "$SMOKE_JOB")"

submitted=0
for wave in train score look rescore; do
    for line in "${ROWS[@]}"; do
        IFS=, read -r run kind labels dates unit seed model batch notes train_args <<<"$line"
        [[ "$kind" == "$wave" ]] || continue
        wanted "$run" "$batch" || continue
        done_file="compare/compare.json"
        [[ "$kind" == "score" || "$kind" == "rescore" ]] && done_file="score/score.json"
        if [[ -f "$CSDV_EXPORT_DIR/pairmodel/$run/$done_file" ]]; then
            echo "$run: done, skipped"
            continue
        fi
        settings=(CSDV_PAIR_LABEL_DIR="$CSDV_EXPORT_DIR/label/$labels"
            CSDV_PAIR_HOLDOUT="$unit" CSDV_PAIR_SPLIT=blocks
            CSDV_PAIR_PREDICT_ARGS="$PREDICT_ARGS")
        if [[ "$kind" == "train" || "$kind" == "score" ]]; then
            options="--seed $seed"
            limit=02:30:00
            if [[ "$dates" == "3" ]]; then
                options="--dates 3 --seed $seed"
                limit=03:30:00
            fi
            settings+=(CSDV_PAIR_TRAIN_ARGS="$options${train_args:+ $train_args}")
            [[ "$kind" == "score" ]] && settings+=(CSDV_PAIR_STAGES="train score")
            job="$(submit "$FIRST" "${settings[@]}" -- --time="$limit" \
                --job-name="pair_$run" --output="$CSDV_LOG_DIR/%x-%j.out" \
                "$SCRIPT" "$run")"
            TRAIN_JOB["$run"]="$job"
        else
            checkpoint="$CSDV_RESULTS_ROOT/pairmodel/$model/best.pt"
            trainer="${TRAIN_JOB[$model]:-}"
            if [[ -z "$trainer" && ! -f "$checkpoint" ]]; then
                echo "$run: no checkpoint $checkpoint and no job that trains" \
                    "$model in this call, skipped" >&2
                continue
            fi
            stages="predict compare"
            [[ "$kind" == "rescore" ]] && stages="score"
            settings+=(CSDV_PAIR_STAGES="$stages"
                CSDV_PAIR_CHECKPOINT="$checkpoint")
            job="$(submit "$(join_ids "$FIRST" "$trainer")" "${settings[@]}" -- \
                --time=01:00:00 --job-name="pair_$run" \
                --output="$CSDV_LOG_DIR/%x-%j.out" "$SCRIPT" "$run")"
        fi
        echo "$run: job $job ($kind, labels $labels, $dates dates, unit $unit)"
        submitted=$((submitted + 1))
    done
done
echo "$submitted jobs$( ((DRY)) && echo " would be") submitted. Watch with: squeue -u \$USER"
echo "Then, on the laptop: scripts/chpc/pull_pair_defoliation.sh"
