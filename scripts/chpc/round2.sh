#!/usr/bin/env bash
# Round 2 of the training labels: one command per CHPC session. Run on a
# login node after `git pull` and `source scripts/chpc/env.sh`.
#
# Usage:
#   scripts/chpc/round2.sh pilot  [--cap 3]  [--without-missing] [--dry-run]
#   scripts/chpc/round2.sh grid              [--without-missing] [--dry-run]
#   scripts/chpc/round2.sh stage1 [--cap 16] [--dry-run]
#   scripts/chpc/round2.sh stage2  --cap 76  [--dry-run]
#   scripts/chpc/round2.sh evaluate [--optional] [--only RUN,RUN] [--dry-run]
#   scripts/chpc/round2.sh status [pilot]
#
# pilot   The whole chain on the few boxes of config/grid/round2_pilot.csv:
#         downloads, GPU workers on every pool of $CSDV_GPU_POOLS, detection
#         and the model map, candidates, the sample, rendering, the calls
#         (cap $3, own label folder round2_pilot), labels and sheets. One
#         submission, then scripts/chpc/round2_pull.sh pilot on the laptop.
# grid    The same for the boxes of config/grid/round2.csv, up to the
#         rendered request images of stage 1. No paid call. Give it again
#         when `status` shows unit-years or units left: it submits only
#         what is missing.
# stage1  The calls, polygons, labels and sheets of stage 1.
# stage2  Reads config/label/round2/stage2_plan.csv (written after the
#         stage-1 check-in, pulled with git), gives those windows to stage
#         2, then renders, calls and builds. The cap is cumulative. Copies
#         config/label/round2/config.yaml to the label folder when it has
#         changed, so that the build of both stages uses it.
# evaluate  The model tests of the round (config/pair/round2.csv), after
#         stage 2 has built its labels. No paid call. It makes the label
#         folder round2_norej (the same answers built without decision 10,
#         one CPU job), merges the chips of scale_test, round2 and
#         stable_forest into merged_r2 and merged_r2_norej, and submits one
#         GPU job per row of the table, each waiting for the folder it
#         reads. Give it again to submit what is not done. --optional and
#         --only are passed to scripts/chpc/submit_pair_defoliation.sh.
# status  Where every box stands (read from disk), what is queued, what the
#         label run has spent.
#
# The sample of a label run is drawn once and kept. `pilot` and `grid`
# therefore stop before the sample when a box has no finished run (the job
# csdv_label_prepare fails and names the boxes): give the same command again,
# which submits what is missing. --without-missing draws the sample without
# those boxes, for boxes that are not coming (too few years, a failed
# download). They stay out of the label run.
#
# Settings, all in scripts/chpc/env.sh: CSDV_SBATCH_CPU, CSDV_GPU_POOLS,
# CSDV_KEYS_FILE. CSDV_ROUND2_CHECKPOINT names the pair model that maps
# every box (default: first_look_blocks2). CSDV_ROUND2_QUEUE_ARGS holds
# further `csdv grid queue` options, for the GPU capacity of a large batch:
#   export CSDV_ROUND2_QUEUE_ARGS="--workers-per-pool 6 --worker-hours 8"
#
# What to look for is printed by each command. Every paid step has a cap
# on the whole label run, refuses to start when its forecast passes the
# cap, and stops the chain behind it when calls fail.

set -euo pipefail

STEP="${1:-}"
[[ -n "$STEP" ]] && shift
CAP="" DRY=() WHICH="" MISSING=() TABLE_ARGS=()
while (($#)); do
    case "$1" in
        --optional) TABLE_ARGS+=(--optional); shift ;;
        --only) TABLE_ARGS+=(--only "${2:?--only needs a list of runs}"); shift 2 ;;
        --cap) CAP="${2:?--cap needs a value}"; shift 2 ;;
        --dry-run) DRY=(--dry-run); shift ;;
        --without-missing) MISSING=(--without-missing); shift ;;
        pilot) WHICH="pilot"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_CPU:?CSDV_SBATCH_CPU is not set in scripts/chpc/env.sh}"
cd "$CSDV_REPO"
mkdir -p "$CSDV_LOG_DIR"
CKPT="${CSDV_ROUND2_CHECKPOINT:-$CSDV_RESULTS_ROOT/pairmodel/first_look_blocks2/best.pt}"
SLURM="$CSDV_REPO/scripts/slurm"
LOG="--output=$CSDV_LOG_DIR/%x-%j.out"

# args_of FILE: the options kept in a tracked one-line file, or nothing.
args_of() { [[ -f "$1" ]] && grep -v '^#' "$1" | tr '\n' ' ' || true; }

# queue NAME: send the batch through the grid queue and print the job that
# everything chained behind it waits for (empty when nothing was submitted).
queue() {
    local name="$1"
    [[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
    csdv grid queue --batch "config/grid/$name.csv" --name "$name" \
        --checkpoint "$CKPT" --priority-intervals "config/label/$name/sites.csv" \
        ${CSDV_ROUND2_QUEUE_ARGS:-} ${DRY[@]+"${DRY[@]}"} >&2
    local last="$CSDV_GRID_DIR/queues/$name/last_finish_job.txt"
    [[ -f "$last" ]] && cat "$last" || true
}

# prepare NAME THEN [sheets options]: candidates, sample and the chain behind.
prepare() {
    local name="$1" then="$2" sheets="${3:-}" after="$4"
    local cmd=(sbatch --parsable)
    [[ -n "$after" ]] && cmd+=("--dependency=afterany:$after")
    # shellcheck disable=SC2206
    cmd+=($CSDV_SBATCH_CPU "$LOG" "$SLURM/run_label_prepare.sbatch"
        "$CSDV_EXPORT_DIR/label/$name" "$CSDV_REPO/config/label/$name" --then "$then")
    [[ -n "$sheets" ]] && cmd+=(--then-sheets "$sheets")
    cmd+=(${MISSING[@]+"${MISSING[@]}"})
    # shellcheck disable=SC2207
    cmd+=($(args_of "config/label/$name/sample.args"))
    echo "${cmd[*]}"
    if ((${#DRY[@]})); then
        echo "Dry run: the label run is not prepared."
    else
        echo "prepare job: $("${cmd[@]}")"
    fi
}

case "$STEP" in
    pilot)
        CAP="${CAP:-3}"
        AFTER=$(queue round2_pilot)
        prepare round2_pilot "--stage 1 --cap $CAP" "--stage 1 --max-pairs 2" "$AFTER"
        # Boxes cut inside a quad that is already processed: their canopy
        # heights against the quad's, by distance from the box's edge. The
        # pairs are the rows of sites.csv with a quad in the column `against`.
        python3 - "config/label/round2_pilot/sites.csv" <<'PY' | while read -r BOX QUAD; do
import csv, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if not l.lstrip().startswith("#")]
for row in csv.DictReader(rows):
    if (row.get("against") or "").strip():
        print(row["unit"].strip(), row["against"].strip())
PY
            CMD=(sbatch --parsable)
            [[ -n "$AFTER" ]] && CMD+=("--dependency=afterany:$AFTER")
            # shellcheck disable=SC2206
            CMD+=($CSDV_SBATCH_CPU "$LOG" "$SLURM/run_csdv_cpu.sbatch" grid compare-units
                --unit "$BOX" --against "$QUAD"
                --out-dir "$CSDV_EXPORT_DIR/checks/round2_pilot")
            echo "${CMD[*]}"
            ((${#DRY[@]})) || echo "compare job: $("${CMD[@]}")"
        done
        cat <<MSG

Pilot submitted. When \`squeue -u \$USER\` is empty:
  scripts/chpc/round2.sh status pilot      # every box exported, spent under \$$CAP
Then on the laptop: scripts/chpc/round2_pull.sh pilot
MSG
        ;;
    grid)
        AFTER=$(queue round2)
        prepare round2 render "" "$AFTER"
        cat <<MSG

Batch submitted. No paid call is in this chain. In the morning:
  scripts/chpc/round2.sh status            # boxes exported, unit-years left
  scripts/chpc/round2.sh grid              # again, if anything is left
Then on the laptop: scripts/chpc/round2_pull.sh grid
MSG
        ;;
    stage1)
        CAP="${CAP:-16}"
        scripts/chpc/submit_label_chain.sh "$CSDV_EXPORT_DIR/label/round2" \
            --stage 1 --cap "$CAP" --sheets "--stage 1 --max-pairs 1" ${DRY[@]+"${DRY[@]}"}
        echo
        echo "Afterwards on the laptop: scripts/chpc/round2_pull.sh stage1"
        ;;
    stage2)
        : "${CAP:?stage2 needs --cap, the cumulative cap of the label run (76 in the plan)}"
        LABEL="$CSDV_EXPORT_DIR/label/round2"
        PLAN="config/label/round2/stage2_plan.csv"
        [[ -f "$PLAN" ]] || { echo "ERROR: no $PLAN. git pull after the stage-1 check-in." >&2; exit 1; }
        WHERE=(--label-dir "$LABEL" --runs-dir "$CSDV_RESULTS_ROOT/disturbance"
            --run-subdir default --run-tag default)
        # The thresholds of the run may have changed at the stage-1 check-in
        # (label policy and routing): the build of stages 1 and 2 reads the
        # copy in the label folder.
        CONFIG="config/label/round2/config.yaml"
        if [[ -f "$CONFIG" ]] && ! cmp -s "$CONFIG" "$LABEL/config.yaml"; then
            echo "config.yaml of the label run differs from $CONFIG:"
            diff "$LABEL/config.yaml" "$CONFIG" || true
            if [[ ${#DRY[@]} -eq 0 ]]; then
                cp "$CONFIG" "$LABEL/config.yaml"
                echo "Copied to $LABEL/config.yaml"
            fi
        fi
        [[ -f "$LABEL/config.yaml" ]] && WHERE+=(--config "$LABEL/config.yaml")
        # The plan names seasons, so the allocation lists the candidates of
        # each season again, with the sites file of the draw.
        csdv label allocate "${WHERE[@]}" --plan "$PLAN" --stage 2 \
            --sites config/label/round2/sites.csv ${DRY[@]+"${DRY[@]}"}
        scripts/chpc/submit_label_chain.sh "$LABEL" --stage 2 --cap "$CAP" \
            --build-stages 1,2 --sheets "--stage 2 --max-pairs 1" ${DRY[@]+"${DRY[@]}"}
        echo
        echo "Afterwards on the laptop: scripts/chpc/round2_pull.sh final"
        echo "Then here: scripts/chpc/round2.sh evaluate (the model tests, no paid call)"
        ;;
    evaluate)
        : "${CSDV_SBATCH_CPU:?set CSDV_SBATCH_CPU in scripts/chpc/env.sh}"
        LABELS="$CSDV_EXPORT_DIR/label"
        BASE="$LABELS/round2"
        VARIANT="$LABELS/round2_norej"
        for name in scale_test round2 stable_forest; do
            [[ -f "$LABELS/$name/chips/chips.parquet" ]] || {
                echo "ERROR: no chips/chips.parquet in $LABELS/$name. For round2:" \
                    "the build of stage 2 has to have ended." >&2
                exit 1
            }
        done
        if squeue -h -u "$USER" -o "%j" | grep '^csdv_label' | grep -qv '_norej$'; then
            echo "ERROR: a job of the label chain is still queued or running." \
                "Wait for it, so that the answers and chips are complete." >&2
            exit 1
        fi
        read -r -a CPU <<<"$CSDV_SBATCH_CPU"
        run() {
            # run COMMAND...: print it in a dry run, do it otherwise.
            if [[ ${#DRY[@]} -gt 0 ]]; then
                printf 'would run:'; printf ' %q' "$@"; echo
            else
                "$@"
            fi
        }
        submit() {
            # submit DEPENDENCY SBATCH-ARGUMENTS...: prints the job ID.
            local after="$1"
            shift
            local cmd=(sbatch --parsable "${CPU[@]}" --output="$CSDV_LOG_DIR/%x-%j.out")
            [[ -n "$after" ]] && cmd+=(--dependency="afterok:$after" --kill-on-invalid-dep=yes)
            cmd+=("$@")
            if [[ ${#DRY[@]} -gt 0 ]]; then
                printf 'would submit:' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2
                echo "<job>"
            else
                "${cmd[@]}"
            fi
        }
        # 1. The labels without decision 10: the same sample and answers,
        #    another config.yaml, the stacks through a link. No call.
        BUILD=""
        if [[ -f "$VARIANT/chips/chips.parquet" ]]; then
            echo "round2_norej is built."
        else
            run mkdir -p "$VARIANT"
            for file in windows.csv units.csv judged.csv answers.jsonl \
                locations.csv sites.csv sample_strata.csv; do
                [[ -f "$BASE/$file" ]] && run cp -p "$BASE/$file" "$VARIANT/"
            done
            [[ -d "$VARIANT/candidates" ]] || run cp -rp "$BASE/candidates" "$VARIANT/"
            [[ -e "$VARIANT/stacks" ]] || run ln -s ../round2/stacks "$VARIANT/stacks"
            run cp config/label/round2_norej/config.yaml "$VARIANT/config.yaml"
            BUILD="$(submit "" --job-name=csdv_label_build_norej \
                "$CSDV_REPO/scripts/slurm/run_label_build.sbatch" "$VARIANT" --stage 1,2)"
            echo "build of round2_norej: job $BUILD"
        fi
        # 2. The merged chip folders. Links only, so a small CPU job each.
        merge() {
            # merge NAME ROUND2-FOLDER DEPENDENCY: prints the job ID, or nothing.
            local name="$1" round="$2" after="$3"
            if [[ -f "$LABELS/$name/chips/chips.parquet" ]]; then
                echo "$name is merged." >&2
                return 0
            fi
            submit "$after" --job-name="csdv_merge_$name" --time=01:00:00 \
                "$CSDV_REPO/scripts/slurm/run_csdv_cpu.sbatch" pair merge-chips \
                --out "$LABELS/$name" --run "scale_test=$LABELS/scale_test" \
                --run "round2=$round" --run "stable_forest=$LABELS/stable_forest"
        }
        MERGE_A="$(merge merged_r2 "$BASE" "")"
        MERGE_B="$(merge merged_r2_norej "$VARIANT" "$BUILD")"
        [[ -n "$MERGE_A" ]] && echo "merge of merged_r2: job $MERGE_A"
        [[ -n "$MERGE_B" ]] && echo "merge of merged_r2_norej: job $MERGE_B"
        # 3. One GPU job per row of the run table, after both merges.
        AFTER=""
        for id in "$MERGE_A" "$MERGE_B"; do
            [[ -n "$id" ]] && AFTER="${AFTER:+$AFTER:}$id"
        done
        scripts/chpc/submit_pair_defoliation.sh --table config/pair/round2.csv \
            --predict-args "" --no-smoke ${AFTER:+--after "$AFTER"} \
            ${TABLE_ARGS[@]+"${TABLE_ARGS[@]}"} ${DRY[@]+"${DRY[@]}"}
        echo
        echo "Afterwards on the laptop: scripts/chpc/round2_pull.sh evaluate"
        ;;
    status)
        NAME="round2${WHICH:+_$WHICH}"
        LABEL="$CSDV_EXPORT_DIR/label/$NAME"
        echo "== Boxes of $NAME that are not exported yet"
        csdv grid queue-status --name "$NAME" --problems || true
        # The whole table, where round2_pull.sh finds it.
        mkdir -p "$CSDV_EXPORT_DIR/checks/$NAME"
        csdv grid queue-status --name "$NAME" \
            >"$CSDV_EXPORT_DIR/checks/$NAME/queue_status.txt" 2>&1 || true
        echo
        echo "== Your jobs in the queue: $(squeue -h -u "$USER" | wc -l)"
        squeue -h -u "$USER" -o "%.12i %.24j %.9T %.10M %R" | sort -k2 | uniq -c -f1 | head -20 || true
        # Workers of pools on other clusters (--clusters=NAME or -M NAME in
        # $CSDV_GPU_POOLS) do not show in the queue of this one.
        for CLUSTER in $(tr ';' '\n' <<<"${CSDV_GPU_POOLS:-}" |
            sed -nE 's/.*(--clusters[= ]|-M ?)([A-Za-z0-9_]+).*/\2/p' | sort -u); do
            echo
            echo "== Your jobs on $CLUSTER"
            squeue -h -M "$CLUSTER" -u "$USER" -o "%.12i %.24j %.9T %.10M %R" |
                grep -v '^CLUSTER:' | head -20 || true
        done
        echo
        echo "== Label run $LABEL"
        if [[ -f "$LABEL/windows.csv" ]]; then
            python - "$LABEL" <<'PY'
import json, sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
windows = pd.read_csv(root / "windows.csv")
print(f"windows by stage: {windows['stage'].value_counts().sort_index().to_dict()}")
done = sorted(p.stem for p in (root / "render_done").glob("*.json")) if (root / "render_done").is_dir() else []
print(f"render markers: {len(done)}")
spent, calls = 0.0, 0
ledger = root / "ledger.jsonl"
if ledger.exists():
    for line in ledger.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        spent += row.get("cost", 0.0)
        calls += 1
print(f"calls: {calls}, spent: ${spent:.2f}")
labels = root / "labels" / "labels.gpkg"
print(f"labels built: {labels.exists()}, sheets: {len(list((root / 'sheets').glob('*.png'))) if (root / 'sheets').is_dir() else 0}")
PY
        else
            echo "not prepared yet (no windows.csv)"
        fi
        ;;
    *)
        sed -n '2,41p' "$0"
        exit 2
        ;;
esac
