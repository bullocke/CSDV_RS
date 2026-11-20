#!/usr/bin/env bash
# Submit one stage of a label run as a chain of jobs: render the request
# images, ask the two vision models, render and ask for the outline
# polygons, build the labels. Run on a CHPC login node after
# `source scripts/chpc/env.sh`.
#
# Usage:
#   scripts/chpc/submit_label_chain.sh LABEL_DIR --stage S --cap DOLLARS
#       [--build-stages LIST] [--after JOBID] [--no-polygons] [--no-build]
#       [--throttle K] [--dry-run]
#
#   LABEL_DIR       Label run folder on CHPC (windows.csv, units.csv,
#                   candidates/, and for a first stage no answers yet).
#   --stage S       Stage of windows.csv to render and judge (1, or 1,2).
#   --cap DOLLARS   Spending cap of the whole label run so far, this stage
#                   included. Required. The verdict jobs refuse to start when
#                   the expected cost would pass it.
#   --build-stages  Stages the build covers. Default: the value of --stage.
#                   For a second stage pass 1,2, because the build writes its
#                   tables anew for the windows it is given.
#   --after JOBID   Start the chain after this job has finished well.
#   --no-polygons   Leave out the polygon pass (labels use the candidate
#                   outlines, with an unlabelled ring where a model doubts them).
#   --no-build      Stop after the calls.
#   --render-only   Render the request images of the stage and stop. No
#                   call is made and --cap is not needed.
#   --sheets ARGS   After the build, draw review sheets with these
#                   `csdv label sheets` options (quoted, e.g. "--stage 1
#                   --max-pairs 1").
#   --throttle K    Render tasks running at one time (default 20).
#   --dry-run       Print the sbatch commands, submit nothing.
#
# The chain, with what stops it:
#   render (array, one task per unit)
#     -> verdicts (data transfer node)   afterany: it counts unrendered
#                                        windows itself and refuses above 10%
#     -> verdicts again                  afterany: sends only what a hung call
#                                        or the walltime left, else nothing
#     -> polygon render (array)          afterok
#     -> polygon calls                   afterany, same guard
#     -> build                           afterok
# A wrong key, a closed account or a cap that is too low ends the first
# verdict job with a non-zero code, and Slurm cancels everything behind it
# (--kill-on-invalid-dep). Nothing is built on missing answers.
#
# Afterwards: `squeue -u $USER`, then the sacct line this script prints.

set -euo pipefail

usage() { sed -n '2,44p' "$0"; }

STAGE="" CAP="" BUILD_STAGES="" AFTER="" POLYGONS=1 BUILD=1 THROTTLE=20 DRY=0
RENDER_ONLY=0 SHEETS=""
POS=()
while (($#)); do
    case "$1" in
        --stage) STAGE="${2:?--stage needs a value}"; shift 2 ;;
        --cap) CAP="${2:?--cap needs a value}"; shift 2 ;;
        --build-stages) BUILD_STAGES="${2:?--build-stages needs a value}"; shift 2 ;;
        --after) AFTER="${2:?--after needs a job ID}"; shift 2 ;;
        --no-polygons) POLYGONS=0; shift ;;
        --no-build) BUILD=0; shift ;;
        --render-only) RENDER_ONLY=1; shift ;;
        --sheets) SHEETS="${2:?--sheets needs the options of csdv label sheets}"; shift 2 ;;
        --throttle) THROTTLE="${2:?--throttle needs a number}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) POS+=("$1"); shift ;;
    esac
done
LABEL="${POS[0]:?$(usage)}"
: "${STAGE:?--stage is required}"
BUILD_STAGES="${BUILD_STAGES:-$STAGE}"

: "${CSDV_REPO:?source scripts/chpc/env.sh first}"
: "${CSDV_SBATCH_CPU:?CSDV_SBATCH_CPU is not set in scripts/chpc/env.sh}"
for needed in windows.csv units.csv; do
    [[ -f "$LABEL/$needed" ]] || { echo "ERROR: $LABEL has no $needed." >&2; exit 1; }
done
if ((!RENDER_ONLY)); then
    : "${CAP:?--cap is required}"
    : "${CSDV_KEYS_FILE:?CSDV_KEYS_FILE is not set in scripts/chpc/env.sh}"
    [[ -f "$CSDV_KEYS_FILE" ]] || { echo "ERROR: no key file at $CSDV_KEYS_FILE." >&2; exit 1; }
fi
mkdir -p "$CSDV_LOG_DIR"

N=$(($(wc -l <"$LABEL/units.csv") - 1))
((N >= 1)) || { echo "ERROR: $LABEL/units.csv names no unit." >&2; exit 1; }
SLURM="$CSDV_REPO/scripts/slurm"
ARRAY="--array=1-${N}%${THROTTLE}"
LOG_ARRAY="--output=$CSDV_LOG_DIR/%x-%A_%a.out"
LOG="--output=$CSDV_LOG_DIR/%x-%j.out"
KILL="--kill-on-invalid-dep=yes"

# submit DEPENDENCY SBATCH_ARGS...: print the command, submit it unless this
# is a dry run, and set JOB to the job ID (a placeholder in a dry run).
N_SUBMITTED=0
submit() {
    local dep="$1"
    shift
    local cmd=(sbatch --parsable)
    [[ -n "$dep" ]] && cmd+=("--dependency=$dep" "$KILL")
    cmd+=("$@")
    echo "${cmd[*]}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
    if ((DRY)); then
        JOB="<job$N_SUBMITTED>"
    else
        JOB=$("${cmd[@]}")
        JOB="${JOB%%;*}"
    fi
    IDS+=("$JOB")
}

IDS=()
echo "Label run $LABEL: $N units, stage $STAGE, cap \$${CAP:-none}, build stages $BUILD_STAGES"

# shellcheck disable=SC2086
submit "${AFTER:+afterok:$AFTER}" $CSDV_SBATCH_CPU $ARRAY "$LOG_ARRAY" \
    "$SLURM/run_label_render.sbatch" "$LABEL" --stage "$STAGE"
RENDER=$JOB
if ((RENDER_ONLY)); then
    echo
    ((DRY)) && echo "Dry run: nothing was submitted." || echo "Submitted the render array $JOB. No call is made."
    exit 0
fi
submit "afterany:$RENDER,singleton" "$LOG" \
    "$SLURM/run_label_verdict.sbatch" "$LABEL" --stage "$STAGE" --cap "$CAP"
submit "afterany:$JOB,singleton" "$LOG" \
    "$SLURM/run_label_verdict.sbatch" "$LABEL" --stage "$STAGE" --cap "$CAP"
LAST=$JOB

if ((POLYGONS)); then
    # shellcheck disable=SC2086
    submit "afterok:$LAST" $CSDV_SBATCH_CPU $ARRAY "$LOG_ARRAY" \
        "$SLURM/run_label_render.sbatch" "$LABEL" --kind polygon --stage "$STAGE"
    submit "afterany:$JOB,singleton" "$LOG" \
        "$SLURM/run_label_verdict.sbatch" "$LABEL" --kind polygon --stage "$STAGE" --cap "$CAP"
    LAST=$JOB
fi

if ((BUILD)); then
    # shellcheck disable=SC2086
    submit "afterok:$LAST" $CSDV_SBATCH_CPU "$LOG" \
        "$SLURM/run_label_build.sbatch" "$LABEL" --stage "$BUILD_STAGES"
    if [[ -n "$SHEETS" ]]; then
        # shellcheck disable=SC2086
        submit "afterok:$JOB" $CSDV_SBATCH_CPU "$LOG" \
            "$SLURM/run_label_sheets.sbatch" "$LABEL" $SHEETS
    fi
fi

LIST=$(IFS=,; echo "${IDS[*]}")
echo
if ((DRY)); then
    echo "Dry run: nothing was submitted."
else
    echo "Submitted $N_SUBMITTED jobs. The last one is $JOB."
    echo "Watch: sacct -j $LIST -X --format=JobID,JobName%24,State,Elapsed,ExitCode"
    echo "$JOB" >"$LABEL/last_chain_job.txt"
fi
