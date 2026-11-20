#!/bin/bash
#SBATCH --job-name=csdv_label_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Start a label run on CHPC from the grid runs of its sites: build the
# candidates, draw the sample by class, and submit what follows. CPU only.
# This script names no allocation: pass $CSDV_SBATCH_CPU at submission.
#
# Usage (after `source scripts/chpc/env.sh`):
#   sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_label_prepare.sbatch LABEL_DIR CONFIG_DIR \
#       [--then render | --then "--stage 1 --cap 3"] [csdv label sample options...]
#
#   LABEL_DIR   Label run folder, created here ($CSDV_EXPORT_DIR/label/NAME).
#   CONFIG_DIR  Folder in the repository with the run's tracked inputs:
#               sites.csv (first column `unit`, then role, interval,
#               location and whatever else the siting wrote) and, optionally,
#               config.yaml (thresholds the run changes).
#   --then      What to submit afterwards with scripts/chpc/submit_label_chain.sh:
#               `render` renders stage 1 and stops before any paid call.
#               A quoted option string is passed on as it is, for a run that
#               may go straight to the calls (the pilot). Default: nothing.
#   --then-sheets  With a chain: the `csdv label sheets` options of the
#               sheets drawn after the build (quoted).
#
# Steps:
#   1. Copy sites.csv to LABEL_DIR/locations.csv (the location text of the
#      prompts) and config.yaml beside it.
#   2. `csdv label candidates` for the sites that have a finished run. Sites
#      without one are named and left out.
#   3. `csdv label sample --design classes --sites ...`, first as a dry run
#      whose table is in the log. A folder that has windows.csv is not
#      sampled again.
#   4. Submit the chain named by --then.
#
# Resources: candidates take a few seconds per box and about a minute per
# whole quad.

set -euo pipefail

USAGE="usage: sbatch $0 LABEL_DIR CONFIG_DIR [--then render|\"chain options\"] [sample options]"
LABEL_DIR="${1:?$USAGE}"
CONFIG_DIR="${2:?$USAGE}"
shift 2
THEN=""
SHEETS=""
SAMPLE_ARGS=()
while (($#)); do
    case "$1" in
        --then) THEN="${2:?--then needs a value}"; shift 2 ;;
        --then-sheets) SHEETS="${2:?--then-sheets needs a value}"; shift 2 ;;
        *) SAMPLE_ARGS+=("$1"); shift ;;
    esac
done

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"

RUN_TAG="${CSDV_LABEL_RUN_TAG:-default}"
SITES="$CONFIG_DIR/sites.csv"
[[ -f "$SITES" ]] || { echo "ERROR: no $SITES" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "label_dir=$LABEL_DIR config_dir=$CONFIG_DIR then=${THEN:-nothing}"

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export MPLBACKEND=Agg
activate_csdv

mkdir -p "$LABEL_DIR"
cp "$SITES" "$LABEL_DIR/locations.csv"
cp "$SITES" "$LABEL_DIR/sites.csv"
WHERE=(--label-dir "$LABEL_DIR"
    --runs-dir "$CSDV_RESULTS_ROOT/disturbance" --run-subdir "$RUN_TAG" --run-tag "$RUN_TAG")
if [[ -f "$CONFIG_DIR/config.yaml" ]]; then
    cp "$CONFIG_DIR/config.yaml" "$LABEL_DIR/config.yaml"
    WHERE+=(--config "$LABEL_DIR/config.yaml")
    echo "thresholds of this run:"
    cat "$LABEL_DIR/config.yaml"
fi

# The sites that have a finished run, from the first column of sites.csv.
READY=()
MISSING=()
while IFS=, read -r UNIT _; do
    UNIT="${UNIT//$'\r'/}"
    [[ -z "$UNIT" || "$UNIT" == "unit" || "$UNIT" == \#* ]] && continue
    if [[ -f "$CSDV_RESULTS_ROOT/disturbance/$UNIT/$RUN_TAG/summary.json" ]]; then
        READY+=("$UNIT")
    else
        MISSING+=("$UNIT")
    fi
done <"$SITES"
echo "${#READY[@]} sites have a run, ${#MISSING[@]} do not: ${MISSING[*]-}"
((${#READY[@]})) || { echo "ERROR: no site has a finished run." >&2; exit 1; }
UNITS=$(IFS=,; echo "${READY[*]}")

csdv label candidates "${WHERE[@]}" --units "$UNITS"

if [[ -f "$LABEL_DIR/windows.csv" ]]; then
    echo "windows.csv exists: the sample is drawn once and is kept."
else
    SAMPLE=(csdv label sample "${WHERE[@]}" --design classes --sites "$SITES"
        --units "$UNITS" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"})
    "${SAMPLE[@]}" --dry-run
    "${SAMPLE[@]}"
fi
wc -l "$LABEL_DIR/windows.csv" "$LABEL_DIR/judged.csv" "$LABEL_DIR/units.csv"

case "$THEN" in
    "") echo "Nothing submitted (no --then)." ;;
    render)
        "$CSDV_REPO/scripts/chpc/submit_label_chain.sh" "$LABEL_DIR" --stage 1 --render-only
        ;;
    *)
        EXTRA=()
        [[ -n "$SHEETS" ]] && EXTRA=(--sheets "$SHEETS")
        # shellcheck disable=SC2086
        "$CSDV_REPO/scripts/chpc/submit_label_chain.sh" "$LABEL_DIR" $THEN \
            ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
esac
echo "done $(date -Is)"
