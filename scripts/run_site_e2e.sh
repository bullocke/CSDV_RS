#!/usr/bin/env bash
# scripts/run_site_e2e.sh — Idempotent end-to-end dispatch for one CSDV site.
#
# Usage:
#   scripts/run_site_e2e.sh SITE YEARS WINDOW_M [--force] [--only STEP] [--skip-download]
#
# Examples:
#   scripts/run_site_e2e.sh SCBI 2014,2018 50
#   scripts/run_site_e2e.sh SCBI 2014,2018 50 --only check
#   scripts/run_site_e2e.sh SCBI 2018 50 --force
#
# Steps run in this order, each skipped if its sentinel output exists:
#   1. csdv check
#   2. csdv download conditioning (once, only if a CHM stage will actually run)
#   3. per year: NAIP download, NAIP-CHM inference, crown segmentation, compute-metrics
#   4. csdv stratify (site-level)
#   5. per year: csdv classify-stages
#   6. csdv classify-trajectories (site-level)
#
# Environment:
#   CSDV_DATA_ROOT, CSDV_RESULTS_ROOT, CSDV_CACHE_ROOT - resolved by `csdv` and
#   used here verbatim; falls back to repo-relative `data/` and `results/`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    sed -n '2,20p' "$0" >&2
    exit "${1:-2}"
}

if [[ $# -lt 3 ]]; then
    usage
fi

SITE="$1"
YEARS_CSV="$2"
WINDOW_M="$3"
shift 3

FORCE=0
ONLY=""
SKIP_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1 ;;
        --only) ONLY="$2"; shift ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
    shift
done

IFS=',' read -r -a YEARS <<<"$YEARS_CSV"

# ---------------------------------------------------------------------------
# Paths + logging
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${CSDV_DATA_ROOT:-$REPO_ROOT/data}"
RESULTS_ROOT="${CSDV_RESULTS_ROOT:-$REPO_ROOT/results}"
RUN_ID="$(date +%Y%m%dT%H%M%S)"
LOG_DIR="$RESULTS_ROOT/logs/$SITE/$RUN_ID"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"

ts() { date +'%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$SUMMARY"; }

log "site=$SITE years=${YEARS[*]} window_m=$WINDOW_M force=$FORCE only=${ONLY:-<all>}"
log "data_root=$DATA_ROOT"
log "results_root=$RESULTS_ROOT"
log "log_dir=$LOG_DIR"

# run_step NAME SENTINEL CMD...
# - If $SENTINEL is non-empty, exists, and FORCE=0, mark as skipped and return.
# - Else run CMD, tee stdout/stderr to <LOG_DIR>/<NAME>.log.
# No --only scope check; callers that need one use `step` instead.
run_step() {
    local name="$1"
    local sentinel="$2"
    shift 2
    if [[ "$FORCE" -eq 0 && -n "$sentinel" && -e "$sentinel" ]]; then
        log "[skip] $name (sentinel exists: $sentinel)"
        return 0
    fi
    local step_log="$LOG_DIR/$name.log"
    local start
    start=$(date +%s)
    log "[run]  $name -> $step_log"
    if "$@" >"$step_log" 2>&1; then
        local elapsed=$(( $(date +%s) - start ))
        log "[ok]   $name (${elapsed}s)"
    else
        local rc=$?
        local elapsed=$(( $(date +%s) - start ))
        log "[fail] $name rc=$rc (${elapsed}s); see $step_log"
        exit "$rc"
    fi
}

# step NAME SENTINEL CMD...
# Like run_step, but first honors the --only scope filter.
step() {
    local name="$1"
    if ! step_in_scope "$name"; then
        log "[skip] $name (not in --only)"
        return 0
    fi
    run_step "$@"
}

# step_in_scope NAME: true if --only is empty or NAME matches the --only prefix.
step_in_scope() {
    local name="$1"
    if [[ -z "$ONLY" ]]; then
        return 0
    fi
    [[ "$name" == "$ONLY" || "$name" == "$ONLY"-* ]]
}

# Skip-or-glob helpers.
naip_tif_for_year() {
    local year="$1"
    local d="$DATA_ROOT/naip/$SITE/$year"
    [[ -d "$d" ]] || return 1
    find "$d" -maxdepth 1 -name '*.tif' -print -quit
}

chm_tif_for_year() {
    local year="$1"
    local d="$DATA_ROOT/naip_chm/$SITE/$year"
    [[ -d "$d" ]] || return 1
    find "$d" -maxdepth 1 -name '*.tif' -print -quit
}

# True if any year's CHM inference is in scope AND has no existing CHM tif
# (i.e. inference will run and therefore needs the conditioning rasters).
chm_inference_needed() {
    local year
    for year in "${YEARS[@]}"; do
        step_in_scope "chm-$year" || continue
        chm_tif_for_year "$year" >/dev/null 2>&1 || return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------

step check "" csdv check \
    --site "$SITE" \
    --years "$YEARS_CSV" \
    --window-m "$WINDOW_M"

# ---------------------------------------------------------------------------
# 1b. Conditioning rasters (static, CONUS-wide, ~1.65 GB, one-time).
# ---------------------------------------------------------------------------
# Ideally pre-fetched on the login node (see docs/workflow_chpc.md); this is
# the in-pipeline safety net. Runs only when a CHM inference will actually
# execute and the rasters are absent. The sentinel is the last raster written.
if [[ "$SKIP_DOWNLOAD" -ne 1 ]] && chm_inference_needed; then
    COND_DIR="${CSDV_CONDITIONING_DIR:-$DATA_ROOT/chm_model/conditioning}"
    run_step "conditioning" "$COND_DIR/ecoregion.tif" \
        csdv download conditioning
fi

# ---------------------------------------------------------------------------
# 2. Per-year input + metric steps
# ---------------------------------------------------------------------------

for year in "${YEARS[@]}"; do
    naip_dir="$DATA_ROOT/naip/$SITE/$year"
    naip_chm_dir="$DATA_ROOT/naip_chm/$SITE/$year"
    crowns_dir="$RESULTS_ROOT/crowns/$SITE/$year"
    crowns_path="$crowns_dir/crowns.gpkg"
    metrics_dir="$RESULTS_ROOT/metrics/$SITE/$year/${WINDOW_M%.*}m"
    metrics_manifest="$metrics_dir/manifest.yaml"

    # 2a. NAIP download (skipped if any tif already exists in the naip dir).
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
        log "[skip] naip-$year (--skip-download)"
    else
        # Sentinel: presence of any *.tif (allows collaborator-supplied DOQQs).
        naip_sentinel=""
        if existing=$(naip_tif_for_year "$year"); then
            naip_sentinel="$existing"
        fi
        mkdir -p "$naip_dir"
        step "naip-$year" "$naip_sentinel" \
            csdv download naip \
                --site "$SITE" \
                --start-date "${year}-04-01" \
                --end-date "${year}-11-30" \
                --out-dir "$naip_dir"
    fi

    # 2b. NAIP-CHM inference (input: NAIP tif from 2a).
    if step_in_scope "chm-$year"; then
        chm_sentinel=""
        if existing=$(chm_tif_for_year "$year"); then
            chm_sentinel="$existing"
        fi
        if [[ -z "$chm_sentinel" ]]; then
            if ! naip_input=$(naip_tif_for_year "$year"); then
                log "[fail] chm-$year: no NAIP tif found in $naip_dir"
                exit 1
            fi
            step "chm-$year" "" \
                csdv chm-inference \
                    --naip-quad "$naip_input" \
                    --output-dir "$naip_chm_dir"
        else
            log "[skip] chm-$year (sentinel exists: $chm_sentinel)"
        fi
    else
        log "[skip] chm-$year (not in --only)"
    fi

    # 2c. Crown segmentation (input: CHM tif from 2b).
    if step_in_scope "crowns-$year"; then
        if ! chm_input=$(chm_tif_for_year "$year"); then
            log "[fail] crowns-$year: no CHM tif found in $naip_chm_dir"
            exit 1
        fi
        mkdir -p "$crowns_dir"
        step "crowns-$year" "$crowns_path" \
            csdv segment-crowns \
                --chm "$chm_input" \
                --out-crowns "$crowns_path"
    else
        log "[skip] crowns-$year (not in --only)"
    fi

    # 2d. Pass-1 metrics.
    step "metrics-$year" "$metrics_manifest" \
        csdv compute-metrics \
            --site "$SITE" \
            --year "$year" \
            --window-m "$WINDOW_M"
done

# ---------------------------------------------------------------------------
# 3. Stratification (site-level)
# ---------------------------------------------------------------------------

strat_sentinel="$RESULTS_ROOT/stratification/$SITE/${WINDOW_M%.*}m/site_type.tif"
step "stratify" "$strat_sentinel" \
    csdv stratify --site "$SITE" --window-m "$WINDOW_M"

# ---------------------------------------------------------------------------
# 4. Per-year stage classification
# ---------------------------------------------------------------------------

for year in "${YEARS[@]}"; do
    stage_sentinel="$RESULTS_ROOT/stages/$SITE/$year/${WINDOW_M%.*}m/stage.tif"
    step "stages-$year" "$stage_sentinel" \
        csdv classify-stages \
            --site "$SITE" \
            --year "$year" \
            --window-m "$WINDOW_M"
done

# ---------------------------------------------------------------------------
# 5. Trajectory classification (site-level, multi-year)
# ---------------------------------------------------------------------------

if [[ "${#YEARS[@]}" -ge 2 ]]; then
    traj_sentinel="$RESULTS_ROOT/trajectories/$SITE/${WINDOW_M%.*}m/trajectory.tif"
    step "trajectories" "$traj_sentinel" \
        csdv classify-trajectories \
            --site "$SITE" \
            --years "$YEARS_CSV" \
            --window-m "$WINDOW_M"
else
    log "[skip] trajectories (need >=2 years; got ${YEARS[*]})"
fi

log "done. logs in $LOG_DIR"
