#!/usr/bin/env bash
# Submit CHM inference (GPU array) and crown segmentation (CPU array) for every
# NAIP GeoTIFF already in the store. Run on a login node.
#
# Usage:
#   source scripts/chpc/env.sh
#   scripts/slurm/submit_chm_crowns.sh SITES [YEARS] [--granite] [--dry-run]
#
#   SITES      Comma-separated site folders under $CSDV_DATA_ROOT/naip/.
#   YEARS      Comma-separated years. Default: every year folder found.
#   --granite  Use the preemptable granite-gpu freecycle partition (L40S),
#              with --requeue, instead of notchpeak-gpu.
#   --dry-run  Write the manifest and print the sbatch commands only.
#
# Examples:
#   scripts/slurm/submit_chm_crowns.sh SantaClara
#   scripts/slurm/submit_chm_crowns.sh SantaClara,SCBI 2018,2022 --dry-run
#
# Images whose CHM or crowns already exist are skipped inside the jobs, so
# resubmitting after a failure only redoes what is missing.

set -euo pipefail

usage() {
    sed -n '2,21p' "$0" >&2
    exit "${1:-2}"
}

[[ $# -ge 1 ]] || usage
: "${CSDV_REPO:?source scripts/chpc/env.sh first}"

SITES_CSV="$1"
shift
YEARS_CSV=""
GPU_OPTS=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --granite)
            GPU_OPTS=(--partition=granite-gpu --qos=granite-gpu-freecycle
                --account=wangj --requeue)
            ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) usage 0 ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) YEARS_CSV="$1" ;;
    esac
    shift
done

IFS=',' read -r -a SITES <<<"$SITES_CSV"
YEARS=()
[[ -n "$YEARS_CSV" ]] && IFS=',' read -r -a YEARS <<<"$YEARS_CSV"

# Preflight: fail here rather than inside a queued job.
for f in elevation climate_pca soil_pca nlcd ecoregion; do
    if [[ ! -f "$CSDV_CONDITIONING_DIR/$f.tif" ]]; then
        echo "ERROR: $CSDV_CONDITIONING_DIR/$f.tif missing. Fetch the conditioning" \
            "rasters first: sbatch scripts/slurm/run_download_dtn.sbatch conditioning" >&2
        exit 1
    fi
done
if [[ ! -f "$NAIPCHM_REPO_DIR/scripts/inference.py" ]]; then
    echo "ERROR: naip-chm repo not found at $NAIPCHM_REPO_DIR" >&2
    exit 1
fi

# Build the manifest: one row per NAIP tif.
mkdir -p "$CSDV_LOG_DIR" "$CSDV_MANIFEST_DIR"
MANIFEST="$CSDV_MANIFEST_DIR/chm_crowns_$(date +%Y%m%dT%H%M%S).csv"
echo "site,year,naip_path,crowns_name" >"$MANIFEST"
for site in "${SITES[@]}"; do
    site_dir="$CSDV_DATA_ROOT/naip/$site"
    if [[ ! -d "$site_dir" ]]; then
        echo "WARNING: no NAIP folder for $site ($site_dir)" >&2
        continue
    fi
    if [[ ${#YEARS[@]} -gt 0 ]]; then
        year_list=("${YEARS[@]}")
    else
        mapfile -t year_list < <(find "$site_dir" -mindepth 1 -maxdepth 1 -type d \
            -printf '%f\n' | sort)
    fi
    for year in "${year_list[@]}"; do
        mapfile -t tifs < <(find "$site_dir/$year" -maxdepth 1 -name '*.tif' \
            2>/dev/null | sort)
        if [[ ${#tifs[@]} -eq 0 ]]; then
            echo "WARNING: no NAIP tif in $site_dir/$year" >&2
            continue
        fi
        for tif in "${tifs[@]}"; do
            if [[ ${#tifs[@]} -eq 1 ]]; then
                crowns_name="crowns.gpkg"
            else
                crowns_name="$(basename "$tif" .tif)_crowns.gpkg"
            fi
            echo "$site,$year,$tif,$crowns_name" >>"$MANIFEST"
        done
    done
done

N=$(($(wc -l <"$MANIFEST") - 1))
if [[ $N -lt 1 ]]; then
    echo "ERROR: no NAIP images found; nothing to submit." >&2
    exit 1
fi
echo "Manifest: $MANIFEST ($N images)"
column -s, -t <"$MANIFEST"

CHM_CMD=(sbatch --parsable --array="1-$N" ${GPU_OPTS[@]+"${GPU_OPTS[@]}"}
    --output="$CSDV_LOG_DIR/%x-%A_%a.out"
    "$CSDV_REPO/scripts/slurm/run_chm_inference.sbatch" "$MANIFEST")
# set_crowns_cmd DEP_JOB: crown task i waits for CHM task i (aftercorr), and is
# cancelled rather than left pending forever if that task fails.
set_crowns_cmd() {
    CROWNS_CMD=(sbatch --parsable --array="1-$N" --dependency="aftercorr:$1"
        --kill-on-invalid-dep=yes --output="$CSDV_LOG_DIR/%x-%A_%a.out"
        "$CSDV_REPO/scripts/slurm/run_segment_crowns.sbatch" "$MANIFEST")
}

if ((DRY_RUN)); then
    set_crowns_cmd "<CHM_JOB>"
    echo "[dry-run] ${CHM_CMD[*]}"
    echo "[dry-run] ${CROWNS_CMD[*]}"
    exit 0
fi

CHM_JOB=$("${CHM_CMD[@]}")
set_crowns_cmd "$CHM_JOB"
CROWNS_JOB=$("${CROWNS_CMD[@]}")

echo "CHM inference job:    $CHM_JOB"
echo "Crown segmentation:   $CROWNS_JOB (task i waits for CHM task i)"
echo "Logs:                 $CSDV_LOG_DIR"
echo "Watch:  squeue -u $USER   |   sacct -j $CHM_JOB,$CROWNS_JOB -X" \
    "--format=JobID,JobName,State,Elapsed,MaxRSS,NodeList"
