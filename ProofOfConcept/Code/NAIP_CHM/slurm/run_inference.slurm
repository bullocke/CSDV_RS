#!/bin/bash
#SBATCH --job-name=naipchm
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/naipchm_%A_%a.out
#SBATCH --error=logs/naipchm_%A_%a.err
#
# Usage:
#   sbatch --array=1-$(($(wc -l < jobs.csv) - 1)) slurm/run_inference.slurm jobs.csv
#
# jobs.csv columns (header on line 1):
#   site,year,date,half_size_km,naip_path,chm_out_dir
#
# Required env vars (set in submission shell or here):
#   NAIPCHM_REPO_DIR  Path to cloned smorf-ntsg/naip-chm
#   NAIPCHM_COND_DIR  Path to conditioning rasters directory
#   NAIPCHM_ENV_NAME  Micromamba env name (default: naipchm)

set -euo pipefail

JOBS_CSV="${1:-jobs.csv}"
ENV_NAME="${NAIPCHM_ENV_NAME:-naipchm}"

mkdir -p logs

if [[ -z "${NAIPCHM_REPO_DIR:-}" || -z "${NAIPCHM_COND_DIR:-}" ]]; then
  echo "ERROR: NAIPCHM_REPO_DIR and NAIPCHM_COND_DIR must be set." >&2
  exit 1
fi

# SLURM_ARRAY_TASK_ID is 1-based. Skip header line.
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE_NUM}p" "$JOBS_CSV")
if [[ -z "$ROW" ]]; then
  echo "ERROR: empty row at line $LINE_NUM of $JOBS_CSV" >&2
  exit 1
fi

IFS=',' read -r SITE YEAR DATE HALFKM NAIP_PATH CHM_OUT <<< "$ROW"

echo "[$(date -Is)] task=$SLURM_ARRAY_TASK_ID site=$SITE year=$YEAR date=$DATE"
echo "  naip : $NAIP_PATH"
echo "  out  : $CHM_OUT"

if [[ ! -f "$NAIP_PATH" ]]; then
  echo "ERROR: NAIP input not found: $NAIP_PATH" >&2
  echo "Hint: run src_local/prepare_naip_aoi.py --jobs-csv $JOBS_CSV before submitting." >&2
  exit 1
fi

mkdir -p "$CHM_OUT"

HERE="$(dirname "$0")/.."
micromamba run -n "$ENV_NAME" python "$HERE/src_local/run_inference.py" \
  --naip-quad "$NAIP_PATH" \
  --output-dir "$CHM_OUT" \
  --repo-dir "$NAIPCHM_REPO_DIR" \
  --cond-dir "$NAIPCHM_COND_DIR"

echo "[$(date -Is)] done."
