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
# CSDV NAIP-CHM inference SLURM array runner.
#
# Usage:
#   sbatch --array=1-$(($(wc -l < jobs.csv) - 1)) \
#          scripts/slurm/run_chm_inference.sbatch jobs.csv
#
# jobs.csv columns (header on line 1):
#   site,year,date,half_size_km,naip_path,chm_out_dir
#
# Required env vars (set in submission shell or here):
#   NAIPCHM_REPO_DIR   Path to cloned smorf-ntsg/naip-chm
#   NAIPCHM_COND_DIR   Path to conditioning rasters directory (optional;
#                      defaults resolved via csdv_core.io.paths if unset)
#   CSDV_ENV_NAME      Micromamba env name (default: csdv)

set -euo pipefail

JOBS_CSV="${1:-jobs.csv}"
ENV_NAME="${CSDV_ENV_NAME:-csdv}"

mkdir -p logs

if [[ -z "${NAIPCHM_REPO_DIR:-}" ]]; then
  echo "ERROR: NAIPCHM_REPO_DIR must be set." >&2
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
  exit 1
fi

mkdir -p "$CHM_OUT"

COND_FLAG=()
if [[ -n "${NAIPCHM_COND_DIR:-}" ]]; then
  COND_FLAG+=(--conditioning-dir "$NAIPCHM_COND_DIR")
fi

micromamba run -n "$ENV_NAME" csdv chm-inference \
  --naip-quad "$NAIP_PATH" \
  --output-dir "$CHM_OUT" \
  --repo-dir "$NAIPCHM_REPO_DIR" \
  "${COND_FLAG[@]}"

echo "[$(date -Is)] done."
