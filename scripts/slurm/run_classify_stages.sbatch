#!/bin/bash
#SBATCH --job-name=stages
#SBATCH --partition=notchpeak
#SBATCH --account=CHANGE_ME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stages_%A_%a.out
#SBATCH --error=logs/stages_%A_%a.err
#
# CSDV stage classification SLURM array runner.
#
# Usage:
#   sbatch --array=1-$(($(wc -l < jobs.csv) - 1)) \
#          scripts/slurm/run_classify_stages.sbatch jobs.csv
#
# jobs.csv columns (header on line 1):
#   site,year,window_m
#
# Required env:
#   CSDV_ENV_NAME      Micromamba env name (default: csdv)

set -euo pipefail

JOBS_CSV="${1:-jobs.csv}"
ENV_NAME="${CSDV_ENV_NAME:-csdv}"

mkdir -p logs

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE_NUM}p" "$JOBS_CSV")
if [[ -z "$ROW" ]]; then
  echo "ERROR: empty row at line $LINE_NUM of $JOBS_CSV" >&2
  exit 1
fi

IFS=',' read -r SITE YEAR WINDOW_M <<< "$ROW"

echo "[$(date -Is)] task=$SLURM_ARRAY_TASK_ID site=$SITE year=$YEAR window=${WINDOW_M}m"

micromamba run -n "$ENV_NAME" csdv classify-stages \
  --site "$SITE" \
  --year "$YEAR" \
  --window-m "$WINDOW_M"

echo "[$(date -Is)] done."
