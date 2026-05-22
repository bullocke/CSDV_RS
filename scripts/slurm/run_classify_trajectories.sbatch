#!/bin/bash
#SBATCH --job-name=trajectories
#SBATCH --partition=notchpeak
#SBATCH --account=CHANGE_ME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/trajectories_%A_%a.out
#SBATCH --error=logs/trajectories_%A_%a.err
#
# CSDV trajectory classification SLURM array runner.
#
# Usage:
#   sbatch --array=1-$(($(wc -l < jobs.csv) - 1)) \
#          scripts/slurm/run_classify_trajectories.sbatch jobs.csv
#
# jobs.csv columns (header on line 1):
#   site,window_m,years
#
# The ``years`` field uses ``;`` to separate years (CSV uses ``,``):
#   SCBI,100,2014;2018;2022
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

IFS=',' read -r SITE WINDOW_M YEARS_SEMI <<< "$ROW"
YEARS="${YEARS_SEMI//;/,}"

echo "[$(date -Is)] task=$SLURM_ARRAY_TASK_ID site=$SITE window=${WINDOW_M}m years=$YEARS"

micromamba run -n "$ENV_NAME" csdv classify-trajectories \
  --site "$SITE" \
  --window-m "$WINDOW_M" \
  --years "$YEARS"

echo "[$(date -Is)] done."
