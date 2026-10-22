#!/bin/bash
#SBATCH --job-name=csdv_chm_fetch
#SBATCH --partition=notchpeak-dtn
#SBATCH --qos=notchpeak-dtn
#SBATCH --account=dtn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Measure NTSG download throughput for the national NAIP-CHM from a CHPC data
# transfer node. The manifest (written by
# analysis/NAIP_National_Crown_AGB_Map/04_chpc_manifests.py) lists about 300
# stratified tiles (~38 GB). They are split into equal chunks, and each chunk is
# fetched at a different concurrency (4, 8, 16, 32 workers), so the reports give
# aggregate MB/s against worker count. The tiles are kept, because the crown
# benchmark (run_crown_benchmark.sbatch) reads a subset of them.
#
# Ask NTSG before running anything larger than this test.
#
# Submit from a login node after `source scripts/chpc/env.sh`:
#   OUT=/scratch/general/vast/$USER/naip_chm_national
#   sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
#       scripts/slurm/run_naipchm_fetch_dtn.sbatch \
#       "$CSDV_MANIFEST_DIR/national_fetch_manifest.csv" "$OUT"
#
# Output: $OUT/tiles/*.tif and $OUT/fetch_reports/report_w<N>.csv.
# VAST scratch is purged after 60 days without access.

set -euo pipefail

MANIFEST="${1:?usage: sbatch $0 MANIFEST.csv OUT_DIR [WORKER_LEVELS]}"
OUT="${2:?usage: sbatch $0 MANIFEST.csv OUT_DIR [WORKER_LEVELS]}"
LEVELS="${3:-4 8 16 32}"

source "${CSDV_REPO:?source scripts/chpc/env.sh before sbatch}/scripts/chpc/env.sh"
activate_csdv

echo "host=$(hostname) job=$SLURM_JOB_ID started=$(date -Is)"
mkdir -p "$OUT/tiles" "$OUT/fetch_reports" "$OUT/fetch_chunks"

# One URL per line, header dropped. The manifest's chm_url column is first.
URLS="$OUT/fetch_chunks/all_urls.txt"
python - "$MANIFEST" "$URLS" <<'PY'
import sys

import pandas as pd

urls = pd.read_csv(sys.argv[1])["chm_url"].astype(str)
urls.to_csv(sys.argv[2], index=False, header=False)
print(f"{len(urls)} URLs")
PY

read -r -a LEVEL_ARR <<<"$LEVELS"
N_LEVELS=${#LEVEL_ARR[@]}
split -n "l/$N_LEVELS" -d "$URLS" "$OUT/fetch_chunks/chunk_"

i=0
for chunk in "$OUT"/fetch_chunks/chunk_*; do
    W=${LEVEL_ARR[$i]}
    echo "== chunk $(basename "$chunk"): $(wc -l <"$chunk") tiles at $W workers"
    START=$(date +%s)
    csdv download naip-chm-http \
        --urls-file "$chunk" \
        --out-dir "$OUT/tiles" \
        --workers "$W" \
        --report "$OUT/fetch_reports/report_w${W}.csv" || true
    echo "chunk wall $(($(date +%s) - START)) s"
    i=$((i + 1))
done

python - "$OUT/fetch_reports" <<'PY'
import sys
from pathlib import Path

import pandas as pd

rows = []
for p in sorted(Path(sys.argv[1]).glob("report_w*.csv")):
    r = pd.read_csv(p)
    ok = r[r["error"].fillna("") == ""]
    moved = ok.loc[~ok["skipped"], "n_bytes"].sum()
    rows.append(
        {
            "workers": int(r["workers"].iloc[0]),
            "tiles": len(r),
            "failed": len(r) - len(ok),
            "gb": moved / 1e9,
            "per_stream_mb_s": (ok["n_bytes"] / ok["seconds"].clip(lower=1e-3)).median()
            / 1e6,
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print("Aggregate MB/s is in each chunk's 'Downloaded ...' log line above.")
PY

echo "done $(date -Is)"
