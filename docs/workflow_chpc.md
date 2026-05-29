# Workflow: CHPC and Colab

This is the operational guide for running CSDV end-to-end on the University of Utah CHPC cluster, with a parallel section for Google Colab when GPU time on CHPC is unavailable. It follows the same step ordering as `scripts/run_site_e2e.sh`.

For the design rationale of each pipeline stage, see [docs/architecture.md](architecture.md). For path layout, see [docs/data_layout.md](data_layout.md). For coding rules, see [docs/conventions.md](conventions.md).

## 1. One-time setup (CHPC)

Login node, about 10 minutes:

```bash
# Repo
cd /uufs/chpc.utah.edu/common/home/dycelab/users/$USER/code
git clone <repo-url> CSDV/CSDV_RS
cd CSDV/CSDV_RS

# Env (micromamba is the project standard)
micromamba env create -f environment.yml -p ./.micromamba/csdv
micromamba activate ./.micromamba/csdv
pip install -e ".[dev,lidr]"

# Verify
python -m pytest -q -m "not slow"
csdv --help

# External: upstream NAIP-CHM repo for the inference subcommand
git clone https://github.com/smorf-ntsg/naip-chm ../naip-chm

# One-time: static CONUS-wide conditioning rasters for NAIP-CHM inference (~1.65 GB).
# Run this on the login node, not inside a GPU job, so the transfer does not
# consume a GPU allocation.
csdv download conditioning
# writes to $CSDV_DATA_ROOT/chm_model/conditioning/ (override with CSDV_CONDITIONING_DIR)
```

Add to your `~/.bashrc` (or a project-local `env.sh` you `source` per session):

```bash
export CSDV_DATA_ROOT=/scratch/general/vast/$USER/csdv/data
export CSDV_RESULTS_ROOT=/scratch/general/vast/$USER/csdv/results
export CSDV_CACHE_ROOT=/scratch/general/vast/$USER/csdv/cache
export NAIPCHM_REPO_DIR=/uufs/chpc.utah.edu/common/home/dycelab/users/$USER/code/CSDV/naip-chm
# Optional: point at a shared, read-only copy of the conditioning rasters.
# export CSDV_CONDITIONING_DIR=/uufs/chpc.utah.edu/common/home/dycelab/shared/naip-chm/conditioning
mkdir -p "$CSDV_DATA_ROOT" "$CSDV_RESULTS_ROOT" "$CSDV_CACHE_ROOT"
```

Verify the environment in one call:

```bash
csdv check
```

A clean report has zero `[FAIL]` rows. `[INFO]` rows for optional imports (`torch`, `rpy2`, `ee`) are fine if you do not need that capability.

## 2. Per-session warm-up

For interactive runs, ask for a small GPU node:

```bash
salloc --account=dycelab --partition=dycelab-shared-gpu \
       --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=02:00:00
# After it lands:
cd /uufs/.../CSDV/CSDV_RS
source env.sh                 # the file you wrote in step 1
micromamba activate ./.micromamba/csdv
csdv check --site SCBI --years 2014,2018 --window-m 50
```

The `csdv check` output is the canonical "what is missing" report. Pipeline-state `[WARN]` rows are expected on a fresh data root. They tell you which steps the e2e script will need to run.

## 3. First end-to-end run on a tiny AOI

Use a 1 to 2 km half-width AOI for the first run so it finishes in minutes, not hours.

```bash
./scripts/run_site_e2e.sh SCBI 2018 50 --only check
./scripts/run_site_e2e.sh SCBI 2018 50
```

What this does:

1. `csdv check --site SCBI --years 2018 --window-m 50`
2. `csdv download conditioning` (only if a CHM stage will run and the conditioning rasters are absent; skipped otherwise)
3. `csdv download naip --site SCBI --start-date 2018-04-01 --end-date 2018-11-30 --out-dir $CSDV_DATA_ROOT/naip/SCBI/2018`
4. `csdv chm-inference --naip-quad <tif from step 3> --output-dir $CSDV_DATA_ROOT/naip_chm/SCBI/2018`
5. `csdv segment-crowns --chm <tif from step 4> --out-crowns $CSDV_RESULTS_ROOT/crowns/SCBI/2018/crowns.gpkg`
6. `csdv compute-metrics --site SCBI --year 2018 --window-m 50`
7. `csdv stratify --site SCBI --window-m 50`
8. `csdv classify-stages --site SCBI --year 2018 --window-m 50`

Trajectory classification is skipped when only one year is requested. Step 8's output, `$CSDV_RESULTS_ROOT/stages/SCBI/2018/50m/stage.tif`, is the deliverable of a single-date run.

Per-step logs land in `$CSDV_RESULTS_ROOT/logs/SCBI/<RUN_ID>/<step>.log`. The summary is `summary.log` in the same directory.

## 4. Iteration loop

The e2e script is idempotent. Each step has a sentinel output file and is skipped when that file already exists. Common patterns:

```bash
# Add a second year. Earlier outputs are reused; new year fills in.
./scripts/run_site_e2e.sh SCBI 2014,2018 50

# Re-run one step only (e.g. after tweaking metrics.yaml).
./scripts/run_site_e2e.sh SCBI 2014,2018 50 --only metrics

# Force re-run everything from scratch.
./scripts/run_site_e2e.sh SCBI 2018 50 --force

# Use a NAIP tif you already have (e.g. DOQQ from collaborator).
mkdir -p $CSDV_DATA_ROOT/naip/SCBI/2018
cp /path/to/m_3807753_*.tif $CSDV_DATA_ROOT/naip/SCBI/2018/
./scripts/run_site_e2e.sh SCBI 2018 50 --skip-download
```

`--only` accepts a stage prefix. `--only metrics` matches `metrics-2014` and `metrics-2018`. `--only stages` matches `stages-2014` and `stages-2018`. Use the literal name (`stratify`, `trajectories`, `check`) for site-level stages.

## 5. Batch submission

Once a config is stable, promote the same call to SLURM with no changes:

```bash
sbatch scripts/slurm/run_site_e2e.sbatch SCBI 2014,2018,2022 50
```

The sbatch wrapper activates the project env, prints the `csdv --version`, then execs `run_site_e2e.sh` with the forwarded arguments. SLURM stdout/stderr live in `results/logs/slurm/csdv_e2e-<jobid>.out`. Per-step e2e logs still go to `$CSDV_RESULTS_ROOT/logs/<SITE>/<RUN_ID>/`.

Run `csdv download conditioning` once on the login node before submitting. The batch job will fetch the conditioning rasters itself if they are absent, but doing it beforehand keeps the 1.65 GB transfer off the GPU allocation.

Resource defaults: 1 GPU, 8 CPU, 48 GB, 6 h. Tune `--gres`, `--mem`, and `--time` in `scripts/slurm/run_site_e2e.sbatch` for larger AOIs.

## 6. Troubleshooting

| Symptom | Where to look | Likely cause |
|---|---|---|
| `csdv check` shows `[FAIL] data_root MISSING` | env vars | `CSDV_DATA_ROOT` unset or pointing at a path you cannot write to. |
| `csdv check` shows `[INFO] import torch (optional) not installed` | env | Only matters if step 3 (`chm-inference`) is in scope. Install with `pip install -e ".[chm_inference]"` or use Colab. |
| `chm-X` step fails with "Upstream naip-chm repo not found" | `$NAIPCHM_REPO_DIR` | Either set the env var or pass `--repo-dir` via the underlying CLI. |
| `chm-X` step fails with "Conditioning rasters missing in ..." | `$CSDV_DATA_ROOT/chm_model/conditioning/` (or `$CSDV_CONDITIONING_DIR`) | Rasters not downloaded, or the conditioning step was skipped under `--skip-download`. Run `csdv download conditioning`. |
| `crowns-X` step fails with "no CHM tif found" | `data/naip_chm/<SITE>/<YEAR>/` | Step 3 did not produce output. Check `chm-<year>.log`. |
| `metrics-X` step fails with "CHM required for ..." | metric registry | A Pass-1 metric needs CHM and none was found. Confirm step 3 ran. |
| `compute-metrics` writes nothing visible | `results/metrics/.../manifest.yaml` | The script writes a metric stack TIF and a manifest. Inspect `manifest.yaml` to see which metrics were computed. |
| Stage raster looks empty | `stages.yaml` thresholds | Rules may be too narrow for this site type. Inspect with `rasterio info`. |
| `csdv check --strict` exits non-zero | any `[WARN]` row | Strict mode treats warnings as failures. Drop `--strict` for normal runs. |

When a step fails, the dispatcher prints the path to its `.log` file. That log holds the full stack trace plus the CLI invocation. Re-run with the same arguments and `--force` once the underlying issue is fixed.

## 7. Colab parity

For ad-hoc experimentation or when CHPC GPU queues are saturated:

```python
!git clone <repo-url> /content/csdv
%cd /content/csdv
!pip install -e ".[dev]"

import os
os.environ["CSDV_DATA_ROOT"] = "/content/drive/MyDrive/csdv/data"
os.environ["CSDV_RESULTS_ROOT"] = "/content/drive/MyDrive/csdv/results"

!csdv check --site SCBI --years 2018 --window-m 50
!bash scripts/run_site_e2e.sh SCBI 2018 50
```

Colab cannot run `lidR`-based crown segmentation. The watershed fallback in `csdv_core.segmentation` is selected automatically when `rpy2` is unavailable. Phase-4 stage classification still produces useful outputs from the Python crown set.

Colab is not appropriate for long batch runs because of session lifetime limits. For anything beyond one (site, year, 50 m window) iteration, move back to CHPC.
