# Workflow: CHPC and Colab

This is the operational guide for running CSDV end-to-end on the University of Utah CHPC cluster, with a parallel section for Google Colab when GPU time on CHPC is unavailable. It follows the same step ordering as `scripts/run_site_e2e.sh`.

For the design rationale of each pipeline stage, see [docs/architecture.md](architecture.md). For path layout, see [docs/data_layout.md](data_layout.md). For coding rules, see [docs/conventions.md](conventions.md).

## 1. One-time setup (CHPC)

CHPC jobs read and write on node-local scratch (`/scratch/local`) and copy data in and out of persistent storage. The project keeps three locations apart. The repository and the upstream `naip-chm` clone live in your user space. Inputs and outputs persist in the group store, `$CSDV_STORE` (for this project `/uufs/chpc.utah.edu/common/home/dycelab/data/CSDV`). Each job works in `/scratch/local/$USER/$SLURM_JOB_ID`, which CHPC deletes when the job ends.

Login node, about 20 minutes:

```bash
cd /uufs/chpc.utah.edu/common/home/dycelab/users/<you>/code
git clone <repo-url> CSDV/CSDV_RS
git clone https://github.com/bullocke/naip-crowns.git CSDV/naip-crowns   # crown segmentation (private: needs GitHub access)
git clone https://github.com/smorf-ntsg/naip-chm CSDV/naip-chm   # code and model weights
cd CSDV/CSDV_RS

# Per-user settings. Edit CSDV_STORE (and NAIPCHM_REPO_DIR if the clone is elsewhere).
cp scripts/chpc/env.example.sh scripts/chpc/env.sh

# Environment with CUDA PyTorch 2.8 (cu126 wheels), named csdv-gpu.
scripts/chpc/create_env.sh
```

`create_env.sh` builds the conda environment from `environment.yml`, installs `torch==2.8.0` and `torchvision==0.23.0` from the official CUDA 12.6 wheels, and installs naipcrowns (from `../naip-crowns`, or `NAIPCROWNS_DIR`) and this package in editable mode with `--no-deps --no-build-isolation`. Do not install PyTorch from conda or add `cpuonly`. The cu126 wheels bundle the CUDA runtime and run on every GPU in the general and freecycle partitions (titanx through h100). They do not support the Blackwell cards on `granite-gpu-guest`. The script prints the torch version and refuses a CPU-only build.

Start every login session with:

```bash
source scripts/chpc/env.sh
activate_csdv
```

`env.sh` exports `CSDV_DATA_ROOT=$CSDV_STORE/data`, `CSDV_RESULTS_ROOT=$CSDV_STORE/results`, the conditioning directory, the log directory and `NAIPCHM_REPO_DIR`. Slurm passes these into each job, and every sbatch script re-sources `env.sh` through `$CSDV_REPO`, so always submit from a shell where `env.sh` is loaded.

### 1a. Downloads run on the data transfer partition

Compute and GPU nodes have no outbound internet, so a download inside a compute job times out with `Errno 110`. Every download goes through `scripts/slurm/run_download_dtn.sbatch`, which runs `csdv download <subcommand>` on the `notchpeak-dtn` partition and writes straight to `$CSDV_STORE`.

The five conditioning rasters (about 1.65 GB, static and CONUS-wide) are fetched once:

```bash
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch conditioning
```

NAIP comes from the Planetary Computer as one mosaic per site and year, at native resolution in the quads' UTM zone, with a 150 m buffer for the canopy height model's unreliable edge. Sites in `config/sites.yaml` need only `--site`. Others take a `--bbox` in longitude and latitude:

```bash
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
    naip-pc --site SantaClara --bbox -86.3125 39.0625 -86.25 39.125 \
    --years 2016,2018,2020,2022
```

Output goes to `$CSDV_DATA_ROOT/naip/<SITE>/<YEAR>/NAIP_<SITE>_<YYYYMMDD>_<res>.tif` with a `.provenance.json` sidecar. The model reads day of year from the eight-digit date, so imagery from any other source (a USDA download, a collaborator's DOQQs) can be dropped into the same folders as long as each filename carries a `YYYYMMDD` token.

The Planetary Computer stops at 2023. Newer years come from the USDA FPAC image service with `naip-rest`, which takes the same options and writes the same grid and file name. The service holds only the most recent collection for each area, so a year disappears from it once the next one is flown:

```bash
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
    naip-rest --site NorthMaine --bbox -69.5656 45.6636 -69.3253 45.7501 --years 2025
```

`naip-auto` takes the same options plus `--start-year` and picks the source of each year by itself, the Planetary Computer when it has the year and the USDA service otherwise. It skips years that cover less than 90% of the box (`--min-coverage`), so a state year that USDA did not finish is left out:

```bash
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_download_dtn.sbatch \
    naip-auto --site NorthMaine --bbox -69.5656 45.6636 -69.3253 45.7501 --start-year 2010
```

A state is not always flown completely in its scheduled year (USDA did not finish Maine, Montana or Idaho in 2023), so check the coverage warning in the log. Both downloaders write area without imagery as zero in every band and flag it in an internal mask, and `csdv chm-inference` sets those CHM pixels, plus a 50 m band around them (`chm_inference.gap_edge_m` in `metrics.yaml`), to nodata. CHMs predicted before that fix can be repaired in place with `scripts/slurm/run_mask_chm_naip_gaps.sbatch SITE YEAR`.

### 1b. Landsat observations (per site, on a data transfer node)

`csdv satellite fetch` reduces forty years of Landsat inside each stand polygon on Earth Engine's servers and writes the result to `$CSDV_DATA_ROOT/satellite/<site>/`. It talks to Google over the network for the whole run, so it has the same constraint as step 1a and cannot run inside a compute job. Run it interactively on a data transfer node, because `earthengine authenticate` needs a browser flow.

```bash
ssh $USER@dtn07.chpc.utah.edu
cd /uufs/chpc.utah.edu/common/home/dycelab/users/<you>/code/CSDV/CSDV_RS
source scripts/chpc/env.sh && activate_csdv
earthengine authenticate           # once per machine, opens a browser flow
csdv satellite fetch --site ElkinsvilleNE
csdv satellite annual --site ElkinsvilleNE
```

The fetch takes roughly twenty minutes for forty stands over 1985 to 2025 and writes about half a megabyte. Check the manifest for failed chunks before moving on. `csdv satellite annual` is pure computation on the cached parquet and can run anywhere, including inside a job.

Both outputs are caches. Delete them and refetch if the quality gates or the index definitions change; nothing downstream needs the Earth Engine session again.

## 2. Check the GPU environment

Before the first batch, confirm that PyTorch sees a GPU on a compute node (about ten minutes):

```bash
salloc --partition=notchpeak-gpu --qos=notchpeak-gpu --account=notchpeak-gpu \
       --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00
source scripts/chpc/env.sh && activate_csdv
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
print((torch.rand(2048, 2048, device='cuda') @ torch.rand(2048, 2048, device='cuda')).sum().item())"
csdv check
```

`csdv check` lists what is missing. Pipeline-state `[WARN]` rows are expected on a fresh data root.

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

## 5. Batch submission: CHM inference and crown segmentation

Once NAIP is in the store, one command processes every image:

```bash
source scripts/chpc/env.sh
scripts/slurm/submit_chm_crowns.sh SantaClara                 # all years found
scripts/slurm/submit_chm_crowns.sh SantaClara,SCBI 2018,2022  # a subset
scripts/slurm/submit_chm_crowns.sh SantaClara --dry-run       # manifest only
```

The wrapper checks the conditioning rasters and the `naip-chm` clone, writes a manifest of every NAIP GeoTIFF to `$CSDV_STORE/manifests/`, and submits two array jobs with one task per image:

1. `run_chm_inference.sbatch` on `notchpeak-gpu` (1 GPU, 8 CPUs, 48 GB, 2 h). Each task copies its NAIP tif and the conditioning rasters to node scratch, runs `csdv chm-inference`, and copies the CHM to `$CSDV_DATA_ROOT/naip_chm/<SITE>/<YEAR>/`. Pass `--granite` to use the preemptable `granite-gpu` freecycle partition (L40S) with `--requeue` when notchpeak is busy.
2. `run_segment_crowns.sbatch` on `wangj-np` (8 CPUs, 64 GB, 4 h). It waits on its own CHM task (`--dependency=aftercorr`), copies the CHM to node scratch, runs `csdv segment-crowns`, and copies the result to `$CSDV_RESULTS_ROOT/crowns/<SITE>/<YEAR>/`. The file is `crowns.gpkg` when a site-year has one image, which is what `compute-metrics` reads, and `<naip stem>_crowns.gpkg` otherwise.

Staging is handled by `scripts/chpc/scratch.sh`. Outputs reach the store only when a task succeeds, because later runs treat an existing CHM or crown file as finished. A failed task's partial output goes to `$CSDV_STORE/failed/<job id>/` and its crown task is cancelled. Resubmitting the same command redoes only the images that are missing. Logs are in `$CSDV_LOG_DIR`, and `sacct -j <job ids> -X --format=JobID,State,Elapsed,MaxRSS` shows run time and peak memory for tuning.

The downstream steps (metrics, stratification, stage and trajectory classification) still run through `scripts/run_site_e2e.sh` with `--skip-download`. `scripts/slurm/run_site_e2e_scratch.sbatch` has not yet been moved onto the staging helpers.

## 6. Disturbance detection

`csdv detect-disturbance` maps forest canopy loss and its onset interval for one site from the NAIP-CHM years that step 5 wrote. It needs no crowns. The method, its parameters and its outputs are described in [docs/guides/disturbance_detection.md](guides/disturbance_detection.md), and the API and file formats in [docs/disturbance_api.md](disturbance_api.md).

One CPU job runs one site (`wangj-np`, 4 CPUs, 24 GB, 3 h):

```bash
source scripts/chpc/env.sh
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_detect_disturbance.sbatch SantaClara

# Other parameters need a new output folder.
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_detect_disturbance.sbatch SantaClara \
    --rule D6s --mmu-ha 1.0 \
    --output-dir "$CSDV_RESULTS_ROOT/disturbance/SantaClara_D6s_1ha"
```

The first argument is the site, and every further argument goes to `csdv detect-disturbance`. Before staging anything, the job checks that `$CSDV_DATA_ROOT/naip_chm/<SITE>/` exists and that the destination does not, because a run never writes into an existing folder. It copies the CHM years (and the NAIP years when the arguments contain `d_ndvi` or `--naip-backdrop`) to node scratch, runs detection there, and copies the run folder to `$CSDV_RESULTS_ROOT/disturbance/<SITE>/` or to the `--output-dir` given. A failed run goes to `$CSDV_STORE/failed/<job id>/out/` with its `detect_disturbance.log`, and the destination stays free for a resubmission.

On WSL the default run on the ElkinsvilleNE quarter quad (3,800 ha, six years) took about one minute and peaked at 0.65 GB of memory, so the request above leaves wide margins. Memory grows with the site area. `--from-run` also works in the job (pass the saved run folder under `$CSDV_RESULTS_ROOT`), and the CHM years are staged as usual because refinement reads them.

The CHMs written by `run_chm_inference.sbatch` are uint16 and are read as centimetres. If the log warns that a CHM year has no nodata value, find the fill value and pass it with `--chm-nodata` in stored units.

To inspect a run before an accuracy assessment, `run_disturbance_chips.sbatch` samples it and draws NAIP RGB and CHM chips of each sampled unit (2 CPUs, 8 GB, 1 h). It reads small windows straight from `$CSDV_DATA_ROOT`, so the NAIP mosaics are not staged, and copies the chips to `<run>/inspection/` or to the `--output-dir` given. See "Inspecting a run with image chips" in the guide.

```bash
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_disturbance_chips.sbatch "$CSDV_RESULTS_ROOT/disturbance/LincolnNH"
```

### Many locations: grid runs

`csdv grid submit` runs this section and step 1a for whole batches of locations. It plans the NAIP years of each unit (a 7.5 minute quad cell, or a box around a point), takes each year from the Planetary Computer or the USDA image service, and submits the download, CHM, detection, chips and export jobs as one dependent chain. `csdv grid status` shows what is done and what failed, and `scripts/chpc/pull_exports.sh` copies the small outputs to a local machine. See [docs/guides/grid_processing.md](guides/grid_processing.md).

```bash
source scripts/chpc/env.sh
activate_csdv # If it does not activate on its own
csdv grid submit --batch config/grid/conus_pilot.csv --dry-run
csdv grid submit --batch config/grid/conus_pilot.csv
csdv grid status
```

`env.sh` files made before the grid commands lack `CSDV_GRID_DIR` and `CSDV_EXPORT_DIR`. Copy those two lines from `env.example.sh`, or the commands fall back to `$CSDV_RESULTS_ROOT/grid` and `$CSDV_RESULTS_ROOT/exports`. 

Some more options and pulling locally:

```bash
source scripts/chpc/env.sh
activate_csdv # If it does not activate on its own
csdv grid submit --batch config/grid/conus_pilot.csv --dry-run   # shows years, sources and jobs
csdv grid submit --cell q4407152                                 # one unit
csdv grid submit --lonlat -71.55 44.16 --buffer-km 6 --label LincolnNH
csdv grid submit --batch config/grid/conus_pilot.csv --run-tag D6s_1ha -- --rule D6s --mmu-ha 1.0
csdv grid status            # or --problems
scripts/chpc/pull_exports.sh [runs/<tag>]                        # run locally
```

## 7. Troubleshooting

| Symptom | Where to look | Likely cause |
|---|---|---|
| `csdv check` shows `[FAIL] data_root MISSING` | env vars | `CSDV_DATA_ROOT` unset or pointing at a path you cannot write to. |
| `csdv check` shows `[INFO] import torch (optional) not installed` | env | The env was not built with `scripts/chpc/create_env.sh`, or `activate_csdv` was not run. |
| CHM task fails with `no CUDA device` | `nvidia-smi -L` in the log | The job landed without a GPU (missing `--gres`), or torch is a CPU build. Rebuild with `create_env.sh`. |
| CHM task fails with `no kernel image is available` | the GPU named in the log | The card is newer than the cu126 build supports (Blackwell). Avoid `granite-gpu-guest`. |
| sbatch fails with `CSDV_REPO: source scripts/chpc/env.sh before sbatch` | the submitting shell | `env.sh` was not sourced before `sbatch`. |
| `chm-X` step fails with "Upstream naip-chm repo not found" | `$NAIPCHM_REPO_DIR` | Either set the env var or pass `--repo-dir` via the underlying CLI. |
| `chm-X` step fails with "Conditioning rasters missing in ..." | `$CSDV_DATA_ROOT/chm_model/conditioning/` (or `$CSDV_CONDITIONING_DIR`) | Rasters not downloaded, or the conditioning step was skipped under `--skip-download`. Pre-fetch them on a data transfer node (step 1a). |
| `conditioning` step fails with `URLError ... Errno 110 Connection timed out` | the SLURM/compute node | Compute nodes have no outbound internet, so the in-job download cannot reach the server. Pre-fetch the rasters on a data transfer node (step 1a); the job will then skip the download. |
| `crowns-X` step fails with "no CHM tif found" | `data/naip_chm/<SITE>/<YEAR>/` | Step 3 did not produce output. Check `chm-<year>.log`. |
| `metrics-X` step fails with "CHM required for ..." | metric registry | A Pass-1 metric needs CHM and none was found. Confirm step 3 ran. |
| `compute-metrics` writes nothing visible | `results/metrics/.../manifest.yaml` | The script writes a metric stack TIF and a manifest. Inspect `manifest.yaml` to see which metrics were computed. |
| Stage raster looks empty | `stages.yaml` thresholds | Rules may be too narrow for this site type. Inspect with `rasterio info`. |
| Disturbance job fails within seconds with `output folder already exists` | the Slurm log | A run never overwrites. Pass a new `--output-dir`, or move the earlier run (or a failed one) away. |
| `csdv check --strict` exits non-zero | any `[WARN]` row | Strict mode treats warnings as failures. Drop `--strict` for normal runs. |

When a step fails, the dispatcher prints the path to its `.log` file. That log holds the full stack trace plus the CLI invocation. Re-run with the same arguments and `--force` once the underlying issue is fixed.

## 8. Colab parity

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

Crown segmentation on Colab needs the naip-crowns repository cloned and installed next to CSDV. Its Python engine needs no R, so `csdv segment-crowns` runs there without lidR.

Colab is not appropriate for long batch runs because of session lifetime limits. For anything beyond one (site, year, 50 m window) iteration, move back to CHPC.
