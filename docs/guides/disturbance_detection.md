# Disturbance detection from NAIP-CHM

`csdv detect-disturbance` maps where forest canopy was lost between consecutive NAIP images and when. It takes the NAIP-derived canopy height models (CHMs) of one site, writes an onset map, a probability map, outlines of every detected object at 19.2 m and at 0.6 m, the standardized layers behind them, figures and a report. This guide explains the method, how to run it on a laptop and on CHPC, how to change it, and how to read and sample its output. Every function, parameter and output file is specified in [../disturbance_api.md](../disturbance_api.md).

## What the method does

The method asks, for every 19.2 m cell that was forest at one NAIP date, whether the change to the next date departs from the change seen in the surrounding forest. It does not use a fixed height-drop threshold. Instead it lets each image pair set its own reference, which makes it less sensitive to the date-to-date differences in CHM quality that come with model-inferred heights.

First, each year's 0.6 m CHM is averaged into cells of 32 by 32 pixels (19.2 m). A cell counts as forest when at least half of its valid pixels are taller than 5 m. The change layer is the mean height at one date minus the mean height at the previous date. A cell is monitored at date t when it was forest at date t − 1, so a clearcut drops out of the forest population by itself once its canopy is gone, and the first image is never monitored.

Each date's change is then converted to a z-score, a Disturbance Index (DI), against the forest of the previous date. The default uses a local window of 9 by 9 cells (172.8 m) and two passes. The first pass computes local z-scores and finds patches of spatially supported loss. Those cells leave the window population, and the second pass recomputes z without them. Without the second pass a large blowdown sets part of its own reference and looks less unusual than it is. The local window also removes the broad, smoothly varying height bias that some CHM years carry (the 2012 ElkinsvilleNE CHM was predicted from 1 m imagery and is shorter everywhere).

The z-scores are turned into probabilities with Efron's empirical null. For each date, a normal curve is fitted to the centre of the monitored z histogram (between its quartiles), giving the null centre, its spread and the share of cells that follow it. The local false discovery rate then gives each cell a probability of loss. Decisions use the Bayes boundary, a probability above 0.5, rather than a tuned z threshold.

Single cells are noisy, so the default rule (D6) asks for spatial support. The z field is smoothed with Gaussian kernels of 10 and 25 m, the null is refitted to each smoothed field, and a cell takes the larger of the two probabilities. A coherent patch stands out from the smoothed null while an isolated noisy cell does not. A cell is detected only when its own evidence also leans toward loss, so smoothing cannot grow a patch beyond its cells. D6 then multiplies that probability by the probability that the next date shows no rebound. A CHM failure that recovers at the next image is therefore suppressed, which is the probabilistic form of a persistence filter. The last interval has no next image and keeps the spatial-support probability without that check.

Each cell's onset is the first date it is detected. Detected cells that share an onset date and touch (including diagonally) form an object, and objects smaller than 1 acre (0.405 ha, 11 cells) are removed. Finally, each object is redrawn at 0.6 m. The CHM change around the object is standardized against a ring of forest just outside it, pixels with a loss probability above 0.5 become openings, and the openings are closed into an envelope that must touch the object and reach the minimum mapping unit. The refined envelopes follow canopy openings rather than the 19.2 m cell edges.

### How well it worked on ElkinsvilleNE

These defaults are the best combination found in `analysis/NAIP_disturbance_detection/2_Disturbance_Index/` on the ElkinsvilleNE quarter quad (six NAIP dates, 2012 to 2022). Against the pooled reference (40 AIS polygons plus 46 confirmed or likely supplemental polygons, 77 in scope), the method detected 70% of polygons (54 of 77) with the correct onset interval. It produced 48 ha of commission in about 3,100 ha of no-change forest (1.5 ha per 100 ha), at an area precision of 0.69. By disturbance group it detected 92% of stand-replacing, 73% of partial or gap, 54% of wind or flood and 75% of clearing polygons. The strict variant D6s detected 61% with 13 ha of commission and a precision of 0.87. The 0.6 m refinement cut commission from 48 to 30 ha and raised area precision to 0.76. Against the AIS polygons alone, the hit rate was 48%, and 37 of the 91 ha counted as commission lay on or next to change that AIS had not mapped.

Read these numbers with three caveats. Every choice (grain, window, rule) was made on this one site, so the scores are optimistic for a new area. The supplemental polygons were found with CHM-change rules, which favours a CHM-change detector in the pooled scores. Some disturbance types were not found at all: a large selection harvest, the flood polygons and an overstory removal changed canopy height too little for NAIP-CHM precision, and small polygons in the last two intervals were missed. Testing the unchanged pipeline on new sites, through an interpretation sample, is the next step.

## Prerequisites

The pipeline needs at least two years of NAIP-CHM GeoTIFFs for the site, one folder per year:

```
$CSDV_DATA_ROOT/naip_chm/<site>/<year>/*.tif
```

File names must carry the acquisition date as `_YYYYMMDD_`. Both local names (`CHM_ElkinsvilleNE_20180812_0p6m.tif`) and the names written by CHPC inference (`NAIP_<site>_<date>_0p6m_chm.tif`) work. Several tiles in one year folder are mosaicked on the fly. All years must share one CRS and pixel size. They need not share an extent or alignment, because the pipeline builds the common grid on the reference year's pixels over the intersection of all years. CHMs stored as float are read as metres. Integer CHMs (the uint16 files from CHPC inference) are read as centimetres and multiplied by 0.01, unless `--chm-scale` says otherwise.

NAIP is needed only for the NDVI input (`--input d_ndvi`) or for an RGB backdrop under the maps (`--naip-backdrop`). It is read from `$CSDV_DATA_ROOT/naip/<site>/<year>/*.tif`, with red in band 1 and near infrared in band 4, and every CHM year must have a NAIP folder.

## Running locally (WSL)

In a VS Code terminal the project environment is already active. WSL has about 8 GB of memory, so keep GDAL's block cache small:

```bash
export GDAL_CACHEMAX=1024
csdv detect-disturbance --site ElkinsvilleNE
```

The run folder is `$CSDV_RESULTS_ROOT/disturbance/ElkinsvilleNE/` (by default `results/disturbance/ElkinsvilleNE/` in the repository). Progress goes to the terminal and to `detect_disturbance.log` in the run folder, and the command prints the folder path when it finishes. On WSL the ElkinsvilleNE quarter quad (3,800 ha, six years) took about one minute with the defaults (65 s wall time, of which 16 s built the layers, 32 s refined the outlines at 0.6 m and 8 s drew the figures) and peaked at 0.65 GB of memory. The NDVI input took about the same time, most of it spent reading NAIP, and a `--from-run` rerun took 35 s. The report's runtime table gives the actual time per stage.

## Running on CHPC

`scripts/slurm/run_detect_disturbance.sbatch` runs the same command as a CPU job on `wangj-np` (4 CPUs, 24 GB, 3 hours). That request is generous: a quarter quad needs about a minute and under 1 GB, and memory grows with the site area, so lower the request for small sites if the queue is long. The first argument is the site, and every further argument is passed to `csdv detect-disturbance`:

```bash
source scripts/chpc/env.sh
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_detect_disturbance.sbatch ElkinsvilleNE
```

The job checks first that the destination does not exist and that `naip_chm/<site>` is present, so a doomed job fails within seconds. It then copies `naip_chm/<site>` to node scratch (and `naip/<site>` when the arguments contain `d_ndvi` or `--naip-backdrop`), runs detection with `CSDV_DATA_ROOT` pointing at scratch, and copies the run folder to `$CSDV_RESULTS_ROOT/disturbance/<site>/`, or to the folder given with `--output-dir`. The Slurm log is `$CSDV_LOG_DIR/csdv_disturbance-<job id>.out`, and the run's own log travels with the run folder. When the job fails, the partial run folder, log included, goes to `$CSDV_STORE/failed/<job id>/out/` instead, and the destination stays free for the next attempt. When `wangj-np` is busy, the header of the sbatch script shows a preemptable freecycle fallback.

The script decides whether to stage NAIP from the command line only. If a `--config` YAML switches to the NDVI input or turns on the backdrop, repeat that choice on the command line (for example `--input d_ndvi`) so NAIP is copied to scratch.

## Changing parameters

The defaults live in `src/csdv_core/config/disturbance.yaml`, where every key is commented. A run resolves its parameters in three layers: that file, then an optional YAML given with `--config`, then explicit command-line options. Later layers win key by key, unknown keys are rejected, and the final values go to `run_params.json` and the report.

The common choices have options:

```bash
# Strict rule and a 1 ha minimum mapping unit
csdv detect-disturbance --site ElkinsvilleNE --rule D6s --mmu-ha 1.0 \
    --output-dir results/disturbance/ElkinsvilleNE_D6s_1ha

# Finer cells and a single-pass window
csdv detect-disturbance --site ElkinsvilleNE --grain-m 9.6 --method local \
    --output-dir results/disturbance/ElkinsvilleNE_9p6_local

# A subset of years, no refinement, no figures (a quick look)
csdv detect-disturbance --site ElkinsvilleNE --years 2014,2016,2018 \
    --no-refine --no-figures --output-dir results/disturbance/ElkinsvilleNE_quick
```

The NDVI input reads NAIP red and near infrared. On ElkinsvilleNE it was much weaker than CHM height (19% of polygons at best), so treat it as a comparison rather than a replacement:

```bash
csdv detect-disturbance --site ElkinsvilleNE --input d_ndvi --method local --rule D5 \
    --output-dir results/disturbance/ElkinsvilleNE_ndvi_D5
```

Keys without an option go in a YAML file that holds only what changes:

```yaml
# my_params.yaml
forest:
  height_m: 3.0
  min_cover: 0.4
decide:
  spatial_sigma_m: [10.0, 40.0]
  min_fit_cells: 500
refine:
  p_loss: 0.7
```

```bash
csdv detect-disturbance --site ElkinsvilleNE --config my_params.yaml \
    --output-dir results/disturbance/ElkinsvilleNE_custom
```

The rules are D1 (z of −3 or lower, the per-cell baseline), D2 (per-cell probability above the boundary), D5 (spatial support), D6 (spatial support and persistence) and the strict D5s and D6s, which also require the cell's own probability to pass the boundary. The methods are `scene` (median and MAD over all forest), `scenemean` (mean and SD), `local` (one window pass) and `local_twopass` (the default).

## The output folder rule

A run never writes into a folder that exists, whether it is the default folder or one given with `--output-dir`. A second run at the same site therefore needs its own `--output-dir`, and the command stops with an error that says so. A failed run keeps its folder and log for inspection, so delete or rename that folder before running into the same path again. Folder names that record the variant (`ElkinsvilleNE_D6s_1ha`) make later comparison easier.

## Re-processing a saved run

The layers and z stacks are the expensive part. `--from-run` reloads them from an earlier run and repeats only the decision, outlines, report and figures:

```bash
csdv detect-disturbance --site ElkinsvilleNE \
    --from-run results/disturbance/ElkinsvilleNE \
    --rule D6s --p-boundary 0.7 \
    --output-dir results/disturbance/ElkinsvilleNE_D6s_p07
```

The `inputs`, `grid`, `forest` and `standardize` sections shaped the saved layers and z stack, so the rerun takes them from the saved run's `run_params.json`, and options given to the first run (for example `--years` or `--window-m`) need not be repeated. Passing one of these options, or setting one of these keys in a `--config` file, is allowed only when the value equals the saved one. A different value stops the command before anything is written, with a list of every difference. The rule, boundary, minimum mapping unit, refinement and figure settings may change. They come from the packaged defaults, the rerun's `--config` file and its options, not from the saved run. The site must also match. The saved layers and z stack are copied into the new folder, so it is complete on its own and can serve as `--from-run` in turn. Refinement (on by default) and the NAIP backdrop still read the 0.6 m source rasters. In that case the site's files are discovered again (file metadata only), and the CHMs, plus NAIP for a backdrop, must still be under `$CSDV_DATA_ROOT`. Add `--no-refine` to rerun without them. The result goes to a new folder, as always.

To redraw only the figures and the report of a run, for example with a NAIP backdrop, rebuild them in place:

```bash
csdv disturbance-figures --run-dir results/disturbance/ElkinsvilleNE --naip-backdrop
```

Without `--naip-backdrop` or `--no-naip-backdrop`, the command follows the saved run's `figures.naip_backdrop` setting. The refined examples and the NAIP backdrop need the source rasters. When they are missing, the command logs a warning and draws the other figures. This overwrites `figures/` and `report.md` and leaves every other file, `summary.json` included, alone.

## Reading the report and figures

Start with `report.md`. The detections-by-interval table gives, for each image pair, the monitored forest area, the detected area after and before the minimum mapping unit, the number of objects and the share of monitored forest detected. A share far above its neighbours, or a large gap between the area before and after the MMU, points at a CHM problem in one of the two images rather than at real loss. The null-fits table gives the fitted centre, spread and null share per date. On ElkinsvilleNE the spread of the ΔCHM DI ranged from about 0.75 to 1.0 and the null share from 0.80 to 1.0. A date outside that range deserves a look at `figures/null_fits.png`. Skipped dates (too few monitored cells) are flagged, and nothing is detected on them.

The figures follow the same order as the stages. `forest_initial.png` and `forest_by_year.png` show the forest mask, and a year with much less forest than its neighbours usually means a weak CHM. `z_maps.png` shows the DI per interval. Broad smooth patches there would indicate bias that the window did not remove. `probability_maps.png` shows the decision probability. Its colour ramp changes from neutral to warm at the boundary (0.5 by default), which marks the boundary on the maps without contour lines around single cells, and cells that were not monitored on a date are grey. `null_fits.png` shows each date's z histogram with the fitted null. Its dashed line is the z where the cell-level posterior (from the histogram, not the smoothed lfdr curve of the decision) reaches the boundary, so read it as approximate. `disturbance_onset.png` and `onset_by_interval.png` show the final map over the first year's CHM height in grey (tall canopy is light), or over NAIP when the backdrop is on. Monitored forest without a detection is left clear so the backdrop shows through, which the legend marks with a hatched swatch, and ground that was never monitored is washed white. `area_by_interval.png` shows the areas, and `refined_examples.png` the 0.6 m outlines of the largest objects over the CHM before and after onset.

Remember that the last interval has no persistence check, so its detections are less certain than the others. The report lists this among its caveats.

## Inspecting a run with image chips

Before a formal accuracy assessment, a quick look at a sample of detections shows whether a run maps real canopy loss. `csdv disturbance-chips` draws a stratified sample from a run folder and renders one figure per sampled unit. Each figure has the NAIP RGB image (top row) and the CHM (bottom row) of every year, centred on the unit, with the detected 19.2 m cells outlined in yellow and, on the CHM row, the refined 0.6 m envelope in orange. The images before and after the onset interval carry blue and red frames. A contact sheet shows the before and after RGB pair of every unit on one page, which is usually enough for a first pass.

The default sample has 24 detected objects and 4 near misses. Detected objects are stratified by NAIP interval: every interval gets at least two objects (or all of them when it has fewer), and the rest are shared out in proportion to the number of objects per interval. On LincolnNH (56 objects) the 24 objects split 2, 3, 7, 2 and 10 over the five intervals. Near misses are objects that the rule flagged but the minimum mapping unit removed, limited to objects of at least half the MMU, so they show what the MMU throws away. `--n-no-detection` adds single cells of monitored forest that were never flagged, which show what undisturbed forest looks like at the same scale. The sample is drawn with a fixed seed (`--seed`), so the same command gives the same sample.

```bash
# Locally, when the NAIP and CHM years are under CSDV_DATA_ROOT
csdv disturbance-chips --run-dir results/disturbance/ElkinsvilleNE

# On CHPC, where the NAIP mosaics live
sbatch --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_disturbance_chips.sbatch "$CSDV_RESULTS_ROOT/disturbance/LincolnNH"

# A larger sample with undisturbed cells and another seed, into a new folder
csdv disturbance-chips --run-dir results/disturbance/LincolnNH \
    --n-detected 40 --n-no-detection 6 --seed 1 \
    --output-dir results/disturbance/LincolnNH/inspection_seed1

# The sample table only, without reading any imagery
csdv disturbance-chips --run-dir results/disturbance/LincolnNH --sample-only
```

The chips read the site's NAIP and CHM years under the current `CSDV_DATA_ROOT`, on the run's saved grid, so the run folder can be inspected on any machine that holds the source rasters. Only small windows around the sampled units are read, and the Slurm script therefore reads straight from the project store instead of staging the NAIP mosaics. On ElkinsvilleNE, 30 units took 53 s and under 1 GB of memory. The outputs go to `<run>/inspection/` by default, and as with detection runs an existing folder is never overwritten.

| File | Contents |
|---|---|
| `samples.csv` | One row per unit: sample id, stratum, object id (blank for near misses and cells), interval, area, probability, mean z, centroid in map and geographic coordinates, chip path, and empty `interp_label` and `interp_notes` columns for your notes |
| `samples.gpkg` | The same units with their polygons, for QGIS |
| `chips/<id>_<stratum>_<interval>.png` | One figure per unit |
| `contact_sheet.png` | Before and after RGB of every unit |
| `index.md` | The contact sheet and a table linking every chip |

NAIP is not atmospherically corrected and each chip is stretched on its own, so colour differences between years (a blue cast on hazy flights, for example) say nothing about the forest. Judge change by crown texture, gaps, bare soil and the CHM row.

## Drawing an interpretation sample

The run does not score itself against reference data. The next step is an interpretation sample on NAIP and the CHM, and the outputs are laid out for that. `detection/objects.csv` lists every object with its onset interval, area, mean and maximum probability, mean z and centroid in both map and geographic coordinates, so detected objects can be stratified and sampled without GIS software. Omission needs samples where nothing was detected, and the probability raster gives those: monitored cells with no detection but a probability near the boundary are the most likely misses.

```python
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

run = "results/disturbance/ElkinsvilleNE/detection"

# Detected objects: up to 10 per interval and probability class.
objects = pd.read_csv(f"{run}/objects.csv")
objects["prob_class"] = pd.cut(objects["mean_prob"], [0.5, 0.7, 0.9, 1.0])
strata = objects.groupby(["interval", "prob_class"], observed=True, group_keys=False)
detected = strata.apply(lambda group: group.sample(min(len(group), 10), random_state=1))

# Non-detected monitored cells, stratified by their highest probability.
with rasterio.open(f"{run}/onset_year.tif") as src:
    onset = src.read(1)
    transform = src.transform
with rasterio.open(f"{run}/probability.tif") as src:
    max_prob = np.nanmax(src.read(), axis=0)
rng = np.random.default_rng(1)
rows = []
for low, high in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)]:
    candidates = np.argwhere((onset == 0) & (max_prob >= low) & (max_prob < high))
    n_picks = min(50, len(candidates))
    picks = candidates[rng.choice(len(candidates), size=n_picks, replace=False)]
    for row, col in picks:
        x, y = xy(transform, row, col)
        stratum = f"p{low}-{high}"
        rows.append({"stratum": stratum, "x": x, "y": y, "max_prob": max_prob[row, col]})
undetected = pd.DataFrame(rows)
```

The strata, sample sizes and the use of the maximum probability over dates are examples. The cell coordinates are cell centres in the grid CRS. Rule D1 writes no probability raster and leaves `mean_prob` empty, so stratify a D1 run by `mean_z` and by the z rasters instead. For an area estimate, record the stratum sizes (object counts per stratum, cell counts per probability band) alongside the sample so the estimates can be weighted. The z rasters in `z/` give the DI behind each sample (divide by 100), and the refined outlines in `disturbances.gpkg` show what the method would map at 0.6 m.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Output folder already exists` | The run never overwrites. Pass a new `--output-dir`, or move the old folder. On CHPC the job stops at the pre-flight check for the same reason |
| Error that names two years with different CRSs or pixel sizes | One year was delivered in another UTM zone or resolution. Reproject that year onto the reference year's CRS at 0.6 m (`gdalwarp -t_srs <crs> -tr 0.6 0.6 -tap -r near`) and rerun |
| `fewer than two years` or a missing year | Year folders must be integers and hold at least one `.tif`. Check `--years` against the folders under `naip_chm/<site>/` |
| A date is listed as skipped | Fewer than 1,000 monitored cells (about 37 ha of forest at 19.2 m) had valid z on that date. Use a larger area, a finer `--grain-m`, or lower `decide.min_fit_cells` in a `--config` YAML (at least 100) |
| Warning that a CHM year has no nodata value | Common for the uint16 CHMs from CHPC inference. Without it, fill pixels are read as heights. Find the fill value (`gdalinfo -stats`, or the value along the image edge) and pass it in stored units, for example `--chm-nodata 65535` |
| Heights look 100 times too large or too small | The scale guess failed. Pass `--chm-scale 1.0` for metres stored as integers or `--chm-scale 0.01` for centimetres |
| NDVI run fails for a missing NAIP folder | Every CHM year needs `naip/<site>/<year>/`. On CHPC, put `--input d_ndvi` on the command line so the job stages NAIP |
| `--from-run` refuses with a list of differences | An option or `--config` key in the `inputs`, `grid`, `forest` or `standardize` section differs from the saved run. Drop it (the saved value is used), or run from scratch |
| Detections in the last interval look doubtful | That interval has no later image, so D6 cannot check it for a rebound. Treat it as unconfirmed until the next NAIP year arrives |
| Process killed on WSL | Memory pressure. Lower `GDAL_CACHEMAX`, close other programs, or run on CHPC |
