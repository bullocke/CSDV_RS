# Disturbance detection from NAIP-CHM

`csdv detect-disturbance` maps where forest canopy was lost between consecutive NAIP images and when. It takes the NAIP-derived canopy height models (CHMs) of one site, writes an onset map, a probability map, outlines of every detected object on 19.2 m cells (32 × 32 pixels of the 0.6 m CHM) and at 0.6 m, the standardized layers behind them, figures and a report. This guide explains the method, how to run it on a laptop and on CHPC, how to change it, and how to read and sample its output. Every function, parameter and output file is specified in [../disturbance_api.md](../disturbance_api.md). The [Glossary](#glossary) at the end defines the terms used here, and its [Rules](#rules) subsection compares the decision rules.

## What the method does

The method asks, for every 19.2 m cell that was forest at one NAIP date, whether the change to the next date departs from the change seen in the surrounding forest. It does not use a fixed height-drop threshold. Instead it lets each image pair set its own reference, which makes it less sensitive to the date-to-date differences in CHM quality that come with model-inferred heights.

First, each year's 0.6 m CHM is averaged into cells of 32 by 32 pixels (19.2 m). A cell counts as forest when at least half of its valid pixels are taller than 5 m. The change layer is the mean height at one date minus the mean height at the previous date. A cell is monitored at date t when it was forest at date t − 1, so a clearcut drops out of the forest population by itself once its canopy is gone, and the first image is never monitored.

Each date's change is then converted to a z-score, a Disturbance Index (DI), against the forest of the previous date. The default uses a local window of 9 by 9 cells (172.8 m) and two passes. The first pass computes local z-scores and finds patches of spatially supported loss. Those cells leave the window population, and the second pass recomputes z without them. Without the second pass a large blowdown sets part of its own reference and looks less unusual than it is. The local window also removes the broad, smoothly varying height bias that some CHM years carry (the 2012 ElkinsvilleNE CHM was predicted from 1 m imagery and is shorter everywhere).

The z-scores are turned into probabilities with Efron's empirical null. For each date, a normal curve is fitted to the centre of the monitored z histogram (between its quartiles), giving the null centre, its spread and the share of cells that follow it. The local false discovery rate then gives each cell a probability of loss. Decisions use the Bayes boundary, a probability above 0.5, rather than a tuned z threshold.

Single cells are noisy, so the default rule (D6, see [Rules](#rules)) asks for spatial support. The z field is smoothed with Gaussian kernels of 10 and 25 m, the null is refitted to each smoothed field, and a cell takes the larger of the two probabilities. A coherent patch stands out from the smoothed null while an isolated noisy cell does not. A cell is detected only when its own evidence also leans toward loss, so smoothing cannot grow a patch beyond its cells. D6 then multiplies that probability by the probability that the next date shows no rebound. A CHM failure that recovers at the next image is therefore suppressed, which is the probabilistic form of a persistence filter. The last interval has no next image and keeps the spatial-support probability without that check.

An optional rule, D7 ("bracketed"), tests each loss against the images on both sides of its interval instead of the gain at the next date. For the interval ending at date t it computes the spatial-support probability three times: for the one-step change h[t] − h[t−1], for the back change h[t] − h[t−2] and for the forward change h[t+1] − h[t−1]. Each two-step change is standardized like the main change (same window, same forest population) and gets its own null fits. The D7 probability is the smallest of the three. A spike in the earlier image vanishes in the back change and a dip in the later image vanishes in the forward change, so neither survives the minimum, while a real step down appears in all three. A check that does not exist is left out, so the first interval uses the forward check only and the last interval the back check only. D7s adds the strict cell test of D5s.

Each cell's onset is the first date it is detected. Detected cells that share an onset date and touch (including diagonally) form an object, and objects smaller than 1 acre (0.405 ha, 11 cells) are removed. Finally, each object is redrawn at 0.6 m. The CHM change around the object is standardized against a ring of forest just outside it, pixels with a loss probability above 0.5 become openings, and the openings are closed into an envelope that must touch the object and reach the minimum mapping unit. The refined envelopes follow canopy openings rather than the 19.2 m cell edges.

### How well it worked on ElkinsvilleNE

These defaults are the best combination found in `analysis/NAIP_disturbance_detection/2_Disturbance_Index/` on the ElkinsvilleNE quarter quad (six NAIP dates, 2012 to 2022). Against the pooled reference (40 AIS polygons plus 46 confirmed or likely supplemental polygons, 77 in scope), the method detected 70% of polygons (54 of 77) with the correct onset interval. It produced 48 ha of commission in about 3,100 ha of no-change forest (1.5 ha per 100 ha), at an area precision of 0.69. By disturbance group it detected 92% of stand-replacing, 73% of partial or gap, 54% of wind or flood and 75% of clearing polygons. The strict variant D6s detected 61% with 13 ha of commission and a precision of 0.87. The 0.6 m refinement cut commission from 48 to 30 ha and raised area precision to 0.76. Against the AIS polygons alone, the hit rate was 48%, and 37 of the 91 ha counted as commission lay on or next to change that AIS had not mapped.

Read these numbers with three caveats. Every choice (grain, window, rule) was made on this one site, so the scores are optimistic for a new area. The supplemental polygons were found with CHM-change rules, which favours a CHM-change detector in the pooled scores. Some disturbance types were not found at all: a large selection harvest, the flood polygons and an overstory removal changed canopy height too little for NAIP-CHM precision, and small polygons in the last two intervals were missed. Testing the unchanged pipeline on new sites, through an interpretation sample, is the next step.

### Temporal artefacts and the bracketed rule

Many detections are not losses at all but artefacts of a single image. A script that follows the median cell CHM height of each object through all six dates (`13_temporal_patterns.py` in the analysis folder) sorts objects into a spike in the earlier image that returns, a dip in the later image that recovers, a persistent step down, and objects in the last interval that have no later image. Of the 56 objects of the default run on LincolnNH, 14 were spikes, 6 dips, 12 persistent drops and 24 lay in the last interval (2021 to 2023). On ElkinsvilleNE the default run had 27 dips, none of them on a reference polygon of their interval, 15 spikes (3 on reference) and 63 persistent steps (34 on reference). D6 catches few of these because its rebound factor only reduces a probability that is often close to 1.

D7 removes nearly all of them. On ElkinsvilleNE it kept 66 of 136 objects, with 1 spike and 1 dip left, and every remaining persistent step lay on a reference polygon. Commission against the pooled reference fell from 47.9 to 3.6 ha and area precision rose from 0.69 to 0.94, but the polygon hit rate fell from 0.70 to 0.42 (AIS: 0.48 to 0.32, commission 91.1 to 23.3 ha). The lost polygons are mostly wind and flood damage of 2016 to 2018 (hit rate 0.54 to 0.15), and 19 of the 21 lost polygons failed the forward check (18 failed only that check). That check compares the September 2016 image with the June 2020 image, and the null fitted to that two-step change took in every cell (null share 1.0), so even a persistent 10 m drop scored low. Two-step changes also carry the error of two CHM pairs, which widens their null. On LincolnNH, re-decided from the saved layers, D7 kept 19 of 56 objects: no spikes or dips, 5 of the 9 persistent steps, the 3 first-interval drops and 11 of the 24 last-interval objects. D6 therefore remains the default until D7 has been checked on more sites. A milder variant that applies only the back check on top of D6 kept the hit rate at 0.64 with 29.9 ha of commission on ElkinsvilleNE and has not been added as a rule.

Acquisition dates matter for the same reason. September CHMs are taller than June CHMs, so an interval that pairs a late-summer image with an early-summer one shows a broad height loss that is partly phenology. The report lists every interval whose two images differ by more than six weeks of season (for LincolnNH, 2021-09-20 against 2023-06-23, 89 days) and counts the objects without a follow-up image. The `follow_up` column of the objects marks them, and the chip titles add "no follow-up image".

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

`--rule` chooses the decision rule: D1, D2, D5, D6 (the default), D7 or one of the strict variants D5s, D6s and D7s. The [Rules](#rules) subsection of the glossary states what each rule requires and compares them. D7 and D7s write two extra z stacks (`_back2` and `_fwd2`) to `z/`. `--method` chooses the standardization: `scene` (median and MAD over the whole site), `scenemean` (mean and standard deviation over the whole site), `local` (one window pass) or `local_twopass` (two window passes, the default). The [Glossary](#glossary) defines each of these terms.

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

The `inputs`, `grid`, `forest` and `standardize` sections shaped the saved layers and z stack, so the rerun takes them from the saved run's `run_params.json`, and options given to the first run (for example `--years` or `--window-m`) need not be repeated. Passing one of these options, or setting one of these keys in a `--config` file, is allowed only when the value equals the saved one. A different value stops the command before anything is written, with a list of every difference. The rule, boundary, minimum mapping unit, refinement and figure settings may change. They come from the packaged defaults, the rerun's `--config` file and its options, not from the saved run. A changed boundary or smoothing scale then affects the decision only. In a fresh run with `local_twopass`, both also choose the cells left out of the second window pass, so a fresh run can give slightly different z and detections than a rerun with the same settings. The site must also match. The saved layers and z stack are copied into the new folder, so it is complete on its own and can serve as `--from-run` in turn. Refinement (on by default) and the NAIP backdrop still read the 0.6 m source rasters. In that case the site's files are discovered again (file metadata only), and the CHMs, plus NAIP for a backdrop, must still be under `$CSDV_DATA_ROOT`. Add `--no-refine` to rerun without them. The result goes to a new folder, as always.

A rerun with `--rule D7` or `--rule D7s` reuses the saved two-step z stacks when the source run has them and otherwise computes them from the saved layers. A D6 run can therefore be re-decided as D7 without the source rasters:

```bash
csdv detect-disturbance --site LincolnNH --from-run results/disturbance/LincolnNH \
    --rule D7 --no-refine --no-naip-backdrop \
    --output-dir results/disturbance/LincolnNH_D7
```

To redraw only the figures and the report of a run, for example with a NAIP backdrop, rebuild them in place:

```bash
csdv disturbance-figures --run-dir results/disturbance/ElkinsvilleNE --naip-backdrop
```

Without `--naip-backdrop` or `--no-naip-backdrop`, the command follows the saved run's `figures.naip_backdrop` setting. The refined examples and the NAIP backdrop need the source rasters. When they are missing, the command logs a warning and draws the other figures. This overwrites `figures/` and `report.md` and leaves every other file, `summary.json` included, alone.

## Reading the report and figures

Start with `report.md`. The detections-by-interval table gives, for each image pair, the monitored forest area, the detected area after and before the minimum mapping unit, the number of objects and the share of monitored forest detected. A share far above its neighbours, or a large gap between the area before and after the MMU, points at a CHM problem in one of the two images rather than at real loss. The null-fits table gives the fitted centre, spread and null share per date. On ElkinsvilleNE the spread of the ΔCHM DI ranged from about 0.75 to 1.0 and the null share from 0.80 to 1.0. A date outside that range deserves a look at `figures/null_fits.png`. Skipped dates (too few monitored cells) are flagged, and nothing is detected on them.

The figures follow the same order as the stages. `forest_initial.png` and `forest_by_year.png` show the forest mask, and a year with much less forest than its neighbours usually means a weak CHM. `z_maps.png` shows the DI per interval. Broad smooth patches there would indicate bias that the window did not remove. `probability_maps.png` shows the decision probability. Its colour ramp changes from neutral to warm at the boundary (0.5 by default), which marks the boundary on the maps without contour lines around single cells, and cells that were not monitored on a date are grey. `null_fits.png` shows each date's z histogram with the fitted null. Its dashed line is the z where the cell-level posterior (from the histogram, not the smoothed lfdr curve of the decision) reaches the boundary, so read it as approximate. `disturbance_onset.png` and `onset_by_interval.png` show the final map over the first year's CHM height in grey (tall canopy is light), or over NAIP when the backdrop is on. Monitored forest without a detection is left clear so the backdrop shows through, which the legend marks with a hatched swatch, and ground that was never monitored is washed white. `area_by_interval.png` shows the areas, and `refined_examples.png` the 0.6 m outlines of the largest objects over the CHM before and after onset.

Remember that the last interval has no persistence check, so its detections are less certain than the others. The report lists this among its caveats, together with the number of objects that have no follow-up image and every interval whose two images were flown more than six weeks of season apart.

## Inspecting a run with image chips

Before a formal accuracy assessment, a quick look at a sample of detections shows whether a run maps real canopy loss. `csdv disturbance-chips` draws a stratified sample from a run folder and renders one figure per sampled unit. Each figure has the NAIP RGB image (top row) and the CHM (bottom row) of every year, centred on the unit, with the detected 19.2 m cells outlined in yellow and, on the CHM row, the refined 0.6 m envelope in orange. The images before and after the onset interval carry blue and red frames. A right-hand column puts the unit in context over an area four times wider than the chips (at least 600 m): the run's onset map on top and the decision probability of the unit's interval below it (the z of that interval for rule D1), with the chip extent shown as a dashed box. A bottom row shows two time series for the 19.2 m cells whose centres fall inside the refined envelope (or inside the detected cells when the object has no envelope): the DI tested by the run's rule for every interval and the cell mean CHM height of every year, each as a median line with 25-75 % and 10-90 % bands and the onset interval shaded. The DI is defined per cell, not per 0.6 m pixel, so these are cell statistics. For the spatial rules (D5, D6, D7 and their strict variants) the DI series shows the z smoothed at the broadest spatial scale (25 m by default), because that is what the rule tests, and a dashed tick at each interval marks the smoothed z where the calibrated probability reaches the decision boundary. The smoothed field has a narrower null than the raw z, so a coherent patch at a raw z near −2.5 can sit well below its cutoff. For D2 the series shows the raw z with its own probability cutoff, and for D1 the raw z with the fixed cut at −3. A single dip in the DI followed by a return to zero, with a height drop that does not recover, is what a real canopy loss looks like. A height drop confined to one image, or a CHM drop under terrain shadow, points to an artefact. A contact sheet shows the before and after RGB pair of every unit on one page, which is usually enough for a first pass.

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
| Detections in the last interval look doubtful | That interval has no later image, so D6 cannot check it for a rebound and D7 checks it against the image two dates back only. Such objects have `follow_up` False. Treat them as unconfirmed until the next NAIP year arrives |
| Many detections in one interval that come back at the next image | An image spike or dip, often with a season mismatch between the two images (see the report caveats). Compare with a `--from-run --rule D7 --no-refine` rerun |
| Process killed on WSL | Memory pressure. Lower `GDAL_CACHEMAX`, close other programs, or run on CHPC |

## Glossary

Terms are listed alphabetically, and the [Rules](#rules) subsection compares the decision rules. Sizes assume the default 19.2 m cell on a 0.6 m CHM. Example values come from the default runs on LincolnNH and NorthMaine (NorthMaine without 2023) and change from date to date.

### Terms

**Back change.** The two-step change h[t] − h[t−2] that rule D7 checks for the interval ending at date t. It skips the earlier image of the interval (t − 1), so a canopy that only looks tall in that image (a spike) shows no loss in the back change. It is standardized like the one-step change and saved as the `_back2` z stack.

**Baseline.** D1 is called the per-cell baseline because it is the simplest rule, a fixed z cut applied to each cell on its own (the NISAR-style rule), and the probabilistic rules were measured against it. In `scripts/compare_disturbance_runs.py` the baseline is instead the first `--run`, the run that every other variant is compared with. Neither use refers to a baseline period of images.

**Boundary (Bayes decision boundary).** The probability a cell must exceed to be called a loss, `decide.p_boundary` (`--p-boundary`, default 0.5). At 0.5 a cell is called when loss is more likely than no loss. The run label p09 means a boundary of 0.9. Probability rises steeply in the tail of the z histogram, so a higher boundary moves the z cutoff only a little: from 0.5 to 0.9 the per-cell cutoff on LincolnNH moved from about −3.5 to −4.1 and the 25 m cutoff from about −1.7 to −1.9. `local_twopass` also uses the boundary to choose the cells it leaves out of its second window pass (see two pass).

**Bracketed spatial support.** The decision probability of rule D7, the smallest of three spatial-support probabilities: one for the one-step change h[t] − h[t−1], one for the back change and one for the forward change. Each comes from its own z stack with its own null fits. A loss passes only when it holds against the images on both sides of the interval. A check that does not exist (the back change in the first interval, the forward change in the last) is left out of the minimum.

**Cell.** The unit of every decision, a square of 32 × 32 pixels of the 0.6 m CHM (19.2 m on a side, 0.0369 ha). Its value is the mean CHM height of its valid pixels, and it is left empty when fewer than half of its pixels are valid (`grid.min_valid_share`). The cell size is the grain.

**Cell test.** The test on a cell's own evidence that the spatial rules (D5, D6 and D7) add to the neighbourhood test, so that smoothing cannot spread a detection into neighbouring forest that shows no loss. The guide calls a cell that passes it one whose evidence "leans toward loss". The plain rules ask that the per-cell probability exceed the prior. With the prior at its 0.01 floor, as on most dates so far, this passes nearly any cell whose z is below the null centre (about z < 0). The strict rules ask that the per-cell probability exceed the boundary, which is the D2 test.

**Date and interval.** A date is one NAIP year, indexed t. An interval is the pair of consecutive dates t − 1 and t and is written with both years (2018-2021). Rasters with one band per date, such as `probability.tif` and the z stacks, store each interval under its later year, and the first date has no interval.

**Disturbance Index (DI).** Another name for z.

**Efron's empirical null.** The distribution of z expected for unchanged forest on one date, fitted from the data instead of assuming that z is standard normal (Efron, 2004, *Journal of the American Statistical Association* 99: 96–104). A normal curve N(μ0, σ0) is fitted to the centre of the histogram of monitored z, between its quartiles (`decide.pct0`), where nearly all cells are unchanged forest. The fit gives the null centre μ0, the null spread σ0 and the null share p0, the share of cells that follow the curve. Each date, each smoothing scale and each D7 check gets its own fit, so a date with noisier CHMs gets a wider null and needs a stronger change for the same probability. The raw z nulls of LincolnNH and NorthMaine had σ0 of 0.95 to 1.09. The report's null-fit table and `null_fits.png` show the fits.

**Forest.** A cell where at least half of the valid pixels are taller than 5 m (`forest.min_cover`, `forest.height_m`).

**Forward change.** The two-step change h[t+1] − h[t−1] that rule D7 checks for the interval ending at date t. It skips the later image of the interval (t), so a canopy that only looks short in that image (a dip) shows no loss in the forward change. It is saved as the `_fwd2` z stack.

**Gain probability.** The probability that a change is a real height gain, from the upper tail of the z histogram. D6 computes it on the smoothed z of the next interval to detect a rebound.

**Grain.** The cell size, `grid.grain_m` (`--grain-m`, default 19.2 m), rounded to a whole number of CHM pixels: 9.6 m is 16 × 16 pixels and 38.4 m is 64 × 64 at 0.6 m. On ElkinsvilleNE the hit rate rose from 0.47 at 2.4 m to 0.70 at 19.2 m and levelled off at 38.4 m, because finer cells broke partial and wind disturbance into fragments smaller than the MMU.

**h.** The cell mean CHM height of one date, so h[t] − h[t−1] is the one-step change of the interval ending at t.

**lfdr (local false discovery rate).** The probability that a cell with a given z belongs to the null: lfdr(z) = p0 f0(z) / f(z), where f0 is the fitted null curve and f is the smoothed histogram of all monitored cells. The probability of loss is P = 1 − lfdr. The test is one-sided, so a z at or above the null centre gets P = 0, and P is forced to rise steadily as z falls.

**Local.** Standardization against the forest in the window around each cell instead of the whole scene (`--method local`, one window pass). It removes broad, smoothly varying height bias between CHM years, which `scene` standardization turns into false patches once z is smoothed.

**MAD.** The median absolute deviation, the median distance of the values from their median, multiplied by 1.4826 so that it equals the standard deviation for normally distributed data. Extreme values such as clearcuts affect it much less than the standard deviation. `scene` divides by the MAD, and the local methods use the scene MAD to trim and floor their window statistics.

**MMU (minimum mapping unit).** The smallest object kept, `mmu.mmu_ha` (`--mmu-ha`, default 0.405 ha or 1 acre, which is 11 cells at 19.2 m). It is applied after the rule, so it drops small objects but never changes which cells pass. Refined envelopes must reach the same area. The run label mmu1 means 1 ha (28 cells, 1.03 ha).

**Monitored.** A cell is monitored in an interval when it was forest at the earlier date. A cleared cell stops being monitored once its canopy is gone, and the first date is never monitored.

**Near miss.** An object that passed the rule but was smaller than the MMU. `csdv disturbance-chips` samples near misses of at least half the MMU.

**Object.** Detected cells that share an onset and touch, diagonally included (8-connected).

**Onset.** The first interval in which a cell is detected. `onset_year.tif` stores the later year of that interval, 0 for monitored cells never detected and −1 for cells never monitored. Each cell has at most one onset, so a second loss on the same cell is not mapped.

**p, P and p0.** Three different quantities. Lowercase p in a run label (p09) is the boundary. Capital P is a probability of loss, 1 − lfdr. p0 is the null share of the empirical null.

**Per-cell probability.** P computed from the cell's own unsmoothed z under the date's empirical null. D2 and the strict cell test compare it with the boundary, and the plain cell test compares it with the prior. On LincolnNH and NorthMaine it exceeded 0.5 at a z of about −2.8 to −3.6, depending on the date.

**Persistence.** The factor that turns D5 into D6. The spatial-support probability of an interval is multiplied by 1 − P_gain, where P_gain is the gain probability of the same cell at the next interval (the larger of its values at the two smoothing scales). A loss that comes back at the next image, as when a CHM failure recovers, is suppressed. Because the factor multiplies, a loss with a probability near 1 survives a moderate rebound probability (0.99 × 0.6 = 0.59). The last interval has no next image and keeps its spatial-support probability.

**Population.** The cells whose statistics set the reference for z on a date: the forest of the earlier date. The local methods use only the population cells inside each window, and the second pass of `local_twopass` also leaves out the cells with supported loss.

**Prior.** The expected share of disturbed cells on a date, 1 − p0, floored at 0.01 (`decide.min_prior`). On most dates of LincolnNH and NorthMaine the null took in every cell (p0 = 1), so the prior sat at the floor.

**Refinement and envelope.** The redraw of each object from the 0.6 m CHM pair. The height change around the object is standardized against a ring of forest outside it, pixels with a loss probability above 0.5 within 10 m of the object become openings, and the openings are closed into an envelope that follows canopy openings rather than cell edges. Envelopes smaller than the MMU are dropped.

**Run label.** The suffix of a run folder that names the one setting changed from the defaults, such as D6s, p09, sig10 and mmu1 (see [Run labels](#run-labels)).

**Scene.** The whole site grid on one date, which is the area that all years cover. `--method scene` standardizes each date with the median and MAD of all population cells, and `scenemean` with their mean and standard deviation. The local methods fall back to the scene statistics in windows with too little forest.

**Sigma (σ, smoothing scale).** The standard deviation, in metres, of the Gaussian kernel that smooths z for spatial support (`decide.spatial_sigma_m`, default 10 and 25 m). The smoothed z of a cell is a weighted mean over the monitored cells around it, with non-forest neighbours left out and the weights rescaled. At 19.2 m cells, σ = 10 m (0.52 cells) gives the cell itself 58% of the weight and its eight neighbours the rest, so it acts nearly as a per-cell test that a strong single cell can pass. σ = 25 m (1.30 cells) gives the cell 9%, its 3 × 3 block 58% and its 5 × 5 block (96 m) 90%, so the loss must be shared by a patch of cells. The run label sig10 means the 10 m scale only. Do not confuse σ with σ0, the spread of the null.

**Spatial support.** The decision probability of rule D5, which D6 and D7 build on. z is smoothed at each sigma, an empirical null is fitted to each smoothed field, and the cell takes the larger of the two probabilities. Smoothing averages noise down (smoothed σ0 of about 0.75 at 10 m and 0.45 at 25 m, against about 1 for raw z), while a patch whose cells all lost height keeps most of its depth. A coherent patch therefore stands far out from the smoothed null and an isolated noisy cell does not. On LincolnNH and NorthMaine a cell passed 0.5 at a smoothed z of about −2.1 to −2.7 at 10 m or −1.2 to −1.8 at 25 m.

**Spike, dip and rebound.** Artefacts of a single image. A spike is canopy that looks taller in one image than in the images on either side, so the interval after it shows a loss that is not real. A dip is canopy that looks shorter in one image, which shows as a loss in the interval ending at that image and a gain (a rebound) in the next. D6 suppresses losses followed by a rebound and so removes dips, while D7 removes spikes with its back check and dips with its forward check.

**Strict.** The s suffix (D5s, D6s and D7s). The cell test uses the boundary instead of the prior, so a cell must pass D2 on its own as well as have spatial support.

**Two pass.** The default method, `local_twopass`. The first window pass gives z, from which the spatial-support probability is computed (without a cell test). Cells above the boundary leave the population, and the second pass recomputes the window mean and standard deviation without them, so a large disturbance does not set its own reference. Because the second pass depends on `decide.p_boundary` and `decide.spatial_sigma_m`, a fresh run that changes either also changes z. On NorthMaine the p09 run changed 31% of the z values, with a median shift of 0.03 and 0.8% of all values shifted by more than 0.5. A `--from-run` rerun keeps the saved z.

**Window.** The square of cells around each cell that the local methods use, `standardize.window_m` (`--window-m`, default 150 m) rounded to an odd number of cells: 9 × 9 cells at 19.2 m (172.8 m, about 3 ha). The window mean and standard deviation come from its population cells, with three guards. Cells beyond ±3 scene z are left out (`trim_z`), windows with less than 20% valid forest use the scene statistics (`min_frac`), and the window standard deviation never falls below a quarter of the scene MAD (`sd_floor`).

**Window pass.** One computation of the window mean and standard deviation over the population, and the z that follows from them. `local` makes one pass and `local_twopass` two.

**z.** The standardized change of a cell over one interval, also called the Disturbance Index (DI). With the default input, z = (Δh − m) / s, where Δh = h[t] − h[t−1], and m and s are the mean and standard deviation of Δh over the population cells in the cell's window (the median and MAD over the scene for `scene`). Negative z means the cell lost more height than the forest around it. On LincolnNH and NorthMaine the median window standard deviation was 0.6 to 1.0 m, so z = −3 meant a drop about 2 to 3 m larger than in the surrounding forest. The z stacks in `z/` store 100 z as int16.

### Rules

Every rule answers the same question for each monitored cell and interval: did the cell lose canopy between dates t − 1 and t? Onset, objects, the MMU and refinement then work the same way for all rules. The rules differ in three things: the test on the cell's own z, whether its neighbourhood must show the loss too, and how many images must agree. The table summarizes them. Its last column gives the hit rate against the pooled ElkinsvilleNE reference (77 polygons) and the commission, both on 19.2 m cells with the 1-acre MMU and before refinement.

| Rule | What must hold for the interval t − 1 to t | Space | Images | ElkinsvilleNE |
|---|---|---|---|---|
| D1 | z ≤ −3 | cell | t − 1, t | 0.49, 70 ha |
| D2 | per-cell P > boundary | cell | t − 1, t | 0.61, 79 ha |
| D3 | D2, then each cluster scored against the mirrored image | cluster | t − 1, t | research only |
| D4 | per-cell evidence summed over dates > boundary | cell | every date up to t | research only |
| D5 | spatial support > boundary, and per-cell P > prior | 10 or 25 m neighbourhood | t − 1, t | 0.70, 116 ha |
| D5s | spatial support > boundary, and per-cell P > boundary | 10 or 25 m neighbourhood | t − 1, t | 0.62, 80 ha |
| D6 (default) | spatial support × (1 − gain at t + 1) > boundary, and per-cell P > prior | 10 or 25 m neighbourhood | t − 1, t, t + 1 | 0.70, 48 ha |
| D6s | as D6, with per-cell P > boundary | 10 or 25 m neighbourhood | t − 1, t, t + 1 | 0.61, 13 ha |
| D7 | bracketed spatial support > boundary, and per-cell P > prior | 10 or 25 m neighbourhood | t − 2 to t + 1 | 0.42, 3.6 ha |
| D7s | as D7, with per-cell P > boundary | 10 or 25 m neighbourhood | t − 2 to t + 1 | not tested |

**D1, fixed cut.** The cell's z is −3 or lower. Space: the cell alone. Time: one image pair. No probability is computed, and the cut stays the same on every date whatever the CHM quality. Many D1 detections are scattered single cells, and the MMU removed 59% of the D1 area on ElkinsvilleNE.

**D2, per-cell probability.** The cell's per-cell probability exceeds the boundary. Space: the cell alone. Time: one image pair. The cutoff adapts to each date through the empirical null (about z ≤ −2.8 to −3.6 on LincolnNH and NorthMaine), but single cells remain noisy, and the MMU removed 68% of the D2 area on ElkinsvilleNE.

**D3 and D4, research only.** Neither is available in `csdv detect-disturbance`. D3 grouped D2 cells into clusters and scored each cluster against the clusters of the mirrored image (2μ0 − z), which shares the image's spatial noise but cannot contain loss. CHM noise produces false losses more often than false gains, so the mirrored image understated false clusters, and after the MMU D3 scored the same as D2. D4 summed per-cell evidence over dates (a CUSUM-style recursion) so that weak but repeated loss would build up, and its variant D4c required the loss to recur at the next date. Both were tested on height and cover rather than on their change and mostly flagged persistent stand structure such as young stands, edges and old gaps. The research scripts also tested D1n2 (z ≤ −3 at two consecutive dates) and D2bh (a Benjamini–Hochberg false discovery rate of 5%).

**D5, spatial support.** The 10 m or the 25 m smoothed z passes the boundary under its own null, and the cell's own z leans toward loss (per-cell P above the prior). Space: at 10 m the cell and its eight neighbours, with most weight on the cell, and at 25 m a block of about 5 × 5 cells (96 m). Time: one image pair.

**D6, spatial support and persistence (default).** D5 holds, and the cell does not regain height at the next interval: the D5 probability times 1 − P_gain of the next interval still exceeds the boundary. Space: as D5. Time: three images (t − 1, t and t + 1). The last interval has no next image and is decided as D5.

**D7, bracketed spatial support.** Spatial support holds for the one-step change and for both two-step changes, so the loss appears whichever image before or after the interval it is compared with. Space: as D5. Time: four images (t − 2 to t + 1). The first interval has only the forward check and the last only the back check. D7 replaces the persistence factor of D6 rather than adding to it.

**Strict variants (D5s, D6s and D7s).** As the plain rule, but the cell test is the D2 test: the cell's per-cell probability must also exceed the boundary. A strict rule therefore detects only cells that D2 would detect on their own and that also have neighbourhood support. Patches shrink to their strongest cells, and weak or partial loss drops out along with most commission.

#### The tests in z

The cutoffs below show what each test asks of z on LincolnNH and NorthMaine. They span the dates of both sites, with NorthMaine at the less negative end. The last column assumes a patch of equal z in continuous, otherwise unchanged forest and gives the z every cell of the patch needs for its centre cell to pass.

| Test | Tested on | Boundary 0.5 | Boundary 0.9 | What it takes at 0.5 |
|---|---|---|---|---|
| D1 | raw z | −3.0 | −3.0 (fixed) | one cell at −3 |
| Plain cell test | raw z | about 0 | about 0 | any cell below the null centre |
| D2, strict cell test | raw z | −2.8 to −3.6 | −3.4 to −4.2 | one cell at the cutoff |
| Spatial support, 10 m | smoothed z | −2.1 to −2.7 | −2.7 to −3.2 | a lone cell at −3.6 to −4.7, or a 3 × 3 patch at −2.1 to −2.7 |
| Spatial support, 25 m | smoothed z | −1.2 to −1.8 | −1.5 to −2.1 | a 3 × 3 patch (0.33 ha) at −2.1 to −3.1, or a 5 × 5 patch (0.92 ha) at −1.3 to −2.0 |

The plain cell test was stricter only on NorthMaine 2018-2021, where the null share fell to 0.94 and the cutoff to about −2.0. The table also explains the run labels. A boundary of 0.9 moves the cutoffs by 0.2 to 0.3 z at 25 m, 0.4 to 0.6 at 10 m and 0.6 to 0.7 on raw z, so p09 trims the edges and the weakest patches. Without the 25 m scale (sig10), a broad patch of moderate loss must reach the 10 m cutoff in each 3 × 3 block, so wide partial harvests drop out. The strict cell test asks each cell to reach −2.8 to −3.6 by itself, which keeps only the cores of strong patches. The MMU acts after the rule and removes whole objects below its size.

#### Run labels

The comparison runs of LincolnNH and NorthMaine are named by the one setting changed from the defaults. Detected areas are after the MMU, and NorthMaine was run without 2023.

| Label | Changed setting | How to set it | LincolnNH (ha) | NorthMaine (ha) |
|---|---|---|---|---|
| D6 | none (defaults) | | 85.2 | 2,747 |
| D6s | strict cell test | `--rule D6s` | 30.8 | 2,022 |
| p09 | boundary of 0.9 | `--p-boundary 0.9` | 54.7 | 2,280 |
| sig10 | 10 m smoothing only | `decide: spatial_sigma_m: [10.0]` in a `--config` YAML | 39.4 | 2,093 |
| mmu1 | MMU of 1 ha | `--mmu-ha 1.0` | 57.3 | 2,605 |

D6s and mmu1 decide on the same z stack as D6. The p09 and sig10 runs were run from scratch, so their z differs slightly from the default (see two pass), and a `--from-run` rerun with the same settings would not reproduce them exactly.
