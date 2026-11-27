# Grid runs: disturbance detection at many locations

`csdv grid` runs the NAIP disturbance workflow (download, CHM inference, detection, chips, export) for any number of locations with one command per batch, tracks what is done, and gathers the small outputs in one folder that can be copied to a laptop. This guide covers the processing units, the commands, the folders they write and the usual ways of working with them on CHPC. The detection method itself is described in [disturbance_detection.md](disturbance_detection.md).

## Processing units

A unit is the area that one chain of jobs processes. Its ID is used as the site name everywhere, so its NAIP years are in `naip/<unit>/`, its CHMs in `naip_chm/<unit>/` and its detection runs in `disturbance/<unit>/<run_tag>/`.

Most units are cells of a fixed CONUS grid. A cell is one USGS 7.5 minute quadrangle (0.125 by 0.125 degrees, NAD83), the cell NAIP quarter quads are cut from, and it carries the quad code that starts every NAIP file name. The White Mountains cell `q4407159` holds the quarter quads `m_4407159_nw`, `_ne`, `_sw` and `_se`. A cell spans 13,000 to 17,500 ha depending on latitude. That is less than the NorthMaine site (19,052 ha), which ran without memory problems, and about the size of LincolnNH (14,115 ha). A cell never straddles a UTM zone. The grid is computed from coordinates and is not stored anywhere. `csdv grid locate --lonlat -71.68 44.06` prints the cell that holds a point and its bounds.

A unit can also be a box around a point, for a site that should be centred on a feature rather than follow the grid. `--lonlat LON LAT --buffer-km K --label NAME` gives the unit `p_NAME` with a half-width of K km. Half-widths above 8 km are refused, because larger boxes may run out of memory.

## Batch files

A batch is a CSV file with the columns `label,lon,lat,buffer_km,cell,notes`. Lines starting with `#` are comments. A row with `cell` uses that cell, a row with `buffer_km` becomes a point unit, and a row with only `lon` and `lat` uses the cell that contains the point. The label and notes are stored with the unit and appear in the status table. [config/grid/conus_pilot.csv](../../config/grid/conus_pilot.csv) is the pilot batch of 20 locations with contrasting disturbance regimes.

## NAIP years and sources

When a unit is first submitted, both NAIP catalogues are asked which years cover it and how much of the unit each year covers. The Planetary Computer is used for every year it covers (through 2023), and the USDA FPAC image service supplies the years only it holds, usually the newest one (2024 or 2025). By default every year from 2010 onward is planned. Years that cover less than 90% of the unit are skipped and logged. This catches states that USDA did not finish in their scheduled year (Maine 2023 covers 52% of the NorthMaine cell), and it also drops years of the neighbouring state from a cell on a state line, because NAIP states are flown in different years. The planned years, their source and coverage, and every skipped year with its reason are stored in the unit record. Later submissions reuse them. Pass `--replan` to ask the catalogues again (for example when a new NAIP year appears), or `--years` to choose years.

The same logic is available without the grid as `csdv download naip-auto`, which takes the options of `naip-pc` plus `--start-year` and `--min-coverage`. The source of each downloaded year is recorded in its `.provenance.json`. Every year of a unit is written in one CRS, that of the mosaics already on disk or else of the first year that arrives. A state can deliver one year in the neighbouring UTM zone (Maryland 2023 is in zone 18 as far west as 79 W), and those quads are warped into the unit's CRS, because detection refuses a unit with two. The USDA service keeps only the newest collection for an area, so download a year from it while it is there.

## Submitting

Submit from a CHPC login node after `source scripts/chpc/env.sh`. The year planning needs internet access, which login nodes have.

```bash
# Look at the plan first: years, sources and the sbatch commands.
csdv grid submit --batch config/grid/conus_pilot.csv --dry-run

# One unit first, then the batch.
csdv grid submit --cell q4407152
csdv grid submit --batch config/grid/conus_pilot.csv

# A point unit.
csdv grid submit --lonlat -71.55 44.16 --buffer-km 6 --label LincolnNH

# Guest nodes
csdv grid submit --batch config/grid/conus_pilot.csv --gpu notchpeak-guest --dry-run   # check, then drop --dry-run
```

For each unit the command submits up to five jobs, each waiting for the one before it.

| Step | Script | Resources | Skipped when |
|---|---|---|---|
| `download` | `run_download_dtn.sbatch naip-auto` | DTN, 2 CPUs, 32 GB, 4 h | every planned year has a NAIP mosaic |
| `chm` | `run_chm_inference.sbatch`, one array task per year | 1 GPU, 48 GB, 2 h per year | every planned year has a CHM |
| `detect` | `run_detect_disturbance.sbatch` | 4 CPUs, 24 GB, 3 h | never (a new run tag is required) |
| `chips` | `run_disturbance_chips.sbatch` | 2 CPUs, 8 GB, 1 h | `--no-chips` |
| `export` | `run_grid_export.sbatch` | 2 CPUs, 8 GB, 1 h | never |

If a job fails, Slurm cancels the jobs that wait on it. The export waits for detection to succeed and for the chips to end, so a chips failure does not hold back the export. `--gpu` chooses where the CHM tasks run (see below). `--steps` limits the chain, for example `--steps download,chm` to prepare data only.

The CHM step is the only one that waits for scarce hardware, so `--gpu` chooses its partition. `notchpeak` (the default) is the general notchpeak GPU pool. `kingspeak` and `lonepeak` are the general pools on the older clusters (Titan X and 1080 Ti cards). They are slower per task, but often idle, and a CHM needs only about 4 GB of GPU memory. `notchpeak-guest` (other groups' idle notchpeak GPUs) and `granite` (granite freecycle, L40S and H100) are preemptable. A preempted task is cancelled and not requeued (verified on CHPC on 2026-10-30), so give the submit command again afterwards: a task whose CHM already exists skips itself. Run `mychpc batch` on a login node to see how idle each pool is before a large submission. A job that waits with the reason `Priority` while GPUs look idle is behind jobs of higher fair-share priority, and those idle GPUs may be held for a larger job, so a guest route often starts sooner. Guest nodes belong to other groups and a faulty card can stay in service there. On notch089 one failed with "Unable to determine the device handle". The CHM task then stops at its GPU check and names the node, and `--exclude-nodes notch089` (comma-separated) keeps the CHM tasks of a resubmission off it. Finished years are skipped on resubmission. `granite-gpu-guest` is not offered because its Blackwell cards need a newer CUDA build than the environment has.

Each submission appends a row to the ledger, `$CSDV_GRID_DIR/submissions.csv`, with the time, unit, run tag, years, job IDs, the detection options and the git commit.

## Many small units: the work queue

`csdv grid submit` gives every unit its own chain and every year its own GPU task. Each task waits in the GPU queue by itself, and one download year that fails cancels the unit's chain. For a few quads that is fine. For a hundred boxes of 4.8 km it means several hundred waits. `csdv grid queue` sends a whole batch through three submissions instead:

```bash
source scripts/chpc/env.sh
csdv grid queue --batch config/grid/round2.csv --dry-run
csdv grid queue --batch config/grid/round2.csv \
    --checkpoint "$CSDV_RESULTS_ROOT/pairmodel/first_look_blocks2/best.pt" \
    --priority-intervals config/label/round2/priority_intervals.csv
csdv grid queue-status --name round2            # read from disk, any time
csdv grid queue-status --name round2 --problems
```

| Submission | Script | What it does |
|---|---|---|
| `download` | `run_grid_download.sbatch`, an array on the data transfer nodes | One task per unit. The task plans the unit's years itself, so nothing is planned on the login node. A year that a catalogue lists and cannot deliver is dropped when at least `--min-years` (4) arrive. A finished unit gets a marker in `<queue>/downloaded/`. |
| `workers` | `run_chm_worker.sbatch`, `--workers-per-pool` (2) jobs on every pool of `$CSDV_GPU_POOLS` | A worker checks its GPU, stages the conditioning rasters once, then takes one unit-year after another that has a mosaic and no CHM, of the units whose download has left its marker. It claims each by creating `<queue>/claims/<unit>_<year>`, which only one job can do. |
| `finish` | `run_grid_finish.sbatch`, a CPU array | One task per unit, after every worker has ended: detection on the years that have a CHM, the pair model's map on CPU (with `--checkpoint`, written to `<run>/model/`), export. |

The queue folder is `$CSDV_GRID_DIR/queues/<name>/`, named after the batch file unless `--name` says otherwise. There is no list of unit-years in it. What is left to do is read from disk every time, by the workers and by the command itself, which has three consequences. Workers can wait on several pools at once, and whichever pool frees a GPU first does the work. A worker that starts late finds less to do, and one that finds nothing exits at once (the last one cancels its siblings that are still pending, so the finish array is not held up by a pool that stays busy). And the same command given again submits only what is missing, which is how a batch recovers from a preempted worker, a faulty card (`--exclude-nodes`) or a failed download.

`$CSDV_GPU_POOLS` is set in `scripts/chpc/env.sh`: one string of sbatch options per pool, with semicolons between pools (see `env.example.sh`). Allocation names stay in that file. `csdv grid queue` asks Slurm about each pool first (`sbatch --test-only`) and submits nothing when a name is not valid. A worker asks for 24 GB and four hours by default (`--worker-mem`, `--worker-hours`), which suits boxes. A whole quad needs `--worker-mem 48G`. A worker stops taking unit-years a quarter of an hour before its walltime.

A unit-year whose inference fails is noted in `<queue>/failed/` and tried once more by another worker. After two failures it is left out, and the unit is finished without that year. A claim that a killed job left behind goes stale after 45 minutes and is taken over.

The finish task refuses a unit, with a message that says why, when its download has no marker, when a CHM is still to be made (give the queue command again), when fewer than `--min-years` are left, or when the unit's interval in `--priority-intervals` (a CSV with the columns `unit` and `interval`, such as `2018-2021`) is no longer a pair of consecutive years. The last check keeps a place that was chosen for one event from being processed without the two images around it. The unit record takes the years that were processed. The export also copies the four cell layers and the model map to `$CSDV_EXPORT_DIR/run_layers/<unit>/<tag>/`, where `csdv label` reads them, so one pull of the exports is enough. The inspection chips of `csdv grid submit` are not made.

### Pools on other clusters

On CHPC, kingspeak, lonepeak and granite are Slurm clusters of their own. From a notchpeak login node `--partition=lonepeak-gpu` is an invalid partition, although `mychpc batch` lists it. A pool reaches another cluster with `--clusters=<name>` in its option string:

```bash
export CSDV_GPU_POOLS="--partition=notchpeak-gpu --qos=notchpeak-gpu --account=notchpeak-gpu --gres=gpu:1;--clusters=lonepeak --partition=lonepeak-gpu --qos=lonepeak-gpu --account=lonepeak-gpu --gres=gpu:1"
```

Job IDs belong to one cluster, so no job can wait for a job of another one. The queue therefore treats such a pool differently in two places. Its workers are submitted without a dependency and start as soon as a GPU is free. A worker takes only units whose download has left its marker, looks again every minute while markers are missing, and gives up when no download has finished and it has found no work for an hour (`CSDV_WORKER_IDLE_S`, in seconds). The limit has to be longer than one download task, and a worker's walltime (`--worker-hours`) has to cover the whole download array, which took 20 minutes or more per box at six tasks at a time in round 2. And one more job is submitted on the login node's cluster, the gate (`run_queue_gate.sbatch`, one core). It asks the other clusters' queues about those workers once a minute and ends when none is pending or running. The finish array waits for the downloads, the local workers and the gate. If the scheduler of the other cluster does not answer, the gate keeps waiting. Its walltime is `--gate-hours` (24), which has to cover the longest wait of such a worker for a GPU plus its own walltime. When it runs out, the finish array starts and refuses the units that are not ready, and the queue command given again takes up what is left.

`<queue>/workers.txt` names the workers of the last submission, with the cluster of each when a pool lies elsewhere, so that the last worker can cancel pending siblings on any cluster. Jobs of another cluster do not show in `squeue` or `sacct` on the login node. Use `squeue -M lonepeak -u $USER`, or `scripts/chpc/round2.sh status`, which lists every cluster the pools name.

Which pools are worth naming changes by the day. `sbatch --test-only <pool options> scripts/slurm/run_chm_worker.sbatch x` prints when a worker would start and submits nothing. On 2026-11-20 the general GPU partitions of lonepeak and kingspeak would have started within minutes, notchpeak-gpu within 13 hours, and the preemptable routes (notchpeak-gpu-guest, granite-gpu freecycle) only after 2 and 16 days. The GTX 1080 Ti cards of lonepeak run the model: a 4.8 km box-year took about 4.4 minutes (263 s, of which about one minute was inference). The cards of kingspeak have not been tried.

### Blocks of adjoining quads

A block is a set of adjoining quads mapped wall to wall, the unit of a regional map. `scripts/chpc/region.sh` sends blocks through the same queue, with each quad mapped by the pair model at the end of its finish task. The first block was Southern Indiana: eight quads around q3908654 (`indiana_block`), a ring of 16 (`indiana_ring`) and 55 more (`indiana_south`), 80 quads with the centre. All 24 quads of the first two went through on the first try. A quad takes about 12 GB on disk (7.5 to 8.1 GB of mosaics and 2.5 to 4.9 GB of height models), about 40 GPU minutes for the height models of eight years and 36 to 51 minutes for its finish task.

```bash
python scripts/grid/block_csv.py blue_ridge --state NC --bbox -84.375 35.0 -81.25 36.625   # laptop: writes the batch files
scripts/chpc/region.sh block blue_ridge --dry-run     # CHPC: free space, links, the jobs
scripts/chpc/region.sh block blue_ridge               # submit, and again the next evening for what is left
scripts/chpc/region.sh status blue_ridge              # quads not exported, disk use, years and model per quad
scripts/chpc/region.sh remap blue_ridge --checkpoint RUN   # map the finished quads again with another model
```

- **One state per block.** `scripts/grid/block_csv.py` keeps the quads of a bounding box that lie wholly inside one state. A quad on a state line loses the years of the neighbouring state (see "NAIP years and sources") and is left for later.
- **Parts.** A block is written as parts of at most 40 quads (`blue_ridge_1.csv`, `blue_ridge_2.csv`, ...), and `region.sh` takes the block's name for all of them. Each part is a queue of its own, because the finish tasks of a queue wait for every height model of that queue.
- **Storage.** With `CSDV_REGION_STORE` set in `scripts/chpc/env.sh`, the mosaics and height models of new quads are kept in that folder, and `$CSDV_DATA_ROOT/naip/<unit>` and `naip_chm/<unit>` are links to them. Nothing else moves. The mosaics stay on disk, because a remap reads them.
- **Another model.** `region.sh remap` runs the finish task with `--remap`: detection is kept, the map is made again and the quad is exported again. The run folder keeps the old map as `model.prev.<job>`. A quad that the named model has mapped already is skipped, so the command also serves for the quads of a block that had a finished run before the block existed: the queue takes them for done and gives them no map. The map's folder names the training run of its model in `checkpoint.txt`, and `export.json` repeats it as `model`.
- **Workers.** The block step gives the workers 48 GB and 8 hours (`CSDV_REGION_QUEUE_ARGS` replaces these options): the height model of a whole quad needs 21 to 34 GB, more than the 24 GB the queue gives a small box.

- **A page of the block.** `scripts/chpc/region.sh page NAME... --page FOLDER` runs `csdv grid webmap` in one CPU job and writes a static page to `$CSDV_PUBLIC_HTML/csdv_blocks/FOLDER` (or under `CSDV_REGION_PAGE_DIR`). The page shows the block as one map: NAIP of every year, and per interval the model's probability of loss and the cause it names from 90% on, with the outlines and image dates of the quads. Each layer is one Web Mercator mosaic (`gdalwarp`, so quads of two UTM zones join) cut into tiles of 512 pixels (`gdal2tiles.py`). The image tiles end at zoom 14, about 3.7 m at 39 degrees north, and sharper online imagery shows when the reader zooms further. Notes placed on the map stay in the browser and download as GeoJSON. A box of 4.8 km with eight image years came to 9 MB, which projects to about 6 MB per quad and image year. The job stops when the page is projected above `CSDV_REGION_PAGE_GB` (default 6): `--naip-zoom 13` is a quarter of the size. The maps of adjoining quads overlap by a few hundred metres, and the page draws each cell from the quad whose box holds its centre, as the patch layer does. In the outer 50 m of a mosaic the model flags about twice as much as the neighbour's map does on the same ground, and the two maps agree on a third of the flagged cells there, against nine tenths from 100 m inwards (24 quads of Southern Indiana). After a remap, `region.sh page ... --model-only` cuts the model's layers again and keeps the images.

- **Patches of the block.** `scripts/chpc/region.sh patches NAME... --page FOLDER` runs `csdv pair patches` in one CPU job of minutes and writes the block's patch layer to `$CSDV_EXPORT_DIR/blocks/FOLDER` (see "Patches" in [pair_model.md](pair_model.md)). Its `seams.csv` compares the maps of adjoining quads on the strip that both cover, and its `background.csv` names the quads and dates on which the height model is off. With `--previous` the step takes the map each quad had before its last remap (`model.prev.<job>` in its run folder) and writes to `FOLDER_previous`, so two models can be compared on the same ground.

`scripts/chpc/region_pull.sh` brings the cell layers and model maps of a block to the laptop, about 25 MB per quad. `scripts/chpc/region_pull.sh --blocks FOLDER` brings the patch layer alone, a few MB.

Three faults and what to do about them:

- **A quad that was processed before its queue existed.** The finish task stops within seconds at "the download has not finished", because the queue has no marker for the quad. `region.sh remap` now writes the marker for every quad with a finished detection. By hand: `touch $CSDV_GRID_DIR/queues/<name>/downloaded/<unit>`.
- **A page job that ran out of time.** Give the same `region.sh page` command again. Finished layers are kept, and a layer that was being cut sits in a folder ending in `.part`, which is cut again. The time limit is `CSDV_REGION_PAGE_HOURS` (24). The log gives the seconds of the mosaic and of the tiles for every layer. The first page (80 quads, seven image years) took 4 hours 45 minutes and came to 1.9 GB. A page that was begun before this rule existed (November 2026) has no `.part` folder: delete the newest `t/naip_<year>` folder before giving the command again.
- **A page above the web folder's space.** The job stops after the first image year when the page is projected above `CSDV_REGION_PAGE_GB`. `--naip-zoom 13` on `csdv grid webmap` gives a quarter of the size.

## Run tags and reruns

Every detection run of a unit has a tag, `default` unless `--run-tag` says otherwise, and lives in `$CSDV_RESULTS_ROOT/disturbance/<unit>/<run_tag>/`. A tag that already exists is refused, as is a tag whose detection job is still queued. To try other parameters, submit again with a new tag and put the `detect-disturbance` options after `--`:

```bash
csdv grid submit --batch config/grid/conus_pilot.csv --run-tag D6s_1ha -- --rule D6s --mmu-ha 1.0
csdv grid submit --cell q4407152 --run-tag strict --config my_disturbance.yaml
```

The NAIP and CHM years exist by then, so only detection, chips and export are submitted. The parameters of every run are in its `run_params.json`, and the ledger links each tag to its command line. To redo only the later steps of an existing run, for example after changing the export, use `--steps chips,export` or `--steps export`.

## Progress

```bash
csdv grid status                 # every unit and run tag
csdv grid status --problems      # only rows that failed or were never submitted
csdv grid status --run-tag D6s_1ha
```

The table has one row per unit and run tag. The `naip` and `chm` columns count finished years, `detect`, `chips` and `export` show which steps of the run are done, and `state` names the first unfinished step with the Slurm state of its job. A failed job's row includes the path of its log. Progress is read from the files, so a step finished by hand counts as done. A full listing also writes `status.csv` to `$CSDV_GRID_DIR` and to the export folder.

## Exports and copying results

The export step writes everything needed to look at a run on a laptop into `$CSDV_EXPORT_DIR` (default `$CSDV_RESULTS_ROOT/exports`):

```
exports/
├── status.csv
├── runs/<run_tag>/<unit>/
│   ├── onset_year_u8.tif        # onset year minus 2000, 0 no detection, 255 never monitored
│   ├── max_probability_u8.tif   # highest loss probability over intervals, percent
│   ├── disturbances.gpkg  objects.csv  summary.json  report.md  run_params.json
│   ├── unit.json  export.json  detect_disturbance.log
│   ├── figures/
│   └── inspection/              # chips, contact sheet, samples.csv
└── quicklooks/<unit>/
    ├── chm_<year>_3m.tif        # uint8, value x 0.25 = height in m, 255 nodata
    └── naip_<year>_3m_rgb.tif   # JPEG RGB with a mask
```

The rasters are Cloud Optimized GeoTIFFs on the run's 19.2 m grid (run files) or a 3 m grid (quicklooks), and the onset map carries a colour table so it displays without styling in QGIS. The CHM quicklook stores its scale in the file, so QGIS reports heights in metres. A run export is a few MB, or tens of MB when the refined outlines are many. NAIP quicklooks take about 14 MB per year per cell and CHM quicklooks a few MB. The 0.6 m NAIP and CHM mosaics and the float32 run layers stay on CHPC. To redo an export by hand, run `csdv grid export --unit <unit> --run-tag <tag>`.

`scripts/chpc/pull_exports.sh` copies the export folder to the local machine with rsync, through a data transfer node. Set two variables once, for example in `~/.bashrc`:

```bash
export CSDV_CHPC_HOST=<uNID>@dtn07.chpc.utah.edu
export CSDV_CHPC_EXPORTS=/uufs/chpc.utah.edu/common/home/dycelab/data/CSDV/exports
```

Then `scripts/chpc/pull_exports.sh` copies everything new to `results/chpc_exports/` (or `$CSDV_LOCAL_EXPORTS`), and `scripts/chpc/pull_exports.sh runs/D6s_1ha` copies one run tag. Only new or changed files are transferred, and nothing is deleted locally.

## Checking a run

`csdv grid check` reads an export folder and checks each exported run for completeness and internal agreement, so it works on a laptop after the export has been copied as well as on CHPC. It compares the years of the unit record, the parameters, the summary and the quicklooks, checks that the byte rasters sit on the run's grid with the onset areas of the summary, that the objects agree in number and area with `objects.csv` and the summary, and that every chip of the inspection sample exists. It also warns about things to look at that are not errors: image pairs more than six weeks of season apart, a CHM year whose median height departs from the others by more than 0.5 m, year-to-year changes of forest area larger than all detections, invalid polygons, and log warnings.

```bash
csdv grid check                                  # every run under $CSDV_EXPORT_DIR, else results/chpc_exports
csdv grid check --unit q4407152 --run-tag default --verbose
```

Each run gets `check.md` and `check.json` in its export folder. The command exits with an error when any check fails.

## Cost

Each cell needs one GPU task per NAIP year (six to eight years for most of CONUS since 2010), plus about ten minutes of CPU time for detection, chips and export. The pilot batch of 20 cells is about 140 GPU tasks. Submit one cell first and check its run time and peak memory with `sacct -j <job ids> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem` before submitting the batch. Leave out `-X`, because MaxRSS is reported on the `.batch` step and not on the job line. CHPC gathers job accounting from cgroups, so MaxRSS includes the page cache of the files a job reads. A step that reads many GB of imagery can therefore report a MaxRSS at its memory request without being short of memory. The export of the test cell q4407152 reported 8.0 of 8 GB, although the quicklook reads, measured outside Slurm on a smaller site, need only the GDAL block cache (`GDAL_CACHEMAX`) plus about 120 MB. On q4407152 (eight years, 2011 to 2025) the download took 37 minutes, each CHM task 5 to 6 minutes, detection 3 minutes (12.8 GB), chips 2 minutes and the export 6 minutes.
