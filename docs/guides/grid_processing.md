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

The same logic is available without the grid as `csdv download naip-auto`, which takes the options of `naip-pc` plus `--start-year` and `--min-coverage`. The source of each downloaded year is recorded in its `.provenance.json`. The USDA service keeps only the newest collection for an area, so download a year from it while it is there.

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
| `workers` | `run_chm_worker.sbatch`, `--workers-per-pool` (2) jobs on every pool of `$CSDV_GPU_POOLS` | A worker checks its GPU, stages the conditioning rasters once, then takes one unit-year after another that has a mosaic and no CHM. It claims each by creating `<queue>/claims/<unit>_<year>`, which only one job can do. |
| `finish` | `run_grid_finish.sbatch`, a CPU array | One task per unit, after every worker has ended: detection on the years that have a CHM, the pair model's map on CPU (with `--checkpoint`, written to `<run>/model/`), export. |

The queue folder is `$CSDV_GRID_DIR/queues/<name>/`, named after the batch file unless `--name` says otherwise. There is no list of unit-years in it. What is left to do is read from disk every time, by the workers and by the command itself, which has three consequences. Workers can wait on several pools at once, and whichever pool frees a GPU first does the work. A worker that starts late finds less to do, and one that finds nothing exits at once (the last one cancels its siblings that are still pending, so the finish array is not held up by a pool that stays busy). And the same command given again submits only what is missing, which is how a batch recovers from a preempted worker, a faulty card (`--exclude-nodes`) or a failed download.

`$CSDV_GPU_POOLS` is set in `scripts/chpc/env.sh`: one string of sbatch options per pool, with semicolons between pools (see `env.example.sh`). Allocation names stay in that file. Every pool has to be a partition of the cluster whose login node the queue is submitted from. On CHPC, kingspeak, lonepeak and granite are Slurm clusters of their own, and a job cannot wait for a job of another cluster, which the queue's chain relies on. `csdv grid queue` asks Slurm about each pool first (`sbatch --test-only`) and submits nothing when a name is not valid. A worker asks for 24 GB and four hours by default (`--worker-mem`, `--worker-hours`), which suits boxes. A whole quad needs `--worker-mem 48G`. A worker stops taking unit-years a quarter of an hour before its walltime.

A unit-year whose inference fails is noted in `<queue>/failed/` and tried once more by another worker. After two failures it is left out, and the unit is finished without that year. A claim that a killed job left behind goes stale after 45 minutes and is taken over.

The finish task refuses a unit, with a message that says why, when its download has no marker, when a CHM is still to be made (give the queue command again), when fewer than `--min-years` are left, or when the unit's interval in `--priority-intervals` (a CSV with the columns `unit` and `interval`, such as `2018-2021`) is no longer a pair of consecutive years. The last check keeps a place that was chosen for one event from being processed without the two images around it. The unit record takes the years that were processed. The export also copies the four cell layers and the model map to `$CSDV_EXPORT_DIR/run_layers/<unit>/<tag>/`, where `csdv label` reads them, so one pull of the exports is enough. The inspection chips of `csdv grid submit` are not made.

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
