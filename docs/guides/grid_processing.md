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
```

For each unit the command submits up to five jobs, each waiting for the one before it.

| Step | Script | Resources | Skipped when |
|---|---|---|---|
| `download` | `run_download_dtn.sbatch naip-auto` | DTN, 2 CPUs, 4 h | every planned year has a NAIP mosaic |
| `chm` | `run_chm_inference.sbatch`, one array task per year | 1 GPU, 48 GB, 2 h per year | every planned year has a CHM |
| `detect` | `run_detect_disturbance.sbatch` | 4 CPUs, 24 GB, 3 h | never (a new run tag is required) |
| `chips` | `run_disturbance_chips.sbatch` | 2 CPUs, 8 GB, 1 h | `--no-chips` |
| `export` | `run_grid_export.sbatch` | 2 CPUs, 8 GB, 1 h | never |

If a job fails, Slurm cancels the jobs that wait on it. The export waits for detection to succeed and for the chips to end, so a chips failure does not hold back the export. `--granite` sends the CHM tasks to the preemptable granite GPUs. `--steps` limits the chain, for example `--steps download,chm` to prepare data only.

Each submission appends a row to the ledger, `$CSDV_GRID_DIR/submissions.csv`, with the time, unit, run tag, years, job IDs, the detection options and the git commit.

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

## Cost

Each cell needs one GPU task per NAIP year (six to eight years for most of CONUS since 2010), plus about ten minutes of CPU time for detection, chips and export. The pilot batch of 20 cells is about 140 GPU tasks. Submit one cell first and check its run time and peak memory with `sacct -j <job ids> -X --format=JobID,JobName,State,Elapsed,MaxRSS` before submitting the batch.
