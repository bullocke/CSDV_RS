# Architecture

## Pipeline

Acquisition feeds two analysis paths. The stand path is the default. The fixed-window raster path is secondary and predates it.

```
              ┌──────────────────────────────────────────────────────┐
              │              csdv_core.config (YAML)                  │
              │  sites · site_types · metrics · satellite · stages ·  │
              │                   trajectories                        │
              └──────────────────────────────────────────────────────┘
                                     │ (typed loader)
                                     ▼
 ┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
 │ download │──▶│ preprocess │──▶│chm_inference │──▶│ segmentation │
 └──────────┘   └────────────┘   └──────────────┘   └──────────────┘
       │                                │                  │
       │  NAIP (RGBN, 0.6 m)            │  CHM             │  crowns
       ▼                                ▼                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                       DEFAULT: stand path                          │
 │  io.stands ──▶ zonal (per stand, per date, no window_m)            │
 │    pixel · crowns · texture · spatial · deltas · record            │
 │                              │                                     │
 │                              ▼  stand_metrics.parquet              │
 └───────────────────────────────────────────────────────────────────┘
                                │
                                │  join on (stand_id, year)
                                │  io.satellite_io.join_satellite_metrics
                                ▲
 ┌───────────────────────────────────────────────────────────────────┐
 │  satellite (Earth Engine)                                          │
 │  sensors ──▶ indices ──▶ extract ──▶ annual                        │
 │  Landsat C2 L2  ·  per-scene rows  ·  satellite_annual.parquet     │
 └───────────────────────────────────────────────────────────────────┘

 ┌───────────────────────────────────────────────────────────────────┐
 │  SECONDARY: fixed-window raster path                               │
 │  metrics (array + GridSpec + window_m)                             │
 │    gap · crown · texture · cover · spatial · deltas · registry     │
 │                              │  <metric>.tif per window            │
 └───────────────────────────────────────────────────────────────────┘
                                │
              both paths ───────┴───────┐
                                        ▼
                             ┌─────────────────────┐
                             │   stratification    │  (site type)
                             └─────────────────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │       stages        │  (per-date assignment)
                             └─────────────────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │    trajectories     │  (multi-date class)
                             └─────────────────────┘

 csdv_core.io          cross-cutting (raster, vector, stands, paths, parquet)
 csdv_core.viz         maps, panels, scatter, shared style
 csdv_core.validation  comparison vs. lidar/field
 csdv_core.examples    stand screening + controlled vocab
 csdv_core.check       `csdv check` preflight
```

`stages` and `trajectories` each expose two entry points. `stand.py` reads a stand record, `classify.py` reads a raster stack.

See [metrics.md](metrics.md) for every metric both paths produce.

## Package map

| Path | Role |
|---|---|
| `config/` | YAML: sites, site types, metric defaults, satellite settings, stage envelopes, trajectory rules. Pydantic models, `extra="forbid"` |
| `io/` | rasterio/geopandas wrappers, AIS stand polygon reader, env-driven path resolver, metric and satellite parquet contracts |
| `download/` | NAIP (GEE and Planetary Computer), NAIP-CHM weights and conditioning rasters. NEON and 3DEP/SSURGO not yet written |
| `preprocess/` | NAIP mosaic and reprojection, CHM smoothing and masking, window tiling |
| `chm_inference/` | NAIP-CHM model wrapper (CHPC + Colab parity) + the 5 static CONUS conditioning rasters |
| `segmentation/` | crown segmentation: watershed (Python), `lidR` (rpy2). DeepForest not yet written |
| `metrics/` | fixed-window raster path. 15 registered metrics, 4 wired in the orchestrator. Keyed to `(array, GridSpec, window_m)` |
| `zonal/` | default path. Per-stand, per-date metrics inside a polygon. Pure `(array, mask) -> float`, no `window_m` |
| `satellite/` | Landsat C2 L2 per-stand series from Earth Engine plus 3 annual metrics. Sensor, index and annual-metric registries. The only networked analysis module |
| `stratification/` | topo + soils → site-type assignment |
| `stages/` | rule engine reading `stages.yaml`. `stand.py` per stand, `classify.py` per pixel |
| `trajectories/` | multi-date sequence assembly + 19-class rule evaluator |
| `examples/` | stand screening for worked examples. Manifest tools and vocab not yet written |
| `change/` | Disturbance Index algorithms shared by the research scripts and `disturbance/`: standardization (`di.py`), empirical null, lfdr and spatial probabilities (`di_decide.py`), 0.6 m outline refinement (`refine.py`) |
| `disturbance/` | `csdv detect-disturbance` pipeline. NAIP-CHM years to cell layers, DI z stacks, decision rule, onset and MMU, objects with 0.6 m outlines, figures and a Markdown and JSON report. Defaults reproduce the best ElkinsvilleNE method. See [disturbance_api.md](disturbance_api.md) |
| `viz/` | maps, metric trajectory panels, height histograms, scatter, shared style |
| `validation/` | scatter, R², RMSE, Kruskal-Wallis, Dunn's |
| `check.py` | `csdv check` preflight: environment, paths, configs, per-site pipeline state |
| `cli.py` | `csdv` console script. Heavy imports stay inside command bodies |

## Analysis folders

`analysis/` holds self-contained research analyses that sit outside the pipeline. Each analysis is a numbered folder (for example `analysis/NAIP_disturbance_detection/1_Data_Exploration/`) with a README, a private `_lib.py`, and numbered scripts that import `csdv_core`. Derived data (`data/`), figures, maps and tables stay in the folder and are not committed. Logic that proves reusable moves into `src/csdv_core`.

## Interactive viewer (`src/rsviewer/`)

`rsviewer` is a general map viewer for the VS Code Interactive Window, built on leafmap (ipyleaflet), localtileserver and Plotly. It shows rasters and vectors over a satellite basemap, swaps layers from dropdowns and passes map clicks to Python for plotting. It does not import `csdv_core`, so it can move to its own repository (see `src/rsviewer/README.md`). Project viewers live next to the analysis they show. The first is `analysis/NAIP_disturbance_detection/2_Disturbance_Index/viewer.py` (guide in `VIEWER.md` there).

## Implementation status

| Area | State |
|---|---|
| `config`, `io`, `preprocess`, `chm_inference`, `metrics`, `zonal`, `satellite`, `stratification`, `stages`, `trajectories`, `change`, `disturbance`, `viz` | implemented |
| `download` | 4 of 6 modules. `neon.py` and `topo_soils.py` are stubs |
| `segmentation` | 3 of 4 modules. `deepforest.py` is a stub |
| `examples` | 1 of 3 modules. `screen.py` works, `index.py` and `manifest.py` are stubs, `vocab/` is empty |
| `validation` | not started. `compare.py` and `stats.py` are stubs |

Every stub carries a TODO header pointing at the legacy file to migrate from.

### Partial implementations

Three areas are implemented but do not yet cover their intended scope.

- `metrics/orchestrator.py` wires only `PASS1_METRICS`, which is 4 of the 15 registered metrics. The rest raise `NotImplementedError` so a caller gets a clear message instead of a silent skip. The raster path can therefore supply 3 of the 7 metrics the stage envelopes need. The full set is reachable only through `zonal/` plus `csdv satellite join`.
- `stages/classify.py` runs a per-pixel Python loop over rows and columns. It is correct but not vectorised, so it does not scale to a full DOQQ.
- Every continuous threshold in `config/site_types.yaml` is `null`, so no stand reaches a real site type and everything falls to `type_00`. `stages/stand.py` names that fallback `UNSTRATIFIED_SITE_TYPE`.

[metrics.md](metrics.md) lists the rest of the known gaps, including two trajectory rules that reference metrics with no implementation.

## CLI surface

```
csdv info
csdv check
csdv download {naip,chm,conditioning}
csdv segment-crowns
csdv chm-inference
csdv stratify
csdv compute-metrics              # raster path
csdv classify-stages
csdv classify-trajectories
csdv satellite {fetch,annual,join,list}
csdv detect-disturbance           # NAIP-CHM disturbance map for one site
csdv disturbance-figures          # rebuild a run's figures and report
csdv disturbance-chips            # sample a run and draw NAIP RGB and CHM chips
```

`detect-disturbance` takes `--site` and writes one run folder (default `$CSDV_RESULTS_ROOT/disturbance/<site>/`, never overwritten). Parameters come from `config/disturbance.yaml`, an optional `--config` YAML and explicit options, and `--from-run` reuses a saved run's layers and z. The guide is [guides/disturbance_detection.md](guides/disturbance_detection.md).
