# Architecture

The library holds two pipelines that share acquisition (NAIP download, mosaics, the NAIP-CHM height model). The disturbance pipeline maps canopy loss between NAIP dates and is the active work. The recovery pipeline computes stand metrics and classifies stages and trajectories, and is a paused proof of concept. Nothing in the recovery pipeline reads the disturbance maps yet.

## Disturbance pipeline

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ grid: one unit = a 7.5 minute quad or a box around a point           │
 │ download ──▶ mosaic ──▶ chm_inference (GPU) ──▶ disturbance ──▶ export│
 │ submit (one unit) · queue (many units, GPU pools) · blocks (regions) │
 └──────────────────────────────────────────────────────────────────────┘
        │ candidates: objects of the Disturbance Index, weaker
        │ detections, height loss, alarms of an earlier pair model
        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ label: candidates ──▶ sample ──▶ render ──▶ verdict (two vision      │
 │ models, cost ledger and cap) ──▶ routing, outlines ──▶ rules ──▶     │
 │ label rasters with a reason per pixel ──▶ chips                      │
 └──────────────────────────────────────────────────────────────────────┘
        │ chips: four-band NAIP per date, change and agent labels
        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ pairmodel: merge ──▶ train (units held out) ──▶ score ──▶ predict    │
 │ loss probability and agent per 9.6 m cell and pair of dates          │
 └──────────────────────────────────────────────────────────────────────┘
        │ maps per unit                         ▲ predict runs inside the
        ▼                                       │ grid chain of a unit
 ┌──────────────────────────────────────────────────────────────────────┐
 │ people: audit (blind reading of labels) · review (sample of          │
 │ detections) · view (maps over NAIP, notes) · grid webmap and label   │
 │ webmap (static pages for sharing)                                    │
 └──────────────────────────────────────────────────────────────────────┘
```

The pair model reads four-band NAIP only. The height model and the Disturbance Index propose and constrain the labels and are never inputs of the model. At mapping time the height model still supplies the forest mask.

Guides: [grid processing](guides/grid_processing.md), [disturbance detection](guides/disturbance_detection.md), [disturbance labels](guides/disturbance_labels.md), [label audit](guides/label_audit.md), [pair model](guides/pair_model.md), [map viewer](guides/map_viewer.md), [disturbance review](guides/disturbance_review.md).

## Recovery pipeline

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
| `download/` | NAIP (GEE, Planetary Computer, USDA image service), NAIP-CHM weights and conditioning rasters. National NAIP-CHM tiles over HTTP moved to naipcrowns (`naipcrowns.national.download`). NEON and 3DEP/SSURGO not yet written |
| `preprocess/` | NAIP mosaic and reprojection, window tiling. CHM smoothing and masking moved to `naipcrowns.chm` |
| `chm_inference/` | NAIP-CHM model wrapper (CHPC + Colab parity) + the 5 static CONUS conditioning rasters |
| external: `naipcrowns` | crown segmentation, crown allometry and the national NAIP-CHM tooling, in the sibling repository [naip-crowns](https://github.com/bullocke/naip-crowns) since October 2026. CSDV installs it with `pip install --no-deps --no-build-isolation -e ../naip-crowns`, exposes `naipcrowns segment` as `csdv segment-crowns`, and reads the crowns it writes in `zonal/crowns.py`. See the [pointer page](guides/crown_segmentation/README.md). DeepForest is not implemented |
| `metrics/` | fixed-window raster path. 15 registered metrics, 4 wired in the orchestrator. Keyed to `(array, GridSpec, window_m)` |
| `zonal/` | default path. Per-stand, per-date metrics inside a polygon. Pure `(array, mask) -> float`, no `window_m` |
| `satellite/` | Landsat C2 L2 per-stand series from Earth Engine plus 3 annual metrics. Sensor, index and annual-metric registries. The only networked analysis module |
| `stratification/` | topo + soils → site-type assignment |
| `stages/` | rule engine reading `stages.yaml`. `stand.py` per stand, `classify.py` per pixel |
| `trajectories/` | multi-date sequence assembly + 19-class rule evaluator |
| `examples/` | stand screening for worked examples. Manifest tools and vocab not yet written |
| `change/` | Disturbance Index algorithms shared by the research scripts and `disturbance/`: standardization (`di.py`), empirical null, lfdr and spatial probabilities (`di_decide.py`), 0.6 m outline refinement (`refine.py`) |
| `disturbance/` | `csdv detect-disturbance` pipeline. NAIP-CHM years to cell layers, DI z stacks, decision rule, onset and MMU, objects with 0.6 m outlines, figures and a Markdown and JSON report. Defaults reproduce the best ElkinsvilleNE method. See [disturbance_api.md](disturbance_api.md) |
| `masks/` | forest masks from canopy height and NDVI |
| `grid/` | the CONUS processing grid. Cells and units (`cells.py`, `units.py`), the Slurm chain of one unit (`submit.py`), a work queue that serves many units from several GPU pools (`queue.py`), blocks of adjoining quads inside one state (`blocks.py`), progress read from disk (`status.py`), the small download set of a run (`export.py`), and a static map page for a block (`blockmap.py`). See [guides/grid_processing.md](guides/grid_processing.md) |
| `label/` | training labels without a photo interpreter. Candidates from the disturbance runs (`candidates.py`), a sample of windows by strata or by class (`sample.py`, `sample_classes.py`), request images (`render.py`, `requests.py`), calls to two vision models with a ledger and a spending cap (`verdicts.py`, `clients.py`, `ledger.py`), agreement and outlines (`routing.py`, `outlines.py`), label rasters with a reason per pixel (`rules.py`, `labels.py`) and chips for training (`chips.py`). See [guides/disturbance_labels.md](guides/disturbance_labels.md) |
| `pairmodel/` | Siamese ResNet-34 for canopy loss between two NAIP dates, with a second head for the agent. Training with held-out units (`train.py`, `data.py`), chip folders of several label runs merged (`merge.py`), scores on units the model did not train on (`score.py`), maps of a unit (`predict.py`). Needs PyTorch. See [guides/pair_model.md](guides/pair_model.md) |
| `audit/` | blind audit of a label run by human interpreters: seeded draw, static page without model answers, a small server that stores answers, scores. See [guides/label_audit.md](guides/label_audit.md) |
| `review/` | Solara app for interpreting a sample of detected disturbances from exported grid runs. See [guides/disturbance_review.md](guides/disturbance_review.md) |
| `view/` | browser viewer for the pair model's maps over NAIP, with derived date, type and probability layers and comment points. See [guides/map_viewer.md](guides/map_viewer.md) |
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
| `masks`, `grid`, `label`, `pairmodel`, `audit`, `review`, `view` | implemented and in use for the regional maps. Not built yet: patches with severity from the pair model's maps, seams between quads, quads on state lines, and the hand-over of patches to `zonal` and `trajectories` |
| `download` | 4 of 6 modules. `neon.py` and `topo_soils.py` are stubs |
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
csdv download {naip,naip-pc,naip-rest,chm,conditioning}
csdv segment-crowns               # naipcrowns segment (needs ../naip-crowns)
csdv chm-inference
csdv stratify
csdv compute-metrics              # raster path
csdv classify-stages
csdv classify-trajectories
csdv satellite {fetch,annual,join,list}
csdv detect-disturbance           # NAIP-CHM disturbance map for one site
csdv disturbance-figures          # rebuild a run's figures and report
csdv disturbance-chips            # sample a run and draw NAIP RGB and CHM chips
csdv grid {locate,submit,queue,queue-status,status,sync-unit,export,check,compare-units,webmap}
csdv label {candidates,sample,allocate,units,render,compare-renders,verdict,build,sheets,webmap}
csdv pair {merge-chips,train,score,predict,compare,crops}
csdv audit {draw,build,app,bundle,model-only,score}
csdv review {sample,extend,app,summary}
csdv view                         # pair-model maps over NAIP in a browser
```

The cluster wrappers in `scripts/chpc/` chain these commands: `round2.sh` (a label round from grid batch to model tests), `region.sh` (a block of quads: map, remap with another model, page, status) and the `*_pull.sh` scripts that copy the small result sets to a laptop. They read their settings from `scripts/chpc/env.sh`.

`detect-disturbance` takes `--site` and writes one run folder (default `$CSDV_RESULTS_ROOT/disturbance/<site>/`, never overwritten). Parameters come from `config/disturbance.yaml`, an optional `--config` YAML and explicit options, and `--from-run` reuses a saved run's layers and z. The guide is [guides/disturbance_detection.md](guides/disturbance_detection.md).
