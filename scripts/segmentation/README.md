# Crown segmentation optimization

## What this is

The scripts that tested and re-tuned crown segmentation. The findings and the chosen parameters are written up in [`docs/guides/segmentation_optimization/`](../../docs/guides/segmentation_optimization/README.md). This file covers how to run them and what each one is for.

Everything writes to `docs/guides/segmentation_optimization/results/`, one tidy table per step, so every number in the report traces back to a row.

## Order to run

Steps depend on earlier output. `tiles.py` first, `sweep.py` second, everything else after that.

```bash
P=.micromamba/envs/CSDV/bin/python

$P scripts/segmentation/tiles.py                     # writes tiles.geojson
$P scripts/segmentation/sweep.py --stage both        # the parameter grid, ~30 min
$P scripts/segmentation/nulls.py                     # geometry and surface nulls
$P scripts/segmentation/sensitivity.py               # synthetic crowns at known CV
$P scripts/segmentation/stability.py --from-sweep    # six-year series, resolution
$P scripts/segmentation/support.py --from-sweep      # bootstrap for MIN_CROWNS
$P scripts/segmentation/discrimination.py --from-sweep
$P scripts/segmentation/compare_r.py                 # lidR parity, needs Rscript
$P scripts/segmentation/figures.py                   # every report figure
$P scripts/segmentation/apply.py --dry-run           # check, then drop --dry-run
```

`--from-sweep` reads the winning parameter set out of `sweep_tune_scored.parquet`. Without it each script falls back to its own flags, which is useful for exploring but should not produce a number that lands in the report.

| Script | Question it answers |
|---|---|
| `_lib.py` | Shared paths, tile reading, published allometry, and the decision rule |
| `tiles.py` | Which ground the sweep is measured on. Writes `tiles.geojson` |
| `sweep.py` | Which parameter set wins under the pre-registered rule |
| `nulls.py` | Is crown diameter CV just tessellation geometry |
| `sensitivity.py` | Can crown_cv recover a crown size distribution it is handed |
| `stability.py` | Do crown metrics hold still across the 1.0 m to 0.6 m resolution break |
| `support.py` | How many crowns a stand needs before crown_cv means anything |
| `discrimination.py` | Do crown metrics separate disturbed stands from undisturbed ones |
| `compare_r.py` | Does the Python engine agree with lidR on matched parameters |
| `figures.py` | Every figure in the report |
| `apply.py` | Re-segment the six Elkinsville years and rebuild stand metrics |

## The decision rule

Written in `_lib.py` before any sweep output was read, so the parameter choice is a rule applied to numbers rather than a number chosen to match a preference.

Hard filters run first. A set has to clear all of them:

| Filter | Band | Why |
|---|---|---|
| `density_per_ha` | 75 to 200 | Published dominant and codominant stem density for eastern hardwood |
| `capped_fraction` | below 0.10 | A set that needs the radius ceiling to look right is being tuned by the ceiling |
| `assigned_fraction` | 0.60 to 0.95 | Below 1.0 means crown extent is set by the canopy rather than by seed spacing |
| `allo_slope` | above 0.05 | Taller trees must get wider crowns, or nothing was learned from height |
| `over_open_grown` | below 0.05 | A stand-grown crown cannot exceed an open-grown crown on the same tree |

Survivors rank on one scalar, `allo_rmse`, lowest first, computed on the tuning tiles only.

Held-out and transfer tiles are read once, after the parameters are fixed. If they disagree with the tuning result, the report says so rather than re-tuning.

## Tiles

`tiles.geojson` holds 18 Elkinsville tiles and 8 transfer tiles, each 300 m square, which is 9 ha. Selection is stratified rather than random, because a random sample of a mostly closed forest returns mostly closed forest and the parameter sets differ most where the canopy is broken.

Strata come from the interpreter labels where a labelled stand covers at least 15 percent of a tile, and from canopy statistics otherwise. Tiles are picked deterministically by how typical they are within their stratum, so the manifest rebuilds identically from the code with no random seed.

Two things about the file are easy to trip over.

The geometry column is WGS84, because one GeoJSON has to hold tiles from two projections. Elkinsville is EPSG:26916 and the NEON sites are EPSG:5070. The authoritative extent is the `minx`, `miny`, `maxx`, `maxy` and `epsg` columns, in each tile's own projection. Read those, not the geometry.

Tile statistics use only crowns lying wholly inside a 15 m inset. An edge-truncated crown is small and irregular, so edge crowns deflate mean diameter and inflate CV, and the size of that bias depends on crown size, which is exactly what the sweep varies.

## Data this depends on

| Item | Where |
|---|---|
| Elkinsville CHM | `data/naip_chm/ElkinsvilleNE/<year>/`, float32 metres, nodata -9999, 0.6 m, EPSG:26916 |
| Elkinsville NAIP | `data/naip/ElkinsvilleNE/<year>/`, 4-band, 0.6 m |
| Stand polygons | `data/calibration/Indiana-ElkinsvilleNE_revised.gdb/Indiana-ElkinsvilleNE_revised.gdb`. The name **is** doubled |
| Transfer sites | `legacy/proof_of_concept/Data/NEON/CHM/`, SCBI and HARV, NEON airborne lidar, 1 m, EPSG:5070 |
| NAIP years | 2012, 2014, 2016, 2018, 2020, 2022. 2012 and 2014 are 1.0 m native, resampled to the common 0.6 m grid |

`legacy/` is gitignored, so the two transfer sites are present on this machine and will be missing on a fresh clone. Scripts warn and carry on without them rather than failing.

## R

`compare_r.py` needs `Rscript` with lidR, terra and sf. All three are installed in `.micromamba/envs/CSDV` but **none are declared in `environment.yml`**, so a fresh environment will not have them. The comparison records the versions it ran against in `compare_r.csv`.

R is the reference implementation, not the production path. Production segmentation runs in Python so the pipeline carries no R dependency.

## Parameters

Every segmentation parameter is a field on `csdv_core.segmentation.params.SegmentationParams`, and every length is in metres. That is what lets one parameter set transfer between the 0.6 m NAIP-CHM and the 1 m lidar CHMs unchanged. A pixel-valued parameter changes physical meaning with resolution, which is the failure this design exists to prevent. lidR's own `max_cr` default is the cautionary example: 10 pixels is a 10 m crown radius on a 1 m CHM and 6 m on a 0.6 m one.

`SegmentationParams.key` is a short stable hash of the whole set. Crown artefacts are written under that key with a sidecar JSON, so a re-run after a parameter change cannot silently reuse the previous run's crowns. The previous cache keyed on filename alone and had exactly that failure mode.

## Writing style

Same as the rest of the project, from [`docs/conventions.md`](../../docs/conventions.md):

- Active voice. Short sentences. Short paragraphs.
- Never use em dashes. Never use semicolons to join two clauses.
- Banned: "straightforward", "leverage" as a verb, "utilize", "facilitate", "comprehensive", "robust" outside a statistical sense, "cutting-edge", "novel", "delve", "harness", "embark", "it's worth noting", "it's important to note".
- Also avoid "landscape", "realm", "tapestry", "pivotal", contrast setups such as "not just X, but Y", and faux-insider transitions such as "Here's why that matters".
- Say what the number is, then say what it does not tell you. The second part is usually the more useful sentence.
