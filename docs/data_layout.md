# Data layout

The library code reads and writes data under three roots resolved at runtime by `csdv_core.io.paths.project_paths()`. Each root is configurable via an environment variable so the same code runs locally, on CHPC, and on Colab.

| Variable | Default (local) | Purpose |
|---|---|---|
| `CSDV_DATA_ROOT` | `<repo>/data` | Inputs: NAIP, NEON, topo/soils, CHM model weights |
| `CSDV_RESULTS_ROOT` | `<repo>/results` | Outputs: derived rasters, tables, figures |
| `CSDV_CACHE_ROOT` | `<repo>/.cache` | Intermediate caches (downloads, tile mosaics) |

## Expected `data/` layout

```
data/
├── naip/<site_code>/<year>/         # NAIP DOQQs (RGBN, 0.6 m)
├── naip_chm/<site_code>/<year>/     # NAIP-CHM rasters
├── neon/<site_code>/
│   ├── als_chm/<year>/              # NEON ALS CHM
│   ├── lidar/<year>/                # NEON point clouds
│   └── field/<year>/                # Vegetation structure tables
├── satellite/<site_code>/           # Per-stand Landsat observations (parquet)
├── topo/<site_code>/                # DEM-derived (slope, aspect, TWI, ...)
├── soils/<site_code>/               # SSURGO joins
└── chm_model/
    └── conditioning/                # 5 static CONUS-wide NAIP-CHM rasters (EPSG:5070)
```

The conditioning rasters are fetched once with `csdv download conditioning`. Their location can be relocated outside this layout (e.g. a shared, read-only copy) by setting `CSDV_CONDITIONING_DIR`.

`data/satellite/<site_code>/` holds `landsat_observations.parquet`, one row per stand per scene from 1985 onward, and a `landsat_observations.manifest.json` sidecar recording the sensors, the date range, the pixel rule and any chunk that failed. It is written by `csdv satellite fetch` and is a cache: it can be deleted and refetched, but refetching costs an Earth Engine session and about twenty minutes for a forty-stand module.

## Expected `results/` layout

```
results/
├── metrics/<site_code>/<year>/<window>m/
├── stages/<site_code>/<year>/
├── stands/<site_code>/              # Per-stand tables, incl. satellite_annual.parquet
├── trajectories/<site_code>/
└── figures/
```

`results/stands/<site_code>/satellite_annual.parquet` is the derived product: one row per stand per year, carrying the annual metrics, the support columns behind each one and a `satellite_unavailable` string naming the gate that voided anything missing. `csdv satellite join` merges its NAIP-year rows into the stand metrics so the stage classifier can read them.

Both `data/` and `results/` are gitignored. On CHPC, symlink them to scratch or project storage. On Colab, point the env vars at a mounted Drive folder.
