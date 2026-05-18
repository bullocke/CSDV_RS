# NAIP-CHM Repository Review

Review of `smorf-ntsg/naip-chm` (commit `2875b7e`, Apr 2026) covering inference, conditioning data, and integration constraints for the CSDV proof-of-concept.

Reference: Morford et al. 2025, *A 0.6-meter resolution canopy height and structure model for the contiguous United States*, bioRxiv 2025.12.12.694075. Repo: https://github.com/smorf-ntsg/naip-chm.

## Repository layout

```
naip-chm/
├── configs/config.yaml                   # model + training config
├── model/model_20251016.pt               # trained weights (~88 MB)
├── scripts/
│   ├── inference.py                      # CLI: one DOQQ -> one CHM GeoTIFF
│   ├── download_conditioning_data.py     # one-time pull of 5 static rasters
│   ├── create_metadata_parquet.py        # training-only
│   ├── train.py                          # training (not used here)
│   └── convert_checkpoint_to_weights.py  # checkpoint utility
├── src/
│   ├── inference_utils.py                # process_naip_quad, StaticRasterHandler, load_model
│   ├── inference_streaming.py            # GEEInferenceStreamer (Colab GEE path)
│   ├── model.py, dataset.py              # UNetFiLM architecture, normalization constants
├── notebooks/gee_inference_colab.ipynb   # full-DOQQ Colab template
└── requirements.txt                      # torch==2.8.0, rasterio, rio-cogeo, pyarrow, ...
```

## Model

U-Net with CBAM attention and multi-stage FiLM conditioning, ~22 M parameters. Image encoder takes 4 channels (RGBN), 432x432 chips. Auxiliary head takes a 15-D continuous vector (6 climate PCAs + 6 soil PCAs + elevation + sin/cos DOY), one NLCD class embedding (9 classes), and one ecoregion embedding (86 classes). Output is one channel of canopy height, predicted in meters and post-processed to UInt16 centimeters clipped to 0-120 m.

## Inference inputs

The CLI `scripts/inference.py` takes:

- `--naip-quad PATH` — a 4-band (R,G,B,NIR) or 5-band (RGBN + mask) GeoTIFF. Resolution can be 0.6 m or 1.0 m. Files at 1.0 m are auto-resampled to 0.6 m via bilinear interpolation. The CRS can be any projected CRS; chip centers are reprojected to EPSG:5070 internally for static raster sampling. **The input does not have to be a full DOQQ** — an arbitrary AOI subset works as long as the filename carries a `YYYYMMDD` token.
- `--output-dir PATH` — destination for the COG plus a JSON report.
- `--model-checkpoint PATH` — `model/model_20251016.pt`.
- `--config PATH` — `configs/config.yaml`.
- `--static-rasters-dir PATH` — directory holding the 5 conditioning rasters listed below.
- `--chip-size INT` (default 432, must be a multiple of 16) and `--chip-overlap FLOAT` (default 0.2).
- `--dry-run` — validate inputs and print metadata without running the model.

## Static conditioning rasters

All five must be present in `--static-rasters-dir`. They are CONUS-wide, EPSG:5070, downloaded once from the UMT rangeland server by `scripts/download_conditioning_data.py`.

| File | Content | Use |
|---|---|---|
| `elevation.tif` | DEM (m), normalized to [0,1] by /4000 | Continuous feature |
| `climate_pca.tif` | 6-band climate PCA, normalized by per-band std | 6 continuous features |
| `soil_pca.tif` | 6-band soil PCA, normalized by per-band std | 6 continuous features |
| `nlcd.tif` | NLCD class (mapped through `NLCD_TO_IDX`, 9 classes) | Categorical embedding |
| `ecoregion.tif` | EPA ecoregion code (86 classes) | Categorical embedding |

Conditioning data is sampled point-wise at each chip center (one row, one column window per raster). Cost is negligible per chip.

## DOY filename constraint

`src/inference_utils.extract_doy_from_filename` regex matches `\d{8}` and parses it as `YYYYMMDD`. The first match wins. If the filename has no 8-digit date, inference fails. Implication for AOI clips: name them with the acquisition date, e.g. `NAIP_SCBI_20180815_aoi5km.tif`.

## Chip pipeline (`process_naip_quad`)

1. Open NAIP, read width/height/CRS/bounds, extract DOY from filename.
2. Resample to 0.6 m if needed.
3. Read RGBN; if a 5th band is present, use it as a validity mask.
4. Build a chip grid: 432 px tiles with 20% overlap (default). Edge chips are shifted inward to maintain full size.
5. For each chip:
   - Reproject chip center to EPSG:5070, sample the 5 static rasters, build the 15-D continuous vector + NLCD index + ecoregion index.
   - Build a distance-weighted blend mask (linear ramp over the overlap region).
   - Run the model on the 432x432x4 tile.
   - Accumulate weighted predictions into a full-image float32 buffer + a weight buffer.
6. Divide accumulator by weight buffer to average overlapping regions.
7. Apply the validity mask, multiply by 100 (m -> cm), clip to [0, 12000], cast to UInt16, set 65535 as NoData.
8. Write a temporary GeoTIFF then call `rio cogeo create` via subprocess to produce a final Cloud-Optimized GeoTIFF with overviews.

A JSON report with chip counts, output path, and elapsed time is written next to the GeoTIFF.

## Two compute paths

1. **Local CLI (`scripts/inference.py`).** Runs on a single GeoTIFF. Best for CHPC SLURM jobs and arbitrary AOIs. Works on CPU but is roughly 30-100x slower than a T4 GPU per chip.
2. **GEE streamer (`src/inference_streaming.GEEInferenceStreamer`).** Used by the Colab notebook. Pulls a full NAIP DOQQ from GEE by lat/lon/year, runs inference on T4, writes to Drive. Full-DOQQ only; cannot accept arbitrary AOIs.

For the PoC we use the **local CLI path on a GEE-clipped AOI** (5 km square per site) from both Colab and CHPC. This keeps one code path across compute environments and reduces processing cost by ~10x vs. a full DOQQ.

## Hardware notes

- Tested with `torch==2.8.0` and Python 3.11.
- Colab T4 (16 GB VRAM) processes a full DOQQ in ~2-6 minutes; a 5 km AOI in well under one minute.
- CHPC: any node with a recent NVIDIA GPU (e.g. notchpeak GPU partition). Memory footprint is modest (~4 GB GPU + ~8 GB host).

## NAIP availability

Model requires 4-band NAIP. The `USDA/NAIP/DOQQ` GEE collection is 4-band for most states from 2009-2010 onward. Some earlier acquisitions are RGB-only and must be filtered out (we filter by `bandNames().length() == 4`). NAIP cycle dates per AOI are listed via the helper in `notebooks/gee_inference_colab.ipynb`; the equivalent local helper is `src_local/list_naip_years.py`.

## Integration points with the existing PoC

- The output filename pattern aligns with `poc_lib.sites.SiteConfig`: extend `SiteConfig` with `predicted_chm_dir` and `predicted_chm_prefix` (default `NAIPCHM_{SITE}`) and existing metric code in `poc_lib.metrics` and `poc_lib.crowns` consumes the new rasters without modification.
- The published CONUS NAIP-CHM tile already downloaded under `ProofOfConcept/Data/NAIP/National_CHM/` lets us regression-test our locally regenerated 2023 prediction against the published product before scaling.

## Risks and known limitations

- Filename DOY parsing is fragile. Any 8-digit substring is treated as a date. Avoid names like `tile_20180001_...` (Jan 0 fails parse) or coordinates that produce spurious matches.
- Static rasters are EPSG:5070 only. If a chip center falls outside CONUS bounds (e.g. near coasts), feature extraction returns None and that chip is skipped, leaving a hole. Not a concern for the inland PoC sites.
- `rio cogeo` must be on PATH for the final COG step. Listed in `requirements.txt` as `rio-cogeo>=3.0.0`.
- Edge chips along the AOI border have larger blend ramps but no neighbors on the outside. Use the upstream `--edge-clip-meters` style trimming or accept a ~50 m edge buffer per AOI side.
