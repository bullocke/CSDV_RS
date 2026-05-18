# NAIP-CHM PoC Inference Workflow

Two pathways for running the pre-trained NAIP-CHM model on PoC sites. The Colab path gets a first result fastest. The CHPC path is the production path for the full site/year matrix. Both call the same upstream CLI (`scripts/inference.py`) on AOI-clipped NAIP GeoTIFFs.

## Output naming convention

NAIP AOI input: `NAIP_{SITE}_{YYYYMMDD}_aoi{km}km.tif`
CHM prediction output: `NAIP_{SITE}_{YYYYMMDD}_aoi{km}km_chm.tif`

`YYYYMMDD` is the median acquisition date of contributing NAIP images and is required by the upstream DOY extractor. `{km}` is the AOI half-edge in kilometers (default 5).

Predictions land in `ProofOfConcept/Data/NAIP/Predicted_CHM/{SITE}/`.

## Path A: Google Colab T4 (quick-start)

For SCBI plus two more NAIP years to validate the pipeline end-to-end before scaling.

1. Open `colab/naip_chm_aoi_inference.ipynb` in Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Edit the config cell: site code, lat/lon, AOI half-size (km), GEE project ID, Drive folder.
4. Run the cells top to bottom. The notebook:
   - Clones `smorf-ntsg/naip-chm`.
   - Installs requirements.
   - Downloads the 5 static conditioning rasters from the UMT server (one-time, ~few GB; cached in Drive on subsequent runs).
   - Authenticates GEE and mounts Drive.
   - Lists available 4-band NAIP years for the AOI.
   - For each requested year: clips NAIP to the AOI, exports a 4-band GeoTIFF named `NAIP_{SITE}_{YYYYMMDD}_aoi{km}km.tif`, then shells out to `scripts/inference.py`.
   - Plots a quick-look montage of all predictions.
5. Download outputs from Drive to `ProofOfConcept/Data/NAIP/Predicted_CHM/{SITE}/` on the laptop.

## Path B: Utah CHPC GPU (production)

For the full site/year matrix once the Colab smoke test passes.

1. **One-time setup on CHPC:**
   ```
   bash setup/clone_and_install.sh        # clones naip-chm to ~/code/naip-chm and creates micromamba env
   bash setup/download_conditioning.sh    # pulls the 5 conditioning rasters to $NAIPCHM_COND_DIR
   ```
2. **Per-run, on the login node (or laptop with GEE auth cached):**
   ```
   python src_local/list_naip_years.py --site SCBI --half-size-km 5
   python src_local/build_job_list.py \
       --sites SCBI HARV TALL \
       --half-size-km 5 \
       --out-dir $SCRATCH/naipchm \
       --jobs-csv jobs.csv
   python src_local/prepare_naip_aoi.py --jobs-csv jobs.csv   # downloads NAIP AOI clips
   ```
3. **Submit SLURM array:**
   ```
   sbatch --array=1-$(($(wc -l < jobs.csv) - 1)) slurm/run_inference.slurm jobs.csv
   ```
4. **Collect outputs:** `rsync` the predicted CHMs from `$SCRATCH/naipchm` back to `ProofOfConcept/Data/NAIP/Predicted_CHM/{SITE}/`.

## Dependencies

- Python 3.11
- PyTorch 2.8.0 with CUDA matching the target node (CHPC supplies CUDA modules; on Colab the T4 driver is fixed)
- `rasterio`, `rio-cogeo`, `pyarrow`, `pyyaml`, `tqdm` (from upstream `requirements.txt`)
- `earthengine-api`, `geemap` (only for the AOI clipping step)
- `click` (CLI helpers in `src_local/`)

## Verification checklist

Run these in order. Stop and debug if any fail.

1. **Conditioning data present:** all five rasters exist in `$NAIPCHM_COND_DIR` (or `data/conditioning_data/` on Colab).
2. **Dry-run on one AOI:**
   ```
   python scripts/inference.py --naip-quad <one_aoi.tif> --output-dir /tmp/dryrun \
     --model-checkpoint model/model_20251016.pt --config configs/config.yaml \
     --static-rasters-dir $NAIPCHM_COND_DIR --dry-run
   ```
   Confirm width, height, CRS, and DOY print correctly.
3. **SCBI 2023 regeneration vs. published product:** run inference on a SCBI 2023 AOI and difference against the published CONUS tile under `ProofOfConcept/Data/NAIP/National_CHM/`. Difference should be small (model is deterministic in eval mode; some I/O resampling differences are expected near edges).
4. **Metric stability check:** compute gap fraction at 25/50/100 m on the predicted SCBI time series using `poc_lib.metrics.gap_fraction`. In undisturbed pixels the distribution should be stable across NAIP cycles within ~5-10 % at 50 m.
