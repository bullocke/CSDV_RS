"""
03_compute_metrics.py — Compute gap fraction and GLCM texture entropy for SCBI.

Outputs
-------
Results/summary_document/intermediate/gap_fraction_25m_SCBI.tif
    Fraction of pixels < 2m height within each 25m non-overlapping window,
    derived from the NEON ALS CHM.

Results/summary_document/intermediate/glcm_entropy_NAIP_SCBI.tif
    GLCM entropy computed from the NAIP NIR band (band index 3) using a 7x7
    pixel sliding window. Computed on a strided grid (every 7 pixels) to keep
    runtime reasonable for 0.6m imagery.

The crown width CV raster is produced by 02_crown_segmentation.R and referenced
here only for path verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from skimage.feature import graycomatrix, graycoprops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
POC = PROJECT_ROOT / "ProofOfConcept"

NEON_CHM = POC / "Data" / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif"
NAIP_RGBN = POC / "Data" / "NAIP" / "Imagery"  # wildcard search below
INTERMEDIATE = POC / "Results" / "summary_document" / "intermediate"

OUT_GAP = INTERMEDIATE / "gap_fraction_25m_SCBI.tif"
OUT_ENTROPY = INTERMEDIATE / "glcm_entropy_NAIP_SCBI.tif"
OUT_CROWNS = INTERMEDIATE / "crown_polygons_SCBI.gpkg"
OUT_CROWN_CV = INTERMEDIATE / "crown_cv_50m_SCBI.tif"


@dataclass
class RasterInfo:
    """Lightweight container for raster metadata."""

    path: Path
    crs: str
    transform: Affine
    shape: tuple[int, int]
    nodata: float | None


def _find_naip(directory: Path) -> Path:
    """Return the first NAIP .tif found in the directory (wxee may add timestamp)."""
    tifs = sorted(directory.glob("NAIP_SCBI*.tif"))
    if not tifs:
        raise FileNotFoundError(
            f"No NAIP GeoTIFF found in {directory}. Run 01_download_data.py first."
        )
    return tifs[0]


def _read_band(path: Path, band_index: int = 1) -> tuple[np.ndarray, RasterInfo]:
    """Read a single band from a raster as float32."""
    with rasterio.open(path) as src:
        data = src.read(band_index).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        info = RasterInfo(
            path=path,
            crs=str(src.crs),
            transform=src.transform,
            shape=(src.height, src.width),
            nodata=nodata,
        )
    return data, info


# ---------------------------------------------------------------------------
# Gap fraction
# ---------------------------------------------------------------------------

def compute_gap_fraction(
    chm_path: Path,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
) -> tuple[np.ndarray, Affine, str]:
    """
    Compute gap fraction within non-overlapping spatial windows.

    Parameters
    ----------
    chm_path : Path
        Canopy height model GeoTIFF (height in meters, nodata as NaN after read).
    window_m : float
        Window size in meters (square).
    height_threshold_m : float
        Pixels below this height are counted as gaps.

    Returns
    -------
    gap_frac : np.ndarray
        2D array of gap fraction values (0–1), shape = (n_rows, n_cols) of windows.
    out_transform : Affine
        Affine transform for the output gap fraction raster.
    crs : str
        CRS string from the input raster.
    """
    chm, info = _read_band(chm_path)
    pixel_size = abs(info.transform.a)  # assumes square pixels
    window_px = max(1, int(round(window_m / pixel_size)))

    logger.info(
        "Gap fraction: CHM shape=%s, pixel_size=%.2fm, window=%dpx (%dm)",
        chm.shape, pixel_size, window_px, window_m,
    )

    rows = chm.shape[0] // window_px
    cols = chm.shape[1] // window_px

    gap_frac = np.full((rows, cols), np.nan, dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            tile = chm[
                r * window_px : (r + 1) * window_px,
                c * window_px : (c + 1) * window_px,
            ]
            valid = tile[~np.isnan(tile)]
            if valid.size == 0:
                continue
            gap_frac[r, c] = np.mean(valid < height_threshold_m)

    # Output transform: origin at top-left corner of input, window_m cell size
    ox, oy = info.transform.c, info.transform.f
    out_transform = Affine(window_m, 0.0, ox, 0.0, -window_m, oy)

    logger.info("Gap fraction grid: %d x %d windows", rows, cols)
    logger.info(
        "Gap fraction range: %.3f – %.3f",
        float(np.nanmin(gap_frac)), float(np.nanmax(gap_frac)),
    )
    return gap_frac, out_transform, info.crs


def save_raster(
    array: np.ndarray,
    path: Path,
    transform: Affine,
    crs: str,
    nodata: float = -9999.0,
) -> None:
    """Write a 2D float32 array to a single-band GeoTIFF."""
    arr = array.astype(np.float32)
    arr[np.isnan(arr)] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# GLCM texture entropy
# ---------------------------------------------------------------------------

def compute_glcm_entropy(
    naip_path: Path,
    nir_band_index: int = 4,
    window_px: int = 7,
    stride: int = 7,
    n_levels: int = 64,
) -> tuple[np.ndarray, Affine, str]:
    """
    Compute GLCM entropy from the NAIP NIR band on a strided grid.

    Entropy is averaged over four directions (0°, 45°, 90°, 135°) at distance=1.
    Computing entropy at every pixel for 0.6m imagery over a km-scale area is
    slow, so we sample every `stride` pixels and write the result at that grid
    spacing. This produces a coarser output but captures the spatial pattern.

    Parameters
    ----------
    naip_path : Path
        4-band NAIP GeoTIFF (R, G, B, N). NIR is band 4 (1-based).
    nir_band_index : int
        1-based band index for NIR (default 4 for RGBN NAIP).
    window_px : int
        GLCM window half-size. A 7x7 window is used.
    stride : int
        Pixel stride for the output grid. stride=7 means one output pixel per
        7x7 input pixels, matching the window size (no overlap).
    n_levels : int
        Number of grey levels for GLCM quantization (reduces to this from 256).

    Returns
    -------
    entropy_grid : np.ndarray
        2D array of GLCM entropy values.
    out_transform : Affine
        Affine transform for the output entropy raster.
    crs : str
    """
    nir, info = _read_band(naip_path, band_index=nir_band_index)
    logger.info("GLCM entropy: NIR shape=%s", nir.shape)

    # Normalize to [0, n_levels-1] uint8 for GLCM
    nir_valid = nir[~np.isnan(nir)]
    vmin, vmax = float(np.percentile(nir_valid, 2)), float(np.percentile(nir_valid, 98))
    nir_norm = np.clip((nir - vmin) / (vmax - vmin + 1e-8) * (n_levels - 1), 0, n_levels - 1)
    nir_uint = nir_norm.astype(np.uint8)

    half = window_px // 2
    rows_in, cols_in = nir.shape

    row_centers = np.arange(half, rows_in - half, stride)
    col_centers = np.arange(half, cols_in - half, stride)
    n_rows = len(row_centers)
    n_cols = len(col_centers)

    entropy_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    logger.info("Computing GLCM entropy on %d x %d grid (stride=%d px)...", n_rows, n_cols, stride)
    for ri, r in enumerate(row_centers):
        for ci, c in enumerate(col_centers):
            patch = nir_uint[r - half : r + half + 1, c - half : c + half + 1]
            if patch.shape != (window_px, window_px):
                continue
            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=angles,
                levels=n_levels,
                symmetric=True,
                normed=True,
            )
            # graycoprops returns shape (1, n_angles); average over angles
            ent = float(np.mean(graycoprops(glcm, "energy")))
            # Convert energy to entropy: H = -sum(p * log(p)) ≈ 1 - energy (approx)
            # Use actual entropy calculation instead
            p = glcm[:, :, 0, :]  # shape (levels, levels, n_angles)
            p = p + 1e-12  # avoid log(0)
            ent_vals = -np.sum(p * np.log2(p), axis=(0, 1))
            entropy_grid[ri, ci] = float(np.mean(ent_vals))

    pixel_size = abs(info.transform.a)
    out_pixel_size = stride * pixel_size
    ox, oy = info.transform.c + half * pixel_size, info.transform.f - half * pixel_size
    out_transform = Affine(out_pixel_size, 0.0, ox, 0.0, -out_pixel_size, oy)

    logger.info(
        "Entropy range: %.4f – %.4f",
        float(np.nanmin(entropy_grid)), float(np.nanmax(entropy_grid)),
    )
    return entropy_grid, out_transform, info.crs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    # -- Gap fraction -------------------------------------------------------
    logger.info("=== Computing Gap Fraction ===")
    if not NEON_CHM.exists():
        logger.error("NEON CHM not found: %s. Run 01_download_data.py first.", NEON_CHM)
        raise SystemExit(1)

    gap_frac, gap_transform, gap_crs = compute_gap_fraction(
        NEON_CHM, window_m=25.0, height_threshold_m=2.0
    )
    save_raster(gap_frac, OUT_GAP, gap_transform, gap_crs)

    # -- GLCM entropy -------------------------------------------------------
    logger.info("=== Computing GLCM Texture Entropy ===")
    try:
        naip_path = _find_naip(NAIP_RGBN)
        logger.info("Using NAIP file: %s", naip_path.name)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    entropy, ent_transform, ent_crs = compute_glcm_entropy(
        naip_path, nir_band_index=4, window_px=7, stride=7, n_levels=64
    )
    save_raster(entropy, OUT_ENTROPY, ent_transform, ent_crs)

    # -- Verify crown outputs -----------------------------------------------
    logger.info("=== Verifying Crown Segmentation Outputs ===")
    for p, label in [(OUT_CROWNS, "Crown polygons"), (OUT_CROWN_CV, "Crown CV raster")]:
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            logger.info("%s: %s (%.1f MB)", label, p.name, size_mb)
        else:
            logger.warning(
                "%s not found at %s. Run 02_crown_segmentation.R first.", label, p
            )

    logger.info("=== Metrics complete ===")


if __name__ == "__main__":
    main()
