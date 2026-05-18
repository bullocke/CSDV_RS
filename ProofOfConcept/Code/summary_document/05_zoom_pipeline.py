"""
05_zoom_pipeline.py — Generate zoomed (bottom-left 1/4 extent) versions of all
demonstration figures for the CSDV summary document.

Clips source rasters to the southwestern quadrant of the SCBI study area, then
runs the full pipeline (crown segmentation via R, gap fraction, GLCM entropy,
figures) on the clipped data. All output files use a '_zoom' infix so no
existing files are overwritten.

Clip logic
----------
"Bottom-left 1/4" means the southwestern quadrant of the NEON CHM raster:
  rows   : height//2 → height   (lower half, i.e. south)
  cols   : 0 → width//2         (left half, i.e. west)
The same geographic bounding box is applied to clip NAIP and NAIP-CHM.

Usage
-----
  python 05_zoom_pipeline.py
"""

from __future__ import annotations

import logging
import subprocess
import sys  # noqa: F401 — kept for sys.exit
from pathlib import Path

import importlib.util

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.windows import Window
from shapely.geometry import box


def _import_script(alias: str, path: Path):
    """Import a Python script by path, bypassing the digit-prefix naming issue."""
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can look up sys.modules[__name__]
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
POC = PROJECT_ROOT / "ProofOfConcept"

# Source rasters (inputs to clip)
NEON_CHM_SRC = POC / "Data" / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif"
NAIP_DIR = POC / "Data" / "NAIP" / "Imagery"
NAIPCHM_DIR = POC / "Data" / "NAIP" / "National_CHM"

# Clipped raster outputs (same directories, _zoom infix)
NEON_CHM_ZOOM = POC / "Data" / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023_zoom.tif"

# Intermediate outputs for zoom
INTERMEDIATE = POC / "Results" / "summary_document" / "intermediate"
CROWNS_ZOOM = INTERMEDIATE / "crown_polygons_SCBI_zoom.gpkg"
CROWN_CV_ZOOM = INTERMEDIATE / "crown_cv_50m_SCBI_zoom.tif"
GAP_ZOOM = INTERMEDIATE / "gap_fraction_25m_SCBI_zoom.tif"
ENTROPY_ZOOM = INTERMEDIATE / "glcm_entropy_NAIP_SCBI_zoom.tif"

# Figure outputs
FIGURES_OUT = POC / "Results" / "summary_document" / "figures"
FIG1_ZOOM = FIGURES_OUT / "fig01_data_sources_zoom.png"
FIG2_ZOOM = FIGURES_OUT / "fig02_derived_metrics_zoom.png"

# R script (accepts optional path args)
R_SCRIPT = HERE.parent / "02_crown_segmentation.R"

MICROMAMBA = Path("/home/bullocke/micromamba/micromamba")
ENV_PATH = Path("/home/bullocke/vscode_projects/csdv/.micromamba/envs/CSDV")


def _mm_run(cmd: list[str]) -> None:
    """Run a command inside the micromamba environment."""
    full_cmd = [str(MICROMAMBA), "run", "-p", str(ENV_PATH)] + cmd
    logger.info("Running: %s", " ".join(str(c) for c in full_cmd))
    result = subprocess.run(full_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {full_cmd}")


# ---------------------------------------------------------------------------
# Clipping helpers
# ---------------------------------------------------------------------------

def compute_zoom_window(src_path: Path) -> tuple[Window, tuple[float, float, float, float]]:
    """
    Return the rasterio Window and (west, south, east, north) bbox for the
    bottom-left 1/4 of the given raster.
    """
    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        row_off = h // 2
        col_off = 0
        row_count = h - row_off
        col_count = w // 2
        window = Window(col_off, row_off, col_count, row_count)
        win_transform = src.window_transform(window)
        left = win_transform.c
        top = win_transform.f
        right = left + col_count * win_transform.a
        bottom = top + row_count * win_transform.e   # e is negative
    return window, (left, bottom, right, top)


def clip_singleband(src_path: Path, window: Window, out_path: Path) -> None:
    """Clip a single-band raster using a pre-computed window and save."""
    with rasterio.open(src_path) as src:
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=transform,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Clipped → %s (%.1f MB)", out_path.name, size_mb)


def clip_multiband_by_bbox(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Clip a multi-band raster to a bbox polygon and save. CRS must match."""
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rasterio_mask(src, [geom], crop=True, all_touched=True)
        profile = src.profile.copy()
        profile.update(
            height=out_data.shape[1],
            width=out_data.shape[2],
            transform=out_transform,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data)
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Clipped → %s (%.1f MB)", out_path.name, size_mb)


def find_latest(directory: Path, prefix: str) -> Path:
    """Return the most recent .tif matching prefix, excluding any _zoom variants."""
    tifs = sorted(
        [p for p in directory.glob(f"{prefix}*.tif") if "_zoom" not in p.stem],
        key=lambda p: p.stat().st_mtime,
    )
    if not tifs:
        raise FileNotFoundError(f"No non-zoom .tif matching '{prefix}*' in {directory}")
    return tifs[-1]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Clip source rasters to bottom-left 1/4
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Clipping source rasters ===")

    if not NEON_CHM_SRC.exists():
        logger.error("NEON CHM not found: %s", NEON_CHM_SRC)
        sys.exit(1)

    window, bbox = compute_zoom_window(NEON_CHM_SRC)
    west, south, east, north = bbox
    logger.info(
        "Zoom bbox (EPSG:5070): W=%.0f S=%.0f E=%.0f N=%.0f  (%.0f x %.0f m)",
        west, south, east, north, east - west, north - south,
    )

    clip_singleband(NEON_CHM_SRC, window, NEON_CHM_ZOOM)

    naip_src = find_latest(NAIP_DIR, "NAIP_SCBI")
    naip_zoom = NAIP_DIR / (naip_src.stem + "_zoom.tif")
    clip_multiband_by_bbox(naip_src, bbox, naip_zoom)
    logger.info("NAIP zoom: %s", naip_zoom.name)

    naipchm_src = find_latest(NAIPCHM_DIR, "NAIPCHM_SCBI")
    naipchm_zoom = NAIPCHM_DIR / (naipchm_src.stem + "_zoom.tif")
    clip_multiband_by_bbox(naipchm_src, bbox, naipchm_zoom)
    logger.info("NAIP-CHM zoom: %s", naipchm_zoom.name)

    # ------------------------------------------------------------------
    # 2. Crown segmentation on clipped NEON CHM (R)
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Crown segmentation (R) ===")
    _mm_run([
        "Rscript", str(R_SCRIPT),
        str(NEON_CHM_ZOOM), str(CROWNS_ZOOM), str(CROWN_CV_ZOOM),
    ])

    # ------------------------------------------------------------------
    # 3. Compute gap fraction and GLCM entropy on clipped data
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Computing metrics ===")

    # Import helper functions from the metrics script (digit-prefix → importlib)
    m03 = _import_script("compute_metrics", HERE.parent / "03_compute_metrics.py")
    compute_gap_fraction = m03.compute_gap_fraction
    compute_glcm_entropy = m03.compute_glcm_entropy
    save_raster = m03.save_raster

    gap_frac, gap_transform, gap_crs = compute_gap_fraction(
        NEON_CHM_ZOOM, window_m=25.0, height_threshold_m=2.0
    )
    save_raster(gap_frac, GAP_ZOOM, gap_transform, gap_crs)

    entropy, ent_transform, ent_crs = compute_glcm_entropy(
        naip_zoom, nir_band_index=4, window_px=7, stride=7, n_levels=64
    )
    save_raster(entropy, ENTROPY_ZOOM, ent_transform, ent_crs)

    # ------------------------------------------------------------------
    # 4. Generate zoom figures
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Generating figures ===")

    m04 = _import_script("make_figures", HERE.parent / "04_make_figures.py")
    make_figure1 = m04.make_figure1
    make_figure2 = m04.make_figure2
    m04.setup_style()

    logger.info("Making Figure 1 (zoom) ...")
    make_figure1(naip_zoom, naipchm_zoom, NEON_CHM_ZOOM, out_path=FIG1_ZOOM)

    logger.info("Making Figure 2 (zoom) ...")
    make_figure2(
        naip_path=naip_zoom,
        gap_frac_path=GAP_ZOOM,
        entropy_path=ENTROPY_ZOOM,
        crowns_path=CROWNS_ZOOM,
        crown_cv_path=CROWN_CV_ZOOM,
        neon_chm_path=NEON_CHM_ZOOM,
        out_path=FIG2_ZOOM,
    )

    logger.info("=== Zoom pipeline complete ===")
    logger.info("  Figure 1 zoom: %s", FIG1_ZOOM)
    logger.info("  Figure 2 zoom: %s", FIG2_ZOOM)


if __name__ == "__main__":
    main()
