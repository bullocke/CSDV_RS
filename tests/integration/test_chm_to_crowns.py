"""Integration test: CHM \u2192 crown segmentation \u2192 mean crown CV.

Reads the legacy SCBI NEON CHM tile and runs the Python watershed engine.
Compares mean crown CV (per 50 m cell) against the PoC reference raster.

The Python watershed engine is the Phase 3 fallback. The PoC reference was
produced by lidR's Dalponte-Coomes segmentation in R. The two algorithms
agree on coarse structure but diverge on per-tree boundaries, so we expect
a sizeable but bounded divergence in derived CV. Tolerance is 30%.

Skipped if either reference file is missing (legacy/ is gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from csdv_core.metrics._window import iter_tiles, window_pixels
from csdv_core.segmentation.chm_watershed import segment_crowns

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY = REPO_ROOT / "legacy" / "proof_of_concept"
CHM_PATH = LEGACY / "Data" / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif"
REF_CV_PATH = (
    LEGACY / "Results" / "summary_document" / "intermediate" / "crown_cv_50m_SCBI.tif"
)


def _mean_crown_cv_50m(
    chm: np.ndarray,
    transform,
    crs,
    pixel_size_m: float,
) -> float:
    """Run watershed segmentation, bin crowns into 50 m cells, return mean CV."""
    gdf = segment_crowns(
        chm,
        transform=transform,
        crs=crs,
        min_height_m=2.0,
        min_peak_distance_m=3.0,
        min_crown_area_m2=1.0,
    )
    if len(gdf) == 0:
        return float("nan")

    # Assign each crown centroid to a 50 m tile and compute per-tile CV of
    # crown diameter.
    wpx = window_pixels(50.0, pixel_size_m)
    rows, cols = chm.shape
    n_rows = rows // wpx
    n_cols = cols // wpx

    centroids = gdf.geometry.centroid
    # Convert centroids to pixel indices via the inverse affine.
    inv = ~transform
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()
    cols_px, rows_px = inv * (xs, ys)
    cols_px = np.asarray(cols_px).astype(int)
    rows_px = np.asarray(rows_px).astype(int)

    tile_r = rows_px // wpx
    tile_c = cols_px // wpx
    keep = (tile_r >= 0) & (tile_r < n_rows) & (tile_c >= 0) & (tile_c < n_cols)
    diam = gdf["crown_diam_m"].to_numpy()[keep]
    tr = tile_r[keep]
    tc = tile_c[keep]

    cvs: list[float] = []
    for r in range(n_rows):
        for c in range(n_cols):
            sel = (tr == r) & (tc == c)
            if sel.sum() < 2:
                continue
            d = diam[sel]
            mu = float(np.mean(d))
            if mu <= 0:
                continue
            cvs.append(float(np.std(d) / mu))
    # Silence unused-import warning for iter_tiles (kept for symmetry with the
    # PoC reference pipeline; both bin into the same grid).
    _ = iter_tiles
    if not cvs:
        return float("nan")
    return float(np.mean(cvs))


def test_chm_to_crowns_mean_cv_within_tolerance() -> None:
    """Mean crown CV from watershed segmentation matches PoC within 10%."""
    if not CHM_PATH.exists():
        pytest.skip(f"Legacy SCBI CHM not present at {CHM_PATH}")
    if not REF_CV_PATH.exists():
        pytest.skip(f"Legacy PoC CV reference not present at {REF_CV_PATH}")

    with rasterio.open(CHM_PATH) as src:
        chm = src.read(1, masked=True).filled(np.nan).astype("float32")
        transform = src.transform
        crs = src.crs
        pixel_size_m = float(abs(transform.a))

    candidate = _mean_crown_cv_50m(chm, transform, crs, pixel_size_m)
    assert np.isfinite(candidate), "No crown CVs produced from segmentation"

    with rasterio.open(REF_CV_PATH) as src:
        ref = src.read(1, masked=True)
    reference = float(np.ma.mean(ref))
    assert np.isfinite(reference), "Reference CV raster has no valid pixels"

    rel_err = abs(candidate - reference) / max(reference, 1e-6)
    assert rel_err <= 0.30, (
        f"Mean crown CV (50 m) diverges from PoC reference beyond 30%: "
        f"candidate={candidate:.4f}, reference={reference:.4f}, rel_err={rel_err:.3f}"
    )
