"""csdv_core.label.screen: a NAIP-only greenness-loss score for a window.

Inputs: two dates of a four-band stack and a mask of reference cells.
Outputs: a z score per cell. Strongly negative means the cell lost more
greenness than reference forest with the same starting greenness.

NAIP is not calibrated between flights, and sensors, sun angle and season
differ. The later date's NDVI is therefore standardised conditional on the
earlier date's NDVI: within each of a few bins of earlier NDVI, the median
and spread of later NDVI among reference cells set the expected value. This
absorbs a non-linear radiometric change and part of the difference between
evergreen and deciduous stands. It does not help when the whole window
changed, because then no reference is left. The score is used as a guard and
a diagnostic, not as a label.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cell_ndvi", "off_season", "residual_z"]


def cell_ndvi(block: np.ndarray, cell_px: int) -> np.ndarray:
    """NDVI of the mean red and near-infrared of each cell.

    Args:
        block: ``(4, rows, cols)`` uint8 (red, green, blue, near-infrared).
            Pixels that are 0 in every band have no imagery.
        cell_px: Pixels per cell side.

    Returns:
        ``(rows // cell_px, cols // cell_px)`` float32, NaN where less than
        half of a cell has imagery.
    """
    rows, cols = block.shape[1] // cell_px, block.shape[2] // cell_px
    trimmed = block[:, : rows * cell_px, : cols * cell_px].astype(np.float32)
    valid = trimmed.any(axis=0)
    shape = (rows, cell_px, cols, cell_px)

    def mean(band: np.ndarray) -> np.ndarray:
        total = np.where(valid, band, 0.0).reshape(shape).sum(axis=(1, 3))
        count = valid.reshape(shape).sum(axis=(1, 3))
        return np.where(
            count >= cell_px * cell_px / 2, total / np.maximum(count, 1), np.nan
        )

    red, nir = mean(trimmed[0]), mean(trimmed[3])
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((nir - red) / (nir + red)).astype(np.float32)


def residual_z(
    pre: np.ndarray,
    post: np.ndarray,
    reference: np.ndarray,
    *,
    n_bins: int = 10,
    min_reference: int = 200,
) -> np.ndarray | None:
    """Standardised later NDVI, given the earlier NDVI.

    Args:
        pre: NDVI at the earlier date, per cell.
        post: NDVI at the later date, per cell.
        reference: Cells taken as unchanged forest.
        n_bins: Bins of earlier NDVI (equal counts of reference cells).
        min_reference: Fewest reference cells that still give a score.

    Returns:
        z per cell (NaN where NDVI is missing), or None when there are too
        few reference cells to say what unchanged forest looks like.
    """
    ok = reference & np.isfinite(pre) & np.isfinite(post)
    if ok.sum() < min_reference:
        return None
    edges = np.quantile(pre[ok], np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(pre, edges)
    z = np.full(pre.shape, np.nan, dtype=np.float32)
    for b in range(n_bins):
        ref = ok & (bins == b)
        if ref.sum() < 5:
            continue
        median = np.median(post[ref])
        spread = np.median(np.abs(post[ref] - median)) * 1.4826 + 1e-4
        members = (bins == b) & np.isfinite(pre) & np.isfinite(post)
        z[members] = (post[members] - median) / spread
    return z


def off_season(date: str, first_month: int = 5, last_month: int = 9) -> bool:
    """Whether an acquisition date falls outside the leaf-on months."""
    month = int(date[5:7])
    return month < first_month or month > last_month
