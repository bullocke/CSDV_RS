"""Shared windowing helpers for metric functions.

Two operations are common across metrics:

1. Non-overlapping square tiles over a raster (gap, crown_fraction, texture).
2. Assigning point centroids to the same tile grid (crown statistics).

Both are exposed as small pure functions so the per-metric files stay terse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.transform import Affine


def window_pixels(window_m: float, pixel_size_m: float) -> int:
    """Number of source pixels along one side of a window. Minimum 1."""
    return max(1, int(round(window_m / pixel_size_m)))


def tile_shape(rows: int, cols: int, window_px: int) -> tuple[int, int]:
    """Output grid shape after floor-dividing into windows of ``window_px``.

    Partial edge tiles are discarded, matching the legacy PoC behavior.
    """
    return rows // window_px, cols // window_px


def window_transform(
    source_transform: Any,
    window_m: float,
) -> Affine:
    """Return an Affine transform with one pixel per window.

    Origin (top-left) is taken from the source transform; pixel size becomes
    ``window_m`` (with the standard north-up y-flip).
    """
    return Affine(window_m, 0.0, source_transform.c, 0.0, -window_m, source_transform.f)


def iter_tiles(
    array: np.ndarray,
    window_px: int,
) -> tuple[int, int, np.ndarray]:  # type: ignore[override]
    """Yield ``(row, col, tile)`` for each non-overlapping window."""
    n_rows, n_cols = tile_shape(array.shape[0], array.shape[1], window_px)
    for r in range(n_rows):
        for c in range(n_cols):
            yield (
                r,
                c,
                array[
                    r * window_px : (r + 1) * window_px,
                    c * window_px : (c + 1) * window_px,
                ],
            )


def assign_points_to_cells(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    window_m: float,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map point coordinates to (row, col) cell indices on a window grid.

    Points outside the grid are returned with index ``-1`` in both axes;
    callers should mask before use.
    """
    col_idx = np.floor((xs - origin_x) / window_m).astype(np.int64)
    row_idx = np.floor((origin_y - ys) / window_m).astype(np.int64)
    in_bounds = (
        (col_idx >= 0) & (col_idx < n_cols) & (row_idx >= 0) & (row_idx < n_rows)
    )
    col_idx = np.where(in_bounds, col_idx, -1)
    row_idx = np.where(in_bounds, row_idx, -1)
    return row_idx, col_idx
