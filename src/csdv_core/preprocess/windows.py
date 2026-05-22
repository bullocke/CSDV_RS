"""csdv_core.preprocess.windows — Analysis-window tiling helpers.

Public re-exports of the lower-level helpers in
:mod:`csdv_core.metrics._window` plus a convenience generator for the
multi-scale window cascade (25, 50, 100, 200 m).
"""

from __future__ import annotations

from collections.abc import Iterator

from csdv_core.io.grids import GridSpec
from csdv_core.metrics._window import (
    assign_points_to_cells,
    iter_tiles,
    tile_shape,
    window_pixels,
    window_transform,
)


def multiscale_grid(
    grid: GridSpec,
    *,
    rows: int,
    cols: int,
    window_sizes_m: list[float],
) -> Iterator[tuple[float, tuple[int, int], object]]:
    """Yield ``(window_m, (n_rows, n_cols), window_transform)`` per scale.

    Args:
        grid: Source grid (pixel size and transform).
        rows: Source raster height in pixels.
        cols: Source raster width in pixels.
        window_sizes_m: Window side lengths in metres.

    Yields:
        Tuples of ``(window_m, tile_shape, window_transform)`` for each
        requested scale, in input order.
    """
    for w in window_sizes_m:
        wpx = window_pixels(w, grid.pixel_size_m)
        shape = tile_shape(rows, cols, wpx)
        wt = window_transform(grid.transform, w)
        yield w, shape, wt


__all__ = [
    "assign_points_to_cells",
    "iter_tiles",
    "multiscale_grid",
    "tile_shape",
    "window_pixels",
    "window_transform",
]
