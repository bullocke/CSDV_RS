"""csdv_core.label.lattice: chips and windows on the global 0.6 m lattice.

Every NAIP mosaic and CHM of a UTM zone shares one 0.6 m pixel lattice with
its origin at x = 0, y = 0 (:mod:`csdv_core.chm_inference.tiling`). Training
chips tile that lattice, so a chip is the same ground whichever unit it is
cut from. A window is a rectangular block of adjacent chips around a seed
candidate. The 19.2 m cell grid of a disturbance run is anchored at the
unit's own origin, so cell layers are mapped to chip pixels by position
(:func:`cells_to_pixels`), never by index.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from affine import Affine

from csdv_core.disturbance.inputs import CellGrid

logger = logging.getLogger(__name__)

#: Pixel size of the lattice, metres.
GSD = 0.6
#: Lattice pixels per metre times ten, to keep edge coordinates exact.
_TENTHS = 6

Bounds = tuple[float, float, float, float]

__all__ = [
    "GSD",
    "Bounds",
    "Chip",
    "Window",
    "cell_index",
    "cells_to_pixels",
    "grid_transform",
    "pixel_slices",
    "window_for",
]


def _edge(index: int, chip_px: int) -> float:
    """Map coordinate of a chip edge (exact to a tenth of a millimetre)."""
    return index * chip_px * _TENTHS / 10.0


@dataclass(frozen=True)
class Chip:
    """One chip of the lattice.

    ``ix`` counts chips east from x = 0 and ``iy`` north from y = 0.
    """

    epsg: int
    ix: int
    iy: int
    chip_px: int = 512

    @property
    def chip_id(self) -> str:
        return f"e{self.epsg}_x{self.ix}_y{self.iy}"

    @property
    def bounds(self) -> Bounds:
        return (
            _edge(self.ix, self.chip_px),
            _edge(self.iy, self.chip_px),
            _edge(self.ix + 1, self.chip_px),
            _edge(self.iy + 1, self.chip_px),
        )

    @property
    def transform(self) -> Affine:
        return grid_transform(self.bounds)


@dataclass(frozen=True)
class Window:
    """A block of ``nx`` by ``ny`` adjacent chips."""

    window_id: str
    unit: str
    epsg: int
    ix0: int
    iy0: int
    nx: int
    ny: int
    chip_px: int = 512

    @property
    def bounds(self) -> Bounds:
        return (
            _edge(self.ix0, self.chip_px),
            _edge(self.iy0, self.chip_px),
            _edge(self.ix0 + self.nx, self.chip_px),
            _edge(self.iy0 + self.ny, self.chip_px),
        )

    def chips(self) -> list[Chip]:
        """Chips of the window, north to south, west to east."""
        return [
            Chip(self.epsg, self.ix0 + i, self.iy0 + j, self.chip_px)
            for j in range(self.ny - 1, -1, -1)
            for i in range(self.nx)
        ]

    def stack_bounds(self, collar_px: int) -> Bounds:
        """Square extent of the imagery kept for the window.

        The square holds the window with ``collar_px`` pixels around its
        longer side and is centred on it, so a view of any candidate in the
        window can be square.
        """
        side_px = max(self.nx, self.ny) * self.chip_px + 2 * collar_px
        x0, y0, x1, y1 = self.bounds
        pad_x = (side_px - self.nx * self.chip_px) // 2
        pad_y = (side_px - self.ny * self.chip_px) // 2
        left = round(x0 / GSD) - pad_x
        bottom = round(y0 / GSD) - pad_y
        return (
            left * _TENTHS / 10.0,
            bottom * _TENTHS / 10.0,
            (left + side_px) * _TENTHS / 10.0,
            (bottom + side_px) * _TENTHS / 10.0,
        )


def grid_transform(bounds: Bounds) -> Affine:
    """North-up 0.6 m transform with its origin at the top-left of ``bounds``."""
    return Affine(GSD, 0.0, bounds[0], 0.0, -GSD, bounds[3])


def window_for(
    bounds: Bounds,
    *,
    window_id: str,
    unit: str,
    epsg: int,
    margin_m: float = 19.2,
    max_chips: int = 3,
    chip_px: int = 512,
    anchor: tuple[float, float] | None = None,
) -> Window:
    """The block of chips that covers ``bounds`` with a margin.

    A block wider than ``max_chips`` is cut to ``max_chips`` around
    ``anchor`` (default: the centre of ``bounds``), so a very large candidate
    is seen through a window placed on it.
    """
    side = chip_px * GSD
    minx, miny, maxx, maxy = bounds
    ax, ay = anchor or ((minx + maxx) / 2, (miny + maxy) / 2)

    def span(lo: float, hi: float, centre: float) -> tuple[int, int]:
        first = math.floor((lo - margin_m) / side)
        last = math.floor((hi + margin_m) / side)
        count = last - first + 1
        if count > max_chips:
            start = math.floor(centre / side) - max_chips // 2
            first = min(max(start, first), last - max_chips + 1)
            count = max_chips
        return first, count

    ix0, nx = span(minx, maxx, ax)
    iy0, ny = span(miny, maxy, ay)
    return Window(window_id, unit, epsg, ix0, iy0, nx, ny, chip_px)


def pixel_slices(inner: Bounds, outer: Bounds) -> tuple[slice, slice]:
    """Row and column slices of ``inner`` within a 0.6 m array covering ``outer``.

    Raises:
        ValueError: If ``inner`` is not inside ``outer``.
    """
    col0 = round((inner[0] - outer[0]) / GSD)
    col1 = round((inner[2] - outer[0]) / GSD)
    row0 = round((outer[3] - inner[3]) / GSD)
    row1 = round((outer[3] - inner[1]) / GSD)
    width = round((outer[2] - outer[0]) / GSD)
    height = round((outer[3] - outer[1]) / GSD)
    if col0 < 0 or row0 < 0 or col1 > width or row1 > height:
        raise ValueError(f"{inner} is not inside {outer}")
    return slice(row0, row1), slice(col0, col1)


def cell_index(grid: CellGrid, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the unit's cell under each pixel row and column.

    Args:
        grid: Cell grid of the run.
        bounds: Extent on the 0.6 m lattice.

    Returns:
        ``(rows, cols)``: the cell row of every pixel row of ``bounds``
        (north to south) and the cell column of every pixel column. Values
        below 0 or beyond the grid mark pixels outside it.
    """
    width = round((bounds[2] - bounds[0]) / GSD)
    height = round((bounds[3] - bounds[1]) / GSD)
    cell = grid.cell_transform
    xs = bounds[0] + (np.arange(width) + 0.5) * GSD
    ys = bounds[3] - (np.arange(height) + 0.5) * GSD
    cols = np.floor((xs - cell.c) / cell.a).astype(int)
    rows = np.floor((cell.f - ys) / -cell.e).astype(int)
    return rows, cols


def cells_to_pixels(
    cells: np.ndarray, grid: CellGrid, bounds: Bounds, fill: float = np.nan
) -> np.ndarray:
    """Spread a cell layer over the 0.6 m pixels of ``bounds`` (nearest cell).

    Args:
        cells: ``(rows, cols)`` or ``(bands, rows, cols)`` on the cell grid.
        grid: Cell grid of the run.
        bounds: Extent on the 0.6 m lattice.
        fill: Value of pixels outside the cell grid.

    Returns:
        An array with the pixel shape of ``bounds`` and the dtype of
        ``cells`` (bool stays bool, with False outside the grid).
    """
    rows, cols = cell_index(grid, bounds)
    height, width = rows.size, cols.size
    in_c = np.flatnonzero((cols >= 0) & (cols < grid.cell_cols))
    in_r = np.flatnonzero((rows >= 0) & (rows < grid.cell_rows))
    stack = cells if cells.ndim == 3 else cells[None]
    fill_value: float | bool = False if stack.dtype == bool else fill
    out = np.full((stack.shape[0], height, width), fill_value, dtype=stack.dtype)
    out[:, in_r[:, None], in_c[None, :]] = stack[
        :, rows[in_r][:, None], cols[in_c][None, :]
    ]
    return out if cells.ndim == 3 else out[0]
