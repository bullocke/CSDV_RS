"""csdv_core.disturbance.windows: read pixel windows on the common grid.

Each function mosaics one band of a year's GeoTIFFs into a window of the
:class:`~csdv_core.disturbance.inputs.CellGrid` pixel array. A file already on
the grid's CRS, pixel size and alignment is read by window, an exact copy.
Any other file goes through a nearest-neighbour ``WarpedVRT``, so tiles with
another alignment or CRS still mosaic onto the grid. Pixels covered by no file
or holding nodata are NaN.

Inputs: GeoTIFF paths, the grid and a window. Outputs: float32 arrays (CHM in
metres once scaled).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from csdv_core.disturbance.inputs import (
    PIXEL_SIZE_RTOL,
    SNAP_TOLERANCE_PX,
    CellGrid,
    SiteInputs,
)

logger = logging.getLogger(__name__)


def read_window(
    paths: Sequence[Path],
    grid: CellGrid,
    row_off: int,
    n_rows: int,
    *,
    band: int = 1,
    scale: float = 1.0,
    nodata: float | None = None,
) -> np.ndarray:
    """Read a full-width strip of the pixel grid from one year's files.

    Each file is warped onto ``grid.pixel_transform`` with nearest-neighbour
    resampling (``WarpedVRT``). Where files overlap, the first file in
    ``paths`` with a valid value wins. Pixels covered by no file or holding
    ``nodata`` are NaN.

    Args:
        paths: GeoTIFFs of one year (CHM or NAIP).
        grid: Common grid.
        row_off: First pixel row of the strip (0-based, grid pixel rows).
        n_rows: Strip height in pixels. ``row_off + n_rows`` must not exceed
            ``grid.pixel_shape[0]``.
        band: 1-based band number.
        scale: Multiplier applied after the nodata mask (0.01 turns uint16 cm
            into metres).
        nodata: Stored nodata value (None = each file's metadata, or no mask
            when the file has none).

    Returns:
        float32 array ``(n_rows, grid.pixel_shape[1])``, NaN = nodata.

    Raises:
        ValueError: If the strip runs outside the grid or ``paths`` is empty.
    """
    pixel_rows, pixel_cols = grid.pixel_shape
    if row_off < 0 or n_rows < 1 or row_off + n_rows > pixel_rows:
        raise ValueError(
            f"Strip rows {row_off}..{row_off + n_rows} fall outside the grid "
            f"(0..{pixel_rows})"
        )
    strip_window = Window(0, row_off, pixel_cols, n_rows)
    return _read_grid_window(
        paths, grid, strip_window, band=band, scale=scale, nodata=nodata
    )


def _read_grid_window(
    paths: Sequence[Path],
    grid: CellGrid,
    window: Window,
    *,
    band: int,
    scale: float,
    nodata: float | None,
) -> np.ndarray:
    """Mosaic one band of ``paths`` into a window of the grid's pixels.

    Each file fills only the grid pixels whose centres fall inside it, so
    pixels outside every file stay NaN even when a file has no nodata value.
    The first file with a valid value wins.

    Args:
        paths: GeoTIFFs of one year.
        grid: Common grid.
        window: Integer window of the grid's pixel array.
        band: 1-based band number.
        scale: Multiplier applied after the nodata mask.
        nodata: Stored nodata value (None = each file's metadata).

    Returns:
        float32 ``(window.height, window.width)``, NaN = nodata.

    Raises:
        ValueError: If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("No raster files to read")
    n_rows = int(window.height)
    n_cols = int(window.width)
    window_transform = _window_transform(window, grid.pixel_transform)
    mosaic = np.full((n_rows, n_cols), np.nan, dtype="float32")
    for path in paths:
        with rasterio.open(path) as src:
            covered = _covered_window(src, grid.crs, window_transform, n_rows, n_cols)
            if covered is None:
                continue
            file_nodata = nodata if nodata is not None else src.nodata
            values = _read_file_window(
                src, grid.crs, window_transform, covered, band, file_nodata
            )
        if file_nodata is not None:
            if math.isnan(file_nodata):
                is_nodata = np.isnan(values)
            else:
                is_nodata = values == np.float32(file_nodata)
            values[is_nodata] = np.nan
        if scale != 1.0:
            values *= np.float32(scale)
        row_start = int(covered.row_off)
        col_start = int(covered.col_off)
        row_slice = slice(row_start, row_start + int(covered.height))
        col_slice = slice(col_start, col_start + int(covered.width))
        target = mosaic[row_slice, col_slice]
        fill = np.isnan(target) & np.isfinite(values)
        target[fill] = values[fill]
    return mosaic


def _covered_window(
    src: rasterio.io.DatasetReader,
    crs: CRS,
    window_transform: Affine,
    n_rows: int,
    n_cols: int,
) -> Window | None:
    """Find the part of a grid window whose pixel centres lie inside a file.

    Args:
        src: Open source dataset.
        crs: Grid CRS.
        window_transform: Transform of the grid window (north up).
        n_rows: Window height in pixels.
        n_cols: Window width in pixels.

    Returns:
        The covered sub-window (offsets relative to the window), or None when
        the file does not reach the window.
    """
    if src.crs == crs:
        left, bottom, right, top = src.bounds
    else:
        left, bottom, right, top = transform_bounds(
            src.crs, crs, *src.bounds, densify_pts=21
        )
    pixel_width = window_transform.a
    pixel_height = -window_transform.e
    col_start_exact = (left - window_transform.c) / pixel_width - 0.5
    col_stop_exact = (right - window_transform.c) / pixel_width - 0.5
    row_start_exact = (window_transform.f - top) / pixel_height - 0.5
    row_stop_exact = (window_transform.f - bottom) / pixel_height - 0.5
    first_inside_col = math.ceil(col_start_exact - SNAP_TOLERANCE_PX)
    last_inside_col = math.floor(col_stop_exact + SNAP_TOLERANCE_PX)
    first_inside_row = math.ceil(row_start_exact - SNAP_TOLERANCE_PX)
    last_inside_row = math.floor(row_stop_exact + SNAP_TOLERANCE_PX)
    col_start = max(first_inside_col, 0)
    col_stop = min(last_inside_col + 1, n_cols)
    row_start = max(first_inside_row, 0)
    row_stop = min(last_inside_row + 1, n_rows)
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def _read_file_window(
    src: rasterio.io.DatasetReader,
    crs: CRS,
    window_transform: Affine,
    covered: Window,
    band: int,
    file_nodata: float | None,
) -> np.ndarray:
    """Read one band of an open file onto a covered sub-window of the grid.

    A file on the grid's CRS, pixel size and pixel alignment is read directly
    by window, which copies it exactly. Other files go through a
    nearest-neighbour ``WarpedVRT``. The direct read is also needed for
    exactness: on real NAIP the warper returned 0 for a uint8 255.

    Args:
        src: Open source dataset.
        crs: Grid CRS.
        window_transform: Transform of the enclosing grid window.
        covered: Sub-window to fill, relative to the enclosing window.
        band: 1-based band number.
        file_nodata: Nodata value to declare to the warper, or None.

    Returns:
        float32 ``(covered.height, covered.width)`` of stored values.
    """
    covered_transform = _window_transform(covered, window_transform)
    source_window = _aligned_source_window(src, crs, covered_transform, covered)
    if source_window is not None:
        stored = src.read(band, window=source_window)
        return stored.astype("float32")
    vrt_options: dict[str, Any] = {
        "crs": crs,
        "transform": covered_transform,
        "width": int(covered.width),
        "height": int(covered.height),
        "resampling": Resampling.nearest,
    }
    if file_nodata is not None:
        vrt_options["src_nodata"] = file_nodata
        vrt_options["nodata"] = file_nodata
    with WarpedVRT(src, **vrt_options) as vrt:
        stored = vrt.read(band)
    return stored.astype("float32")


def _aligned_source_window(
    src: rasterio.io.DatasetReader,
    crs: CRS,
    covered_transform: Affine,
    covered: Window,
) -> Window | None:
    """Return the source window matching a covered grid window, if aligned.

    Args:
        src: Open source dataset.
        crs: Grid CRS.
        covered_transform: Transform of the covered grid window.
        covered: Covered grid window (for its size).

    Returns:
        The integer source window, or None when the file's CRS, pixel size,
        rotation or pixel alignment differs from the grid.
    """
    source_transform = src.transform
    if src.crs != crs:
        return None
    same_width = math.isclose(
        source_transform.a, covered_transform.a, rel_tol=PIXEL_SIZE_RTOL
    )
    same_height = math.isclose(
        source_transform.e, covered_transform.e, rel_tol=PIXEL_SIZE_RTOL
    )
    unrotated = source_transform.b == 0 and source_transform.d == 0
    if not (same_width and same_height and unrotated):
        return None
    col_exact = (covered_transform.c - source_transform.c) / source_transform.a
    row_exact = (covered_transform.f - source_transform.f) / source_transform.e
    col_off = round(col_exact)
    row_off = round(row_exact)
    col_aligned = abs(col_exact - col_off) < SNAP_TOLERANCE_PX
    row_aligned = abs(row_exact - row_off) < SNAP_TOLERANCE_PX
    if not (col_aligned and row_aligned):
        return None
    return Window(col_off, row_off, int(covered.width), int(covered.height))


def _window_transform(window: Window, transform: Affine) -> Affine:
    """Return the affine transform of a window of a north-up raster.

    Same result as ``rasterio.windows.transform``, written with ``@`` to avoid
    the ``affine`` deprecation warning for ``*`` composition.

    Args:
        window: Integer window.
        transform: Transform of the full raster.

    Returns:
        Transform whose origin is the window's top-left corner.
    """
    offset = Affine.translation(window.col_off, window.row_off)
    return transform @ offset


def read_bounds(
    paths: Sequence[Path],
    grid: CellGrid,
    bounds: tuple[float, float, float, float],
    *,
    band: int = 1,
    scale: float = 1.0,
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine]:
    """Read a pixel-grid window covering ``bounds``, clipped to the grid extent.

    The window is ``from_bounds(...).round_offsets().round_lengths()``, as in
    ``08_refine_best._read``: the column and row offsets are floored to the
    pixel edge at or before the left and top bounds, and the width and height
    are rounded to the nearest whole pixel. The right and bottom edges can
    therefore fall up to half a pixel inside or outside ``bounds``. The window
    is then intersected with the grid. Used by outline refinement and by
    figure panels.

    Args:
        paths: GeoTIFFs of one year.
        grid: Common grid.
        bounds: ``(left, bottom, right, top)`` in grid CRS units.
        band: 1-based band number.
        scale: Multiplier applied after the nodata mask.
        nodata: Stored nodata value (None = file metadata).

    Returns:
        ``(array, transform)``: float32 ``(rows, cols)`` with NaN nodata, and
        the window's affine transform.

    Raises:
        ValueError: If ``bounds`` does not overlap the grid.
    """
    pixel_rows, pixel_cols = grid.pixel_shape
    grid_window = Window(0, 0, pixel_cols, pixel_rows)
    requested = from_bounds(*bounds, transform=grid.pixel_transform)
    snapped = requested.round_offsets().round_lengths()
    try:
        clipped = snapped.intersection(grid_window)
    except WindowError as error:
        raise ValueError(f"Bounds {bounds} do not overlap the grid") from error
    if clipped.width < 1 or clipped.height < 1:
        raise ValueError(f"Bounds {bounds} do not overlap the grid")
    values = _read_grid_window(
        paths, grid, clipped, band=band, scale=scale, nodata=nodata
    )
    window_transform = _window_transform(clipped, grid.pixel_transform)
    return values, window_transform


def read_chm_bounds(
    site_inputs: SiteInputs, year: int, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, Affine]:
    """Read one year's CHM in metres over ``bounds`` with :func:`read_bounds`.

    Args:
        site_inputs: Inputs and grid of the run.
        year: NAIP year to read.
        bounds: ``(left, bottom, right, top)`` in grid CRS units.

    Returns:
        ``(chm, transform)``: float32 heights (NaN nodata) and the window's
        affine transform.
    """
    year_input = site_inputs.year_input(year)
    return read_bounds(
        year_input.chm_paths,
        site_inputs.grid,
        bounds,
        scale=year_input.chm_scale,
        nodata=year_input.chm_nodata,
    )


__all__ = ["read_bounds", "read_chm_bounds", "read_window"]
