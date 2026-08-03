"""csdv_core.zonal.mask — read a raster window for one stand and mask it.

Every zonal metric starts from the same two things: the block of pixels covering
a stand's bounding box, and a boolean array saying which of those pixels are
inside the stand. :class:`StandWindow` carries both, along with the support
bookkeeping the classification document asks to be reported next to any metric
that needs a rectangular footprint.

Pixel membership uses :func:`rasterio.features.geometry_mask` with
``all_touched=False``, which is rasterio's pixel-centre rule. That is exactly
the definition in Section 2: every pixel whose centre falls inside the polygon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

__all__ = ["StandWindow", "read_stand_array", "stand_window"]


@dataclass(frozen=True)
class StandWindow:
    """A stand's bounding-box footprint on a raster grid.

    Attributes:
        stand_id: Identifier carried through for logging and error messages.
        geometry: The stand geometry, in the raster CRS.
        window: The rasterio window covering the stand's bounding box.
        transform: Affine transform of the windowed block.
        mask: Boolean array, True for pixels whose centre is inside the stand.
        pixel_size_m: Pixel side length in metres.
        n_inside: Count of True pixels in ``mask``.
        bbox_fill_fraction: ``n_inside / mask.size``. Long thin stands fill only
            a small part of their bounding box, which limits how much a texture
            or spatial metric computed over that box can be trusted.
    """

    stand_id: str
    geometry: BaseGeometry
    window: Window
    transform: Any
    mask: np.ndarray
    pixel_size_m: float
    n_inside: int
    bbox_fill_fraction: float

    @property
    def area_m2(self) -> float:
        """Area of the in-stand pixels, which may differ slightly from the
        geometry area because of the pixel-centre rule."""
        return float(self.n_inside) * self.pixel_size_m**2

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the windowed block."""
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))


def stand_window(
    geometry: BaseGeometry,
    transform: Any,
    shape: tuple[int, int],
    *,
    stand_id: str = "",
    pad_px: int = 0,
    all_touched: bool = False,
) -> StandWindow:
    """Build the windowed footprint and in-stand mask for one geometry.

    Args:
        geometry: Stand geometry in the raster CRS. MultiPolygon is fine.
        transform: Affine transform of the full raster.
        shape: ``(height, width)`` of the full raster.
        stand_id: Identifier used in messages.
        pad_px: Extra pixels to include on every side. Padding widens the block
            without widening the mask, which is what a figure panel or a crown
            segmentation buffer needs.
        all_touched: Passed to :func:`rasterio.features.geometry_mask`. Leave
            False for the pixel-centre rule the classification document defines.

    Returns:
        A :class:`StandWindow`.

    Raises:
        ValueError: If the geometry does not overlap the raster, or if the
            resulting block would be empty.
    """
    height, width = int(shape[0]), int(shape[1])
    minx, miny, maxx, maxy = geometry.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=transform)
    row_off = int(np.floor(win.row_off)) - pad_px
    col_off = int(np.floor(win.col_off)) - pad_px
    row_stop = int(np.ceil(win.row_off + win.height)) + pad_px
    col_stop = int(np.ceil(win.col_off + win.width)) + pad_px

    row_off = max(0, row_off)
    col_off = max(0, col_off)
    row_stop = min(height, row_stop)
    col_stop = min(width, col_stop)
    if row_stop <= row_off or col_stop <= col_off:
        raise ValueError(
            f"Stand {stand_id!r} does not overlap the raster "
            f"(bounds {geometry.bounds}, raster shape {shape})"
        )

    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=col_stop - col_off,
        height=row_stop - row_off,
    )
    win_transform = rasterio.windows.transform(window, transform)
    out_shape = (int(window.height), int(window.width))
    mask = geometry_mask(
        [geometry],
        out_shape=out_shape,
        transform=win_transform,
        invert=True,
        all_touched=all_touched,
    )
    n_inside = int(mask.sum())
    if n_inside == 0:
        raise ValueError(
            f"Stand {stand_id!r} contains no pixel centres at this resolution "
            f"(block {out_shape}, pixel {abs(transform.a):.3f} m)"
        )
    return StandWindow(
        stand_id=stand_id,
        geometry=geometry,
        window=window,
        transform=win_transform,
        mask=mask,
        pixel_size_m=float(abs(transform.a)),
        n_inside=n_inside,
        bbox_fill_fraction=float(n_inside) / float(mask.size),
    )


def read_stand_array(
    path: Path | str,
    geometry: BaseGeometry,
    *,
    band: int = 1,
    pad_px: int = 0,
    scale: float = 1.0,
    all_touched: bool = False,
    stand_id: str = "",
) -> tuple[np.ndarray, StandWindow]:
    """Read one band over a stand's bounding box and return it with its mask.

    Only the stand's block is read, which matters when the source raster covers
    a whole mapping module. The raster nodata value becomes NaN, and the values
    are multiplied by ``scale`` (use 0.01 to convert a centimetre canopy height
    product to metres).

    Returns:
        ``(array, stand_window)`` where ``array`` is float32 of the same shape
        as ``stand_window.mask``.

    Raises:
        ValueError: If the stand does not overlap the raster.
    """
    with rasterio.open(path) as src:
        sw = stand_window(
            geometry,
            src.transform,
            (src.height, src.width),
            stand_id=stand_id,
            pad_px=pad_px,
            all_touched=all_touched,
        )
        arr = src.read(band, window=sw.window).astype(np.float32)
        nodata = src.nodatavals[band - 1]
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == np.float32(nodata), np.nan, arr)
    if scale != 1.0:
        arr = arr * np.float32(scale)
    return arr, sw
