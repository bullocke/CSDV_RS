"""csdv_core.viz.maps — image panels with a stand outline drawn on them.

The panels a worked example needs are a true-colour NAIP view and a canopy
height view of the same ground, repeated across dates, with the stand outlined
so a reader can see what was measured and what was not.

:func:`draw_stand_outline` replaces the legacy overlay helper, which read
``polygon.exterior`` and therefore raised on a MultiPolygon. Every polygon in
the calibration delivery is a MultiPolygon, and an impact polygon may also have
holes, so both cases are handled here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from matplotlib.axes import Axes
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from csdv_core.viz.style import INK

logger = logging.getLogger(__name__)

CHM_VMAX_M = 30.0

__all__ = [
    "CHM_VMAX_M",
    "chm_panel",
    "draw_stand_outline",
    "read_rgb_window",
    "rgb_panel",
    "stretch_rgb",
]


def stretch_rgb(
    rgb: np.ndarray,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    gamma: float = 1.0,
):
    """Scale a NAIP RGB block to 0-1 for display.

    All three bands are normalised against one shared range rather than
    separately. Stretching each band to its own percentiles equalises the
    colour balance, which looks tidier across a strip of dates but is not what
    the sensor recorded, and on a scene that is almost entirely forest it also
    amplifies noise in the low-contrast blue band.

    NAIP is not atmospherically corrected, so a hazy acquisition reads blue.
    Over this module the 2014 and 2022 flights carry visibly more blue than
    2018 for that reason, not because anything changed in the forest. A caption
    for a multi-date strip should say so. The crown texture and the gap pattern,
    which is what these panels are shown for, survive the cast.

    ``gamma`` above 1 lightens the midtones, which helps on dark closed canopy.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size == 0:
        return arr
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    lo, hi = np.nanpercentile(arr, list(percentiles))
    out = (
        np.clip(arr, 0.0, 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6
        else np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    )
    if gamma != 1.0:
        out = np.power(out, 1.0 / gamma, dtype=np.float32)
    return out


def _decimated(
    window: Window, max_px: int | None
) -> tuple[tuple[int, int] | None, float]:
    """Return the out_shape and scale factor for reading ``window`` decimated.

    A whole mapping module is over a hundred megapixels, far more than any
    printed panel can show. Reading it decimated keeps an overview figure
    within memory and costs nothing visible at figure resolution.
    """
    height, width = int(window.height), int(window.width)
    if max_px is None or max(height, width) <= max_px:
        return None, 1.0
    factor = max(height, width) / float(max_px)
    return (max(1, int(height / factor)), max(1, int(width / factor))), factor


def read_rgb_window(
    path: Path | str,
    bounds: tuple[float, float, float, float],
    *,
    bands: tuple[int, int, int] = (1, 2, 3),
    max_px: int | None = 2400,
) -> tuple[np.ndarray, Any]:
    """Read three bands over map ``bounds`` and return ``(H, W, 3)`` with its transform.

    ``max_px`` caps the longer side of the returned block, decimating on read.
    The returned transform describes the decimated grid, so an overlay drawn
    with it still lands in the right place.
    """
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(
            Window(col_off=0, row_off=0, width=src.width, height=src.height)
        )
        out_shape, factor = _decimated(window, max_px)
        block = np.dstack(
            [src.read(b, window=window, out_shape=out_shape) for b in bands]
        )
        transform = src.window_transform(window)
    if factor != 1.0:
        transform = transform * transform.scale(factor, factor)
    return block, transform


def read_band_window(
    path: Path | str,
    bounds: tuple[float, float, float, float],
    *,
    band: int = 1,
    scale: float = 1.0,
    max_px: int | None = 2400,
) -> tuple[np.ndarray, Any]:
    """Read one band over map ``bounds``, nodata as NaN, scaled to metres.

    ``max_px`` caps the longer side, decimating on read, as in
    :func:`read_rgb_window`.
    """
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(
            Window(col_off=0, row_off=0, width=src.width, height=src.height)
        )
        out_shape, factor = _decimated(window, max_px)
        arr = src.read(band, window=window, out_shape=out_shape).astype(np.float32)
        transform = src.window_transform(window)
        nodata = src.nodatavals[band - 1]
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == np.float32(nodata), np.nan, arr)
    if factor != 1.0:
        transform = transform * transform.scale(factor, factor)
    return arr * np.float32(scale), transform


def _rings(geometry: BaseGeometry):
    """Yield every exterior and interior ring of a polygon or multipolygon."""
    parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        yield part.exterior
        yield from part.interiors


def draw_stand_outline(
    ax: Axes,
    geometry: BaseGeometry,
    transform: Any,
    *,
    color: str = "white",
    linewidth: float = 1.2,
    halo: bool = True,
    zorder: int = 5,
) -> None:
    """Draw a stand boundary on an image axis.

    Handles MultiPolygon and interior rings, unlike the legacy helper. Map
    coordinates are converted to the pixel coordinates that
    :func:`matplotlib.axes.Axes.imshow` uses.

    Args:
        ax: Axis already showing an array read with ``transform``.
        geometry: Stand geometry in the raster CRS.
        transform: Affine transform of the displayed block.
        color: Line colour.
        linewidth: Line width.
        halo: Draw a dark line underneath, so the outline stays visible over
            both bright bare ground and dark shadow.
        zorder: Drawing order.
    """
    import matplotlib.patheffects as pe

    # A stand often runs past the edge of the panel. Freeze the limits so the
    # outline is clipped to the image instead of shrinking it.
    frozen = bool(ax.images)
    if frozen:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

    effects = (
        [pe.withStroke(linewidth=linewidth + 1.4, foreground=INK)] if halo else None
    )
    inverse = ~transform
    for ring in _rings(geometry):
        xs, ys = ring.xy
        pixels = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
        ax.plot(
            [p[0] for p in pixels],
            [p[1] for p in pixels],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            path_effects=effects,
            zorder=zorder,
        )
    if frozen:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def rgb_panel(
    ax: Axes,
    path: Path | str,
    bounds: tuple[float, float, float, float],
    *,
    geometry: BaseGeometry | None = None,
    title: str | None = None,
    gamma: float = 1.0,
    max_px: int | None = 2400,
) -> Any:
    """Draw a NAIP true-colour panel, optionally with a stand outlined."""
    block, transform = read_rgb_window(path, bounds, max_px=max_px)
    ax.imshow(stretch_rgb(block, gamma=gamma), interpolation="nearest")
    if geometry is not None:
        draw_stand_outline(ax, geometry, transform)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return transform


def chm_panel(
    ax: Axes,
    path: Path | str,
    bounds: tuple[float, float, float, float],
    *,
    geometry: BaseGeometry | None = None,
    title: str | None = None,
    vmax: float = CHM_VMAX_M,
    scale: float = 1.0,
    cmap: str = "viridis",
    max_px: int | None = 2400,
):
    """Draw a canopy height panel, optionally with a stand outlined.

    Returns the image handle so a caller can attach one shared colourbar to a
    whole row rather than one per panel.
    """
    arr, transform = read_band_window(path, bounds, scale=scale, max_px=max_px)
    image = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    if geometry is not None:
        draw_stand_outline(ax, geometry, transform)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return image, transform


def padded_bounds(
    geometry: BaseGeometry,
    *,
    pad_fraction: float = 0.25,
    min_pad_m: float = 20.0,
) -> tuple[float, float, float, float]:
    """Bounds of a stand widened by a fraction of its longer side."""
    minx, miny, maxx, maxy = geometry.bounds
    pad = max(min_pad_m, pad_fraction * max(maxx - minx, maxy - miny))
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)
