"""csdv_core.review.tiles: map tiles of the exported rasters, as PNG bytes.

The review server renders the small exported rasters itself with rio-tiler,
so the app needs no second tile server and works wherever the page is served
from. Every function returns None for a tile outside the raster.

- NAIP quicklooks (3 m RGB with a mask) as they are.
- CHM quicklooks (uint8, value x 0.25 m, 255 nodata) on the canopy colour
  ramp from 0 to :data:`CHM_VMAX_M`.
- Height change between two CHM quicklooks, red for loss and blue for gain,
  transparent below :data:`DCHM_MIN_M`. It is computed per tile, so nothing
  is cached on disk.
- The onset raster with its own colour table, with "no detection" transparent
  so it can lie over the imagery.
- The highest loss probability on a magma ramp, transparent below 10%.

All rasters are read with nearest-neighbour resampling, so 19.2 m cells and
3 m pixels stay sharp at high zoom.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.models import ImageData

from csdv_core.grid.export import CHM_STEP_M, NODATA_U8

TILE_SIZE = 256
CHM_VMAX_M = 30.0
DCHM_LIMIT_M = 15.0
DCHM_MIN_M = 2.0
PROB_MIN = 10

__all__ = [
    "chm_change_tile",
    "chm_tile",
    "onset_tile",
    "probability_tile",
    "rgb_tile",
]

Lut = dict[int, tuple[int, int, int, int]]


def _read(path: Path, z: int, x: int, y: int) -> ImageData | None:
    try:
        with Reader(str(path)) as src:
            return src.tile(x, y, z, tilesize=TILE_SIZE)
    except TileOutsideBounds:
        return None


def _rgba(cmap_name: str, values: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps

    return (np.asarray(colormaps[cmap_name](values)) * 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _chm_lut() -> Lut:
    from csdv_core.viz.relief import canopy_cmap

    cmap = canopy_cmap()
    lut: Lut = {}
    for code in range(NODATA_U8):
        r, g, b, _ = cmap(min(code * CHM_STEP_M / CHM_VMAX_M, 1.0))
        lut[code] = (int(255 * r), int(255 * g), int(255 * b), 255)
    lut[NODATA_U8] = (0, 0, 0, 0)
    return lut


@lru_cache(maxsize=1)
def _change_lut() -> Lut:
    """Codes 0 to 254 for -15 to +15 m, 255 transparent."""
    codes = np.arange(NODATA_U8)
    delta = (codes / (NODATA_U8 - 1) * 2 - 1) * DCHM_LIMIT_M
    colours = _rgba("RdBu", codes / (NODATA_U8 - 1))
    lut: Lut = {}
    for code, d, c in zip(codes, delta, colours, strict=True):
        alpha = 0 if abs(d) < DCHM_MIN_M else 230
        lut[int(code)] = (int(c[0]), int(c[1]), int(c[2]), alpha)
    lut[NODATA_U8] = (0, 0, 0, 0)
    return lut


@lru_cache(maxsize=1)
def _probability_lut() -> Lut:
    colours = _rgba("magma", np.linspace(0.2, 1.0, 101))
    lut: Lut = {
        p: (int(c[0]), int(c[1]), int(c[2]), 0 if p < PROB_MIN else 200)
        for p, c in enumerate(colours)
    }
    for code in range(101, 256):
        lut[code] = (0, 0, 0, 0)
    return lut


def rgb_tile(path: Path, z: int, x: int, y: int) -> bytes | None:
    """A NAIP quicklook tile."""
    img = _read(path, z, x, y)
    return None if img is None else img.render(img_format="PNG")


def chm_tile(path: Path, z: int, x: int, y: int) -> bytes | None:
    """A CHM quicklook tile on the canopy colour ramp."""
    img = _read(path, z, x, y)
    if img is None:
        return None
    return img.render(img_format="PNG", colormap=_chm_lut(), add_mask=False)


def chm_change_tile(before: Path, after: Path, z: int, x: int, y: int) -> bytes | None:
    """Height change ``after - before`` of two CHM quicklooks."""
    a, b = _read(before, z, x, y), _read(after, z, x, y)
    if a is None or b is None:
        return None
    h0 = a.data[0].astype("float32")
    h1 = b.data[0].astype("float32")
    bad = (h0 == NODATA_U8) | (h1 == NODATA_U8)
    delta = (h1 - h0) * CHM_STEP_M
    scaled = (np.clip(delta, -DCHM_LIMIT_M, DCHM_LIMIT_M) / DCHM_LIMIT_M + 1) / 2
    code = np.rint(scaled * (NODATA_U8 - 1)).astype(np.uint8)
    code[bad] = NODATA_U8
    out = ImageData(code[np.newaxis], bounds=a.bounds, crs=a.crs)
    return out.render(img_format="PNG", colormap=_change_lut(), add_mask=False)


def onset_tile(path: Path, z: int, x: int, y: int) -> bytes | None:
    """The onset raster in its own colours, with no detection transparent."""
    with Reader(str(path)) as src:
        table = dict(src.colormap or {})
        try:
            img = src.tile(x, y, z, tilesize=TILE_SIZE)
        except TileOutsideBounds:
            return None
    table[0] = (0, 0, 0, 0)
    table[NODATA_U8] = (0, 0, 0, 0)
    lut = {int(k): tuple(int(c) for c in v) for k, v in table.items()}
    for code in range(256):
        lut.setdefault(code, (0, 0, 0, 0))
    return img.render(img_format="PNG", colormap=lut, add_mask=False)


def probability_tile(path: Path, z: int, x: int, y: int) -> bytes | None:
    """The highest loss probability (percent), transparent below 10%."""
    img = _read(path, z, x, y)
    if img is None:
        return None
    return img.render(img_format="PNG", colormap=_probability_lut(), add_mask=False)
