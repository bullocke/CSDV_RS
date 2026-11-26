"""csdv_core.view.tiles: map tiles of the derived layers, as PNG bytes.

The rasters of a run are small (about 1200 by 1500 cells at 9.6 m), so a
layer is derived once at full resolution (:mod:`csdv_core.view.derive`) and
kept in memory. A tile samples it with nearest neighbour: the tile's pixel
centres are projected to the raster's CRS on a coarse grid and interpolated,
which is accurate to well under a cell and takes a few milliseconds.

Thresholding the full-resolution cells, and not the averaged probability
overviews of the files, keeps small patches at their true size at every zoom.

The 3 m quicklooks (NAIP, canopy height) are too large to hold in memory.
Their tiles read only the window under the tile (:func:`_under_tile`).
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from csdv_core.review.imagery import tile_bounds_3857
from csdv_core.view import derive
from csdv_core.view.catalog import RunInfo, interval_years

logger = logging.getLogger(__name__)

TILE_SIZE = 256
#: Points per side of the grid on which a tile is projected.
_GRID = 9
#: Probabilities below this (percent) are not drawn.
PROB_MIN = 10
#: Years at the ends of the date colour scale, the same at every site.
YEAR_RANGE = (2010, 2025)

#: Colour of each agent code. This is the palette of the shared figures
#: (analysis/vlm_labeling/2_scale_test/share_figures/common.py), checked for
#: colour-vision deficiency. The colour table inside the exported type rasters
#: is csdv_core.pairmodel.predict.AGENT_COLOURS.
AGENT_COLOURS: dict[int, str] = {
    1: "#e0702e",
    2: "#1ba6b5",
    3: "#a3201c",
    4: "#cfb500",
    5: "#4b55c9",
    6: "#b04a9c",
}
AGENT_LABELS: dict[int, str] = {
    1: "Harvest",
    2: "Wind",
    3: "Fire",
    4: "Insects / disease",
    5: "Conversion",
    6: "Other",
}
LAYERS = ("date", "type", "prob")

__all__ = [
    "AGENT_COLOURS",
    "AGENT_LABELS",
    "LAYERS",
    "Layer",
    "chm_change_tile",
    "chm_tile",
    "image_tile",
    "layer_of",
    "legend",
    "point_values",
    "render_tile",
    "sample_tile",
    "year_colour",
]

_lock = threading.RLock()
_local = threading.local()
#: Derived layers by (run, kind, interval, percent, forest, file time), the
#: least recently used first. One layer is under 2 MB.
_layers: dict[tuple, Layer] = {}
_MAX_LAYERS = 32


@dataclass(frozen=True)
class Stack:
    """The bands of a raster with its grid."""

    data: np.ndarray  # (bands, rows, cols) uint8
    transform: tuple[float, ...]  # affine coefficients a, b, c, d, e, f
    epsg: int


@dataclass(frozen=True)
class Layer:
    """One derived layer on its grid."""

    data: np.ndarray  # (rows, cols) uint8
    transform: tuple[float, ...]
    epsg: int
    kind: str  # "date", "type" or "prob"


@lru_cache(maxsize=10)
def _stack(path: str, version: float) -> Stack:
    import rasterio

    with rasterio.open(path) as src:
        data = src.read()
        epsg = src.crs.to_epsg()
        transform = tuple(src.transform)[:6]
    logger.debug("Read %s %s", Path(path).name, data.shape)
    return Stack(data, transform, epsg)


def _read(path: Path) -> Stack:
    return _stack(str(path), path.stat().st_mtime)


def _prob_path(run: RunInfo, forest: bool) -> Path:
    name = "prob_forest" if forest and run.has_forest else "prob"
    return run.map_dir / f"{name}_{run.unit}.tif"


def _band(run: RunInfo, interval: str) -> int | None:
    if interval == "all":
        return None
    try:
        return run.intervals.index(interval)
    except ValueError:
        raise FileNotFoundError(f"{run.run} has no interval {interval}") from None


def _build_layer(
    run: RunInfo, kind: str, interval: str, percent: int, forest: bool
) -> Layer:
    if run.family == "di":
        return _di_layer(run, kind, interval)
    band = _band(run, interval)
    prob = _read(_prob_path(run, forest))
    threshold = percent / 100
    if kind == "date":
        years = [interval_years(i)[1] for i in run.intervals]
        data = derive.date_layer(prob.data, years, threshold, band)
    elif kind == "type":
        agent = _read(run.map_dir / f"agent_{run.unit}.tif")
        data = derive.type_layer(prob.data, agent.data, threshold, band)
    else:
        data = derive.prob_layer(prob.data, band)
    return Layer(np.ascontiguousarray(data), prob.transform, prob.epsg, kind)


def _di_layer(run: RunInfo, kind: str, interval: str) -> Layer:
    if kind == "date":
        onset = _read(run.map_dir / "onset_year_u8.tif")
        year = None if interval == "all" else interval_years(interval)[1]
        data = derive.onset_layer(onset.data[0], year)
        return Layer(data, onset.transform, onset.epsg, kind)
    # The DI export holds the highest probability over intervals only.
    prob = _read(run.map_dir / "max_probability_u8.tif")
    return Layer(prob.data[0], prob.transform, prob.epsg, kind)


def layer_of(
    run: RunInfo,
    kind: str,
    interval: str = "all",
    percent: int = 50,
    forest: bool = False,
) -> Layer:
    """A derived layer of a run, from the cache when it was asked for before.

    Args:
        run: The run (a pair-model run or the DI baseline of a unit).
        kind: ``date``, ``type`` or ``prob``.
        interval: ``all`` or the name of one interval.
        percent: Threshold in percent. Ignored by ``prob`` and by the DI.
        forest: Use the probability masked to forest, where the run has it.
    """
    if kind not in run.layers:
        raise FileNotFoundError(f"{run.run} has no {kind} layer")
    if kind == "prob" or run.family == "di":
        percent = 0
    key = (run.run, kind, interval, int(percent), bool(forest), run.version)
    with _lock:  # a new layer is asked for by many tiles at once
        layer = _layers.pop(key, None)
        if layer is None:
            layer = _build_layer(run, kind, interval, int(percent), bool(forest))
        _layers[key] = layer  # the newest entry is the last one
        while len(_layers) > _MAX_LAYERS:
            _layers.pop(next(iter(_layers)))
        return layer


def _transformer(src_epsg: int | str, dst_epsg: int | str):
    """A pyproj transformer of this thread (they are not shared safely)."""
    from pyproj import Transformer

    cache = getattr(_local, "transformers", None)
    if cache is None:
        cache = _local.transformers = {}
    key = (src_epsg, dst_epsg)
    if key not in cache:
        cache[key] = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return cache[key]


def _stretch(grid: np.ndarray, size: int) -> np.ndarray:
    """Bilinear interpolation of a square grid to ``size`` pixel centres."""
    k = grid.shape[0]
    t = (np.arange(size) + 0.5) / size * (k - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, k - 2)
    w = t - i0
    rows = grid[i0] * (1 - w)[:, None] + grid[i0 + 1] * w[:, None]
    return rows[:, i0] * (1 - w) + rows[:, i0 + 1] * w


def _tile_grid(
    crs: int | str, transform: Sequence[float], z: int, x: int, y: int
) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of a raster under a coarse grid over an XYZ tile.

    Both are ``(_GRID, _GRID)`` and fractional, from the tile's top left
    corner to its bottom right corner.
    """
    left, bottom, right, top = tile_bounds_3857(z, x, y)
    gx, gy = np.meshgrid(
        np.linspace(left, right, _GRID), np.linspace(top, bottom, _GRID)
    )
    ux, uy = _transformer(3857, crs).transform(gx, gy)
    a, b, c, d, e, f = transform
    det = a * e - b * d
    col = (e * (ux - c) - b * (uy - f)) / det
    row = (a * (uy - f) - d * (ux - c)) / det
    return col, row


def sample_tile(
    layer: Layer, z: int, x: int, y: int, size: int = TILE_SIZE
) -> tuple[np.ndarray, np.ndarray] | None:
    """The cells of a layer under the pixels of an XYZ tile.

    Returns:
        ``(values, inside)``, both ``(size, size)``, or None when the tile
        lies outside the layer.
    """
    col, row = _tile_grid(layer.epsg, layer.transform, z, x, y)
    rows, cols = layer.data.shape
    if col.max() < 0 or col.min() >= cols or row.max() < 0 or row.min() >= rows:
        return None
    ci = np.floor(_stretch(col, size)).astype(np.intp)
    ri = np.floor(_stretch(row, size)).astype(np.intp)
    inside = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
    values = np.zeros((size, size), dtype=layer.data.dtype)
    values[inside] = layer.data[ri[inside], ci[inside]]
    return values, inside


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _hex(rgb: Sequence[int]) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb)


def year_colour(year: int) -> tuple[int, int, int]:
    """Colour of a year on the date scale: early yellow to late purple."""
    from matplotlib import colormaps

    first, last = YEAR_RANGE
    share = min(max((year - first) / (last - first), 0.0), 1.0)
    r, g, b, _ = colormaps["plasma_r"](0.08 + 0.92 * share)
    return int(255 * r), int(255 * g), int(255 * b)


@lru_cache(maxsize=1)
def _luts() -> dict[str, np.ndarray]:
    """RGBA lookup table of each layer kind, indexed by cell value."""
    from csdv_core.review.tiles import _probability_lut

    date = np.zeros((256, 4), dtype=np.uint8)
    for code in range(1, derive.NODATA):
        date[code] = (*year_colour(derive.YEAR_BASE + code), 255)
    agent = np.zeros((256, 4), dtype=np.uint8)
    for code, colour in AGENT_COLOURS.items():
        agent[code] = (*_hex_rgb(colour), 255)
    prob = np.zeros((256, 4), dtype=np.uint8)
    for code, colour in _probability_lut().items():
        prob[code] = (*colour[:3], 255 if PROB_MIN <= code <= 100 else 0)
    return {"date": date, "type": agent, "prob": prob}


def render_tile(layer: Layer, z: int, x: int, y: int) -> bytes | None:
    """A PNG tile of a layer, or None when nothing is drawn on it."""
    sampled = sample_tile(layer, z, x, y)
    if sampled is None:
        return None
    values, inside = sampled
    rgba = _luts()[layer.kind][values]
    rgba[~inside] = 0
    return _png(rgba) if rgba[..., 3].any() else None


def _cell(stack: Stack, lon: float, lat: float) -> tuple[int, int] | None:
    ux, uy = _transformer(4326, stack.epsg).transform(lon, lat)
    a, b, c, d, e, f = stack.transform
    det = a * e - b * d
    col = int(np.floor((e * (ux - c) - b * (uy - f)) / det))
    row = int(np.floor((a * (uy - f) - d * (ux - c)) / det))
    _, rows, cols = stack.data.shape
    return (row, col) if 0 <= row < rows and 0 <= col < cols else None


def _values(stack: Stack, cell: tuple[int, int] | None) -> list[int | None]:
    if cell is None:
        return [None] * stack.data.shape[0]
    column = stack.data[:, cell[0], cell[1]]
    return [None if v == derive.NODATA else int(v) for v in column]


def point_values(run: RunInfo, lon: float, lat: float) -> dict:
    """What a run says at a position: values per interval, None where no data."""
    if run.family == "di":
        onset = _read(run.map_dir / "onset_year_u8.tif")
        code = _values(onset, _cell(onset, lon, lat))[0]
        out: dict = {"onset_year": derive.YEAR_BASE + code if code else None}
        path = run.map_dir / "max_probability_u8.tif"
        if path.exists():
            prob = _read(path)
            out["max_prob"] = _values(prob, _cell(prob, lon, lat))[0]
        return out
    prob = _read(_prob_path(run, forest=False))
    cell = _cell(prob, lon, lat)
    out = {"intervals": list(run.intervals), "prob": _values(prob, cell)}
    if "type" in run.layers:
        agent = _read(run.map_dir / f"agent_{run.unit}.tif")
        out["agent"] = _values(agent, cell)
    if run.has_forest:
        out["prob_forest"] = _values(_read(_prob_path(run, forest=True)), cell)
    return out


def legend() -> dict:
    """Colours of the page legend, the same ones the tiles use."""
    first, last = YEAR_RANGE
    return {
        "years": {str(y): _hex(year_colour(y)) for y in range(first, last + 1)},
        "agents": [
            {"code": code, "label": AGENT_LABELS[code], "colour": AGENT_COLOURS[code]}
            for code in sorted(AGENT_COLOURS)
        ],
        "prob": [_hex(_luts()["prob"][p][:3]) for p in (10, 25, 40, 55, 70, 85, 100)],
        "prob_min": PROB_MIN,
    }


def _under_tile(
    path: Path, z: int, x: int, y: int, *, smooth: bool = False, masked: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """The pixels of a raster file under an XYZ tile.

    Only the window under the tile is read, at about the tile's resolution, so
    GDAL takes it from the overview that fits. This is several times faster
    than a warped read of the same tile.

    Args:
        path: A raster with overviews.
        z: Zoom of the tile.
        x: Column of the tile.
        y: Row of the tile.
        smooth: Read with bilinear resampling (imagery). Otherwise nearest.
        masked: Also read the dataset mask and count masked pixels as outside.

    Returns:
        ``(values, inside)`` with shapes ``(bands, size, size)`` and
        ``(size, size)``, or None when the tile lies outside the raster.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        col, row = _tile_grid(src.crs.to_string(), tuple(src.transform)[:6], z, x, y)
        c0, c1 = max(int(np.floor(col.min())), 0), min(
            int(np.ceil(col.max())), src.width
        )
        r0, r1 = max(int(np.floor(row.min())), 0), min(
            int(np.ceil(row.max())), src.height
        )
        if c1 <= c0 or r1 <= r0:
            return None
        # Raster pixels per tile pixel. Never read finer than the file.
        step = max((col.max() - col.min()) / TILE_SIZE, 1.0)
        width = max(int(round((c1 - c0) / step)), 1)
        height = max(int(round((r1 - r0) / step)), 1)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(
            window=window,
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear if smooth else Resampling.nearest,
        )
        mask = None
        if masked:
            mask = src.dataset_mask(window=window, out_shape=(height, width))
    ci = np.floor((_stretch(col, TILE_SIZE) - c0) * width / (c1 - c0)).astype(np.intp)
    ri = np.floor((_stretch(row, TILE_SIZE) - r0) * height / (r1 - r0)).astype(np.intp)
    inside = (ci >= 0) & (ci < width) & (ri >= 0) & (ri < height)
    ci, ri = np.clip(ci, 0, width - 1), np.clip(ri, 0, height - 1)
    if mask is not None:
        inside &= mask[ri, ci] > 0
    return data[:, ri, ci], inside


def _png(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def image_tile(path: Path, z: int, x: int, y: int) -> tuple[bytes, str] | None:
    """A tile of an RGB quicklook: JPEG where it is full, PNG at the edges.

    Returns:
        The image and its media type, or None outside the quicklook.
    """
    from PIL import Image

    got = _under_tile(path, z, x, y, smooth=True, masked=True)
    if got is None or not got[1].any():
        return None
    values, inside = got
    rgb = np.ascontiguousarray(np.moveaxis(values[:3], 0, -1))
    if inside.all():
        buffer = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buffer, "JPEG", quality=85)
        return buffer.getvalue(), "image/jpeg"
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    return _png(np.dstack([rgb, alpha])), "image/png"


@lru_cache(maxsize=2)
def _review_lut(name: str) -> np.ndarray:
    """A colour table of the review app as an array indexed by cell value."""
    from csdv_core.review import tiles as review_tiles

    table = {"chm": review_tiles._chm_lut, "change": review_tiles._change_lut}[name]()
    lut = np.zeros((256, 4), dtype=np.uint8)
    for code, colour in table.items():
        lut[code] = colour
    return lut


def chm_tile(path: Path, z: int, x: int, y: int) -> bytes | None:
    """A tile of a canopy height quicklook on the canopy colour ramp."""
    got = _under_tile(path, z, x, y)
    if got is None:
        return None
    values, inside = got
    rgba = _review_lut("chm")[values[0]]
    rgba[~inside] = 0
    return _png(rgba) if rgba[..., 3].any() else None


def chm_change_tile(before: Path, after: Path, z: int, x: int, y: int) -> bytes | None:
    """Height change ``after - before``: red for loss and blue for gain.

    The colours and limits are those of
    :func:`csdv_core.review.tiles.chm_change_tile`.
    """
    from csdv_core.grid.export import CHM_STEP_M
    from csdv_core.review.tiles import DCHM_LIMIT_M

    a, b = _under_tile(before, z, x, y), _under_tile(after, z, x, y)
    if a is None or b is None:
        return None
    h0, h1 = a[0][0].astype("float32"), b[0][0].astype("float32")
    bad = (a[0][0] == derive.NODATA) | (b[0][0] == derive.NODATA) | ~a[1] | ~b[1]
    delta = (h1 - h0) * CHM_STEP_M
    scaled = (np.clip(delta, -DCHM_LIMIT_M, DCHM_LIMIT_M) / DCHM_LIMIT_M + 1) / 2
    code = np.rint(scaled * (derive.NODATA - 1)).astype(np.uint8)
    code[bad] = derive.NODATA
    rgba = _review_lut("change")[code]
    return _png(rgba) if rgba[..., 3].any() else None
