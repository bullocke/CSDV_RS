"""csdv_core.label.naip: four-band NAIP for a window, from any source.

Inputs: a window on the 0.6 m lattice, a year, and one of three sources: the
unit's mosaics on disk (CHPC), Planetary Computer, or the USDA image service.
Outputs: a ``(4, rows, cols)`` uint8 array (red, green, blue, near-infrared)
and what was read (date, native resolution, item IDs).

Every source goes through :func:`warp_sources`, which averages imagery finer
than 0.6 m and interpolates coarser imagery, so the two machines produce the
same pixels from the same quads.

The acquisition date of a window is the date of the quarter quad that fills
most of it. A unit's mosaic on disk is named after the median date of all its
quads, which can be months from the date of one quad, so the date of a window
read from a mosaic comes from the mosaic's ``.provenance.json`` sidecar
(:func:`provenance_date`).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.label.lattice import GSD, Bounds, grid_transform

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"_(\d{8})_")
_RES_RE = re.compile(r"_(\d+)(?:p(\d+))?m(?:\.|_)")
#: A quarter-quad name inside an item ID. Planetary Computer puts a state code
#: in front (``nh_m_4407152_nw_19_060_20180923_20190225``), the USDA service
#: does not (``m_4506921_nw_19_030_20250815``). The resolution token is one of
#: ``1``, ``.6``, ``.5``, ``h``, ``060`` and ``030``.
_ITEM_RE = re.compile(
    r"(?:^|_)m_(?P<quad>\d{7})_(?P<quarter>[ns][ew])_\d{1,2}_(?P<res>[^_]+)"
    r"_(?P<date>\d{8})"
)
#: Metres past its nominal edge within which a quad listed earlier is taken
#: to fill a window. Delivered quads overlap their neighbours by 300 m or
#: more, and the mosaic keeps the first quad that reaches a pixel. Half of
#: that width is used. On the 40 example windows of the scale test, 0, 150
#: and 300 m each gave the true date in 281 of 282 window-years.
QUAD_OVERLAP_M = 150.0
#: A window is flagged as holding two dates when the quads of the reported
#: date fill less than this share of it.
MIXED_DATE_SHARE = 0.75
_METRES_PER_DEGREE = 111_320.0
_warned_mosaics: set[str] = set()

__all__ = [
    "MIXED_DATE_SHARE",
    "QUAD_OVERLAP_M",
    "NaipRead",
    "QuadDate",
    "local_mosaics",
    "parse_item_id",
    "provenance_date",
    "read_naip",
    "read_planetary_computer",
    "read_usda",
    "warp_sources",
    "window_quad",
]


@dataclass(frozen=True)
class NaipRead:
    """One year of a window.

    Attributes:
        data: ``(4, rows, cols)`` uint8. Pixels without imagery are 0 in
            every band.
        date: Acquisition date (``YYYY-MM-DD``) of the quad that fills most
            of the window.
        source: ``local``, ``pc`` or ``rest``.
        native_m: Ground sample distance of that quad, metres, when known.
        item_ids: Quads or files read.
        date_rule: How the date was found. ``quad``: from the quads listed
            in a mosaic's sidecar. ``mosaic_name``: from the file name of a
            mosaic without a usable sidecar (the median date of the unit).
            ``item``: from the quad that filled most pixels of a remote read.
        date_share: Share of the window filled by quads flown on ``date``,
            when known.
        other_dates: Other dates of quads that fill part of the window.
        quad: Item ID of the quad that gave the date, when known.
    """

    data: np.ndarray
    date: str
    source: str
    native_m: float | None
    item_ids: tuple[str, ...]
    date_rule: str = ""
    date_share: float | None = None
    other_dates: tuple[str, ...] = ()
    quad: str = ""

    @property
    def coverage(self) -> float:
        return float(self.data.any(axis=0).mean())

    @property
    def mixed_dates(self) -> bool:
        """True when quads of another date fill a large part of the window."""
        return self.date_share is not None and self.date_share < MIXED_DATE_SHARE


@dataclass(frozen=True)
class QuadDate:
    """The quarter quad that fills most of a window, from a list of item IDs.

    Attributes:
        item_id: Item ID of that quad.
        date: Its acquisition date (``YYYY-MM-DD``).
        native_m: Its ground sample distance, metres, when the ID states it.
        date_share: Share of the window filled by quads flown on ``date``.
        other_dates: Other dates of quads that fill part of the window.
    """

    item_id: str
    date: str
    native_m: float | None
    date_share: float
    other_dates: tuple[str, ...]


def _iso(date8: str) -> str:
    return f"{date8[:4]}-{date8[4:6]}-{date8[6:]}"


def _paste(src: Any, epsg: int, bounds: Bounds) -> np.ndarray:
    """One raster resampled onto the 0.6 m grid of ``bounds`` (zeros outside it).

    The pixels are read into an array first and warped from there. A warp
    straight from the file would treat a fourth band flagged as alpha as
    transparency, and the mosaics written by ``csdv download`` flag their
    near-infrared band that way.
    """
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.errors import WindowError
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    width = round((bounds[2] - bounds[0]) / GSD)
    height = round((bounds[3] - bounds[1]) / GSD)
    out = np.zeros((4, height, width), dtype="uint8")
    dst_crs = CRS.from_epsg(epsg)
    same_crs = src.crs == dst_crs
    box = bounds if same_crs else transform_bounds(dst_crs, src.crs, *bounds, 21)
    res = abs(src.transform.a)
    pad = 4 * max(res, GSD)
    window = from_bounds(
        box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad, transform=src.transform
    )
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    try:
        window = window.intersection(Window(0, 0, src.width, src.height))
    except WindowError:
        return out
    data = src.read([1, 2, 3, 4], window=window)
    if not data.any():
        return out
    reproject(
        source=data,
        destination=out,
        src_transform=src.window_transform(window),
        src_crs=src.crs,
        dst_transform=grid_transform(bounds),
        dst_crs=dst_crs,
        resampling=Resampling.average if res < 0.9 * GSD else Resampling.bilinear,
    )
    return out


def warp_sources(
    sources: list[str],
    epsg: int,
    bounds: Bounds,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Paste rasters onto the 0.6 m grid of ``bounds``, first source first.

    A source finer than the grid is averaged, a coarser one is interpolated
    (bilinear). Later sources fill only pixels the earlier ones left empty.

    Args:
        sources: File paths or URLs of four-band rasters.
        epsg: EPSG code of the lattice.
        bounds: Extent on the lattice.
        keep: Pixels to fill, ``(rows, cols)`` bool. The others stay 0 and
            are not counted. Default: every pixel.

    Returns:
        The ``(4, rows, cols)`` uint8 mosaic, and the number of pixels each
        source filled.
    """
    import rasterio

    width = round((bounds[2] - bounds[0]) / GSD)
    height = round((bounds[3] - bounds[1]) / GSD)
    out = np.zeros((4, height, width), dtype="uint8")
    filled: list[int] = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for source in sources:
            empty = ~out.any(axis=0)
            if not empty.any():
                filled.append(0)
                continue
            with rasterio.open(source) as src:
                if src.count < 4:
                    logger.warning("%s has %d bands, skipped", source, src.count)
                    filled.append(0)
                    continue
                block = _paste(src, epsg, bounds)
            fill = empty & block.any(axis=0)
            if keep is not None:
                fill &= keep
            out[:, fill] = block[:, fill]
            filled.append(int(fill.sum()))
    return out, filled


def _token_gsd(token: str) -> float | None:
    """Ground sample distance stated by the resolution token of an item ID."""
    if re.fullmatch(r"\d{3}", token):
        return int(token) / 100.0
    if re.fullmatch(r"\d", token) or re.fullmatch(r"\.\d+", token):
        return float(token)
    return None


def parse_item_id(item_id: str) -> tuple[str, str, str, float | None] | None:
    """Quad code, quarter, date and resolution of a NAIP item ID.

    Returns:
        ``(quad, quarter, date, native_m)`` with the date as ``YYYY-MM-DD``,
        or None when ``item_id`` holds no quarter-quad name. ``native_m`` is
        None for a resolution token that is not a number (``h``).
    """
    match = _ITEM_RE.search(item_id)
    if match is None:
        return None
    return (
        match["quad"],
        match["quarter"],
        _iso(match["date"]),
        _token_gsd(match["res"]),
    )


def window_quad(
    item_ids: Sequence[str],
    epsg: int,
    bounds: Bounds,
    *,
    overlap_m: float = QUAD_OVERLAP_M,
    extent: Bounds | None = None,
    samples: int = 24,
) -> QuadDate | None:
    """The quarter quad that fills most of a window of a first-come mosaic.

    A mosaic written by ``csdv download`` pastes its quads in the order of
    the sidecar's ``item_ids``, and a pixel keeps the first quad that reaches
    it. Quads reach ``overlap_m`` past their nominal edge, so near a seam the
    earlier quad of the list fills ground of its neighbour. The same order is
    followed here on a grid of sample points, without reading any pixel.

    Args:
        item_ids: Item IDs in the order they were pasted.
        epsg: EPSG code of ``bounds``.
        bounds: The window.
        overlap_m: Metres a quad reaches past its nominal edge. 0 gives the
            nominal quarter quads.
        extent: Extent of the mosaic in the same CRS. Points outside it are
            left out, since the mosaic holds nothing there.
        samples: Sample points along each side of the window.

    Returns:
        None when no listed quad reaches the window.
    """
    from pyproj import Transformer

    from csdv_core.grid.cells import quarter_quad_bounds

    x0, y0, x1, y1 = bounds
    xs = x0 + (np.arange(samples) + 0.5) * (x1 - x0) / samples
    ys = y0 + (np.arange(samples) + 0.5) * (y1 - y0) / samples
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    if extent is not None:
        inside = (
            (gx >= extent[0])
            & (gx <= extent[2])
            & (gy >= extent[1])
            & (gy <= extent[3])
        )
        gx, gy = gx[inside], gy[inside]
    if gx.size == 0:
        return None
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(gx, gy)
    pad_y = overlap_m / _METRES_PER_DEGREE
    pad_x = pad_y / max(float(np.cos(np.radians(lat.mean()))), 0.1)
    free = np.ones(gx.size, dtype=bool)
    claims: list[tuple[int, str, str, float | None]] = []
    for item_id in item_ids:
        parsed = parse_item_id(item_id)
        if parsed is None:
            continue
        quad, quarter, date, gsd = parsed
        try:
            west, south, east, north = quarter_quad_bounds(quad, quarter)
        except ValueError:
            continue
        reach = (
            (lon >= west - pad_x)
            & (lon <= east + pad_x)
            & (lat >= south - pad_y)
            & (lat <= north + pad_y)
        )
        taken = reach & free
        if taken.any():
            claims.append((int(taken.sum()), item_id, date, gsd))
            free &= ~taken
    if not claims:
        return None
    total = sum(n for n, *_ in claims)
    count, item_id, date, gsd = max(claims, key=lambda claim: claim[0])
    same = sum(n for n, _, d, _ in claims if d == date)
    others = sorted({d for _, _, d, _ in claims if d != date})
    return QuadDate(item_id, date, gsd, round(same / total, 4), tuple(others))


def _mosaic_extent(sidecar: dict[str, Any], epsg: int) -> Bounds | None:
    """Extent of a mosaic from its sidecar, when it is on the window's CRS."""
    try:
        if int(str(sidecar["crs"]).rsplit(":", 1)[-1]) != epsg:
            return None
        a, _, c, _, e, f = (float(v) for v in sidecar["transform"][:6])
        rows, cols = (int(v) for v in sidecar["shape"])
    except (KeyError, TypeError, ValueError):
        return None
    return (c, f + rows * e, c + cols * a, f)


def provenance_date(mosaic: Path, epsg: int, bounds: Bounds) -> QuadDate | None:
    """Date of a window read from a unit's mosaic, from the mosaic's sidecar.

    Args:
        mosaic: The mosaic. Its sidecar is ``<stem>.provenance.json``.
        epsg: EPSG code of ``bounds``.
        bounds: The window.

    Returns:
        None when the sidecar is missing, cannot be read, or lists no quad
        that reaches the window.
    """
    path = Path(mosaic).with_suffix(".provenance.json")
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        item_ids = [str(item) for item in sidecar["item_ids"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return window_quad(item_ids, epsg, bounds, extent=_mosaic_extent(sidecar, epsg))


def local_mosaics(data_root: Path, unit: str, year: int) -> list[Path]:
    """NAIP mosaics of a unit and year under ``data_root/naip/<unit>/<year>/``."""
    folder = Path(data_root) / "naip" / unit / str(year)
    return sorted(
        p for p in folder.glob("*.tif") if "_chm" not in p.name and p.is_file()
    )


def _read_local(
    data_root: Path,
    unit: str,
    year: int,
    epsg: int,
    bounds: Bounds,
    keep: np.ndarray | None = None,
) -> NaipRead:
    paths = local_mosaics(data_root, unit, year)
    if not paths:
        raise FileNotFoundError(
            f"No NAIP mosaic for {unit} {year} under {data_root}/naip"
        )
    data, filled = warp_sources([str(p) for p in paths], epsg, bounds, keep)
    best = paths[int(np.argmax(filled))]
    date = _DATE_RE.search(best.name)
    res = _RES_RE.search(best.name)
    native = None
    if res:
        native = float(f"{res[1]}.{res[2]}") if res[2] else float(res[1])
    items = tuple(p.name for p in paths)
    quad = provenance_date(best, epsg, bounds)
    if quad is None:
        # The file name carries the median date of every quad of the unit.
        if str(best) not in _warned_mosaics:
            _warned_mosaics.add(str(best))
            logger.warning(
                "%s: no usable .provenance.json, so windows take the date in "
                "the file name (the unit's median date)",
                best.name,
            )
        return NaipRead(
            data=data,
            date=_iso(date[1]) if date else f"{year}-07-01",
            source="local",
            native_m=native,
            item_ids=items,
            date_rule="mosaic_name",
        )
    return NaipRead(
        data=data,
        date=quad.date,
        source="local",
        native_m=quad.native_m if quad.native_m is not None else native,
        item_ids=items,
        date_rule="quad",
        date_share=quad.date_share,
        other_dates=quad.other_dates,
        quad=quad.item_id,
    )


def _bbox_4326(epsg: int, bounds: Bounds) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    to_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = to_geo.transform(
        [bounds[0], bounds[2], bounds[0], bounds[2]],
        [bounds[1], bounds[1], bounds[3], bounds[3]],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def read_planetary_computer(
    year: int, epsg: int, bounds: Bounds, keep: np.ndarray | None = None
) -> NaipRead:
    """Read a window of program year ``year`` from Planetary Computer."""
    from csdv_core.download.naip_pc import search_naip_items

    items = search_naip_items(_bbox_4326(epsg, bounds), year=year)
    data, filled = warp_sources(
        [item.assets["image"].href for item in items], epsg, bounds, keep
    )
    best: Any = items[int(np.argmax(filled))]
    gsd = best.properties.get("gsd")
    date = best.datetime.strftime("%Y-%m-%d")
    dates = [item.datetime.strftime("%Y-%m-%d") for item in items]
    total = sum(filled)
    same = sum(n for n, d in zip(filled, dates, strict=True) if d == date)
    return NaipRead(
        data=data,
        date=date,
        source="pc",
        native_m=round(float(gsd), 2) if gsd is not None else None,
        item_ids=tuple(item.id for item, n in zip(items, filled, strict=True) if n),
        date_rule="item",
        date_share=round(same / total, 4) if total else None,
        other_dates=tuple(
            sorted({d for n, d in zip(filled, dates, strict=True) if n and d != date})
        ),
        quad=str(best.id),
    )


def read_usda(
    year: int, epsg: int, bounds: Bounds, keep: np.ndarray | None = None
) -> NaipRead:
    """Read a window of ``year`` from the USDA image service.

    The service resamples on its side (bilinear), in one request. The date
    is that of the first quad the catalogue lists for the window.
    """
    from csdv_core.download.naip_rest import _export_tile, find_quads, parse_quad_name

    quads = find_quads(_bbox_4326(epsg, bounds), year=year)
    if not quads:
        raise ValueError(f"No USDA NAIP quads for {year} over {bounds}")
    width = round((bounds[2] - bounds[0]) / GSD)
    height = round((bounds[3] - bounds[1]) / GSD)
    ids = ",".join(str(q["OBJECTID"]) for q in quads)
    params = {
        "bboxSR": str(epsg),
        "imageSR": str(epsg),
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
        "bbox": ",".join(f"{v:.3f}" for v in bounds),
        "size": f"{width},{height}",
        "mosaicRule": (
            f'{{"mosaicMethod":"esriMosaicLockRaster","lockRasterIds":[{ids}]}}'
        ),
    }
    block = _export_tile(params, 4, f"{year} {bounds}")
    if block.shape[0] < 4:
        raise ValueError(f"USDA returned {block.shape[0]} bands for {year}")
    name = parse_quad_name(quads[0]["Name"])
    data = np.ascontiguousarray(block[:4, :height, :width]).astype("uint8")
    if keep is not None:
        data[:, ~keep] = 0
    dates = {parse_quad_name(q["Name"]).iso_date for q in quads}
    return NaipRead(
        data=data,
        date=name.iso_date,
        source="rest",
        native_m=name.gsd,
        item_ids=tuple(str(q["Name"]) for q in quads),
        date_rule="item",
        other_dates=tuple(sorted(dates - {name.iso_date})),
        quad=str(quads[0]["Name"]),
    )


def read_naip(
    source: str,
    *,
    unit: str,
    year: int,
    epsg: int,
    bounds: Bounds,
    data_root: Path | None = None,
    keep: np.ndarray | None = None,
) -> NaipRead:
    """Read one year of a window from ``source`` (``local``, ``pc`` or ``rest``).

    Args:
        keep: Pixels to read, ``(rows, cols)`` bool on the grid of ``bounds``.
            The others come back as 0 (no imagery). Default: every pixel.

    Raises:
        ValueError: If ``source`` is unknown or has no imagery.
        FileNotFoundError: If a local mosaic is missing. A compute node has
            no internet, so a missing mosaic never falls back to a download.
    """
    if source == "local":
        if data_root is None:
            from csdv_core.io.paths import project_paths

            data_root = project_paths().data_root
        return _read_local(data_root, unit, year, epsg, bounds, keep)
    if source == "pc":
        return read_planetary_computer(year, epsg, bounds, keep)
    if source == "rest":
        return read_usda(year, epsg, bounds, keep)
    raise ValueError(f"Unknown NAIP source {source!r}")
