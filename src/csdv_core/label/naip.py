"""csdv_core.label.naip: four-band NAIP for a window, from any source.

Inputs: a window on the 0.6 m lattice, a year, and one of three sources: the
unit's mosaics on disk (CHPC), Planetary Computer, or the USDA image service.
Outputs: a ``(4, rows, cols)`` uint8 array (red, green, blue, near-infrared)
and what was read (date, native resolution, item IDs).

Every source goes through :func:`warp_sources`, which averages imagery finer
than 0.6 m and interpolates coarser imagery, so the two machines produce the
same pixels from the same quads.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.label.lattice import GSD, Bounds, grid_transform

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"_(\d{8})_")
_RES_RE = re.compile(r"_(\d+)(?:p(\d+))?m(?:\.|_)")

__all__ = [
    "NaipRead",
    "local_mosaics",
    "read_naip",
    "read_planetary_computer",
    "read_usda",
    "warp_sources",
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
    """

    data: np.ndarray
    date: str
    source: str
    native_m: float | None
    item_ids: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return float(self.data.any(axis=0).mean())


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
    sources: list[str], epsg: int, bounds: Bounds
) -> tuple[np.ndarray, list[int]]:
    """Paste rasters onto the 0.6 m grid of ``bounds``, first source first.

    A source finer than the grid is averaged, a coarser one is interpolated
    (bilinear). Later sources fill only pixels the earlier ones left empty.

    Args:
        sources: File paths or URLs of four-band rasters.
        epsg: EPSG code of the lattice.
        bounds: Extent on the lattice.

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
            out[:, fill] = block[:, fill]
            filled.append(int(fill.sum()))
    return out, filled


def local_mosaics(data_root: Path, unit: str, year: int) -> list[Path]:
    """NAIP mosaics of a unit and year under ``data_root/naip/<unit>/<year>/``."""
    folder = Path(data_root) / "naip" / unit / str(year)
    return sorted(
        p for p in folder.glob("*.tif") if "_chm" not in p.name and p.is_file()
    )


def _read_local(
    data_root: Path, unit: str, year: int, epsg: int, bounds: Bounds
) -> NaipRead:
    paths = local_mosaics(data_root, unit, year)
    if not paths:
        raise FileNotFoundError(
            f"No NAIP mosaic for {unit} {year} under {data_root}/naip"
        )
    data, filled = warp_sources([str(p) for p in paths], epsg, bounds)
    best = paths[int(np.argmax(filled))]
    date = _DATE_RE.search(best.name)
    res = _RES_RE.search(best.name)
    native = None
    if res:
        native = float(f"{res[1]}.{res[2]}") if res[2] else float(res[1])
    return NaipRead(
        data=data,
        date=_iso(date[1]) if date else f"{year}-07-01",
        source="local",
        native_m=native,
        item_ids=tuple(p.name for p in paths),
    )


def _bbox_4326(epsg: int, bounds: Bounds) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    to_geo = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = to_geo.transform(
        [bounds[0], bounds[2], bounds[0], bounds[2]],
        [bounds[1], bounds[1], bounds[3], bounds[3]],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def read_planetary_computer(year: int, epsg: int, bounds: Bounds) -> NaipRead:
    """Read a window of program year ``year`` from Planetary Computer."""
    from csdv_core.download.naip_pc import search_naip_items

    items = search_naip_items(_bbox_4326(epsg, bounds), year=year)
    data, filled = warp_sources(
        [item.assets["image"].href for item in items], epsg, bounds
    )
    best: Any = items[int(np.argmax(filled))]
    gsd = best.properties.get("gsd")
    return NaipRead(
        data=data,
        date=best.datetime.strftime("%Y-%m-%d"),
        source="pc",
        native_m=float(gsd) if gsd is not None else None,
        item_ids=tuple(item.id for item, n in zip(items, filled, strict=True) if n),
    )


def read_usda(year: int, epsg: int, bounds: Bounds) -> NaipRead:
    """Read a window of ``year`` from the USDA image service.

    The service resamples on its side (bilinear), in one request.
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
    return NaipRead(
        data=np.ascontiguousarray(block[:4, :height, :width]).astype("uint8"),
        date=name.iso_date,
        source="rest",
        native_m=name.gsd,
        item_ids=tuple(str(q["Name"]) for q in quads),
    )


def read_naip(
    source: str,
    *,
    unit: str,
    year: int,
    epsg: int,
    bounds: Bounds,
    data_root: Path | None = None,
) -> NaipRead:
    """Read one year of a window from ``source`` (``local``, ``pc`` or ``rest``).

    Raises:
        ValueError: If ``source`` is unknown or has no imagery.
        FileNotFoundError: If a local mosaic is missing. A compute node has
            no internet, so a missing mosaic never falls back to a download.
    """
    if source == "local":
        if data_root is None:
            from csdv_core.io.paths import project_paths

            data_root = project_paths().data_root
        return _read_local(data_root, unit, year, epsg, bounds)
    if source == "pc":
        return read_planetary_computer(year, epsg, bounds)
    if source == "rest":
        return read_usda(year, epsg, bounds)
    raise ValueError(f"Unknown NAIP source {source!r}")
