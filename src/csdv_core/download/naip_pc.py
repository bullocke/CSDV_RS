"""csdv_core.download.naip_pc — NAIP from the Microsoft Planetary Computer.

The Planetary Computer serves NAIP as public cloud-optimised GeoTIFFs, one per
digital orthophoto quarter quadrangle, in the imagery's native UTM zone at its
native resolution. For a study module that is often the least troublesome way to
get the imagery: no account, no export quota, no reprojection to undo, and the
files land in the same CRS as a state's photo-interpretation delivery.

Two details matter enough to be worth stating.

Access tokens expire after about an hour, so a job spanning several dates must
sign its links date by date rather than all at once up front.

And every date has to land on one grid. A difference between two dates is only
meaningful pixel for pixel, so each year's quads are read through a warped
virtual raster targeted at a shared grid, which makes the mosaic step a paste
and keeps resampling to a single pass.

Earth Engine remains the route to imagery older than the Planetary Computer
archive; see :mod:`csdv_core.download.naip_gee`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
NAIP_ASSET = "image"
NAIP_BANDS = ("red", "green", "blue", "nir")

__all__ = [
    "NAIP_BANDS",
    "SNAP_M",
    "aoi_bbox_from_center",
    "available_years",
    "download_naip_year",
    "footprint_coverage",
    "naip_output_path",
    "resolution_tag",
    "NAIP_COLLECTION",
    "PC_STAC_URL",
    "GridSpecOut",
    "grid_from_bounds",
    "median_date_tag",
    "naip_mosaic",
    "search_naip_items",
    "snap_bounds",
]


class GridSpecOut(tuple):
    """``(transform, width, height)`` for a target grid, with named access."""

    __slots__ = ()

    def __new__(cls, transform: Any, width: int, height: int) -> GridSpecOut:
        return super().__new__(cls, (transform, int(width), int(height)))

    @property
    def transform(self) -> Any:
        return self[0]

    @property
    def width(self) -> int:
        return self[1]

    @property
    def height(self) -> int:
        return self[2]


def snap_bounds(
    bounds: tuple[float, float, float, float],
    *,
    snap_m: float,
    pad_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """Widen ``bounds`` by ``pad_m`` and snap outward to a multiple of ``snap_m``.

    Snapping to a common multiple of every resolution in play, 3 m for NAIP's
    0.6 m and 1.0 m products, makes the two grids nest so a coarse date and a
    fine date share pixel corners.
    """
    if snap_m <= 0:
        raise ValueError(f"snap_m must be positive, got {snap_m}")
    minx, miny, maxx, maxy = bounds
    return (
        math.floor((minx - pad_m) / snap_m) * snap_m,
        math.floor((miny - pad_m) / snap_m) * snap_m,
        math.ceil((maxx + pad_m) / snap_m) * snap_m,
        math.ceil((maxy + pad_m) / snap_m) * snap_m,
    )


def grid_from_bounds(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> GridSpecOut:
    """Return the transform and shape of a north-up grid covering ``bounds``."""
    from affine import Affine

    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounds {bounds} give an empty grid at {resolution} m")
    transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)
    return GridSpecOut(transform, width, height)


def median_date_tag(datetimes: Sequence[str | datetime]) -> str:
    """Return the median acquisition date of a set of items as ``YYYYMMDD``.

    A mapping module usually spans several quads, and the quads are not always
    flown on the same day. The canopy height model reads day of year from the
    filename, so the mosaic gets one representative date. Record every
    individual date in the run's provenance, because the model conditions on
    that single value for the whole mosaic.
    """
    if not datetimes:
        raise ValueError("No acquisition dates supplied")
    parsed = sorted(
        (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        for value in datetimes
    )
    return parsed[len(parsed) // 2].strftime("%Y%m%d")


def search_naip_items(
    bbox_4326: tuple[float, float, float, float],
    *,
    year: int,
    catalog: Any | None = None,
) -> list[Any]:
    """Find every NAIP quad intersecting ``bbox_4326`` in ``year``.

    A catalogue opened with the signing modifier returns items whose asset
    links already carry a token. Open it inside the per-year loop rather than
    once for a whole job, because the tokens expire.

    Raises:
        ValueError: If the year has no imagery over the box.
    """
    import planetary_computer
    import pystac_client

    if catalog is None:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
    search = catalog.search(
        collections=[NAIP_COLLECTION],
        bbox=list(bbox_4326),
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No NAIP imagery for {year} over {bbox_4326}")
    resolutions = {item.properties.get("gsd") for item in items}
    logger.info(
        "%d: %d NAIP quads, ground sample distance %s",
        year,
        len(items),
        sorted(r for r in resolutions if r is not None),
    )
    return items


def footprint_coverage(
    bbox_4326: tuple[float, float, float, float], footprints: Sequence[Any]
) -> float:
    """Fraction of ``bbox_4326`` covered by the union of ``footprints``.

    Areas are taken in degrees, which is close enough for a box a few
    kilometres across.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    aoi = box(*bbox_4326)
    if not footprints or aoi.area == 0:
        return 0.0
    covered = unary_union(list(footprints)).intersection(aoi)
    return float(covered.area / aoi.area)


def available_years(
    bbox_4326: tuple[float, float, float, float],
    *,
    catalog: Any | None = None,
) -> dict[int, float]:
    """Planetary Computer NAIP years over ``bbox_4326`` and the share they cover.

    One search without a date filter lists every quad that touches the box.
    Quads that only touch an edge add nothing to the coverage, so a year whose
    quads fall outside the box reports a share near zero.

    Returns:
        ``{year: covered fraction of the box}``, sorted by year.
    """
    import pystac_client
    from shapely.geometry import shape

    if catalog is None:
        catalog = pystac_client.Client.open(PC_STAC_URL)
    search = catalog.search(collections=[NAIP_COLLECTION], bbox=list(bbox_4326))
    by_year: dict[int, list[Any]] = {}
    for item in search.items():
        year = item.properties.get("naip:year") or item.datetime.year
        by_year.setdefault(int(year), []).append(shape(item.geometry))
    return {
        year: footprint_coverage(bbox_4326, geoms)
        for year, geoms in sorted(by_year.items())
    }


def naip_mosaic(
    items: Sequence[Any],
    out_path: Path | str,
    *,
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    resampling: str = "bilinear",
    compress: str = "DEFLATE",
    strip_rows: int = 1024,
) -> tuple[Path, dict[str, Any]]:
    """Mosaic NAIP quads onto a target grid and write a four-band GeoTIFF.

    Each quad is opened through a warped virtual raster already targeted at the
    output grid, so the merge is a paste and every pixel is resampled once.

    Args:
        items: STAC items, already signed.
        out_path: Destination GeoTIFF.
        dst_crs: Target CRS.
        dst_transform: Target affine transform.
        dst_width: Target width in pixels.
        dst_height: Target height in pixels.
        resampling: Resampling method for the warp.
        compress: GeoTIFF compression.
        strip_rows: Rows read from each warped quad at a time. A whole 0.6 m
            quad warped in one read is about half a gigabyte, enough to be
            killed on a small machine, so the paste proceeds in strips.

    Returns:
        ``(path, provenance)``. The provenance dictionary records the item
        identifiers, every acquisition date, the ground sample distances found
        and the grid written, which is what a later reader needs to judge
        whether two dates are comparable.

    Pixels no quad reaches stay zero in every band and are flagged in an
    internal dataset mask (0 = no imagery), so readers can tell a gap from
    black imagery.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method = Resampling[resampling]

    mosaic = np.zeros((4, dst_height, dst_width), dtype="uint8")
    written = np.zeros((dst_height, dst_width), dtype=bool)
    full = Window(col_off=0, row_off=0, width=dst_width, height=dst_height)

    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "3",
    }
    with rasterio.Env(**env):
        for item in items:
            href = item.assets[NAIP_ASSET].href
            with rasterio.open(href) as src:
                # Warp only into the part of the output this quad can reach.
                # Reading the whole grid once per quad would multiply the work
                # by the number of quads for no benefit.
                quad = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
                window = from_bounds(*quad, transform=dst_transform)
                window = window.round_offsets().round_lengths()
                try:
                    window = window.intersection(full)
                except rasterio.errors.WindowError:
                    logger.debug("%s falls outside the target grid", item.id)
                    continue
                if window.width < 1 or window.height < 1:
                    continue
                sub_transform = rasterio.windows.transform(window, dst_transform)
                width, height = int(window.width), int(window.height)
                row_off, col_off = int(window.row_off), int(window.col_off)
                cols = slice(col_off, col_off + width)
                with WarpedVRT(
                    src,
                    crs=dst_crs,
                    transform=sub_transform,
                    width=width,
                    height=height,
                    resampling=method,
                ) as vrt:
                    for start in range(0, height, strip_rows):
                        n_rows = min(strip_rows, height - start)
                        block = vrt.read(window=Window(0, start, width, n_rows))
                        rows = slice(row_off + start, row_off + start + n_rows)
                        covered = block.any(axis=0)
                        fill = covered & ~written[rows, cols]
                        target = mosaic[:, rows, cols]
                        target[:, fill] = block[:, fill]
                        written[rows, cols] |= covered

    coverage = float(written.mean())
    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 4,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(out_path, "w", **profile) as dst,
    ):
        dst.write(mosaic)
        dst.write_mask(written)
        dst.descriptions = NAIP_BANDS

    dates = [str(item.properties["datetime"]) for item in items]
    provenance = {
        "n_items": len(items),
        "item_ids": [item.id for item in items],
        "acq_dates": sorted(dates),
        "date_tag": median_date_tag(dates),
        "gsd": sorted({item.properties.get("gsd") for item in items} - {None}),
        "crs": str(dst_crs),
        "transform": list(dst_transform)[:6],
        "shape": [dst_height, dst_width],
        "coverage_fraction": coverage,
        "path": str(out_path),
    }
    if coverage < 0.999:
        logger.warning(
            "%s covers only %.3f of the target grid; check the search bounds",
            out_path.name,
            coverage,
        )
    logger.info("Wrote %s from %d quads", out_path.name, len(items))
    return out_path, provenance


# ---------------------------------------------------------------------------
# Site-year download
# ---------------------------------------------------------------------------

# 3 m is a whole number of pixels at both 0.6 m and 1.0 m, so grids built for
# different NAIP dates over one AOI share pixel corners.
SNAP_M = 3.0
_KM_PER_DEG_LAT = 111.32


def aoi_bbox_from_center(
    center_lonlat: tuple[float, float], half_km: float
) -> tuple[float, float, float, float]:
    """Return an EPSG:4326 box of half-width ``half_km`` around a point.

    The box is approximate (a spherical degree length), which is fine for a
    search box. The output grid is built in the quads' UTM CRS afterwards.
    """
    if half_km <= 0:
        raise ValueError(f"half_km must be positive, got {half_km}")
    lon, lat = center_lonlat
    dlat = half_km / _KM_PER_DEG_LAT
    dlon = half_km / (_KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def resolution_tag(gsd: float) -> str:
    """Format a ground sample distance for a filename, e.g. ``0p6m``."""
    return f"{gsd:g}".replace(".", "p") + "m"


def naip_output_path(
    out_root: Path | str, site: str, year: int, date_tag: str, gsd: float
) -> Path:
    """Return ``<out_root>/<site>/<year>/NAIP_<site>_<YYYYMMDD>_<res>.tif``.

    The eight-digit date token is required: the canopy height model reads day
    of year from the filename.
    """
    return (
        Path(out_root)
        / site
        / str(year)
        / f"NAIP_{site}_{date_tag}_{resolution_tag(gsd)}.tif"
    )


def _dominant(values: Sequence[Any], what: str, year: int) -> Any:
    """Return the most common value, warning when the set is mixed."""
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        raise ValueError(f"{year}: no {what} recorded on the NAIP items")
    if len(counts) > 1:
        logger.warning("%d: quads carry mixed %s %s", year, what, sorted(counts))
    return max(counts, key=lambda value: counts[value])


def _item_epsg(item: Any) -> int:
    epsg = item.properties.get("proj:epsg")
    if epsg is None:
        code = item.properties.get("proj:code", "")
        if code.upper().startswith("EPSG:"):
            epsg = int(code.split(":", 1)[1])
    if epsg is None:
        raise ValueError(f"NAIP item {item.id} carries no projection code")
    return int(epsg)


def download_naip_year(
    bbox_4326: tuple[float, float, float, float],
    *,
    site: str,
    year: int,
    out_root: Path | str,
    pad_m: float = 150.0,
    overwrite: bool = False,
    attempts: int = 3,
) -> Path:
    """Mosaic one year of Planetary Computer NAIP over a box at native GSD.

    The grid uses the quads' own UTM CRS and resolution, so the canopy height
    model sees imagery with the statistics it was trained on. The box is
    widened by ``pad_m`` before snapping because the model's outermost ~50 m
    is unreliable. A ``<stem>.provenance.json`` sidecar records the quads and
    every acquisition date.

    Returns:
        Path of the written (or already present) GeoTIFF.
    """
    from rasterio.warp import transform_bounds

    year_dir = Path(out_root) / site / str(year)
    existing = sorted(year_dir.glob("*.tif"))
    if existing and not overwrite:
        logger.info("%d: %s exists, skipping", year, existing[0].name)
        return existing[0]

    items = search_naip_items(bbox_4326, year=year)
    epsg = _dominant([_item_epsg(item) for item in items], "CRS", year)
    gsd = float(_dominant([item.properties.get("gsd") for item in items], "GSD", year))
    dst_crs = f"EPSG:{epsg}"
    utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_4326, densify_pts=21)
    grid = grid_from_bounds(
        snap_bounds(utm_bounds, snap_m=SNAP_M, pad_m=pad_m), resolution=gsd
    )
    dates = [str(item.properties["datetime"]) for item in items]
    out_path = naip_output_path(out_root, site, year, median_date_tag(dates), gsd)

    # Remote tile reads fail now and then. Retry with a fresh search, which
    # also refreshes the signed links, before giving up on the year.
    for attempt in range(1, attempts + 1):
        try:
            path, provenance = naip_mosaic(
                items,
                out_path,
                dst_crs=dst_crs,
                dst_transform=grid.transform,
                dst_width=grid.width,
                dst_height=grid.height,
            )
            break
        except Exception as exc:  # noqa: BLE001 - rasterio raises several types
            if attempt == attempts:
                raise
            logger.warning("%d: attempt %d failed (%s), retrying", year, attempt, exc)
            items = search_naip_items(bbox_4326, year=year)

    provenance.update(
        {"source": "planetary_computer", "site": site, "bbox_4326": list(bbox_4326)}
    )
    out_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return path


def _resolve_aoi(
    site: str, bbox: tuple[float, ...] | None, half_km: float
) -> tuple[float, float, float, float]:
    if bbox:
        return tuple(bbox)  # type: ignore[return-value]
    from csdv_core.config import load_sites

    try:
        entry = load_sites().get(site)
    except KeyError as exc:
        raise click.UsageError(
            f"Site {site!r} is not in sites.yaml; pass --bbox W S E N."
        ) from exc
    if entry.center_lonlat is None:
        raise click.UsageError(f"Site {site!r} has no center_lonlat; pass --bbox.")
    return aoi_bbox_from_center(entry.center_lonlat, half_km)


def _resolve_years(site: str, years: str | None) -> list[int]:
    if years:
        return [int(token) for token in years.split(",") if token.strip()]
    from csdv_core.config import load_sites

    try:
        listed = load_sites().get(site).naip.years
    except KeyError:
        listed = []
    if not listed:
        raise click.UsageError(f"No NAIP years listed for {site!r}; pass --years.")
    return list(listed)


@click.command("naip-pc")
@click.option("--site", required=True, help="Site label; output goes to naip/<site>/.")
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=None,
    help="W S E N in EPSG:4326. Default: sites.yaml center +/- --half-km.",
)
@click.option("--years", default=None, help="Comma-separated. Default: sites.yaml.")
@click.option("--half-km", default=2.5, show_default=True, type=float)
@click.option(
    "--pad-m",
    default=150.0,
    show_default=True,
    type=float,
    help="Buffer added before snapping; covers the CHM edge artefact.",
)
@click.option(
    "--out-root",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Default: $CSDV_DATA_ROOT/naip.",
)
@click.option("--overwrite", is_flag=True, default=False)
def cli(
    site: str,
    bbox: tuple[float, float, float, float] | None,
    years: str | None,
    half_km: float,
    pad_m: float,
    out_root: Path | None,
    overwrite: bool,
) -> None:
    """Download NAIP from the Planetary Computer, one mosaic per year.

    Needs outbound internet. On CHPC run it on a data transfer node.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    bbox_4326 = _resolve_aoi(site, bbox, half_km)
    year_list = _resolve_years(site, years)
    if out_root is None:
        from csdv_core.io.paths import project_paths

        out_root = project_paths().data_root / "naip"

    failed = []
    for year in year_list:
        try:
            path = download_naip_year(
                bbox_4326,
                site=site,
                year=year,
                out_root=out_root,
                pad_m=pad_m,
                overwrite=overwrite,
            )
            click.echo(str(path))
        except ValueError as exc:
            logger.error("%d: %s", year, exc)
            failed.append(year)
    if failed:
        raise click.ClickException(f"No mosaic written for {failed}")
