"""csdv_core.download.naip_gee — NAIP RGBN acquisition via Earth Engine / wxee.

Downloads 4-band NAIP DOQQ imagery for a configured site and date range,
mosaics overlapping tiles, and writes a single GeoTIFF with a date-tagged
filename suitable for downstream NAIP-CHM inference (which extracts DOY
from the filename).

Migration source: ``legacy/proof_of_concept/Code/download_gee.py``.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import shutil
import tempfile
from pathlib import Path

import click

from csdv_core.download._ee import (
    export_image_to_tif,
    initialize_ee,
    resolve_region,
)

# Earth Engine's getDownloadURL hard cap is 48 MiB. Keep each tile
# request well under that to leave headroom for response overhead.
_EE_SOFT_MAX_BYTES = 32 * 1024 * 1024
_NAIP_BYTES_PER_PIXEL = 4  # 4 bands x uint8

logger = logging.getLogger(__name__)

NAIP_COLLECTION = "USDA/NAIP/DOQQ"


def _aoi_bounds_in_crs(region, crs: str) -> tuple[float, float, float, float]:
    """Return AOI (minx, miny, maxx, maxy) in the target projected CRS."""

    coords = region.bounds().coordinates().getInfo()[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    from pyproj import Transformer

    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tx.transform(lons, lats)
    return min(xs), min(ys), max(xs), max(ys)


def _required_tiles(width_m: float, height_m: float, scale: float) -> int:
    """How many tiles per side keep each request under the EE limit."""
    nx = max(1, math.ceil(width_m / scale))
    ny = max(1, math.ceil(height_m / scale))
    total_bytes = nx * ny * _NAIP_BYTES_PER_PIXEL
    if total_bytes <= _EE_SOFT_MAX_BYTES:
        return 1
    return math.ceil(math.sqrt(total_bytes / _EE_SOFT_MAX_BYTES))


def _export_tiled(
    image,
    *,
    out_path: Path,
    description: str,
    region,
    scale: float,
    crs: str,
) -> Path:
    """Download an EE image, tiling if the AOI exceeds the EE request limit.

    Tiles are downloaded into a temp dir and merged with rasterio.
    """
    import ee
    import rasterio
    from rasterio.merge import merge as rio_merge

    minx, miny, maxx, maxy = _aoi_bounds_in_crs(region, crs)
    n = _required_tiles(maxx - minx, maxy - miny, scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if n == 1:
        logger.info("AOI fits in a single EE request; downloading whole image.")
        export_image_to_tif(
            image,
            out_dir=out_path.parent,
            description=description,
            region=region,
            scale=scale,
            crs=crs,
        )
        return out_path

    logger.info(
        "AOI exceeds EE request limit; splitting into %dx%d=%d tiles",
        n,
        n,
        n * n,
    )
    dx = (maxx - minx) / n
    dy = (maxy - miny) / n
    # Co-locate the tile dir next to the output so it lives on the same
    # filesystem and stays visible if a step fails. We only clean it up
    # on success; on failure it remains for inspection.
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_path.stem}_tiles_", dir=out_path.parent))
    logger.info("Tile directory: %s", tmp)
    tile_paths: list[Path] = []
    success = False
    try:
        for i in range(n):
            for j in range(n):
                t_minx = minx + i * dx
                t_maxx = minx + (i + 1) * dx
                t_miny = miny + j * dy
                t_maxy = miny + (j + 1) * dy
                tile_region = ee.Geometry.Rectangle(
                    [t_minx, t_miny, t_maxx, t_maxy],
                    proj=crs,
                    geodesic=False,
                )
                tile_desc = f"{description}_tile_{i}_{j}"
                logger.info("  tile %d/%d (%s)", len(tile_paths) + 1, n * n, tile_desc)
                before = set(tmp.glob("*.tif"))
                export_image_to_tif(
                    image,
                    out_dir=tmp,
                    description=tile_desc,
                    region=tile_region,
                    scale=scale,
                    crs=crs,
                )
                after = set(tmp.glob("*.tif"))
                new_files = sorted(after - before)
                if not new_files:
                    raise RuntimeError(
                        f"wxee export produced no .tif for tile {tile_desc}. "
                        f"Temp dir contents: {sorted(p.name for p in tmp.iterdir())}"
                    )
                if len(new_files) > 1:
                    logger.warning(
                        "tile %s produced %d files; using %s",
                        tile_desc,
                        len(new_files),
                        new_files[0].name,
                    )
                tile_paths.append(new_files[0])

        logger.info("Merging %d tiles -> %s", len(tile_paths), out_path)
        srcs = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, transform = rio_merge(srcs)
            profile = srcs[0].profile
        finally:
            for s in srcs:
                s.close()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
        success = True
    finally:
        if success:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            logger.warning("Leaving tile directory in place for inspection: %s", tmp)
    return out_path


def _median_date_tag(coll) -> str:  # type: ignore[no-untyped-def]
    """Return ``YYYYMMDD`` for the median acquisition timestamp in ``coll``."""
    dates = sorted(coll.aggregate_array("acq_ms").getInfo())
    if not dates:
        raise RuntimeError("NAIP ImageCollection is empty.")
    median_ms = dates[len(dates) // 2]
    return dt.datetime.utcfromtimestamp(median_ms / 1000).strftime("%Y%m%d")


def download_naip_rgbn(
    site: str,
    start_date: str,
    end_date: str,
    out_dir: Path | str,
    *,
    half_km: float = 2.5,
    scale: float = 1.0,
    crs: str = "EPSG:5070",
    project: str = "dyce-biomass",
    require_4band: bool = True,
) -> Path:
    """Download a NAIP RGBN mosaic for a site and date range.

    Args:
        site: Site code (case insensitive); resolved through ``sites.yaml``.
        start_date: ISO start date string (e.g. ``"2021-01-01"``).
        end_date: ISO end date string (inclusive of NAIP's filter semantics).
        out_dir: Output directory; created if missing.
        half_km: AOI half-side length in kilometres.
        scale: Output pixel size in CRS units (metres).
        crs: Output CRS.
        project: GEE project ID.
        require_4band: If True, drop images missing the N band.

    Returns:
        Path to the written GeoTIFF.
    """
    import ee

    initialize_ee(project=project)
    region = resolve_region(site=site, half_km=half_km)

    coll = (
        ee.ImageCollection(NAIP_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    coll = coll.map(
        lambda img: img.set(
            {
                "band_count": img.bandNames().length(),
                "acq_ms": img.date().millis(),
            }
        )
    )
    if require_4band:
        coll = coll.filter(ee.Filter.eq("band_count", 4))

    size = coll.size().getInfo()
    if size == 0:
        raise RuntimeError(
            f"No NAIP imagery for {site} in {start_date}..{end_date} "
            f"(require_4band={require_4band})."
        )
    logger.info("NAIP collection size: %d images", size)

    date_tag = _median_date_tag(coll)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    description = f"NAIP_{site.upper()}_{date_tag}_aoi{int(half_km * 2)}km"

    image = (
        coll.select(["R", "G", "B", "N"]).mosaic().set("system:time_start", 0).toUint8()
    )

    out_path = out_dir / f"{description}.tif"
    logger.info("Exporting %s ...", description)
    _export_tiled(
        image,
        out_path=out_path,
        description=description,
        region=region,
        scale=scale,
        crs=crs,
    )
    return out_path
    # return out_dir / f"{description}.tif"


@click.command("naip-gee")
@click.option("--site", required=True, help="Site code (SCBI, HARV, ...).")
@click.option("--start-date", required=True, help="ISO start date.")
@click.option("--end-date", required=True, help="ISO end date.")
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--half-km", default=2.5, show_default=True, type=float)
@click.option("--scale", default=1.0, show_default=True, type=float)
@click.option("--crs", default="EPSG:5070", show_default=True)
@click.option("--project", default="dyce-biomass", show_default=True)
def cli(
    site: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
    half_km: float,
    scale: float,
    crs: str,
    project: str,
) -> None:
    """Download NAIP RGBN for a site / date range."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    path = download_naip_rgbn(
        site=site,
        start_date=start_date,
        end_date=end_date,
        out_dir=out_dir,
        half_km=half_km,
        scale=scale,
        crs=crs,
        project=project,
    )
    click.echo(str(path))
