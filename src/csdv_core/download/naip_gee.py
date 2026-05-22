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
from pathlib import Path

import click

from csdv_core.download._ee import (
    export_image_to_tif,
    initialize_ee,
    resolve_region,
)

logger = logging.getLogger(__name__)

NAIP_COLLECTION = "USDA/NAIP/DOQQ"


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

    logger.info("Exporting %s ...", description)
    export_image_to_tif(
        image,
        out_dir=out_dir,
        description=description,
        region=region,
        scale=scale,
        crs=crs,
    )
    return out_dir / f"{description}.tif"


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
