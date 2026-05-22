"""csdv_core.download.chm_model — Pre-computed NAIP-CHM rasters via GEE.

This module downloads the NAIP-CHM product (Morford et al., 2025) published
as an Earth Engine ImageCollection. To run the NAIP-CHM model on a fresh
NAIP quad, use :mod:`csdv_core.chm_inference.infer` instead.

Migration source: ``legacy/proof_of_concept/Code/download_gee.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from csdv_core.download._ee import (
    export_image_to_tif,
    initialize_ee,
    resolve_region,
)

logger = logging.getLogger(__name__)

NAIPCHM_COLLECTION = "projects/naip-chm/assets/conus-structure-model"


def download_naip_chm_raster(
    site: str,
    start_date: str,
    end_date: str,
    out_dir: Path | str,
    *,
    half_km: float = 2.5,
    scale: float = 1.0,
    crs: str = "EPSG:5070",
    project: str = "dyce-biomass",
) -> Path:
    """Download a NAIP-CHM mosaic for a site and date range.

    Args:
        site: Site code (case insensitive); resolved through ``sites.yaml``.
        start_date: ISO start date string.
        end_date: ISO end date string.
        out_dir: Output directory; created if missing.
        half_km: AOI half-side length in kilometres.
        scale: Output pixel size in CRS units (metres).
        crs: Output CRS.
        project: GEE project ID.

    Returns:
        Path to the written GeoTIFF.

    Notes:
        The product is stored as uint16 with a 0.01 m scale factor. Use
        :func:`csdv_core.io.raster.clip_and_convert_naip_chm` to convert to
        float32 metres.
    """
    import ee

    initialize_ee(project=project)
    region = resolve_region(site=site, half_km=half_km)

    coll = (
        ee.ImageCollection(NAIPCHM_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    size = coll.size().getInfo()
    if size == 0:
        raise RuntimeError(
            f"No NAIP-CHM imagery for {site} in {start_date}..{end_date}."
        )
    logger.info("NAIP-CHM collection size: %d images", size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    description = (
        f"NAIPCHM_{site.upper()}_{start_date[:4]}_{end_date[:4]}"
        f"_aoi{int(half_km * 2)}km"
    )

    image = coll.mosaic().set("system:time_start", 0)
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


@click.command("chm-model")
@click.option("--site", required=True)
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
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
    """Download a NAIP-CHM mosaic from Earth Engine."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    path = download_naip_chm_raster(
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
