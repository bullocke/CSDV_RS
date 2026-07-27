"""csdv_core.download.chm_model — Pre-computed NAIP-CHM rasters via GEE.

This module downloads the NAIP-CHM product (Morford et al., 2025) published
as an Earth Engine ImageCollection. To run the NAIP-CHM model on a fresh
NAIP quad, use :mod:`csdv_core.chm_inference.infer` instead.

Migration source: ``legacy/proof_of_concept/Code/download_gee.py``.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import click

from csdv_core.chm_inference.conditioning import (
    CONDITIONING_BASE_URL,
    REQUIRED_RASTERS,
    resolve_conditioning_dir,
)
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


def download_conditioning_rasters(
    out_dir: Path | str | None = None,
    *,
    base_url: str = CONDITIONING_BASE_URL,
    force: bool = False,
) -> Path:
    """Download the 5 static NAIP-CHM conditioning rasters over HTTP.

    These are CONUS-wide, EPSG:5070 rasters required by NAIP-CHM inference
    (see :mod:`csdv_core.chm_inference.conditioning`). They total roughly
    1.65 GB and only need to be fetched once.

    Args:
        out_dir: Destination directory. Defaults to
            :func:`csdv_core.chm_inference.conditioning.resolve_conditioning_dir`.
        base_url: HTTP directory holding the rasters.
        force: Re-download files that already exist.

    Returns:
        Path to the directory holding the rasters.

    Raises:
        FileNotFoundError: If any raster is missing after the download.
    """
    out_dir = Path(out_dir) if out_dir is not None else resolve_conditioning_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_RASTERS:
        target = out_dir / name
        if target.exists() and not force:
            logger.info("skip %s (already present)", name)
            continue
        url = f"{base_url}/{name}"
        tmp = target.with_suffix(target.suffix + ".part")
        logger.info("downloading %s -> %s", url, target)
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
            shutil.copyfileobj(response, fh)
        tmp.replace(target)

    missing = [name for name in REQUIRED_RASTERS if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Conditioning rasters still missing in {out_dir} after download: "
            f"{missing}."
        )
    logger.info(
        "All %d conditioning rasters present in %s", len(REQUIRED_RASTERS), out_dir
    )
    return out_dir


@click.command("conditioning")
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination directory; defaults to the resolved conditioning dir.",
)
@click.option("--base-url", default=CONDITIONING_BASE_URL, show_default=True)
@click.option(
    "--force", is_flag=True, default=False, help="Re-download existing files."
)
def conditioning_cli(out_dir: Path | None, base_url: str, force: bool) -> None:
    """Download the 5 static NAIP-CHM conditioning rasters (~1.65 GB)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    path = download_conditioning_rasters(out_dir, base_url=base_url, force=force)
    click.echo(str(path))
