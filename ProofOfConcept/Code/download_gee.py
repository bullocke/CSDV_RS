"""
download_gee.py — Reusable Google Earth Engine downloader for the CSDV project.

Downloads a single GEE Image or a filtered/mosaicked ImageCollection as a
GeoTIFF using wxee. Supports named NEON/NAIP sites and all major data sources
used in this project.

Known GEE collection IDs
------------------------
NAIP RGBN imagery         : USDA/NAIP/DOQQ
NEON ALS canopy height    : projects/neon-prod-earthengine/assets/CHM/001
NEON RGB orthomosaic      : projects/neon-prod-earthengine/assets/RGB/001
NAIP-CHM (Morford 2025)   : projects/naip-chm/assets/conus-structure-model
NEON site boundaries      : projects/dyce-biomass/assets/NEON/Geometry/FieldSiteBoundaries

Usage examples
--------------
# NEON CHM for SCBI (single image by full asset path):
python download_gee.py \\
    --collection projects/neon-prod-earthengine/assets/CHM/001/2023_SCBI_6 \\
    --site scbi --scale 1 --out-dir data/neon/chm --description NEON_CHM_SCBI_2023

# NAIP RGBN for SCBI (ImageCollection, mosaic over region, most recent cycle):
python download_gee.py \\
    --collection USDA/NAIP/DOQQ \\
    --site scbi --start-date 2021-01-01 --end-date 2023-12-31 \\
    --bands R,G,B,N --mosaic --scale 1 \\
    --out-dir data/naip/imagery --description NAIP_SCBI_2022

# NAIP-CHM for SCBI (scale factor 100 converts UInt16 to meters):
python download_gee.py \\
    --collection projects/naip-chm/assets/conus-structure-model \\
    --site scbi --start-date 2021-01-01 --end-date 2023-12-31 \\
    --mosaic --scale 1 \\
    --out-dir data/naip/national_chm --description NAIPCHM_SCBI_2022

# Custom bbox (WGS84 decimal degrees: west south east north):
python download_gee.py \\
    --collection USDA/NAIP/DOQQ \\
    --bbox -78.153 38.888 -78.135 38.899 \\
    --start-date 2022-01-01 --end-date 2022-12-31 \\
    --bands R,G,B,N --mosaic --scale 1 \\
    --out-dir data/naip --description NAIP_custom
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import ee
import wxee  # noqa: F401 — registers .wx accessor on ee objects

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named site bounding boxes (WGS84: west, south, east, north)
# ---------------------------------------------------------------------------
SITE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "scbi": (-78.15303514106559, 38.88783134643515, -78.13483903510856, 38.89905430300793),
    "harv": (-72.244, 42.432, -72.224, 42.444),
    "tall": (-87.41, 32.93, -87.36, 32.97),
    "mlbs": (-80.53, 37.36, -80.49, 37.40),
}


def bbox_to_polygon(west: float, south: float, east: float, north: float) -> ee.Geometry:
    """Convert a bounding box to an EE Polygon geometry."""
    return ee.Geometry.Polygon(
        [[[west, north], [west, south], [east, south], [east, north]]]
    )


def _initialize_ee(project: str) -> None:
    """Initialize Earth Engine with the high-volume endpoint."""
    ee.Initialize(
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    logger.info("Earth Engine initialized (project=%s)", project)


def _resolve_region(
    bbox: Optional[tuple[float, float, float, float]],
    site: Optional[str],
) -> ee.Geometry:
    """Return an EE Geometry from either explicit bbox or a named site."""
    if bbox is not None:
        west, south, east, north = bbox
        logger.info("Using explicit bbox: W=%.4f S=%.4f E=%.4f N=%.4f", west, south, east, north)
        return bbox_to_polygon(west, south, east, north)
    if site is not None:
        site_lower = site.lower()
        if site_lower not in SITE_BBOXES:
            raise click.BadParameter(
                f"Unknown site '{site}'. Valid options: {', '.join(SITE_BBOXES)}",
                param_hint="--site",
            )
        coords = SITE_BBOXES[site_lower]
        logger.info("Using named site '%s': %s", site_lower, coords)
        return bbox_to_polygon(*coords)
    raise click.UsageError("Provide either --site or --bbox to define the region.")


def _load_image_or_collection(
    collection: str,
    region: ee.Geometry,
    start_date: Optional[str],
    end_date: Optional[str],
    bands: Optional[list[str]],
    mosaic: bool,
) -> tuple[ee.Image | ee.ImageCollection, bool]:
    """
    Load a GEE asset as either an Image or ImageCollection.

    Returns (asset, is_single_image).  For collections with --mosaic, returns a
    mosaicked Image so the caller can treat it uniformly.
    """
    # Heuristic: if the ID has no slashes beyond the 3rd segment, try as single Image first.
    # Users can always supply a full image path for single-image downloads.
    try:
        img = ee.Image(collection)
        # Trigger a lightweight server-side call to validate it exists as an Image.
        img.bandNames().getInfo()
        logger.info("Loaded as single Image: %s", collection)
        if bands:
            img = img.select(bands)
        return img, True
    except Exception:
        pass  # Fall through to ImageCollection

    logger.info("Loading as ImageCollection: %s", collection)
    coll = ee.ImageCollection(collection).filterBounds(region)

    if start_date and end_date:
        coll = coll.filterDate(start_date, end_date)
        logger.info("Date filter: %s to %s", start_date, end_date)

    size = coll.size().getInfo()
    logger.info("Collection size after filters: %d images", size)
    if size == 0:
        raise RuntimeError(
            f"ImageCollection '{collection}' is empty after filtering. "
            "Check --start-date / --end-date and --site / --bbox."
        )

    if bands:
        coll = coll.select(bands)

    if mosaic:
        logger.info("Mosaicking collection to single image")
        # wxee requires system:time_start; mosaic() drops it, so set a dummy value
        mosaicked = coll.mosaic().set("system:time_start", 0)
        return mosaicked, True

    return coll, False


@click.command()
@click.option(
    "--collection", "-c",
    required=True,
    help="GEE Image or ImageCollection asset ID (e.g. 'USDA/NAIP/DOQQ').",
)
@click.option(
    "--out-dir", "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write output GeoTIFF(s).",
)
@click.option(
    "--description", "-d",
    default="gee_download",
    show_default=True,
    help="Output filename prefix (without extension).",
)
@click.option(
    "--site",
    default=None,
    help=(
        f"Named site shorthand. Options: {', '.join(SITE_BBOXES)}. "
        "Mutually exclusive with --bbox."
    ),
)
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=None,
    metavar="WEST SOUTH EAST NORTH",
    help="Bounding box in WGS84 decimal degrees. Mutually exclusive with --site.",
)
@click.option("--start-date", default=None, help="Start date ISO string (e.g. 2021-01-01).")
@click.option("--end-date", default=None, help="End date ISO string (e.g. 2023-12-31).")
@click.option(
    "--scale",
    default=1.0,
    show_default=True,
    type=float,
    help="Output spatial resolution in CRS units (meters).",
)
@click.option(
    "--crs",
    default="EPSG:5070",
    show_default=True,
    help="Output coordinate reference system.",
)
@click.option(
    "--bands",
    default=None,
    help="Comma-separated band names to select (e.g. 'R,G,B,N'). Selects all if omitted.",
)
@click.option(
    "--mosaic",
    is_flag=True,
    default=False,
    help="Mosaic an ImageCollection into a single image before downloading.",
)
@click.option(
    "--project",
    default="dyce-biomass",
    show_default=True,
    help="Google Earth Engine project ID.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG logging.",
)
def main(
    collection: str,
    out_dir: str,
    description: str,
    site: Optional[str],
    bbox: Optional[tuple[float, float, float, float]],
    start_date: Optional[str],
    end_date: Optional[str],
    scale: float,
    crs: str,
    bands: Optional[str],
    mosaic: bool,
    project: str,
    verbose: bool,
) -> None:
    """Download a GEE Image or ImageCollection as GeoTIFF(s).

    Provide either --site (named location) or --bbox (custom bounding box) to
    define the spatial region. For ImageCollections, use --start-date /
    --end-date to filter by acquisition date, and --mosaic to combine
    overlapping tiles into a single output file.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    band_list: Optional[list[str]] = bands.split(",") if bands else None

    logger.info("=== CSDV GEE Downloader ===")
    logger.info("Collection : %s", collection)
    logger.info("Output dir : %s", out_path.resolve())
    logger.info("Description: %s", description)
    logger.info("Scale      : %s m", scale)
    logger.info("CRS        : %s", crs)

    _initialize_ee(project)
    region = _resolve_region(bbox, site)
    asset, is_image = _load_image_or_collection(
        collection, region, start_date, end_date, band_list, mosaic
    )

    if is_image:
        logger.info("Downloading image → %s/%s.tif", out_path, description)
        files = asset.wx.to_tif(  # type: ignore[attr-defined]
            out_dir=str(out_path),
            description=description,
            region=region,
            scale=scale,
            crs=crs,
        )
        logger.info("Downloaded: %s", files)
    else:
        # ImageCollection without --mosaic: download each image separately
        logger.info("Downloading collection images individually")
        files = asset.wx.to_tif(  # type: ignore[attr-defined]
            out_dir=str(out_path),
            prefix=f"{description}_",
            region=region,
            scale=scale,
            crs=crs,
        )
        logger.info("Downloaded %d file(s)", len(files) if hasattr(files, "__len__") else 1)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
