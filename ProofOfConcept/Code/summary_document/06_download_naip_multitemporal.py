"""
06_download_naip_multitemporal.py — Download one per-year NAIP mosaic for a site.

Queries GEE to discover which years have NAIP coverage for the target region,
then downloads each year as a separate 4-band (RGBN) mosaicked GeoTIFF.
Outputs land in Data/NAIP/Imagery/ as clean `NAIP_{SITE}_{YEAR}.tif` files
(no wxee timestamp suffix).  Already-downloaded years are skipped.

Usage
-----
  # Download all available years for SCBI (default):
  python 06_download_naip_multitemporal.py --site scbi

  # Custom bbox, explicit year range:
  python 06_download_naip_multitemporal.py \\
      --bbox -78.153 38.888 -78.135 38.899 \\
      --start-year 2010 --end-year 2022 --site-name SCBI

  # Different site:
  python 06_download_naip_multitemporal.py --site harv
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import ee
import wxee  # noqa: F401 — registers .wx accessor

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve()
CODE_ROOT = HERE.parents[1]     # ProofOfConcept/Code
POC_ROOT = HERE.parents[2]      # ProofOfConcept/
DEFAULT_OUT_DIR = POC_ROOT / "Data" / "NAIP" / "Imagery"


# ---------------------------------------------------------------------------
# importlib helper (same pattern as 05_zoom_pipeline.py)
# ---------------------------------------------------------------------------

def _import_script(alias: str, path: Path):
    """Import a Python script by file path (bypasses digit-prefix naming)."""
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def _query_available_years(
    collection: str,
    region: ee.Geometry,
    start_year: int,
    end_year: int,
) -> list[int]:
    """Return sorted list of years with data in collection over region."""
    coll = (
        ee.ImageCollection(collection)
        .filterBounds(region)
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
    )
    size = coll.size().getInfo()
    if size == 0:
        logger.warning("No images found for %d–%d", start_year, end_year)
        return []
    logger.info("Found %d tiles in %d–%d", size, start_year, end_year)

    timestamps = coll.aggregate_array("system:time_start").getInfo()
    years: set[int] = set()
    for ts in timestamps:
        if ts is not None and ts > 0:
            years.add(datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year)
    return sorted(years)


def _download_year(
    collection: str,
    region: ee.Geometry,
    year: int,
    out_dir: Path,
    site_name: str,
    scale: float,
    crs: str,
    bands: list[str],
    load_fn,
) -> Optional[Path]:
    """
    Download one year's NAIP mosaic.  Returns Path to output file, or None on failure.

    wxee appends a timestamp suffix to every output file.  This function renames
    the downloaded file to the clean convention `NAIP_{SITE_NAME}_{YEAR}.tif`.
    """
    target = out_dir / f"NAIP_{site_name}_{year}.tif"
    if target.exists():
        logger.info("  %d  already downloaded (%s)", year, target.name)
        return target

    logger.info("  %d  downloading …", year)
    description = f"NAIP_{site_name}_{year}_tmp"

    try:
        asset, _ = load_fn(
            collection=collection,
            region=region,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            bands=bands,
            mosaic=True,
        )
        asset.wx.to_tif(
            out_dir=str(out_dir),
            description=description,
            region=region,
            scale=scale,
            crs=crs,
        )
    except Exception as exc:
        logger.warning("  %d  download failed: %s", year, exc)
        return None

    # Find the file wxee wrote (it appends a timestamp)
    candidates = sorted(out_dir.glob(f"{description}*.tif"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.warning("  %d  no output file found after download", year)
        return None

    candidates[-1].rename(target)
    size_mb = target.stat().st_size / 1e6
    logger.info("  %d  saved → %s (%.1f MB)", year, target.name, size_mb)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--site", default=None,
    help="Named site shorthand (scbi, harv, tall, mlbs). Mutually exclusive with --bbox.",
)
@click.option(
    "--bbox", nargs=4, type=float, default=None, metavar="WEST SOUTH EAST NORTH",
    help="Bounding box in WGS84. Mutually exclusive with --site.",
)
@click.option(
    "--site-name", default=None,
    help="Uppercase label used in output filenames (e.g. SCBI). "
         "Inferred from --site if not given.",
)
@click.option("--start-year", default=2008, show_default=True, type=int)
@click.option("--end-year", default=2024, show_default=True, type=int)
@click.option(
    "--out-dir", default=None, type=click.Path(file_okay=False),
    help=f"Output directory (default: {DEFAULT_OUT_DIR})",
)
@click.option("--collection", default="USDA/NAIP/DOQQ", show_default=True)
@click.option("--scale", default=1.0, show_default=True, type=float)
@click.option("--crs", default="EPSG:5070", show_default=True)
@click.option("--bands", default="R,G,B,N", show_default=True)
@click.option("--project", default="dyce-biomass", show_default=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    site: Optional[str],
    bbox: Optional[tuple[float, float, float, float]],
    site_name: Optional[str],
    start_year: int,
    end_year: int,
    out_dir: Optional[str],
    collection: str,
    scale: float,
    crs: str,
    bands: str,
    project: str,
    verbose: bool,
) -> None:
    """Download one per-year NAIP mosaic per available year for a site."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import helpers from download_gee.py
    gee_mod = _import_script("download_gee", CODE_ROOT / "download_gee.py")
    _initialize_ee = gee_mod._initialize_ee
    _resolve_region = gee_mod._resolve_region
    _load_image_or_collection = gee_mod._load_image_or_collection

    # Resolve site name for filenames
    if site_name is None:
        site_name = site.upper() if site else "SITE"

    out_path = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    band_list = bands.split(",")

    logger.info("=== NAIP Multi-Temporal Download ===")
    logger.info("Site       : %s", site_name)
    logger.info("Collection : %s", collection)
    logger.info("Years      : %d – %d", start_year, end_year)
    logger.info("Output dir : %s", out_path.resolve())

    _initialize_ee(project)
    region = _resolve_region(bbox, site)

    # 1. Discover available years
    logger.info("Querying available years …")
    years = _query_available_years(collection, region, start_year, end_year)
    if not years:
        logger.error("No NAIP data found for this region and date range.")
        raise SystemExit(1)
    logger.info("Available years: %s", years)

    # 2. Download each year
    results: dict[int, Optional[Path]] = {}
    for year in years:
        results[year] = _download_year(
            collection=collection,
            region=region,
            year=year,
            out_dir=out_path,
            site_name=site_name,
            scale=scale,
            crs=crs,
            bands=band_list,
            load_fn=_load_image_or_collection,
        )

    # 3. Summary
    logger.info("=== Summary ===")
    ok = [y for y, p in results.items() if p is not None]
    fail = [y for y, p in results.items() if p is None]
    logger.info("Downloaded : %s", ok)
    if fail:
        logger.warning("Failed     : %s", fail)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
