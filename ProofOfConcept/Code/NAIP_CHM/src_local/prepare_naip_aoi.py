"""Download NAIP AOI clips for inference.

Pulls 4-band NAIP imagery from Google Earth Engine, clipped to a square AOI
around a PoC site center, and writes a single GeoTIFF per (site, year) with a
date-tagged filename so that the upstream ``extract_doy_from_filename`` works.

Two modes:

1. ``--site SITE --years 2018 2021 2023`` — download a specific set of years.
2. ``--jobs-csv jobs.csv`` — read pre-built rows of (site, year, date, ...) and
   download each. Used by the SLURM driver.

Output filename pattern:
    NAIP_{SITE}_{YYYYMMDD}_aoi{km}km.tif
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import sys
from pathlib import Path

import click
import ee
import geemap

from site_centers import get_site  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _aoi(lon: float, lat: float, half_km: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat]).buffer(half_km * 1000.0).bounds()


def _median_date_tag(year_coll: ee.ImageCollection) -> str:
    dates = sorted(year_coll.aggregate_array("acq_ms").getInfo())
    if not dates:
        raise RuntimeError("Year collection is empty.")
    median = dates[len(dates) // 2]
    return dt.datetime.utcfromtimestamp(median / 1000).strftime("%Y%m%d")


def download_one(
    site_code: str,
    year: int,
    half_km: float,
    out_dir: Path,
    project: str,
) -> Path:
    """Download a single (site, year) NAIP AOI clip. Returns the output path."""
    center = get_site(site_code)
    aoi = _aoi(center.lon, center.lat, half_km)

    coll = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(aoi)
        .filter(ee.Filter.calendarRange(year, year, "year"))
    )
    coll = coll.map(lambda img: img.set({
        "band_count": img.bandNames().length(),
        "acq_ms": img.date().millis(),
    })).filter(ee.Filter.eq("band_count", 4))

    if coll.size().getInfo() == 0:
        raise RuntimeError(f"No 4-band NAIP imagery for {site_code} in {year}.")

    date_tag = _median_date_tag(coll)
    fname = f"NAIP_{site_code}_{date_tag}_aoi{int(half_km * 2)}km.tif"
    out_path = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info("Cached: %s", out_path)
        return out_path

    image = coll.select(["R", "G", "B", "N"]).mosaic().clip(aoi).toUint8()
    logger.info("Exporting %s ...", fname)
    geemap.ee_export_image(
        image,
        filename=str(out_path),
        scale=0.6,
        region=aoi,
        crs="EPSG:5070",
        file_per_band=False,
    )
    return out_path


@click.command()
@click.option("--site", default=None, help="Site code (SCBI, HARV, TALL, MLBS).")
@click.option("--years", default=None, multiple=True, type=int, help="One or more years.")
@click.option("--half-size-km", default=2.5, show_default=True, type=float)
@click.option("--out-dir", default="ProofOfConcept/Data/NAIP/AOI_Imagery", show_default=True, type=click.Path())
@click.option("--jobs-csv", default=None, type=click.Path(exists=True), help="Use a job list CSV instead of --site/--years.")
@click.option("--project", default="dyce-biomass", show_default=True)
def main(site, years, half_size_km, out_dir, jobs_csv, project) -> None:
    """Download NAIP AOI clips."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")

    out_root = Path(out_dir)

    if jobs_csv:
        with open(jobs_csv) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            site_code = row["site"]
            year = int(row["year"])
            site_dir = out_root / site_code
            try:
                download_one(site_code, year, half_size_km, site_dir, project)
            except Exception as e:
                logger.error("FAILED %s %s: %s", site_code, year, e)
        return

    if not site or not years:
        logger.error("Provide --site and --years, or --jobs-csv.")
        sys.exit(2)

    site_dir = out_root / site.upper()
    for year in years:
        try:
            download_one(site.upper(), year, half_size_km, site_dir, project)
        except Exception as e:
            logger.error("FAILED %s %s: %s", site, year, e)


if __name__ == "__main__":
    main()
