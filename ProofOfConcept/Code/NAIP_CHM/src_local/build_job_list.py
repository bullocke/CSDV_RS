"""Build a jobs.csv enumerating (site, year) inference tasks for SLURM.

Queries GEE to enumerate all available 4-band NAIP years for each requested
site, and writes one row per (site, year). Each row contains the expected
NAIP AOI filename and the expected CHM output path so the SLURM template can
work entirely from environment variables and array indices.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path

import click
import ee

from site_centers import SITES, get_site  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _aoi(lon: float, lat: float, half_km: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat]).buffer(half_km * 1000.0).bounds()


@click.command()
@click.option("--sites", multiple=True, default=("SCBI",), show_default=True, help="One or more site codes.")
@click.option("--half-size-km", default=2.5, show_default=True, type=float)
@click.option("--aoi-dir", required=True, type=click.Path(path_type=Path), help="Where downloaded NAIP AOIs will live.")
@click.option("--chm-dir", required=True, type=click.Path(path_type=Path), help="Where predicted CHMs will be written (one subdir per site).")
@click.option("--jobs-csv", default="jobs.csv", show_default=True, type=click.Path(path_type=Path))
@click.option("--years", multiple=True, type=int, default=(), help="Restrict to these years (default: all available).")
@click.option("--project", default="dyce-biomass", show_default=True)
def main(sites, half_size_km, aoi_dir, chm_dir, jobs_csv, years, project) -> None:
    """Write a jobs.csv with one row per (site, year) inference task."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")

    requested_years = set(years) if years else None
    half_tag = int(half_size_km * 2)

    rows = []
    for code in sites:
        center = get_site(code)
        aoi = _aoi(center.lon, center.lat, half_size_km)
        coll = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(aoi)
        coll = coll.map(lambda img: img.set({
            "band_count": img.bandNames().length(),
            "year": img.date().get("year"),
            "acq_ms": img.date().millis(),
        })).filter(ee.Filter.eq("band_count", 4))

        avail = sorted(set(coll.aggregate_array("year").getInfo()))
        if requested_years is not None:
            avail = [y for y in avail if y in requested_years]
        logger.info("%s: years=%s", code, avail)

        for year in avail:
            yc = coll.filter(ee.Filter.calendarRange(year, year, "year"))
            dates = sorted(yc.aggregate_array("acq_ms").getInfo())
            median = dates[len(dates) // 2]
            date_tag = dt.datetime.utcfromtimestamp(median / 1000).strftime("%Y%m%d")
            naip_name = f"NAIP_{code}_{date_tag}_aoi{half_tag}km.tif"
            chm_subdir = chm_dir / code / f"{date_tag}"
            rows.append({
                "site": code,
                "year": year,
                "date": date_tag,
                "half_size_km": half_size_km,
                "naip_path": str(aoi_dir / code / naip_name),
                "chm_out_dir": str(chm_subdir),
            })

    jobs_csv = Path(jobs_csv)
    jobs_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site", "year", "date", "half_size_km", "naip_path", "chm_out_dir"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d job rows to %s", len(rows), jobs_csv)


if __name__ == "__main__":
    main()
