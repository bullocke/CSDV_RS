"""List available 4-band NAIP years for a PoC site AOI.

Usage
-----
    python list_naip_years.py --site SCBI --half-size-km 2.5

Output is a CSV-style line per available year:
    SCBI,2018,20180815,3
where the columns are site, year, median acquisition date (YYYYMMDD), and the
number of contributing NAIP images.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

import click
import ee

from site_centers import get_site  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _aoi_from_center(lon: float, lat: float, half_size_km: float) -> ee.Geometry:
    """Return a rectangular AOI by buffering a point and taking bounds."""
    return ee.Geometry.Point([lon, lat]).buffer(half_size_km * 1000.0).bounds()


@click.command()
@click.option("--site", required=True, help="Site code, e.g. SCBI.")
@click.option("--half-size-km", default=2.5, show_default=True, type=float)
@click.option("--project", default="dyce-biomass", show_default=True)
def main(site: str, half_size_km: float, project: str) -> None:
    """Print available 4-band NAIP years for the site AOI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    center = get_site(site)
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")

    aoi = _aoi_from_center(center.lon, center.lat, half_size_km)
    coll = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(aoi)

    def _tag(img: ee.Image) -> ee.Image:
        return img.set({
            "band_count": img.bandNames().length(),
            "year": img.date().get("year"),
            "acq_ms": img.date().millis(),
        })

    tagged = coll.map(_tag).filter(ee.Filter.eq("band_count", 4))
    info = tagged.aggregate_array("year").getInfo()
    if not info:
        logger.warning("No 4-band NAIP imagery found at %s.", site)
        sys.exit(1)

    years = sorted(set(info))
    rows = []
    for year in years:
        yc = tagged.filter(ee.Filter.calendarRange(year, year, "year"))
        n = yc.size().getInfo()
        dates_ms = sorted(yc.aggregate_array("acq_ms").getInfo())
        median = dates_ms[len(dates_ms) // 2]
        median_tag = dt.datetime.utcfromtimestamp(median / 1000).strftime("%Y%m%d")
        rows.append((center.code, year, median_tag, n))

    click.echo("site,year,date,n_images")
    for r in rows:
        click.echo(f"{r[0]},{r[1]},{r[2]},{r[3]}")


if __name__ == "__main__":
    main()
