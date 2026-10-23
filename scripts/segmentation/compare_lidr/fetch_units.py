"""Cache a CHM chip around every NEON field unit, and the units' field summaries.

Each chip covers the 20 x 20 m unit plus a 50 m buffer, enough for the largest
crown radius cap to resolve crowns whose tree top lies in the unit.

``naip``  the national NAIP-CHM (0.6 m), read over HTTP from the tile whose
          nominal quarter-quad cell holds the unit centroid.
``neon``  the NEON AOP lidar CHM (DP3.30015.001, 1 m) from the Earth Engine
          asset ``projects/neon-prod-earthengine/assets/CHM/001``, the image
          of the lidar year nearest the field survey.

Also writes ``tables/neon_units.parquet``: unit attributes from refAGB (field
AGBD, split, biome) plus field canopy-tree counts, height and crown diameter.

Run::

    export REFAGB_METHODS_DIR=<refAGB methods-paper repo>
    P=.micromamba/envs/CSDV/bin/python
    $P scripts/segmentation/compare_lidr/fetch_units.py naip     # ~5 min
    $P scripts/segmentation/compare_lidr/fetch_units.py neon     # Earth Engine
"""

from __future__ import annotations

import argparse
import time

import _lib as L
import numpy as np
import pandas as pd
import rasterio

log = L.setup_logging("fetch_units")

BUFFER_M = 50.0
INDEX_CSV = L.PATHS.data_root / "naip_chm_national" / "index.csv"


def unit_table() -> pd.DataFrame:
    """Unit attributes plus field canopy-tree summaries, cached as Parquet."""
    out = L.TABLE_DIR / "neon_units.parquet"
    if out.exists():
        return pd.read_parquet(out)
    units = L.neon_units()
    field = pd.concat(
        [L.field_canopy_trees(site, grp) for site, grp in units.groupby("site")],
        ignore_index=True,
    )
    table = units.drop(columns="geometry").merge(field, on="unit_id", how="left")
    L.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    log.info("Wrote %s (%d units)", out, len(table))
    return table


def assign_naip_tiles(units: pd.DataFrame) -> pd.Series:
    """CHM URL of the tile whose nominal cell holds each unit centroid."""
    from csdv_core.download.naip_chm_http import fetch_index
    from csdv_core.national.inventory import (
        drop_duplicate_quarter_quads,
        quarter_quad_bounds,
        read_index,
    )

    inventory = L.PATHS.results_root / "national_crowns" / "inventory.parquet"
    if inventory.exists():
        cols = ["chm_url", "cell_west", "cell_south", "cell_east", "cell_north"]
        cells = pd.read_parquet(inventory, columns=cols)
    else:
        if not INDEX_CSV.exists():
            fetch_index(INDEX_CSV.parent)
        index, _ = drop_duplicate_quarter_quads(
            read_index(INDEX_CSV).drop(columns="geo")
        )
        cells = pd.concat(
            [index[["chm_url"]].reset_index(drop=True),
             quarter_quad_bounds(index["quarter_quad"]).reset_index(drop=True)],
            axis=1,
        )  # fmt: skip
    out = {}
    for uid, lon, lat in zip(units["unit_id"], units["lon"], units["lat"], strict=True):
        hit = cells[
            (cells.cell_west <= lon) & (lon < cells.cell_east)
            & (cells.cell_south <= lat) & (lat < cells.cell_north)
        ]  # fmt: skip
        out[uid] = hit.iloc[0].chm_url if len(hit) else None
    return pd.Series(out, name="chm_url")


def fetch_naip() -> None:
    from rasterio.windows import from_bounds
    from rasterio.windows import transform as win_transform

    from csdv_core.national.census import VSICURL_OPTIONS

    units = L.neon_units()
    urls = assign_naip_tiles(units)
    log.info("%d of %d units have a tile", urls.notna().sum(), len(urls))
    out_dir = L.DATA_DIR / "units" / "naip"
    out_dir.mkdir(parents=True, exist_ok=True)
    units = units.assign(chm_url=units["unit_id"].map(urls)).dropna(subset=["chm_url"])
    for url, grp in units.groupby("chm_url"):
        todo = [u for u in grp["unit_id"] if not L.unit_chip("naip", u).exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        with rasterio.Env(**VSICURL_OPTIONS), rasterio.open("/vsicurl/" + url) as src:
            polys = grp[grp["unit_id"].isin(todo)].to_crs(src.crs)
            for uid, poly in zip(polys["unit_id"], polys.geometry, strict=True):
                x0, y0, x1, y1 = poly.bounds
                win = (
                    from_bounds(x0 - BUFFER_M, y0 - BUFFER_M, x1 + BUFFER_M,
                                y1 + BUFFER_M, src.transform)
                    .round_offsets()
                    .round_lengths()
                )  # fmt: skip
                data = src.read(1, window=win, boundless=True, fill_value=65535)
                profile = {
                    "driver": "GTiff",
                    "width": data.shape[1],
                    "height": data.shape[0],
                    "count": 1,
                    "dtype": "uint16",
                    "crs": src.crs,
                    "transform": win_transform(win, src.transform),
                    "nodata": 65535,
                    "compress": "deflate",
                }
                with rasterio.open(L.unit_chip("naip", uid), "w", **profile) as dst:
                    dst.write(data, 1)
                    dst.scales = (L.NAIP_SCALE,)
                    dst.update_tags(chm_url=url, unit_id=uid)
        log.info(
            "%s: %d units in %.1f s",
            url.rsplit("/", 1)[-1],
            len(todo),
            time.perf_counter() - t0,
        )


def fetch_neon() -> None:
    import ee
    from csdv_core.download.ee_auth import initialize_ee

    initialize_ee()
    units = L.neon_units()
    table = unit_table().set_index("unit_id")
    out_dir = L.DATA_DIR / "units" / "neon"
    out_dir.mkdir(parents=True, exist_ok=True)
    coll = ee.ImageCollection("projects/neon-prod-earthengine/assets/CHM/001")
    rows = []
    for site, grp in units.groupby("site"):
        site_coll = coll.filter(ee.Filter.eq("NEON_SITE", site))
        years = sorted(
            int(y) for y in site_coll.aggregate_array("NEON_YEAR").getInfo() or []
        )
        if not years:
            log.warning("%s: no lidar CHM in Earth Engine", site)
            continue
        t0 = time.perf_counter()
        n_new = 0
        for uid, poly, epsg in zip(
            grp["unit_id"], grp.geometry, grp["epsg"], strict=True
        ):
            path = L.unit_chip("neon", uid)
            survey = table.loc[uid, "survey_year"]
            year = (
                min(years, key=lambda y: abs(y - survey))
                if np.isfinite(survey)
                else years[-1]
            )
            rows.append({"unit_id": uid, "lidar_chm_year": year})
            if path.exists():
                continue
            utm = utm_bounds(poly, units.crs, int(epsg), BUFFER_M)
            img = site_coll.filter(ee.Filter.eq("NEON_YEAR", str(year))).mosaic()
            download_ee_chip(img, utm, int(epsg), path)
            n_new += 1
        log.info(
            "%s: %d new chips in %.1f s (years %s)",
            site,
            n_new,
            time.perf_counter() - t0,
            years,
        )
    pd.DataFrame(rows).to_csv(L.TABLE_DIR / "neon_unit_lidar_years.csv", index=False)


def utm_bounds(
    poly, crs, epsg: int, buffer_m: float
) -> tuple[float, float, float, float]:
    """Unit bounds plus a buffer in the unit's UTM zone, snapped to whole metres."""
    import geopandas as gpd

    x0, y0, x1, y1 = gpd.GeoSeries([poly], crs=crs).to_crs(epsg).total_bounds
    return (
        float(np.floor(x0 - buffer_m)),
        float(np.floor(y0 - buffer_m)),
        float(np.ceil(x1 + buffer_m)),
        float(np.ceil(y1 + buffer_m)),
    )


def download_ee_chip(img, bounds, epsg: int, path) -> None:
    """Download one 1 m chip on a whole-metre UTM grid as float32 GeoTIFF."""
    import io
    import urllib.request
    import zipfile

    import ee

    x0, y0, x1, y1 = bounds
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], f"EPSG:{epsg}", False)
    url = img.getDownloadURL(
        {
            "region": region,
            "crs": f"EPSG:{epsg}",
            "crsTransform": [1, 0, x0, 0, -1, y1],
            "dimensions": f"{int(x1 - x0)}x{int(y1 - y0)}",
            "format": "GEO_TIFF",
        }
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            payload = zf.read(next(n for n in zf.namelist() if n.endswith(".tif")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", choices=["naip", "neon", "table"])
    args = ap.parse_args()
    unit_table()
    if args.source == "naip":
        fetch_naip()
    elif args.source == "neon":
        fetch_neon()


if __name__ == "__main__":
    main()
