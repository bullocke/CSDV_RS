"""Cache a CHM chip around every NEON field unit, and the units' field summaries.

Each chip covers the 20 x 20 m unit plus a 50 m buffer, enough for the largest
crown radius cap to resolve crowns whose tree top lies in the unit.

``naip``  the national NAIP-CHM (0.6 m), read over HTTP from the tile whose
          nominal quarter-quad cell holds the unit centroid.
``neon``  the NEON AOP lidar CHM (DP3.30015.001, 1 m) from the Earth Engine
          asset ``projects/neon-prod-earthengine/assets/CHM/001``, the image
          of the unit's refAGB lidar year.
``blocks`` one 1024 x 1024 m block of the same lidar CHM per NEON site,
          centred on the site's units, the lidar counterpart of the NAIP-CHM
          strata blocks. Listed in ``tables/lidar_blocks.csv``.

Also writes ``tables/neon_units.parquet``: unit attributes from refAGB (field
AGBD, split, biome) plus field canopy-tree counts, height and crown diameter.

Run::

    export REFAGB_METHODS_DIR=<refAGB methods-paper repo>
    P=.micromamba/envs/CSDV/bin/python
    $P scripts/segmentation/compare_lidr/fetch_units.py naip     # ~5 min
    $P scripts/segmentation/compare_lidr/fetch_units.py neon     # ~1 min, Earth Engine
    $P scripts/segmentation/compare_lidr/fetch_units.py blocks   # ~1 min, Earth Engine
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


NEON_CHM = "projects/neon-prod-earthengine/assets/CHM/001"
EE_WORKERS = 8


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


def neon_chm_image(site: str, lon: float, lat: float, year: int):
    """The site's NEON CHM image covering a point, nearest ``year`` within two."""
    import ee

    coll = (
        ee.ImageCollection(NEON_CHM)
        .filter(ee.Filter.eq("NEON_SITE", site))
        .filterBounds(ee.Geometry.Point([lon, lat]))
    )
    for y in sorted({year + d for d in (0, -1, 1, -2, 2)}, key=lambda v: abs(v - year)):
        img = coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
        if img.size().getInfo():
            return ee.Image(img.first()), y
    return None, None


def download_chip(img, bounds, epsg: int, path) -> None:
    """One 1 m chip on a whole-metre UTM grid, through ``computePixels``."""
    import ee

    x0, y0, x1, y1 = bounds
    request = {
        "expression": img.select(0),
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": int(x1 - x0), "height": int(y1 - y0)},
            "affineTransform": {
                "scaleX": 1.0, "shearX": 0, "translateX": x0,
                "shearY": 0, "scaleY": -1.0, "translateY": y1,
            },
            "crsCode": f"EPSG:{epsg}",
        },
    }  # fmt: skip
    payload = ee.data.computePixels(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def fetch_neon() -> None:
    """Chips of the NEON lidar CHM, the image of each unit's refAGB lidar year."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from csdv_core.download._ee import initialize_ee

    initialize_ee()
    units = L.neon_units()
    rows, todo = [], []
    for unit in units.itertuples():
        path = L.unit_chip("neon", unit.unit_id)
        rows.append({"unit_id": unit.unit_id, "requested_year": int(unit.lidar_year)})
        if not path.exists():
            todo.append(unit)
    log.info("%d of %d NEON chips to fetch", len(todo), len(units))

    def one(unit) -> dict:
        img, year = neon_chm_image(unit.site, unit.lon, unit.lat, int(unit.lidar_year))
        if img is None:
            return {"unit_id": unit.unit_id, "lidar_chm_year": None}
        bounds = utm_bounds(unit.geometry, units.crs, int(unit.epsg), BUFFER_M)
        for attempt in range(4):
            try:
                download_chip(
                    img, bounds, int(unit.epsg), L.unit_chip("neon", unit.unit_id)
                )
                break
            except Exception as exc:  # EE quota and transient HTTP errors
                if attempt == 3:
                    log.warning("%s: %s", unit.unit_id, exc)
                    return {"unit_id": unit.unit_id, "lidar_chm_year": None}
                time.sleep(2.0 * (attempt + 1))
        return {"unit_id": unit.unit_id, "lidar_chm_year": year}

    t0 = time.perf_counter()
    got = []
    with ThreadPoolExecutor(max_workers=EE_WORKERS) as pool:
        futures = [pool.submit(one, u) for u in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            got.append(fut.result())
            if i % 50 == 0:
                log.info(
                    "%d / %d chips (%.0f s)", i, len(todo), time.perf_counter() - t0
                )
    years = pd.DataFrame(rows).merge(
        pd.DataFrame(got, columns=["unit_id", "lidar_chm_year"]),
        on="unit_id",
        how="left",
    )
    old = L.TABLE_DIR / "neon_unit_lidar_years.csv"
    if old.exists():
        prev = pd.read_csv(old)
        years = (
            years.set_index("unit_id")
            .combine_first(prev.set_index("unit_id"))
            .reset_index()
        )
    years.to_csv(old, index=False)
    missing = sum(1 for g in got if g["lidar_chm_year"] is None)
    log.info(
        "Done in %.0f s, %d units without a chip", time.perf_counter() - t0, missing
    )


LIDAR_BLOCK_M = 1024


def fetch_lidar_blocks() -> None:
    """One 1024 m lidar CHM block per site, centred on the site's units."""
    import ee
    import geopandas as gpd

    from csdv_core.download._ee import initialize_ee

    initialize_ee()
    units = L.neon_units()
    out_dir = L.DATA_DIR / "lidar_blocks"
    rows = []
    for site, grp in units.groupby("site"):
        epsg = int(grp["epsg"].iloc[0])
        year = int(grp["lidar_year"].mode().iloc[0])
        cx, cy = (
            gpd.GeoSeries(grp.geometry, crs=units.crs)
            .to_crs(epsg)
            .union_all()
            .centroid.coords[0]
        )
        x0 = float(np.floor(cx - LIDAR_BLOCK_M / 2))
        y0 = float(np.floor(cy - LIDAR_BLOCK_M / 2))
        bounds = (x0, y0, x0 + LIDAR_BLOCK_M, y0 + LIDAR_BLOCK_M)
        path = out_dir / f"{site}.tif"
        site_coll = ee.ImageCollection(NEON_CHM).filter(ee.Filter.eq("NEON_SITE", site))
        for y in sorted(
            {year + d for d in (0, -1, 1, -2, 2)}, key=lambda v: abs(v - year)
        ):
            sub = site_coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
            if sub.size().getInfo():
                img, year = sub.mosaic(), y
                break
        else:
            log.warning("%s: no lidar CHM within two years of %d", site, year)
            continue
        if not path.exists():
            download_chip(img, bounds, epsg, path)
        with rasterio.open(path) as src:
            data = src.read(1)
        rows.append(
            {
                "site": site,
                "biome": grp["biome_long"].iloc[0],
                "epsg": epsg,
                "year": year,
                "x0": x0,
                "y0": y0,
                "covered_frac": float(np.mean(data > 0)),
                "canopy_frac": float(np.mean(data >= 2.0)),
            }
        )
        log.info(
            "%s: block %s, %.0f%% canopy",
            site,
            path.name,
            100 * rows[-1]["canopy_frac"],
        )
    pd.DataFrame(rows).to_csv(L.TABLE_DIR / "lidar_blocks.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", choices=["naip", "neon", "blocks", "table"])
    args = ap.parse_args()
    unit_table()
    if args.source == "naip":
        fetch_naip()
    elif args.source == "neon":
        fetch_neon()
    elif args.source == "blocks":
        fetch_lidar_blocks()


if __name__ == "__main__":
    main()
