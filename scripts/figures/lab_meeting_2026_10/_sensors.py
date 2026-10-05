"""Landsat, Sentinel-2 and NAIP inputs for the sensor comparison figure.

Three things are fetched or computed here and cached on disk, so the figure
script only draws:

``fetch_chip``
    A growing-season median composite over a square view round the stand, on
    the sensor's own pixel grid. Pixels are requested at native size with
    nearest-neighbour resampling, so a 30 m Landsat pixel in the figure is a
    real 30 m Landsat pixel.

``s2_observations``
    Per-scene stand-mean Sentinel-2 NDVI in the same column layout as the
    Landsat observation table, so the quality gates and the annual mean that
    the Landsat series uses apply unchanged.

``naip_ndvi`` and ``structure_series``
    NAIP NDVI over the stand, and the four NAIP canopy structure metrics.

Sentinel-2 uses the harmonized Level-2A surface reflectance collection, so its
NDVI is on the same footing as Landsat Collection 2 surface reflectance. The
archive over Indiana thins out before 2017, which is why the pre-harvest year
rests on few scenes. Clouds are masked with Cloud Score+.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "examples" / "docs" / "metrics"))

from _lib import (  # noqa: E402
    ExampleSite,
    disturbance_years,
    load_satellite_observations,
    load_stand_crowns,
    stand_mask,
    stand_metric_series,
    stand_row,
)

from csdv_core.zonal.crowns import MIN_CROWNS, crown_diameter_stats  # noqa: E402

logger = logging.getLogger(__name__)

YEARS = (2012, 2014, 2016, 2018, 2020, 2022)
CACHE_DIR = REPO / "results" / "figures" / "lab_meeting_2026_10" / "cache" / "sensors"
CRS = "EPSG:26916"
#: Growing season used for both composites and annual means (DOY 152 to 258).
SEASON = ("06-01", "09-15")
DOY_WINDOW = (152, 258)
WIDE_SEASON = ("05-01", "10-15")
MIN_OBS = 3
S2_FIRST_YEAR = 2015
S2_CLOUD_SCORE = 0.60
#: Sentinel-2 tiles are on a whole-metre 10 m grid. Landsat Collection 2 pixel
#: edges sit at 15 m modulo 30 in UTM.
GRID = {"landsat": (30.0, 15.0), "s2": (10.0, 0.0)}
#: Below this many crowns the stand metrics table voids crown CV. The figure
#: recomputes it with a lower floor and draws those dates hollow.
CV_FALLBACK_MIN_CROWNS = 10


@dataclass
class Chip:
    """One composite on its native grid, in map coordinates."""

    sensor: str
    year: int
    rgb: np.ndarray | None  # (rows, cols, 3) reflectance, None when no data
    ndvi: np.ndarray | None
    extent: tuple[float, float, float, float]  # left, right, bottom, top
    res: float
    n_scenes: int


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def square_view(geometry, *, pad_m: float = 40.0, min_side_m: float = 300.0):
    """Square map bounds round the stand, snapped to the 30 m Landsat grid."""
    minx, miny, maxx, maxy = geometry.bounds
    side = max(maxx - minx, maxy - miny, min_side_m - 2 * pad_m) + 2 * pad_m
    side = np.ceil(side / 30.0) * 30.0
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    x0 = np.floor((cx - side / 2 - 15.0) / 30.0) * 30.0 + 15.0
    y0 = np.floor((cy - side / 2 - 15.0) / 30.0) * 30.0 + 15.0
    return (float(x0), float(y0), float(x0 + side), float(y0 + side))


def _native_grid(view, sensor: str):
    """Grid covering ``view`` on the sensor's pixel lattice, one pixel wider."""
    res, offset = GRID[sensor]
    left = np.floor((view[0] - offset) / res) * res + offset - res
    bottom = np.floor((view[1] - offset) / res) * res + offset - res
    right = np.ceil((view[2] - offset) / res) * res + offset + res
    top = np.ceil((view[3] - offset) / res) * res + offset + res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    return left, bottom, right, top, width, height, res


def pixels_in_stand(geometry, view, sensor: str) -> int:
    """Native pixels whose centre falls inside the stand."""
    left, bottom, right, top, width, height, res = _native_grid(view, sensor)
    transform = rasterio.transform.from_origin(left, top, res, res)
    return int(stand_mask(geometry, (height, width), transform).sum())


def native_inside(geometry, chip: Chip) -> np.ndarray:
    """Pixel-centre stand mask for a chip."""
    left, _, _, top = chip.extent[0], chip.extent[1], chip.extent[2], chip.extent[3]
    shape = (chip.rgb if chip.rgb is not None else chip.ndvi).shape[:2]
    transform = rasterio.transform.from_origin(left, top, chip.res, chip.res)
    return stand_mask(geometry, shape, transform)


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------


def _ee():
    import ee

    from csdv_core.download._ee import initialize_ee

    if not getattr(_ee, "_ready", False):
        initialize_ee()
        _ee._ready = True
    return ee


def _view_geometry(view):
    ee = _ee()
    return ee.Geometry.Rectangle(list(view), proj=CRS, geodesic=False)


def _landsat_composite(view, year: int):
    """Median of every clear Landsat SR pixel in the season, harmonized bands."""
    from csdv_core.config import load_satellite
    from csdv_core.satellite.sensors import (
        get_sensor,
        sensor_collection,
        sensors_for_year,
    )

    ee = _ee()
    cfg = load_satellite()
    region = _view_geometry(view)
    start, end = f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}"
    parts = [
        sensor_collection(get_sensor(name), region, start, end, cfg).select(
            ["red", "green", "blue", "nir"]
        )
        for name in sensors_for_year(year)
        if name != "L4"
    ]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part)
    merged = ee.ImageCollection(merged)
    return merged.median(), merged.size()


def _s2_collection(region, start: str, end: str):
    """Sentinel-2 L2A surface reflectance, Cloud Score+ masked, as red/green/blue/nir."""
    ee = _ee()
    scores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def _prep(image):
        clear = image.select("cs_cdf").gte(S2_CLOUD_SCORE)
        return (
            image.select(["B4", "B3", "B2", "B8"], ["red", "green", "blue", "nir"])
            .divide(10000)
            .updateMask(clear)
            .copyProperties(image, ["system:time_start", "system:index"])
        )

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .linkCollection(scores, ["cs_cdf"])
        .map(_prep)
    )


def _download(image, view, sensor: str, path: Path) -> None:
    ee = _ee()
    left, bottom, right, top, width, height, res = _native_grid(view, sensor)
    payload = ee.data.computePixels(
        {
            "expression": image.toFloat(),
            "fileFormat": "GEO_TIFF",
            "grid": {
                "dimensions": {"width": width, "height": height},
                "affineTransform": {
                    "scaleX": res,
                    "shearX": 0,
                    "translateX": left,
                    "shearY": 0,
                    "scaleY": -res,
                    "translateY": top,
                },
                "crsCode": CRS,
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def fetch_chip(
    sensor: str, stand_id: str, year: int, view, *, refresh: bool = False
) -> Chip:
    """Growing-season median composite on the sensor's native grid, cached."""
    left, bottom, right, top, _, _, res = _native_grid(view, sensor)
    extent = (left, right, bottom, top)
    if sensor == "s2" and year < S2_FIRST_YEAR:
        return Chip(sensor, year, None, None, extent, res, 0)

    path = CACHE_DIR / stand_id / f"{sensor}_{year}.tif"
    if refresh or not path.exists():
        if sensor == "landsat":
            image, size = _landsat_composite(view, year)
        else:
            region = _view_geometry(view)
            col = _s2_collection(region, f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}")
            if col.size().getInfo() < MIN_OBS:
                # The 2016 archive is thin. Widen to May to mid-October rather
                # than show a single, mostly cloudy scene.
                start, end = WIDE_SEASON
                col = _s2_collection(region, f"{year}-{start}", f"{year}-{end}")
                logger.info("%s s2 %d: widened to %s..%s", stand_id, year, *WIDE_SEASON)
            image, size = col.median(), col.size()
        n_scenes = int(size.getInfo())
        logger.info("%s %s %d: %d scenes", stand_id, sensor, year, n_scenes)
        if n_scenes == 0:
            return Chip(sensor, year, None, None, extent, res, 0)
        image = image.set("n_scenes", n_scenes)
        _download(image.select(["red", "green", "blue", "nir"]), view, sensor, path)
        path.with_suffix(".n").write_text(str(n_scenes))

    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        bands[bands == nodata] = np.nan
    red, green, blue, nir = bands
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    n_scenes = int(path.with_suffix(".n").read_text())
    return Chip(
        sensor, year, np.dstack([red, green, blue]), ndvi, extent, res, n_scenes
    )


def s2_observations(
    stands, stand_ids: list[str], *, refresh: bool = False
) -> pd.DataFrame:
    """Per-scene stand-mean Sentinel-2 NDVI, filtered like the Landsat table.

    Scenes from overlapping tiles of the same overpass are averaged into one
    observation per date.
    """
    from csdv_core.config import load_satellite
    from csdv_core.satellite.annual import filter_observations
    from csdv_core.satellite.extract import aoi_geometry, stands_to_fc

    path = CACHE_DIR / "s2_observations.parquet"
    if refresh or not path.exists():
        ee = _ee()
        subset = stands[stands["stand_id"].isin(stand_ids)]
        fc = stands_to_fc(subset)
        aoi = aoi_geometry(subset, buffer_m=200.0)
        reducer = (
            ee.Reducer.mean()
            .combine(ee.Reducer.count().unweighted(), sharedInputs=True)
            .combine(ee.Reducer.sum(), sharedInputs=True)
        )
        frames = []
        for year in range(S2_FIRST_YEAR, 2026):
            col = _s2_collection(aoi, f"{year}-05-01", f"{year}-10-31")

            def _per_image(image):
                ndvi = image.normalizedDifference(["nir", "red"]).rename("ndvi")
                stacked = ndvi.addBands(ndvi.multiply(0).add(1).rename("valid"))
                reduced = stacked.reduceRegions(
                    collection=fc, reducer=reducer, scale=10
                )
                return reduced.map(
                    lambda f: f.set(
                        {
                            "image_id": image.get("system:index"),
                            "time_ms": image.get("system:time_start"),
                        }
                    )
                )

            table = ee.FeatureCollection(col.map(_per_image)).flatten()
            frame = ee.data.computeFeatures(
                {"expression": table, "fileFormat": "PANDAS_DATAFRAME"}
            )
            logger.info("Sentinel-2 %d: %d stand-scenes", year, len(frame))
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True).rename(
            columns={
                "ndvi_mean": "ndvi",
                "ndvi_count": "ndvi_n_pixels",
                "valid_count": "n_pixels",
                "valid_sum": "pixel_weight_sum",
            }
        )
        raw = raw.drop(columns=[c for c in ("geo",) if c in raw.columns])
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path)

    raw = pd.read_parquet(path)
    areas = stands.set_index("stand_id").geometry.area
    raw["date"] = pd.to_datetime(raw["time_ms"], unit="ms").dt.normalize()
    raw["area_m2"] = raw["stand_id"].map(areas)
    raw["expected_pixels"] = raw["area_m2"] / 100.0
    raw["coverage_fraction"] = raw["pixel_weight_sum"] / raw["expected_pixels"]
    raw["sensor"] = "S2"
    # One observation per overpass: tiles overlap, so a date can appear twice.
    obs = (
        raw.groupby(["stand_id", "date"], as_index=False)
        .agg(
            ndvi=("ndvi", "mean"),
            n_pixels=("n_pixels", "max"),
            ndvi_n_pixels=("ndvi_n_pixels", "max"),
            pixel_weight_sum=("pixel_weight_sum", "max"),
            coverage_fraction=("coverage_fraction", "max"),
            area_m2=("area_m2", "first"),
            expected_pixels=("expected_pixels", "first"),
        )
        .assign(
            year=lambda d: d["date"].dt.year,
            doy=lambda d: d["date"].dt.dayofyear,
        )
    )
    # Pixel-count gates are in Landsat pixels, so scale them to 10 m pixels.
    cfg = load_satellite()
    kept, counts = filter_observations(
        obs,
        index="ndvi",
        min_pixels=cfg.qa.min_pixels * 9,
        min_effective_pixels=cfg.qa.min_effective_pixels * 9,
        min_coverage_fraction=cfg.qa.min_coverage_fraction,
        min_area_m2=cfg.qa.min_area_m2,
        valid_range=tuple(cfg.qa.index_valid_range),
    )
    logger.info("Sentinel-2: %d observations kept, dropped %s", len(kept), counts)
    return kept.sort_values(["stand_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Series for the time-series panels
# --------------------------------------------------------------------------


def decimal_year(dates) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    return (dates.dt.year + (dates.dt.dayofyear - 0.5) / 365.25).to_numpy(float)


def annual_ndvi(obs: pd.DataFrame) -> pd.DataFrame:
    """Growing-season mean per year, hollow below ``MIN_OBS`` observations."""
    doy = pd.to_numeric(obs["doy"], errors="coerce")
    season = obs.loc[(doy >= DOY_WINDOW[0]) & (doy <= DOY_WINDOW[1])]
    out = (
        season.groupby("year")["ndvi"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "ndvi", "count": "n_obs"})
        .reset_index()
    )
    out["t"] = out["year"] + (sum(DOY_WINDOW) / 2) / 365.25
    out["enough"] = out["n_obs"] >= MIN_OBS
    return out


def season_obs(obs: pd.DataFrame) -> pd.DataFrame:
    doy = pd.to_numeric(obs["doy"], errors="coerce")
    season = obs.loc[(doy >= DOY_WINDOW[0]) & (doy <= DOY_WINDOW[1])].copy()
    season["t"] = decimal_year(season["date"])
    return season


def landsat_observations(site: ExampleSite, stand_id: str) -> pd.DataFrame:
    return load_satellite_observations(site, stand_id)


def read_naip(site: ExampleSite, year: int, view) -> tuple[np.ndarray, object]:
    """Four NAIP bands over ``view`` as float, with the window transform."""
    with rasterio.open(site.naip(year)) as src:
        window = from_bounds(*view, transform=src.transform)
        window = window.round_offsets().round_lengths()
        arr = src.read([1, 2, 3, 4], window=window).astype(np.float32)
        transform = src.window_transform(window)
    return np.moveaxis(arr, 0, -1), transform


def naip_ndvi(block: np.ndarray, inside: np.ndarray) -> float:
    red, nir = block[..., 0], block[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    return float(np.nanmean(ndvi[inside]))


def structure_series(context, site: ExampleSite, stand_id: str) -> pd.DataFrame:
    """Median height, trees per ha, crown width CV and gap fraction by NAIP date.

    Crown CV is voided in the stand metrics table below ``MIN_CROWNS`` crowns.
    Those dates are recomputed here from the crown polygons with a floor of
    ``CV_FALLBACK_MIN_CROWNS`` and flagged as low support.
    """
    stand = stand_row(context.stands, stand_id)
    rows = stand_metric_series(context.metrics, stand_id)
    area_ha = stand.geometry.area / 1e4
    out = pd.DataFrame(
        {
            "year": rows["year"].astype(int),
            "t": decimal_year(rows["date"]) if "date" in rows else rows["year"] + 0.5,
            "height_median": rows["height_median"],
            "trees_per_ha": rows["crown_count"] / area_ha,
            "crown_cv": rows["crown_cv"],
            "gap_fraction": rows["gap_fraction"],
            "n_crowns": rows["crown_count"],
        }
    )
    out["cv_low_support"] = out["n_crowns"] < MIN_CROWNS
    for i, row in out.iterrows():
        if np.isnan(row["crown_cv"]):
            crowns = load_stand_crowns(site, int(row["year"]), stand.geometry)
            stats = crown_diameter_stats(crowns, min_crowns=CV_FALLBACK_MIN_CROWNS)
            out.loc[i, "crown_cv"] = stats.cv
            logger.info(
                "%s %d: crown CV recomputed from %d crowns = %.3f",
                stand_id,
                int(row["year"]),
                stats.n_crowns,
                stats.cv,
            )
    return out


def stand_info(context, stand_id: str) -> dict:
    stand = stand_row(context.stands, stand_id)
    last_pre, first_post = disturbance_years(stand)
    return {
        "geometry": stand.geometry,
        "label": str(stand.get("dist_label", "")),
        "area_ha": stand.geometry.area / 1e4,
        "last_pre": last_pre,
        "first_post": first_post,
    }
