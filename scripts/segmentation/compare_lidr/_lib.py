"""Shared paths, inputs and metrics for the Python-against-lidR comparison.

Inputs
    strata_blocks.csv (here)  36 benchmark blocks of 1024 x 1024 px in 12
                              NAIP-CHM tiles chosen to span CONUS forest types.
                              Tiles are fetched from the NTSG server on first
                              use into ``data_root/naip_chm_national/tiles``.
    REFAGB_METHODS_DIR        Root of the refAGB methods-paper repository,
                              which holds the 626 NEON 20 m field units
                              (``analysis/naip_chm``) and the NEON field trees
                              (``results/neon_sites/raw``). Needed only by the
                              NEON unit steps.

Outputs go to ``results_root/segmentation_compare/`` (tables, figures) and
cached unit chips to ``data_root/segmentation_compare/units/<source>/``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GDAL_CACHEMAX", "256")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.features import geometry_mask, rasterize  # noqa: E402

from csdv_core.biomass.allometry import jucker_agb_kg, tallo_model  # noqa: E402
from csdv_core.io.paths import project_paths  # noqa: E402

HERE = Path(__file__).resolve().parent
PATHS = project_paths()
DATA_DIR = PATHS.data_root / "segmentation_compare"
OUT_DIR = PATHS.results_root / "segmentation_compare"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TILE_DIR = PATHS.data_root / "naip_chm_national" / "tiles"
STRATA_CSV = HERE / "strata_blocks.csv"
NAIP_SCALE = 0.01

#: Field canopy trees: live, mapped, at least 5 m tall, in one of these
#: canopy positions. Same definition as the national feasibility study.
CANOPY_POSITIONS = {"Full sun", "Open grown", "Partially shaded"}
MIN_TREE_HEIGHT_M = 5.0


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("csdv_core.segmentation").setLevel(logging.ERROR)
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Strata blocks
# ---------------------------------------------------------------------------
def strata_blocks() -> pd.DataFrame:
    """The benchmark blocks, one row per block."""
    return pd.read_csv(STRATA_CSV)


def tile_path(url: str) -> Path:
    """Local copy of a NAIP-CHM tile, downloaded on first use."""
    path = TILE_DIR / Path(url).name
    if not path.exists():
        from csdv_core.download.naip_chm_http import download_tile

        TILE_DIR.mkdir(parents=True, exist_ok=True)
        download_tile(url, path)
    return path


@dataclass(frozen=True)
class Block:
    """One haloed block of a tile, in metres, with its interior bounds."""

    stratum: str
    chm: np.ndarray
    transform: object
    crs: object
    interior: tuple[float, float, float, float]
    inner: tuple[slice, slice]


def read_block(row: pd.Series, halo_px: int) -> Block:
    """Read one strata block plus a halo, as float32 metres with NaN nodata."""
    from csdv_core.national.benchmark import _read_block

    n = int(row["block_px"])
    r0, c0 = int(row["row0"]), int(row["col0"])
    with rasterio.open(tile_path(row["chm_url"])) as src:
        chm, transform, bounds = _read_block(
            src, (r0, c0, r0 + n, c0 + n), halo_px, NAIP_SCALE
        )
        crs = src.crs
    dr, dc = r0 - max(0, r0 - halo_px), c0 - max(0, c0 - halo_px)
    inner = (slice(dr, dr + n), slice(dc, dc + n))
    return Block(row["stratum"], chm, transform, crs, bounds, inner)


# ---------------------------------------------------------------------------
# NEON units
# ---------------------------------------------------------------------------
def refagb_dir() -> Path:
    root = os.environ.get("REFAGB_METHODS_DIR")
    if not root:
        raise RuntimeError(
            "Set REFAGB_METHODS_DIR to the refAGB methods-paper repository root."
        )
    return Path(root)


def neon_units() -> gpd.GeoDataFrame:
    """The 626 NEON units with field AGBD, split, biome and geometry."""
    base = refagb_dir() / "analysis" / "naip_chm"
    geom = gpd.read_file(base / "derived" / "units.gpkg")[["unit_id", "geometry"]]
    attrs = pd.read_csv(base / "results" / "units_naip_chm.csv")
    return geom.merge(attrs, on="unit_id")


def field_canopy_trees(site: str, units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-unit counts, mean height and mean crown diameter of field canopy trees."""
    path = (
        refagb_dir()
        / "results"
        / "neon_sites"
        / "raw"
        / site
        / "field_calibration"
        / "individual_trees.csv"
    )
    t = pd.read_csv(path, low_memory=False)
    t = t[t["adjEasting"].notna() & t["stemDistance"].notna()]
    t = t[t["plantStatus"].astype(str).str.startswith("Live")]
    t = t.assign(h=pd.to_numeric(t.get("height_m", t["height"]), errors="coerce"))
    epsg = int(units["epsg"].iloc[0])
    if "utmZone" in t:
        zs = t["utmZone"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        t = t[zs.isna() | (zs == epsg % 100)]
    pts = gpd.GeoDataFrame(
        t, geometry=gpd.points_from_xy(t["adjEasting"], t["adjNorthing"]), crs=epsg
    )
    rows = []
    polys = units.to_crs(epsg)
    for uid, poly, year in zip(
        polys["unit_id"], polys.geometry, polys["survey_year"], strict=True
    ):
        inside = pts[pts.within(poly)]
        if np.isfinite(year) and "survey_year" in inside and len(inside):
            yrs = inside["survey_year"].dropna().unique()
            if len(yrs):
                inside = inside[
                    inside["survey_year"] == yrs[np.argmin(np.abs(yrs - year))]
                ]
        if "individualID" in inside:
            inside = inside.drop_duplicates("individualID")
        canopy = inside[
            (inside["h"] >= MIN_TREE_HEIGHT_M)
            & inside["canopyPosition"].isin(CANOPY_POSITIONS)
        ]
        cd = pd.to_numeric(canopy.get("maxCrownDiameter"), errors="coerce")
        rows.append(
            {
                "unit_id": uid,
                "n_canopy": len(canopy),
                "field_canopy_h": canopy["h"].mean() if len(canopy) else np.nan,
                "field_canopy_cd": cd.mean() if len(canopy) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def unit_chip(source: str, unit_id: str) -> Path:
    return DATA_DIR / "units" / source / f"{unit_id}.tif"


def read_chip(path: Path) -> tuple[np.ndarray, object, object]:
    """A cached chip as float32 metres with NaN nodata."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        scale = src.scales[0] if src.scales and src.scales[0] != 1.0 else 1.0
        if data.dtype.kind in "ui" and scale == 1.0:
            scale = NAIP_SCALE
        chm = data.astype("float32").filled(np.nan) * np.float32(scale)
        return chm, src.transform, src.crs


# ---------------------------------------------------------------------------
# Crown summaries
# ---------------------------------------------------------------------------
def crown_label_raster(crowns: gpd.GeoDataFrame, shape, transform) -> np.ndarray:
    """Crown polygons burnt back onto the CHM grid (1..n, 0 elsewhere)."""
    if crowns.empty:
        return np.zeros(shape, dtype="int32")
    return rasterize(
        zip(crowns.geometry, np.arange(1, len(crowns) + 1), strict=True),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )


def canopy_assigned(
    crowns: gpd.GeoDataFrame,
    chm: np.ndarray,
    transform,
    *,
    region: np.ndarray | None = None,
    min_height_m: float = 2.0,
) -> float:
    """Share of canopy pixels (raw CHM at or above ``min_height_m``) in a crown."""
    canopy = np.nan_to_num(chm, nan=0.0) >= min_height_m
    if region is not None:
        canopy &= region
    if not canopy.any():
        return float("nan")
    lab = crown_label_raster(crowns, chm.shape, transform)
    return float((lab[canopy] > 0).mean())


def unit_mask(poly, shape, transform) -> np.ndarray:
    return ~geometry_mask([poly], shape, transform, all_touched=False)


def crown_agb_kg(crowns: pd.DataFrame, biome: str) -> dict[str, np.ndarray]:
    """Per-crown AGB from the biome's Tallo model and from Jucker et al. (2017)."""
    h = crowns["apex_h_m"].to_numpy()
    cd = crowns["crown_diam_m"].to_numpy()
    out = {
        "jucker": jucker_agb_kg(h, cd, gymnosperm=biome == "Temperate Conifer Forests")
    }
    try:
        out["tallo"] = tallo_model(biome).agb_kg(h, cd)
    except KeyError:
        out["tallo"] = np.full(len(crowns), np.nan)
    return out


def fit_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """R² against the 1:1 line, RMSE, RMSE % and bias %, as in refAGB."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[ok], pred[ok]
    if len(o) < 3:
        return {
            "n": int(ok.sum()),
            "r2": np.nan,
            "rmse": np.nan,
            "rmse_pct": np.nan,
            "bias_pct": np.nan,
        }
    ss_res = float(np.sum((o - p) ** 2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    return {
        "n": int(ok.sum()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse": rmse,
        "rmse_pct": 100.0 * rmse / o.mean(),
        "bias_pct": 100.0 * (p.mean() - o.mean()) / o.mean(),
    }


def line_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares intercept and slope (x may be 2-D for several predictors)."""
    x = np.column_stack([np.ones(len(y)), np.asarray(x).reshape(len(y), -1)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def line_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(x)), np.asarray(x).reshape(len(x), -1)])
    return x @ coef
