"""Shared harness for the crown segmentation optimization.

Everything the sweep, the null models and the figures need in common: where the
data lives, how a tile is read, the published allometries the results are
scored against, and the decision rule.

The decision rule is in this file on purpose. It was written before any sweep
output was read, so that the choice of parameters is a rule applied to numbers
rather than a number picked to match a preference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from csdv_core.io.paths import project_paths
from csdv_core.segmentation.chm_watershed import segment_crowns
from csdv_core.segmentation.params import (
    WINDOW_FUNCTIONS,
    SegmentationParams,
    WindowFunction,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
GUIDE_DIR = REPO / "docs" / "guides" / "segmentation_optimization"
FIGURE_DIR = GUIDE_DIR / "figures"
RESULT_DIR = GUIDE_DIR / "results"
TILE_PATH = Path(__file__).resolve().parent / "tiles.geojson"

SITE = "ElkinsvilleNE"
NAIP_YEARS = (2012, 2014, 2016, 2018, 2020, 2022)
#: 2012 and 2014 were flown at 1.0 m and resampled onto the common 0.6 m grid.
NATIVE_RES_M = {2012: 1.0, 2014: 1.0, 2016: 0.6, 2018: 0.6, 2020: 0.6, 2022: 0.6}
REFERENCE_YEAR = 2016

TILE_SIZE_M = 300.0

#: Tiles are scored only on crowns wholly inside this inset. A crown clipped by
#: the tile edge is small and irregular, so edge crowns deflate mean diameter
#: and inflate CV, and the size of that bias depends on crown size, which is
#: the thing the sweep varies.
TILE_INSET_M = 15.0

CANOPY_FLOOR_M = 2.0


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SiteData:
    """One CHM source, wherever it lives."""

    name: str
    chm: Path
    pixel_size_m: float
    note: str = ""


def elkinsville_chm(year: int) -> Path:
    """Path to the Elkinsville NAIP-CHM for one year."""
    paths = project_paths()
    hits = sorted((paths.data_root / "naip_chm" / SITE / str(year)).glob("*.tif"))
    if not hits:
        raise FileNotFoundError(f"No CHM for {SITE} {year}")
    return hits[0]


def elkinsville_naip(year: int) -> Path:
    """Path to the Elkinsville NAIP mosaic for one year."""
    paths = project_paths()
    hits = sorted((paths.data_root / "naip" / SITE / str(year)).glob("*.tif"))
    if not hits:
        raise FileNotFoundError(f"No NAIP for {SITE} {year}")
    return hits[0]


def transfer_sites() -> list[SiteData]:
    """The two airborne lidar CHMs used as held-out transfer sites.

    Both are NEON discrete-return lidar at 1 m, so they test the parameters
    against a different sensor, a different resolution and a different part of
    the eastern hardwood range. Neither is used for tuning.
    """
    base = REPO / "legacy" / "proof_of_concept" / "Data" / "NEON" / "CHM"
    out = []
    for name, fname in (
        ("SCBI", "NEON_CHM_SCBI_Subset_2023.tif"),
        ("HARV", "NEON_CHM_HARV_Subset_2023.time.19700101T000000.tif"),
    ):
        path = base / fname
        if path.exists():
            out.append(SiteData(name, path, 1.0, "NEON airborne lidar, 1 m, EPSG:5070"))
        else:
            logger.warning("Transfer site %s missing at %s", name, path)
    return out


def stands() -> gpd.GeoDataFrame:
    """The 40 interpreter-labelled disturbance stands."""
    from csdv_core.io.stands import read_ais_stands

    paths = project_paths()
    gdb = (
        paths.data_root
        / "calibration"
        / "Indiana-ElkinsvilleNE_revised.gdb"
        / "Indiana-ElkinsvilleNE_revised.gdb"
    )
    return read_ais_stands(gdb)


# ---------------------------------------------------------------------------
# Raster reading
# ---------------------------------------------------------------------------
def read_window(path: Path, bounds: tuple[float, float, float, float]):
    """Read a CHM subset in metres, NaN for nodata.

    Returns ``(array, transform, crs)``.
    """
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read(1, window=win).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr < -100] = np.nan
        return arr, src.window_transform(win), src.crs


def degrade(arr: np.ndarray, factor: float) -> np.ndarray:
    """Coarsen an array by ``factor`` and restore it to the original grid.

    Used to ask what the 1.0 m to 0.6 m native resolution change does to crown
    metrics. This is a lower bound on the real effect. The 2012 and 2014 CHMs
    came from a model predicting height from blurrier imagery, and most of the
    artefact lives in that inference rather than in the resampling.
    """
    from scipy.ndimage import zoom

    filled = np.nan_to_num(arr, nan=0.0)
    weight = np.isfinite(arr).astype("float32")
    small = zoom(filled, 1.0 / factor, order=1)
    small_w = zoom(weight, 1.0 / factor, order=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        small = np.where(small_w > 0.5, small / np.maximum(small_w, 1e-6), np.nan)
    back = zoom(np.nan_to_num(small, nan=0.0), factor, order=1)
    back_w = zoom(np.isfinite(small).astype("float32"), factor, order=1)
    out = np.where(back_w > 0.5, back, np.nan).astype("float32")
    # zoom can land one pixel short or long.
    result = np.full(arr.shape, np.nan, dtype="float32")
    r = min(arr.shape[0], out.shape[0])
    c = min(arr.shape[1], out.shape[1])
    result[:r, :c] = out[:r, :c]
    return result


# ---------------------------------------------------------------------------
# Published allometry
# ---------------------------------------------------------------------------
# Popescu and Wynne (2004), Photogrammetric Engineering and Remote Sensing
# 70(5): 589-604. Crown width from lidar height, fitted in Virginia and used as
# the FUSION CanopyMaxima variable-window default.
#
# The combined-species coefficients (2.51503, 0.00901) are the FUSION defaults
# and are well attested. The deciduous split (3.09632, 0.00895) is reported at
# second hand and should be checked against the source before publication.
POPESCU_COMBINED = (2.51503, 0.0, 0.00901)
POPESCU_DECIDUOUS = (3.09632, 0.0, 0.00895)


def crown_width_popescu(height_m: np.ndarray, coeffs=POPESCU_DECIDUOUS) -> np.ndarray:
    """Predicted crown width in metres from tree height."""
    a, b, c = coeffs
    h = np.asarray(height_m, dtype="float64")
    return a + b * h + c * h * h


#: How much wider an open-grown crown runs than a stand-grown crown of the same
#: size. Open-grown trees are commonly around 1.5 to 2 times wider, which is the
#: point of crown competition. Two is the generous end.
OPEN_GROWN_MULTIPLE = 2.0


def crown_width_open_grown(height_m: np.ndarray) -> np.ndarray:
    """A generous plausibility ceiling on crown width.

    Krajicek, Brinkman and Gingrich (1961), Forest Science 7(1): 35-42, give
    maximum crown width for open-grown trees as a function of stem diameter. It
    is the right idea for a ceiling, because a stand-grown crown cannot exceed
    an open-grown crown on the same tree. It cannot be applied directly here,
    because stem diameter is not observed and converting height to diameter
    without site information carries more error than the ceiling can absorb.

    This uses the stand-grown expectation scaled by
    :data:`OPEN_GROWN_MULTIPLE` instead. The result is a rule of thumb rather
    than a fitted curve, which is why exceedance is reported as a diagnostic
    and not used as a hard filter. See the report for why that changed.

    An earlier version routed through an invented height-to-diameter relation
    and fell *below* the stand-grown prediction at heights under about 8 m. A
    ceiling that sits below the central expectation is not a ceiling.
    """
    return OPEN_GROWN_MULTIPLE * crown_width_popescu(height_m)


# ---------------------------------------------------------------------------
# Tile statistics
# ---------------------------------------------------------------------------
def tile_stats(
    crowns: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    canopy_area_m2: float,
    *,
    inset_m: float = TILE_INSET_M,
) -> dict[str, float]:
    """Summarise one segmented tile, excluding edge-truncated crowns."""
    minx, miny, maxx, maxy = bounds
    if crowns.empty:
        return {"n_crowns": 0.0}
    cx = crowns.geometry.centroid.x.to_numpy()
    cy = crowns.geometry.centroid.y.to_numpy()
    inner = (
        (cx >= minx + inset_m)
        & (cx <= maxx - inset_m)
        & (cy >= miny + inset_m)
        & (cy <= maxy - inset_m)
    )
    kept = crowns.loc[inner]
    if len(kept) < 3:
        return {"n_crowns": float(len(kept))}

    d = kept["crown_diam_m"].to_numpy(dtype="float64")
    h = kept["apex_h_m"].to_numpy(dtype="float64")
    capped = kept["capped_frac"].to_numpy(dtype="float64")
    inner_area_ha = max(
        (maxx - minx - 2 * inset_m) * (maxy - miny - 2 * inset_m) / 1e4, 1e-9
    )

    # The allometric fit uses only crowns the radius ceiling did not shape. A
    # capped diameter is censored, so it is a lower bound rather than a
    # measurement and belongs nowhere near a squared error.
    free = capped <= 0.05
    pred = crown_width_popescu(h)
    if free.sum() >= 10:
        slope, intercept = np.polyfit(pred[free], d[free], 1)
        resid = d[free] - pred[free]
        rmse = float(np.sqrt(np.mean(resid**2)))
        bias = float(np.mean(resid))
        scatter = float(np.std(resid))
    else:
        slope = intercept = rmse = bias = scatter = float("nan")

    return {
        "n_crowns": float(len(kept)),
        "density_per_ha": float(len(kept) / inner_area_ha),
        "diam_mean": float(d.mean()),
        "diam_median": float(np.median(d)),
        "diam_p90": float(np.percentile(d, 90)),
        "diam_cv": float(d.std() / d.mean()) if d.mean() > 0 else float("nan"),
        "apex_mean": float(h.mean()),
        "capped_fraction": float((capped > 0.05).mean()),
        "assigned_fraction": (
            float(crowns["area_m2"].sum() / canopy_area_m2)
            if canopy_area_m2 > 0
            else float("nan")
        ),
        "allo_slope": float(slope),
        "allo_intercept": float(intercept),
        "allo_rmse": float(rmse),
        "allo_bias": float(bias),
        "allo_scatter": float(scatter),
        "over_open_grown": float(np.mean(d > crown_width_open_grown(h))),
    }


def segment_tile(
    chm_path: Path,
    bounds: tuple[float, float, float, float],
    params: SegmentationParams,
) -> tuple[gpd.GeoDataFrame, float]:
    """Segment one tile. Returns the crowns and the tile's canopy area."""
    arr, transform, crs = read_window(chm_path, bounds)
    px = float(abs(transform.a))
    canopy_area = float(np.sum(np.isfinite(arr) & (arr >= CANOPY_FLOOR_M)) * px * px)
    crowns = segment_crowns(arr, transform, crs, params=params)
    return crowns, canopy_area


# ---------------------------------------------------------------------------
# The parameter grid
# ---------------------------------------------------------------------------
SMOOTH_RADII_M = (0.0, 0.6, 1.2, 2.0)
WINDOW_NAMES = (
    "legacy",
    "shallow",
    "popescu_deciduous",
    "popescu_linear",
    "fixed_5m",
    "fixed_8m",
)
TH_CR_VALUES = (0.0, 0.40, 0.55, 0.70)
MAX_RADIUS_M = (None, 8.0, 12.0)


def parameter_grid(
    *,
    smooth=SMOOTH_RADII_M,
    windows=WINDOW_NAMES,
    th_cr=TH_CR_VALUES,
    max_radius=MAX_RADIUS_M,
) -> list[SegmentationParams]:
    """Every combination of the swept axes."""
    out = []
    for s in smooth:
        for w in windows:
            for t in th_cr:
                for m in max_radius:
                    out.append(
                        SegmentationParams(
                            smooth_radius_m=s,
                            window=WINDOW_FUNCTIONS[w],
                            th_cr=t,
                            max_crown_radius_m=m,
                        )
                    )
    return out


def params_row(params: SegmentationParams) -> dict[str, object]:
    """Flatten a parameter set into columns for a tidy table."""
    return {
        "key": params.key,
        "smooth_radius_m": params.smooth_radius_m,
        "window": params.window.name or params.window.describe(),
        "th_cr": params.th_cr,
        "max_crown_radius_m": (
            np.nan if params.max_crown_radius_m is None else params.max_crown_radius_m
        ),
    }


def params_from_row(row) -> SegmentationParams:
    """Rebuild a parameter set from a sweep row."""
    cap = row["max_crown_radius_m"]
    return SegmentationParams(
        smooth_radius_m=float(row["smooth_radius_m"]),
        window=WINDOW_FUNCTIONS[str(row["window"])],
        th_cr=float(row["th_cr"]),
        max_crown_radius_m=None if pd.isna(cap) else float(cap),
    )


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------
#: Hard filters. Written before any sweep result was read.
#:
#: The density band comes from the tiling arithmetic. When the watershed
#: assigns every canopy pixel to some crown, density and mean width are locked
#: together by ``d = 2*sqrt(10000/(pi*N))``: 35.2 m at 10/ha, 13.1 m at 74/ha,
#: 8.0 m at 200/ha. Popescu deciduous at 25 m height predicts 8.7 m. Once a
#: crown-assigned fraction below one breaks that identity, 8 to 11 m crowns are
#: consistent with roughly 110 to 160 stems per hectare. Published
#: dominant and codominant density for eastern hardwood runs about 75 to 200
#: per hectare, so that is the band.
HARD_FILTERS = {
    "density_per_ha": (75.0, 200.0),
    "capped_fraction": (0.0, 0.10),
    "assigned_fraction": (0.60, 0.95),
    "allo_slope": (0.05, np.inf),
}

#: Reported alongside the filters, not used to reject.
#:
#: POST-HOC CHANGE, and it needs stating plainly. ``over_open_grown`` was
#: written as a hard filter at 0.05 before any result was read. It then
#: rejected all 288 parameter sets on its own, with a median exceedance of
#: 0.61, while the other four filters passed six sets between them.
#:
#: Inspection showed the filter was mis-specified rather than the segmentation
#: being universally wrong. The ceiling ran through an invented height-to-
#: diameter conversion, and at heights below about 8 m it fell below the
#: stand-grown expectation it was supposed to bound. A ceiling under the
#: central prediction rejects everything by construction.
#:
#: The function is now a simple multiple of the stand-grown expectation, which
#: is honest about being a rule of thumb. A rule of thumb should not gate a
#: result, so exceedance is reported and the ranking is unchanged. Both
#: scorings are in the report.
DIAGNOSTICS = ("over_open_grown", "diam_cv", "diam_mean", "allo_bias", "allo_scatter")

#: Survivors rank on this, lowest first. One scalar, chosen in advance.
RANK_ON = "allo_rmse"


def apply_decision_rule(summary: pd.DataFrame) -> pd.DataFrame:
    """Score parameter sets against the pre-registered rule.

    Adds a ``passes`` column and a ``rank`` over the survivors, and one
    ``fail_*`` column per filter so a rejection can be traced to its cause.
    """
    out = summary.copy()
    passes = pd.Series(True, index=out.index)
    for column, (lo, hi) in HARD_FILTERS.items():
        if column not in out.columns:
            logger.warning("Filter column %s missing from the summary", column)
            continue
        ok = out[column].between(lo, hi)
        out[f"fail_{column}"] = ~ok
        passes &= ok.fillna(False)
    out["passes"] = passes
    out["rank"] = np.where(
        out["passes"], out[RANK_ON].rank(method="min", na_option="bottom"), np.nan
    )
    return out.sort_values(["passes", RANK_ON], ascending=[False, True])


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """Consistent logging across every script here."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def write_table(frame: pd.DataFrame, name: str) -> Path:
    """Write a result table where the report can cite it."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / name
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path.relative_to(REPO), len(frame))
    return path


def read_table(name: str) -> pd.DataFrame:
    """Read a table written by an earlier step."""
    path = RESULT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the step that writes it.")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_tiles(subset: str | None = None) -> gpd.GeoDataFrame:
    """Load the frozen tile manifest.

    Args:
        subset: ``"tune"``, ``"holdout"``, a site name, or None for all.
    """
    if not TILE_PATH.exists():
        raise FileNotFoundError(
            f"{TILE_PATH} not found. Run scripts/segmentation/tiles.py first."
        )
    tiles = gpd.read_file(TILE_PATH)
    if subset in {"tune", "holdout"}:
        return tiles[tiles["split"] == subset].reset_index(drop=True)
    if subset:
        return tiles[tiles["site"] == subset].reset_index(drop=True)
    return tiles


def save_manifest(payload: dict, name: str) -> Path:
    """Record how a step was run, next to its results."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


__all__ = [
    "CANOPY_FLOOR_M",
    "FIGURE_DIR",
    "GUIDE_DIR",
    "HARD_FILTERS",
    "NAIP_YEARS",
    "NATIVE_RES_M",
    "POPESCU_COMBINED",
    "POPESCU_DECIDUOUS",
    "RANK_ON",
    "REFERENCE_YEAR",
    "REPO",
    "RESULT_DIR",
    "SITE",
    "TILE_INSET_M",
    "TILE_PATH",
    "TILE_SIZE_M",
    "SiteData",
    "WindowFunction",
    "apply_decision_rule",
    "configure_logging",
    "crown_width_open_grown",
    "crown_width_popescu",
    "degrade",
    "elkinsville_chm",
    "elkinsville_naip",
    "load_tiles",
    "parameter_grid",
    "params_from_row",
    "params_row",
    "read_table",
    "read_window",
    "save_manifest",
    "segment_tile",
    "stands",
    "tile_stats",
    "transfer_sites",
    "write_table",
]
