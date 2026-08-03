"""csdv_core.io.stands — read and decode AIS disturbance polygon deliveries.

The photo-interpretation team delivers one file geodatabase per DOQQ mapping
module. Each feature is an *impact polygon*: a disturbance footprint, optionally
subdivided where percent cover varies inside it. That polygon is the base unit
the classification system works on, so this module loads it, decodes the
protocol's integer code lists into readable labels and numeric cover bounds, and
assigns each feature a stable identifier.

Code lists come from the project's photo-interpretation protocol,
``Project Documentation/Markdown/FInalDRAFT_Eastern Forest Disturbance Protocol
Work Plan_3-31-2026.md``: disturbance types in section 4 (Disturbance Footprint
Attributes) and percent cover classes in section 4 (Percent Cover Attributes).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

DEFAULT_LAYER = "DisturbancePoly"

#: Disturbance type codes. Undifferentiated parents end in 0.
DIST_TYPE_LABELS: dict[int, str] = {
    110: "Fire, undifferentiated",
    111: "Crown fire",
    120: "Harvest, undifferentiated",
    121: "Clearcut",
    122: "Clearcut with reserves",
    123: "Overstory removal",
    124: "Seed tree establishment cut",
    125: "Shelterwood establishment cut",
    126: "Strip thinning",
    127: "Uneven-age / selection harvest",
    128: "Two-age harvest",
    130: "Weather, undifferentiated",
    131: "Wind",
    132: "Snow / ice",
    133: "Flood",
    140: "Mining, undifferentiated",
    141: "Mining, active",
    142: "Mining, inactive",
    970: "Unknown disturbance",
    971: "Tree mortality",
    972: "Crown discoloration",
    973: "Defoliation",
    974: "Multiple characteristics",
    999: "Non-disturbance area",
}

#: Coarse grouping, keyed by the hundreds digit of the disturbance type.
DIST_GROUP_LABELS: dict[int, str] = {
    110: "Fire",
    120: "Harvest",
    130: "Weather",
    140: "Mining",
    970: "Unknown",
    990: "Non-disturbance",
}

#: Percent cover class -> (lower, upper) as fractions. 9 is "not assessed".
COVER_CLASS_RANGE: dict[int, tuple[float, float]] = {
    0: (0.00, 0.01),
    1: (0.01, 0.05),
    2: (0.05, 0.10),
    3: (0.10, 0.20),
    4: (0.20, 0.40),
    5: (0.40, 0.60),
    6: (0.60, 0.80),
    7: (0.80, 1.00),
    9: (np.nan, np.nan),
}

#: Class midpoints, for plotting a measured value against the interpreter bin.
COVER_CLASS_MIDPOINT: dict[int, float] = {
    code: float(np.mean(bounds)) for code, bounds in COVER_CLASS_RANGE.items()
}

#: Tree type affected, assigned only for unknown disturbance agents (970-974).
TREE_TYPE_LABELS: dict[int, str] = {
    500: "Deciduous",
    600: "Coniferous",
    700: "All tree types",
    999: "Not applicable",
}

#: Whether the disturbance footprint is fully inside the mapping module.
WITHIN_MAPPING_AREA_LABELS: dict[int, str] = {
    0: "Extends beyond module",
    1: "Complete within module",
    9: "Not applicable",
}

#: Additional disturbance between the base and follow-up imagery.
ADDITIONAL_DISTURBANCE_LABELS: dict[int, str] = {
    0: "None",
    1: "Present",
    8: "Undetermined",
    9: "Not assessed",
}

NON_DISTURBANCE_CODE = 999

#: The four dates at which the interpreter recorded percent cover. The tuples
#: give (label, year column, tree cover column, woody regeneration column).
#: Regeneration is only assessed on the base and follow-up imagery.
COVER_SERIES_FIELDS: tuple[tuple[str, str, str, str | None], ...] = (
    ("pre-disturbance", "LastImageryPreDist", "PercentTreePreDist", None),
    ("first post-disturbance", "FirstImageryPostDist", "PercentTreeFirstPost", None),
    ("base", "BaseDate", "PercentTreeBase", "PercentWoodyRegenBase"),
    ("follow-up", "FollowUpDate", "PercentTreeFollowUp", "PercentWoodyRegenFollowUp"),
)

#: Year sentinel meaning "not applicable" in the imagery date fields.
YEAR_NOT_APPLICABLE = 9999

__all__ = [
    "ADDITIONAL_DISTURBANCE_LABELS",
    "COVER_CLASS_MIDPOINT",
    "COVER_CLASS_RANGE",
    "COVER_SERIES_FIELDS",
    "DIST_GROUP_LABELS",
    "DIST_TYPE_LABELS",
    "TREE_TYPE_LABELS",
    "WITHIN_MAPPING_AREA_LABELS",
    "cover_bounds",
    "cover_midpoint",
    "cover_series_frame",
    "interpreter_cover_series",
    "module_footprint",
    "read_ais_stands",
    "stands_by_id",
]


def cover_midpoint(code: int | float | None) -> float:
    """Return the midpoint of a percent cover class as a fraction.

    Unknown or not-assessed codes return NaN rather than raising, because the
    delivery uses 9 for every attribute outside a disturbance footprint.
    """
    if code is None or (isinstance(code, float) and not np.isfinite(code)):
        return float("nan")
    return COVER_CLASS_MIDPOINT.get(int(code), float("nan"))


def cover_bounds(code: int | float | None) -> tuple[float, float]:
    """Return the (lower, upper) fractional bounds of a percent cover class."""
    if code is None or (isinstance(code, float) and not np.isfinite(code)):
        return (float("nan"), float("nan"))
    return COVER_CLASS_RANGE.get(int(code), (float("nan"), float("nan")))


def _dist_group(dist_type: int) -> str:
    """Map a disturbance type to its coarse group label."""
    if dist_type == NON_DISTURBANCE_CODE:
        return DIST_GROUP_LABELS[990]
    parent = (int(dist_type) // 10) * 10
    return DIST_GROUP_LABELS.get(parent, "Unknown")


def _module_code(doqq: str) -> str:
    """Compress a DOQQ name into a short uppercase identifier prefix.

    ``"ElkinsvilleNE"`` becomes ``"ELKNE"``: the first three letters of the
    name plus any trailing quarter-quad suffix.
    """
    name = str(doqq or "UNK")
    suffix = ""
    while name and name[-1].isupper() and name[-1] in "NSEW":
        suffix = name[-1] + suffix
        name = name[:-1]
    return (name[:3] + suffix).upper()


def _assign_stand_ids(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Build a stable, unique identifier for each impact polygon.

    The protocol says the disturbance UID combined with the DOQQ name forms a
    project-wide unique identifier for a *footprint*. A footprint subdivided
    into several impact polygons therefore repeats, so a letter suffix ordered
    by descending area disambiguates the pieces.
    """
    prefix = gdf["DOQQ"].map(_module_code)
    base = (
        prefix
        + "-U"
        + gdf["UID1"].astype(int).astype(str)
        + "-"
        + gdf["UID2"].astype(int).astype(str)
        + "-"
        + gdf["UID3"].astype(int).astype(str)
    )
    ids = base.copy()
    order = gdf.geometry.area.rank(ascending=False, method="first")
    for key, idx in base.groupby(base).groups.items():
        if len(idx) < 2:
            continue
        ranked = order.loc[idx].sort_values().index
        for position, row in enumerate(ranked):
            ids.loc[row] = f"{key}{chr(ord('a') + position)}"
        logger.info("Footprint %s has %d impact polygons", key, len(idx))
    return ids


def read_ais_stands(
    path: Path | str,
    *,
    layer: str = DEFAULT_LAYER,
    drop_non_disturbance: bool = True,
) -> gpd.GeoDataFrame:
    """Read an AIS disturbance polygon delivery and decode its code lists.

    Geometries are kept as delivered, including MultiPolygon, because an impact
    polygon is one stand even when it has several parts. Invalid geometries are
    repaired with :func:`shapely.make_valid` and the count is logged.

    Args:
        path: Path to the file geodatabase.
        layer: Layer name. Defaults to ``"DisturbancePoly"``.
        drop_non_disturbance: Drop features coded 999, which fill the module
            outside any disturbance footprint. Use :func:`module_footprint` to
            recover the module outline.

    Returns:
        A GeoDataFrame with the delivered attributes plus ``stand_id``,
        ``dist_label``, ``dist_group``, ``tree_type_label``, ``area_m2``,
        ``perimeter_m`` and ``bbox_fill``.

    Raises:
        ValueError: If the layer is empty or has no projected CRS.
    """
    path = Path(path)
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise ValueError(f"Layer {layer!r} in {path} is empty")
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError(
            f"Layer {layer!r} needs a projected CRS for area metrics, got {gdf.crs}"
        )

    invalid = ~gdf.geometry.is_valid
    if int(invalid.sum()):
        logger.warning("Repairing %d invalid geometries", int(invalid.sum()))
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    gdf = gdf.reset_index(drop=True)
    gdf["stand_id"] = _assign_stand_ids(gdf)
    gdf["dist_label"] = gdf["DistType"].map(DIST_TYPE_LABELS)
    gdf["dist_group"] = gdf["DistType"].map(_dist_group)
    gdf["tree_type_label"] = gdf["TreeType"].map(TREE_TYPE_LABELS)
    gdf["area_m2"] = gdf.geometry.area
    gdf["perimeter_m"] = gdf.geometry.length

    bounds = gdf.geometry.bounds
    bbox_area = (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"])
    gdf["bbox_fill"] = np.where(bbox_area > 0, gdf["area_m2"] / bbox_area, np.nan)

    n_total = len(gdf)
    if drop_non_disturbance:
        gdf = gdf[gdf["DistType"] != NON_DISTURBANCE_CODE].reset_index(drop=True)
    logger.info(
        "Read %d features from %s (%d disturbance polygons, CRS %s)",
        n_total,
        path.name,
        len(gdf),
        gdf.crs,
    )
    return gdf


def _fill_holes(geom: BaseGeometry) -> BaseGeometry:
    """Return ``geom`` with every interior ring removed."""
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    filled = [Polygon(part.exterior) for part in parts]
    return filled[0] if len(filled) == 1 else MultiPolygon(filled)


def module_footprint(
    path: Path | str,
    *,
    layer: str = DEFAULT_LAYER,
) -> BaseGeometry:
    """Return the outline of the DOQQ mapping module.

    The delivery covers the module wall to wall, so the union of every feature
    is the module. Sliver gaps survive that union where adjacent polygons do
    not share exact vertices, so interior rings are dropped: a mapping module
    has no legitimate holes.
    """
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise ValueError(f"Layer {layer!r} in {path} is empty")
    footprint = _fill_holes(gdf.geometry.union_all())
    logger.info(
        "Module footprint: %.2f km2 from %d features",
        footprint.area / 1e6,
        len(gdf),
    )
    return footprint


def interpreter_cover_series(stand: pd.Series) -> pd.DataFrame:
    """Return the interpreter's four-point cover record for one stand.

    The photo interpreters record tree cover on four images: the last one
    before the disturbance, the first one after it, the base image used for
    delineation, and a follow-up. Woody regeneration is recorded on the last
    two only. Together these give an independent reference series to compare a
    NAIP-derived crown fraction against.

    Args:
        stand: One row of the frame returned by :func:`read_ais_stands`.

    Returns:
        A four-row frame with columns ``which``, ``year``, ``tree_class``,
        ``tree_lo``, ``tree_hi``, ``tree_mid``, ``regen_class``, ``regen_lo``,
        ``regen_hi`` and ``regen_mid``. Cover bounds are fractions. A year of
        9999 becomes NaN.
    """
    rows: list[dict[str, float | str | None]] = []
    for which, year_field, tree_field, regen_field in COVER_SERIES_FIELDS:
        year = stand.get(year_field)
        year = float("nan") if year in (None, YEAR_NOT_APPLICABLE) else float(year)
        tree_class = stand.get(tree_field)
        tree_lo, tree_hi = cover_bounds(tree_class)
        row: dict[str, float | str | None] = {
            "which": which,
            "year": year,
            "tree_class": tree_class,
            "tree_lo": tree_lo,
            "tree_hi": tree_hi,
            "tree_mid": cover_midpoint(tree_class),
        }
        regen_class = stand.get(regen_field) if regen_field else None
        regen_lo, regen_hi = cover_bounds(regen_class)
        row.update(
            {
                "regen_class": regen_class,
                "regen_lo": regen_lo,
                "regen_hi": regen_hi,
                "regen_mid": cover_midpoint(regen_class),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cover_series_frame(stands: gpd.GeoDataFrame) -> pd.DataFrame:
    """Stack :func:`interpreter_cover_series` for every stand in ``stands``."""
    frames: list[pd.DataFrame] = []
    for _, stand in stands.iterrows():
        frame = interpreter_cover_series(stand)
        frame.insert(0, "stand_id", stand["stand_id"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def stands_by_id(
    stands: gpd.GeoDataFrame,
    stand_ids: Sequence[str],
) -> gpd.GeoDataFrame:
    """Select stands by identifier, preserving the order given.

    Raises:
        KeyError: If any requested identifier is absent.
    """
    lookup = stands.set_index("stand_id")
    missing = [sid for sid in stand_ids if sid not in lookup.index]
    if missing:
        raise KeyError(f"Unknown stand_id values: {missing}")
    return lookup.loc[list(stand_ids)].reset_index()
