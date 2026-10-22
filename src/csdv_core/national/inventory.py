"""csdv_core.national.inventory — the national NAIP-CHM tile inventory.

The NTSG index lists one row per CHM tile. A few quarter quads appear more than
once, from overlapping NAIP years, so the inventory keeps the latest
acquisition of each and records how many it dropped.

Two geometries matter. The index ``.geo`` polygon is the raster footprint,
which runs about 200 m past the quarter quad on every side and so overlaps its
neighbours. The nominal quarter quad, a 3.75 minute cell in NAD83 geographic
coordinates, follows from the tile name and tiles CONUS without overlap. A crown
belongs to the tile whose nominal cell holds its tree top, and the 200 m
margin means that crown is always segmented whole inside that tile.

Inputs: the NTSG ``index.csv``, optional state and EPA ecoregion polygons.
Outputs: a GeoDataFrame with one row per quarter quad, plus a summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from csdv_core.download.naip_chm_http import https_url

logger = logging.getLogger(__name__)

#: Equal-area CRS for footprint areas and region joins.
ALBERS_CRS = "EPSG:5070"

INDEX_COLUMNS = (
    "system:index",
    "acquisition_date",
    "chm_url",
    "quad_id",
    "quarter",
    "scale_factor",
    "source_doqq",
    "utm_zone",
    "year",
    ".geo",
)

#: Columns kept from the EPA Level III ecoregion layer.
ECOREGION_COLUMNS = ("NA_L1CODE", "NA_L1NAME", "NA_L2CODE", "NA_L2NAME", "US_L3CODE")

#: Side of a quarter-quad cell in degrees (3.75 minutes).
QUARTER_QUAD_DEG = 0.0625

#: Geographic CRS of the nominal cells (NAIP DOQQs are cut on NAD83).
CELL_CRS = "EPSG:4269"

__all__ = [
    "ALBERS_CRS",
    "CELL_CRS",
    "InventorySummary",
    "attach_regions",
    "build_inventory",
    "drop_duplicate_quarter_quads",
    "quarter_quad_bounds",
    "read_index",
    "source_resolution_m",
]


@dataclass(frozen=True)
class InventorySummary:
    """Counts describing one inventory build.

    Attributes:
        n_index_rows: Rows in the index.
        n_tiles: Tiles kept, one per quarter quad.
        n_duplicates_dropped: Older acquisitions of a repeated quarter quad.
        footprint_km2: Summed nominal footprint area of the kept tiles.
        by_year: Kept tiles per NAIP year.
        by_source_res: Kept tiles per source NAIP resolution in metres.
    """

    n_index_rows: int
    n_tiles: int
    n_duplicates_dropped: int
    footprint_km2: float
    by_year: dict[int, int] = field(default_factory=dict)
    by_source_res: dict[float, int] = field(default_factory=dict)


def source_resolution_m(source_doqq: pd.Series) -> pd.Series:
    """Source NAIP resolution in metres from the DOQQ name.

    The fifth underscore field is ``060`` or ``030`` (hundredths of a metre) on
    almost every tile. A few 2017 tiles carry ``1`` for 1 m. Anything else is
    returned as NaN.
    """
    token = source_doqq.astype(str).str.split("_").str[4]
    out = pd.to_numeric(token, errors="coerce")
    return out.where(token.str.len() != 3, out / 100.0)


def quarter_quad_bounds(quarter_quad: pd.Series) -> pd.DataFrame:
    """Nominal NAD83 bounds of quarter quads named like ``3807504_ne``.

    The first five digits give the latitude and longitude of the 1 degree
    block's south-east corner. The next two number its 64 USGS 7.5 minute
    quads from 01 at the north-west corner, row by row eastward. The suffix
    picks one quarter of that quad.

    Returns:
        Columns ``cell_west``, ``cell_south``, ``cell_east``, ``cell_north`` in
        decimal degrees (west negative).
    """
    q = quarter_quad.astype(str).str
    lat0 = q[0:2].astype(int).to_numpy()
    lon0 = q[2:5].astype(int).to_numpy()
    cell = q[5:7].astype(int).to_numpy()
    quarter = q[8:10]
    row, col = (cell - 1) // 8, (cell - 1) % 8
    north = lat0 + 1 - row * 2 * QUARTER_QUAD_DEG
    west = -(lon0 + 1) + col * 2 * QUARTER_QUAD_DEG
    north = north - np.where(quarter.str[0] == "s", QUARTER_QUAD_DEG, 0.0)
    west = west + np.where(quarter.str[1] == "e", QUARTER_QUAD_DEG, 0.0)
    return pd.DataFrame(
        {
            "cell_west": west,
            "cell_south": north - QUARTER_QUAD_DEG,
            "cell_east": west + QUARTER_QUAD_DEG,
            "cell_north": north,
        },
        index=quarter_quad.index,
    )


def read_index(path: Path | str) -> pd.DataFrame:
    """Read the NTSG index into typed columns.

    Args:
        path: ``index.csv`` from :func:`csdv_core.download.naip_chm_http.fetch_index`.

    Returns:
        A DataFrame with ``tile_id``, ``quarter_quad``, ``acquisition_date``
        (datetime), ``year``, ``utm_zone``, ``source_res_m``, ``scale_factor``
        (multiplier to metres, 0.01),
        ``chm_url`` (https) and the GeoJSON footprint string ``geo``.
    """
    frame = pd.read_csv(path, usecols=list(INDEX_COLUMNS))
    out = pd.DataFrame(
        {
            "tile_id": frame["system:index"].astype(str),
            "quarter_quad": frame["quad_id"].astype(int).astype(str).str.zfill(7)
            + "_"
            + frame["quarter"].astype(str),
            "acquisition_date": pd.to_datetime(frame["acquisition_date"]),
            "year": frame["year"].astype(int),
            "utm_zone": frame["utm_zone"].astype(int),
            "source_doqq": frame["source_doqq"].astype(str),
            "source_res_m": source_resolution_m(frame["source_doqq"]),
            # The index stores the divisor (100). The rest of csdv_core uses a
            # multiplier from stored values to metres, as io.raster does.
            "scale_factor": 1.0 / frame["scale_factor"].astype(float),
            "chm_url": frame["chm_url"].astype(str).map(https_url),
            "geo": frame[".geo"].astype(str),
        }
    )
    logger.info("Read %d index rows from %s", len(out), path)
    return out


def drop_duplicate_quarter_quads(index: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the latest acquisition of each quarter quad.

    Returns:
        ``(kept, n_dropped)``. Ties on date keep the later index row.
    """
    ordered = index.sort_values(["quarter_quad", "acquisition_date"], kind="stable")
    kept = ordered.drop_duplicates("quarter_quad", keep="last")
    n_dropped = len(index) - len(kept)
    return kept.sort_index(), n_dropped


def attach_regions(
    tiles: gpd.GeoDataFrame,
    *,
    states: gpd.GeoDataFrame | None = None,
    ecoregions: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Label each tile with the state and ecoregion at its footprint centre.

    A centre point rather than the largest overlap keeps the join cheap over
    two hundred thousand tiles, and a quarter quad is small enough (about
    40 km²) that the two rarely disagree. Tiles whose centre falls outside
    every polygon (coastal tiles over water) get missing labels.

    Args:
        tiles: Tiles with footprint geometry in any CRS.
        states: Polygons with a ``STUSPS`` column.
        ecoregions: EPA Level III polygons with :data:`ECOREGION_COLUMNS`.

    Returns:
        A copy of ``tiles`` with ``state`` and the ecoregion columns added.
    """
    out = tiles.copy()
    centres = gpd.GeoDataFrame(
        geometry=tiles.to_crs(ALBERS_CRS).geometry.centroid,
        index=tiles.index,
        crs=ALBERS_CRS,
    )
    if states is not None:
        joined = gpd.sjoin(
            centres, states[["STUSPS", "geometry"]].to_crs(ALBERS_CRS), how="left"
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        out["state"] = joined["STUSPS"].reindex(out.index)
    if ecoregions is not None:
        cols = [c for c in ECOREGION_COLUMNS if c in ecoregions.columns]
        joined = gpd.sjoin(
            centres, ecoregions[cols + ["geometry"]].to_crs(ALBERS_CRS), how="left"
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        for col in cols:
            out[col.lower()] = joined[col].reindex(out.index)
    return out


def build_inventory(
    index_path: Path | str,
    *,
    states_path: Path | str | None = None,
    ecoregions_path: Path | str | None = None,
) -> tuple[gpd.GeoDataFrame, InventorySummary]:
    """Build the national tile inventory from the NTSG index.

    Args:
        index_path: NTSG ``index.csv``.
        states_path: Optional state polygons (anything geopandas reads,
            including ``zip://`` paths).
        ecoregions_path: Optional EPA Level III ecoregion polygons.

    Returns:
        ``(tiles, summary)``. ``tiles`` is in EPSG:4326 with the raster
        footprint as geometry, ``footprint_km2``, the nominal cell bounds from
        :func:`quarter_quad_bounds`, and region labels when polygons were given.
    """
    index = read_index(index_path)
    kept, n_dropped = drop_duplicate_quarter_quads(index)
    geometry = shapely.from_geojson(kept["geo"].to_numpy())
    tiles = gpd.GeoDataFrame(
        kept.drop(columns="geo"), geometry=geometry, crs="EPSG:4326"
    )
    tiles["footprint_km2"] = tiles.to_crs(ALBERS_CRS).geometry.area / 1e6
    tiles = tiles.join(quarter_quad_bounds(tiles["quarter_quad"]))

    states = gpd.read_file(states_path) if states_path is not None else None
    ecoregions = gpd.read_file(ecoregions_path) if ecoregions_path else None
    if states is not None or ecoregions is not None:
        tiles = attach_regions(tiles, states=states, ecoregions=ecoregions)
    tiles = tiles.reset_index(drop=True)

    res = tiles["source_res_m"].fillna(-1.0)
    summary = InventorySummary(
        n_index_rows=len(index),
        n_tiles=len(tiles),
        n_duplicates_dropped=n_dropped,
        footprint_km2=float(tiles["footprint_km2"].sum()),
        by_year={int(k): int(v) for k, v in tiles["year"].value_counts().items()},
        by_source_res={
            float(k): int(v) for k, v in res.value_counts().items() if np.isfinite(k)
        },
    )
    logger.info(
        "Inventory: %d tiles from %d index rows (%d older duplicates dropped), "
        "%.0f km² of footprint",
        summary.n_tiles,
        summary.n_index_rows,
        summary.n_duplicates_dropped,
        summary.footprint_km2,
    )
    return tiles, summary
