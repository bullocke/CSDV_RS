"""csdv_core.disturbance.sampling: draw an inspection sample from a disturbance run.

The sample is meant for a quick visual check of a run before a formal accuracy
assessment. It has three strata:

``detected``
    Objects of ``detection/disturbances.gpkg`` (after the MMU), stratified by
    NAIP interval. Every interval gets at least ``min_per_interval`` objects
    (or all of them when it has fewer), and the rest of the sample is shared
    out in proportion to the number of objects per interval.
``below_mmu``
    Near misses: objects that the decision rule flagged but the minimum
    mapping unit removed in every interval (``onset_year_nommu > 0`` and
    ``onset_year == 0``), grouped by the interval the rule first passed and
    limited to objects of at least half the MMU so the chips show more than a
    single cell. They show what the MMU throws away. Their ``object_id`` is
    blank because they are not part of ``disturbances.gpkg``.
``no_detection``
    Single cells of monitored forest that were never flagged, each paired
    with a random interval in which the cell was monitored. They show what
    undisturbed forest looks like at the same scale.

Within a stratum, units are drawn at random without replacement with a fixed
seed, so the same design and run always give the same sample.

Inputs: a saved run (:class:`csdv_core.disturbance.store.SavedRun`). Outputs:
a GeoDataFrame with :data:`SAMPLE_COLUMNS` in the grid CRS.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from csdv_core.disturbance.decide import onset_stack
from csdv_core.disturbance.outlines import (
    detection_objects,
    interval_label,
    objects_table,
)
from csdv_core.disturbance.store import OBJECTS_LAYER, SavedRun, read_year_stack

logger = logging.getLogger(__name__)

#: Stratum of objects kept after the minimum mapping unit.
STRATUM_DETECTED = "detected"
#: Stratum of flagged objects removed by the minimum mapping unit.
STRATUM_BELOW_MMU = "below_mmu"
#: Stratum of monitored forest cells never flagged.
STRATUM_NO_DETECTION = "no_detection"
#: Strata in the order they appear in the sample table.
STRATA: tuple[str, ...] = (STRATUM_DETECTED, STRATUM_BELOW_MMU, STRATUM_NO_DETECTION)

#: Smallest below-MMU object drawn, as a share of the minimum mapping unit.
NEAR_MISS_MMU_SHARE = 0.5

#: Columns of the sample table (plus ``geometry`` in the GeoDataFrame).
SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "stratum",
    "object_id",
    "interval",
    "year_before",
    "onset_year",
    "n_cells",
    "area_ha",
    "mean_prob",
    "max_prob",
    "mean_z",
    "follow_up",
    "centroid_x",
    "centroid_y",
    "centroid_lon",
    "centroid_lat",
)


@dataclass(frozen=True)
class SampleDesign:
    """Sample sizes and seed of an inspection sample.

    Attributes:
        n_detected: Objects drawn from the ``detected`` stratum.
        n_below_mmu: Objects drawn from the ``below_mmu`` stratum.
        n_no_detection: Cells drawn from the ``no_detection`` stratum.
        min_per_interval: Minimum ``detected`` objects per interval (fewer
            when an interval has fewer objects).
        seed: Seed of the random generator.
    """

    n_detected: int = 24
    n_below_mmu: int = 4
    n_no_detection: int = 0
    min_per_interval: int = 2
    seed: int = 0


def allocate_by_stratum(
    counts: Mapping[str, int], n_total: int, min_per_stratum: int
) -> dict[str, int]:
    """Share a sample size among strata, with a floor per stratum.

    Each stratum first gets ``min(min_per_stratum, count)``. When those floors
    alone exceed ``n_total``, one unit at a time goes to the largest strata
    instead. The remaining units are handed out one at a time to the stratum
    with the largest ``count / (allocated + 1)`` (the D'Hondt rule), which
    keeps the allocation close to proportional. No stratum gets more units
    than it has. When ``n_total`` covers every unit, every unit is taken.

    Args:
        counts: Number of units per stratum.
        n_total: Total sample size.
        min_per_stratum: Floor per stratum.

    Returns:
        Units per stratum, with the same keys as ``counts``.
    """
    allocation = {name: 0 for name in counts}
    available_total = sum(counts.values())
    n_target = min(n_total, available_total)
    floors = {name: min(min_per_stratum, count) for name, count in counts.items()}
    if sum(floors.values()) <= n_target:
        allocation.update(floors)
    n_left = n_target - sum(allocation.values())
    while n_left > 0:
        best_name = None
        best_score = -1.0
        for name, count in counts.items():
            if allocation[name] >= count:
                continue
            score = count / (allocation[name] + 1)
            if score > best_score:
                best_name = name
                best_score = score
        allocation[best_name] += 1
        n_left -= 1
    return allocation


def _draw_rows(
    frame: pd.DataFrame, n_rows: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw ``n_rows`` rows of ``frame`` at random without replacement.

    Args:
        frame: Rows to draw from.
        n_rows: Number of rows to keep (all rows when it exceeds the length).
        rng: Random generator.

    Returns:
        The drawn rows, in their original order.
    """
    n_keep = min(n_rows, len(frame))
    positions = rng.choice(len(frame), size=n_keep, replace=False)
    sorted_positions = np.sort(positions)
    return frame.iloc[sorted_positions]


def sample_objects(
    objects: gpd.GeoDataFrame,
    n_samples: int,
    min_per_interval: int,
    rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Draw objects stratified by interval with :func:`allocate_by_stratum`.

    Args:
        objects: Objects with an ``interval`` column.
        n_samples: Total number of objects to draw.
        min_per_interval: Floor per interval.
        rng: Random generator.

    Returns:
        The drawn objects, ordered by onset year and then object id.
    """
    if objects.empty or n_samples <= 0:
        return objects.iloc[0:0]
    counts_series = objects.groupby("interval").size()
    counts = counts_series.to_dict()
    allocation = allocate_by_stratum(counts, n_samples, min_per_interval)
    logger.info("Objects per interval %s, drawn %s", counts, allocation)
    drawn_parts = []
    for interval, interval_objects in objects.groupby("interval"):
        drawn = _draw_rows(interval_objects, allocation[interval], rng)
        drawn_parts.append(drawn)
    combined = pd.concat(drawn_parts)
    ordered = combined.sort_values(["onset_year", "object_id"])
    return gpd.GeoDataFrame(ordered, crs=objects.crs)


def _read_band(path: object) -> np.ndarray:
    """Read band 1 of a single-band raster.

    Args:
        path: Raster path.

    Returns:
        The band array.
    """
    with rasterio.open(path) as src:
        return src.read(1)


def below_mmu_objects(saved: SavedRun) -> gpd.GeoDataFrame:
    """Vectorize cells flagged by the rule but removed by the MMU.

    Args:
        saved: The saved run.

    Returns:
        Objects with the columns of
        :func:`csdv_core.disturbance.outlines.detection_objects`.
    """
    run_paths = saved.run_paths
    onset_year = _read_band(run_paths.onset_year)
    onset_year_nommu = _read_band(run_paths.onset_year_nommu)
    removed = (onset_year_nommu > 0) & (onset_year <= 0)
    onset_index = np.full(onset_year.shape, -1, dtype=np.int8)
    for date_index, year in enumerate(saved.years):
        year_cells = removed & (onset_year_nommu == year)
        onset_index[year_cells] = date_index
    probability = None
    if run_paths.probability.exists():
        probability_stack = read_year_stack(run_paths.probability)
        probability = probability_stack.values
    removed_stack = onset_stack(onset_index, len(saved.years))
    return detection_objects(
        removed_stack, probability, saved.z, saved.grid, saved.years
    )


def no_detection_cells(
    saved: SavedRun, n_samples: int, rng: np.random.Generator
) -> gpd.GeoDataFrame:
    """Draw monitored forest cells that were never flagged.

    A cell qualifies when it was monitored in at least one interval and its
    ``onset_year_nommu`` is 0. Each drawn cell is paired with a random
    interval in which it was monitored.

    Args:
        saved: The saved run.
        n_samples: Number of cells to draw.
        rng: Random generator.

    Returns:
        One square cell polygon per drawn cell, with the object columns
        (``object_id``, ``mean_prob`` and ``max_prob`` are NaN).
    """
    grid = saved.grid
    onset_year_nommu = _read_band(saved.run_paths.onset_year_nommu)
    forest = saved.layers.forest
    monitored = np.zeros_like(forest)
    monitored[1:] = forest[:-1]
    ever_monitored = monitored[1:].any(axis=0)
    candidates = ever_monitored & (onset_year_nommu == 0)
    candidate_rows, candidate_cols = np.nonzero(candidates)
    n_keep = min(n_samples, len(candidate_rows))
    picks = rng.choice(len(candidate_rows), size=n_keep, replace=False)
    rows = []
    for pick in picks:
        row = int(candidate_rows[pick])
        col = int(candidate_cols[pick])
        monitored_dates = np.nonzero(monitored[:, row, col])[0]
        date_index = int(rng.choice(monitored_dates))
        year = saved.years[date_index]
        year_before = saved.years[date_index - 1]
        left, top = grid.cell_transform * (col, row)
        right, bottom = grid.cell_transform * (col + 1, row + 1)
        cell_polygon = box(left, bottom, right, top)
        rows.append(
            {
                "object_id": np.nan,
                "onset_year": year,
                "year_before": year_before,
                "interval": interval_label(year_before, year),
                "n_cells": 1,
                "area_ha": grid.cell_ha,
                "mean_prob": np.nan,
                "max_prob": np.nan,
                "mean_z": float(saved.z[date_index, row, col]),
                "geometry": cell_polygon,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=grid.crs)


def draw_samples(saved: SavedRun, design: SampleDesign) -> gpd.GeoDataFrame:
    """Draw the inspection sample of a run.

    Args:
        saved: The saved run.
        design: Sample sizes and seed.

    Returns:
        GeoDataFrame with :data:`SAMPLE_COLUMNS` and ``geometry`` (grid CRS).
        ``follow_up`` is True when the unit's onset year is not the run's
        last year (computed here, so older runs without the objects column
        get it too). ``sample_id`` is ``S01``, ``S02``, ... in table order: detected
        objects, then below-MMU objects, then no-detection cells, each
        ordered by onset year.
    """
    rng = np.random.default_rng(design.seed)
    objects = gpd.read_file(saved.run_paths.disturbances_gpkg, layer=OBJECTS_LAYER)
    detected = sample_objects(objects, design.n_detected, design.min_per_interval, rng)
    parts = {STRATUM_DETECTED: detected}
    if design.n_below_mmu > 0:
        removed = below_mmu_objects(saved)
        min_near_miss_ha = NEAR_MISS_MMU_SHARE * saved.params.mmu.mmu_ha
        near_misses = removed[removed["area_ha"] >= min_near_miss_ha]
        drawn_near_misses = sample_objects(near_misses, design.n_below_mmu, 1, rng)
        drawn_near_misses = drawn_near_misses.assign(object_id=np.nan)
        parts[STRATUM_BELOW_MMU] = drawn_near_misses
    if design.n_no_detection > 0:
        parts[STRATUM_NO_DETECTION] = no_detection_cells(
            saved, design.n_no_detection, rng
        )
    tables = []
    for stratum, stratum_frame in parts.items():
        if stratum_frame.empty:
            logger.warning("Stratum %s has no units", stratum)
            continue
        table = objects_table(stratum_frame)
        table["stratum"] = stratum
        table["geometry"] = stratum_frame.geometry.to_numpy()
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined["object_id"] = combined["object_id"].astype("Int64")
    last_year = saved.years[-1]
    combined["follow_up"] = combined["onset_year"] < last_year
    n_samples = len(combined)
    combined["sample_id"] = [f"S{number:02d}" for number in range(1, n_samples + 1)]
    ordered_columns = [*SAMPLE_COLUMNS, "geometry"]
    samples = gpd.GeoDataFrame(
        combined[ordered_columns], geometry="geometry", crs=saved.grid.crs
    )
    stratum_counts = samples["stratum"].value_counts().to_dict()
    logger.info("Drew %d samples %s", n_samples, stratum_counts)
    return samples


__all__ = [
    "SAMPLE_COLUMNS",
    "STRATA",
    "STRATUM_BELOW_MMU",
    "STRATUM_DETECTED",
    "STRATUM_NO_DETECTION",
    "NEAR_MISS_MMU_SHARE",
    "SampleDesign",
    "allocate_by_stratum",
    "below_mmu_objects",
    "draw_samples",
    "no_detection_cells",
    "sample_objects",
]
