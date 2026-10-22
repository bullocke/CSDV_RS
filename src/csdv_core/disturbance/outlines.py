"""csdv_core.disturbance.outlines: vectorize detected cells and refine them at 0.6 m.

Detected cells (after the MMU) are grouped into objects: 8-connected cells
detected in the same interval, dissolved into one (multi)polygon each. A cell
detected in two intervals belongs to one object in each, so objects of
different intervals can overlap. With refinement on, every object is redrawn
from the 0.6 m CHM before and after its interval, following
``08_refine_best.refine_object``: the CHM change (with a 3 x 3 maximum on the
later CHM) is standardized against a ring of forest around the object,
Efron's empirical null is fitted on the ring, and pixels with loss
probability above ``refine.p_loss`` near the object become openings.
:mod:`csdv_core.change.refine` closes the openings into an envelope that
must touch the object and reach the MMU (the growth minimum for objects kept
by the growth rule). Refinement always reads the CHM, whatever the
detection input.

Inputs: detections per interval, probability and z stacks, the grid, CHM
windows.
Outputs: GeoDataFrames of objects, refined envelopes and openings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from csdv_core.change import di, di_decide
from csdv_core.change import refine as change_refine
from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.decide import EIGHT_CONNECTED
from csdv_core.disturbance.inputs import M2_PER_HA, CellGrid, SiteInputs
from csdv_core.disturbance.windows import read_chm_bounds

logger = logging.getLogger(__name__)

#: Columns of the objects GeoDataFrame, in order (geometry last).
OBJECT_COLUMNS: tuple[str, ...] = (
    "object_id",
    "onset_year",
    "year_before",
    "interval",
    "onset_index",
    "n_cells",
    "area_ha",
    "earlier_share",
    "growth",
    "mean_prob",
    "max_prob",
    "mean_z",
    "follow_up",
    "geometry",
)
#: Columns of the refined outlines GeoDataFrame, in order (geometry last).
REFINED_COLUMNS: tuple[str, ...] = (
    "object_id",
    "onset_year",
    "year_before",
    "interval",
    "envelope_ha",
    "openings_ha",
    "opening_share",
    "n_openings",
    "n_ring_px",
    "geometry",
)
#: Columns of the refined openings GeoDataFrame, in order (geometry last).
OPENING_COLUMNS: tuple[str, ...] = (
    "object_id",
    "onset_year",
    "openings_ha",
    "geometry",
)
#: Extra columns that :func:`objects_table` adds for sampling.
CENTROID_COLUMNS: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "centroid_lon",
    "centroid_lat",
)
#: Geographic CRS of the sampling centroids.
LONLAT_CRS = "EPSG:4326"


@dataclass(frozen=True)
class RefinedOutline:
    """0.6 m outline of one detected object.

    Attributes:
        envelope: Envelope polygon (closed openings, holes filled, smoothed by
            ``refine.smooth_m`` and simplified by ``refine.simplify_m``).
        openings: Union of the opening polygons inside the envelope (not
            smoothed).
        envelope_ha: Envelope polygon area (ha).
        openings_ha: Opening pixel area inside the envelope (ha).
        opening_share: Opening pixels / envelope pixels.
        n_openings: Number of 8-connected opening components.
        n_ring_px: Reference ring pixels used for the null.
    """

    envelope: BaseGeometry
    openings: BaseGeometry
    envelope_ha: float
    openings_ha: float
    opening_share: float
    n_openings: int
    n_ring_px: int


@dataclass
class RefinementResult:
    """Refinement of all objects of a run.

    Attributes:
        outlines: GeoDataFrame with :data:`REFINED_COLUMNS`, one row per
            refined object, grid CRS.
        openings: GeoDataFrame with :data:`OPENING_COLUMNS`, one row per
            refined object, grid CRS.
        n_objects: Objects submitted.
        n_refined: Objects with an envelope.
        skipped_object_ids: Objects without an envelope (ring too small, no
            usable statistics, or empty envelope).
    """

    outlines: gpd.GeoDataFrame
    openings: gpd.GeoDataFrame
    n_objects: int
    n_refined: int
    skipped_object_ids: list[int]


def interval_label(year_before: int, year: int) -> str:
    """Interval label used in tables and figures.

    Args:
        year_before: Earlier image year.
        year: Later image year.

    Returns:
        ``"<year_before>-<year>"``, e.g. ``"2017-2019"``.
    """
    return f"{year_before}-{year}"


def _frame(
    rows: list[dict[str, object]], columns: Sequence[str], crs: object
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from row dicts, keeping ``columns`` when empty.

    Args:
        rows: One dict per feature, each holding ``columns``.
        columns: Column names in output order, ending with ``"geometry"``.
        crs: Coordinate reference system of the frame.

    Returns:
        The GeoDataFrame with exactly ``columns``.
    """
    column_list = list(columns)
    if not rows:
        return gpd.GeoDataFrame(columns=column_list, geometry="geometry", crs=crs)
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return frame[column_list]


def _label_polygons(labels: np.ndarray, transform: Affine) -> dict[int, BaseGeometry]:
    """Polygonize a label raster and dissolve the polygons of each label.

    Args:
        labels: int32 label raster, 0 = background.
        transform: Affine transform of the raster.

    Returns:
        Mapping from label to its dissolved (multi)polygon.
    """
    parts_by_label: dict[int, list[BaseGeometry]] = {}
    foreground = labels > 0
    polygon_iter = rasterio.features.shapes(
        labels, mask=foreground, transform=transform, connectivity=8
    )
    for geojson, value in polygon_iter:
        label = int(value)
        polygon = shape(geojson)
        parts_by_label.setdefault(label, []).append(polygon)
    geometry_by_label = {}
    for label, parts in parts_by_label.items():
        geometry_by_label[label] = unary_union(parts)
    return geometry_by_label


def _masked_stat(
    values: np.ndarray | None,
    mask: np.ndarray,
    reducer: Callable[[np.ndarray], float],
) -> float:
    """Reduce the finite values of ``values[mask]``, NaN when none is left.

    Args:
        values: 2-D float array, or None.
        mask: Boolean mask of the object cells.
        reducer: Reduction over a 1-D array, e.g. ``np.mean`` or ``np.max``.

    Returns:
        The reduced value as a float, NaN if ``values`` is None or all NaN.
    """
    if values is None:
        return float("nan")
    selected = values[mask]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def detection_objects(
    detected: np.ndarray,
    probability: np.ndarray | None,
    z: np.ndarray,
    grid: CellGrid,
    years: Sequence[int],
    *,
    min_cells: int | None = None,
) -> gpd.GeoDataFrame:
    """Vectorize the detections of every interval into objects.

    The cells detected at date ``t`` are labelled 8-connected
    (``scipy.ndimage.label``), polygonized on ``grid.cell_transform`` with
    ``rasterio.features.shapes`` and dissolved per label. ``object_id`` runs
    from 1 in order of date, then label. For an object at date ``t``:
    ``onset_year = years[t]`` (the later image of its interval),
    ``year_before = years[t - 1]``, ``area_ha = n_cells * grid.cell_ha``,
    ``earlier_share`` = share of its cells detected at an earlier date,
    ``growth`` = fewer cells than ``min_cells`` (kept by the growth rule of
    :func:`csdv_core.disturbance.decide.interval_detections`, False when
    ``min_cells`` is None), ``mean_prob`` / ``max_prob`` over
    ``probability[t]`` (NaN when ``probability`` is None), ``mean_z`` over
    ``z[t]`` (NaN cells ignored) and ``follow_up = t < T - 1`` (a later image
    exists to confirm that the loss persists).

    Args:
        detected: bool ``(T, rows, cols)`` detections per interval (after the
            MMU), e.g. :attr:`csdv_core.disturbance.decide.DecisionResult.detected`.
        probability: float32 ``(T, rows, cols)`` or None (rule D1).
        z: float32 ``(T, rows, cols)`` quantized z.
        grid: Common grid.
        years: NAIP years, length ``T``.
        min_cells: Cells an object needs without the growth rule
            (:func:`csdv_core.disturbance.decide.mmu_cells`), or None.

    Returns:
        GeoDataFrame with :data:`OBJECT_COLUMNS` in grid CRS (empty, with the
        same columns, when nothing is detected).
    """
    rows = []
    object_id = 0
    n_dates_used = 0
    earlier = np.zeros(detected.shape[1:], dtype=bool)
    for date_index in range(detected.shape[0]):
        date_mask = detected[date_index]
        if not date_mask.any():
            continue
        n_dates_used += 1
        labels, n_labels = ndimage.label(date_mask, structure=EIGHT_CONNECTED)
        labels = labels.astype("int32")
        cell_counts = np.bincount(labels.ravel())
        earlier_counts = np.bincount(
            labels[earlier & date_mask], minlength=n_labels + 1
        )
        geometry_by_label = _label_polygons(labels, grid.cell_transform)
        probability_date = None if probability is None else probability[date_index]
        z_date = z[date_index]
        onset_year = int(years[date_index])
        year_before = int(years[date_index - 1])
        has_follow_up = date_index < len(years) - 1
        for label in range(1, n_labels + 1):
            object_mask = labels == label
            n_cells = int(cell_counts[label])
            is_growth = min_cells is not None and n_cells < min_cells
            object_id += 1
            rows.append(
                {
                    "object_id": object_id,
                    "onset_year": onset_year,
                    "year_before": year_before,
                    "interval": interval_label(year_before, onset_year),
                    "onset_index": date_index,
                    "n_cells": n_cells,
                    "area_ha": n_cells * grid.cell_ha,
                    "earlier_share": int(earlier_counts[label]) / n_cells,
                    "growth": is_growth,
                    "mean_prob": _masked_stat(probability_date, object_mask, np.mean),
                    "max_prob": _masked_stat(probability_date, object_mask, np.max),
                    "mean_z": _masked_stat(z_date, object_mask, np.mean),
                    "follow_up": has_follow_up,
                    "geometry": geometry_by_label[label],
                }
            )
        earlier |= date_mask
    objects = _frame(rows, OBJECT_COLUMNS, grid.crs)
    logger.info("%d detection objects over %d intervals", len(objects), n_dates_used)
    return objects


def _burn(
    geometry: BaseGeometry, window_shape: tuple[int, ...], transform: Affine
) -> np.ndarray:
    """Rasterize a geometry onto a window (pixel centres inside count).

    Args:
        geometry: Polygon in the window's CRS.
        window_shape: ``(rows, cols)`` of the window.
        transform: Affine transform of the window.

    Returns:
        Boolean mask, True inside ``geometry``.
    """
    burned = rasterio.features.rasterize(
        [(geometry, 1)],
        out_shape=window_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def refine_object(
    chm_before: np.ndarray,
    chm_after: np.ndarray,
    transform: Affine,
    geometry: BaseGeometry,
    params: DisturbanceConfig,
    *,
    min_area_ha: float | None = None,
) -> RefinedOutline | None:
    """Redraw one object at the CHM pixel size (port of ``08_refine_best``).

    The change is the later CHM (3 x 3 maximum for ``refine.tolerance_px``
    = 1) minus the earlier CHM. It is standardized on a ring of earlier
    forest outside the object buffered by ``refine.ring_gap_m``, an empirical
    null and lfdr curve give each pixel a loss probability, and forest pixels
    within ``refine.near_m`` of the object above ``refine.p_loss`` are loss.
    :mod:`csdv_core.change.refine` cleans the loss into openings and closes
    them into an envelope that must touch the object and reach
    ``min_area_ha`` (the MMU by default).

    Args:
        chm_before: float32 ``(rows, cols)`` CHM (m) at the year before onset,
            NaN nodata, covering the object bounds buffered by ``refine.pad_m``.
        chm_after: float32 ``(rows, cols)`` CHM (m) at the onset year, same
            window.
        transform: Affine transform of the window.
        geometry: The object polygon (grid CRS).
        params: Run parameters (``refine``, ``decide``, ``mmu``).
        min_area_ha: Least envelope area (ha), or None for ``mmu.mmu_ha``.

    Returns:
        The refined outline, or None when the object cannot be refined.

    Raises:
        ValueError: If the two CHM windows differ in shape.
    """
    if chm_before.shape != chm_after.shape:
        raise ValueError(
            f"CHM windows differ in shape: {chm_before.shape} vs {chm_after.shape}"
        )
    refine = params.refine
    decide = params.decide
    envelope_min_ha = params.mmu.mmu_ha if min_area_ha is None else min_area_ha
    pixel_m = abs(transform.a)
    window_shape = chm_before.shape

    filter_size = 2 * refine.tolerance_px + 1
    after_filled = np.nan_to_num(chm_after, nan=-1.0)
    after_max = ndimage.maximum_filter(after_filled, size=filter_size)
    both_finite = np.isfinite(chm_before) & np.isfinite(chm_after)
    change = np.where(both_finite, after_max - chm_before, np.nan)

    before_filled = np.nan_to_num(chm_before)
    forest_before = before_filled > refine.forest_height_m
    object_mask = _burn(geometry, window_shape, transform)
    near_geometry = geometry.buffer(refine.near_m)
    near_mask = _burn(near_geometry, window_shape, transform)
    gap_geometry = geometry.buffer(refine.ring_gap_m)
    gap_mask = _burn(gap_geometry, window_shape, transform)
    ring = forest_before & ~gap_mask
    n_ring_px = int(ring.sum())

    stats = di.population_stats(change, ring)
    if stats is None or n_ring_px < refine.min_ring_px:
        logger.debug("Ring too small or unusable (%d pixels)", n_ring_px)
        return None
    z = (change - stats.centre) / stats.scale
    null = di_decide.fit_null(z, ring, pct0=decide.pct0, bin_w=decide.bin_w)
    model_mask = forest_before & np.isfinite(z)
    model = di_decide.lfdr_model(
        z,
        model_mask,
        null=null,
        bin_w=decide.bin_w,
        smooth_bins=decide.smooth_bins,
    )
    loss_probability = model.prob(z)
    loss = forest_before & near_mask & (loss_probability > refine.p_loss)

    openings = change_refine.opening_footprint(
        loss,
        pixel_m,
        open_radius_m=refine.open_radius_m,
        min_area_m2=refine.min_opening_m2,
    )
    envelope_clip = forest_before | openings
    envelope = change_refine.envelope_mask(
        openings,
        pixel_m,
        close_radius_m=refine.close_radius_m,
        fill_holes_m2=refine.fill_holes_m2,
        min_area_m2=envelope_min_ha * M2_PER_HA,
        seed=object_mask,
        clip=envelope_clip,
    )
    if not envelope.any():
        return None

    envelope_geometry = change_refine.polygonize(
        envelope, transform, smooth_m=refine.smooth_m, simplify_m=refine.simplify_m
    )
    openings_inside = openings & envelope
    openings_geometry = change_refine.polygonize(openings_inside, transform)
    n_opening_px = int(openings_inside.sum())
    n_envelope_px = int(envelope.sum())
    _, n_openings = ndimage.label(openings_inside, structure=EIGHT_CONNECTED)
    return RefinedOutline(
        envelope=envelope_geometry,
        openings=openings_geometry,
        envelope_ha=float(envelope_geometry.area / M2_PER_HA),
        openings_ha=float(n_opening_px * pixel_m**2 / M2_PER_HA),
        opening_share=float(n_opening_px / n_envelope_px),
        n_openings=int(n_openings),
        n_ring_px=n_ring_px,
    )


def refine_objects(
    objects: gpd.GeoDataFrame,
    site_inputs: SiteInputs,
    params: DisturbanceConfig,
) -> RefinementResult:
    """Refine every object, reading CHM windows from the site inputs.

    For each object, the CHMs of ``year_before`` and ``onset_year`` are read
    over the object bounds buffered by ``refine.pad_m``. Objects kept by the
    growth rule (column ``growth``) need an envelope of
    ``min(mmu.growth_mmu_ha, mmu.mmu_ha)``, the others ``mmu.mmu_ha``.
    Objects that cannot be refined are listed in ``skipped_object_ids``.

    Args:
        objects: From :func:`detection_objects`.
        site_inputs: Inputs and grid of the run.
        params: Run parameters.

    Returns:
        The refinement result (empty GeoDataFrames when nothing is refined).
    """
    grid = site_inputs.grid
    growth_min_ha = min(params.mmu.growth_mmu_ha, params.mmu.mmu_ha)
    outline_rows = []
    opening_rows = []
    skipped_object_ids: list[int] = []
    for row in objects.itertuples(index=False):
        object_id = int(row.object_id)
        onset_year = int(row.onset_year)
        year_before = int(row.year_before)
        padded = row.geometry.buffer(params.refine.pad_m)
        window_bounds = padded.bounds
        chm_before, transform = read_chm_bounds(site_inputs, year_before, window_bounds)
        chm_after, _ = read_chm_bounds(site_inputs, onset_year, window_bounds)
        is_growth = bool(getattr(row, "growth", False))
        min_area_ha = growth_min_ha if is_growth else params.mmu.mmu_ha
        outline = refine_object(
            chm_before,
            chm_after,
            transform,
            row.geometry,
            params,
            min_area_ha=min_area_ha,
        )
        if outline is None:
            skipped_object_ids.append(object_id)
            continue
        outline_rows.append(
            {
                "object_id": object_id,
                "onset_year": onset_year,
                "year_before": year_before,
                "interval": row.interval,
                "envelope_ha": outline.envelope_ha,
                "openings_ha": outline.openings_ha,
                "opening_share": outline.opening_share,
                "n_openings": outline.n_openings,
                "n_ring_px": outline.n_ring_px,
                "geometry": outline.envelope,
            }
        )
        opening_rows.append(
            {
                "object_id": object_id,
                "onset_year": onset_year,
                "openings_ha": outline.openings_ha,
                "geometry": outline.openings,
            }
        )
    outlines = _frame(outline_rows, REFINED_COLUMNS, grid.crs)
    openings = _frame(opening_rows, OPENING_COLUMNS, grid.crs)
    n_objects = len(objects)
    n_refined = len(outline_rows)
    logger.info(
        "Refined %d of %d objects (%d skipped)",
        n_refined,
        n_objects,
        len(skipped_object_ids),
    )
    return RefinementResult(
        outlines=outlines,
        openings=openings,
        n_objects=n_objects,
        n_refined=n_refined,
        skipped_object_ids=skipped_object_ids,
    )


def objects_table(objects: gpd.GeoDataFrame) -> pd.DataFrame:
    """Objects as a flat table for ``objects.csv`` (interpretation sampling).

    Drops the geometry and adds :data:`CENTROID_COLUMNS`: the polygon
    centroid in grid CRS units and in EPSG:4326 degrees.

    Args:
        objects: From :func:`detection_objects`.

    Returns:
        DataFrame with the non-geometry object columns plus the centroids.

    Raises:
        ValueError: If ``objects`` has no CRS.
    """
    if objects.crs is None:
        raise ValueError("objects has no CRS, so centroids cannot be projected")
    centroids = objects.geometry.centroid
    centroids_lonlat = centroids.to_crs(LONLAT_CRS)
    attributes = objects.drop(columns="geometry")
    table = pd.DataFrame(attributes)
    table["centroid_x"] = centroids.x.to_numpy()
    table["centroid_y"] = centroids.y.to_numpy()
    table["centroid_lon"] = centroids_lonlat.x.to_numpy()
    table["centroid_lat"] = centroids_lonlat.y.to_numpy()
    return table


__all__ = [
    "OBJECT_COLUMNS",
    "OPENING_COLUMNS",
    "REFINED_COLUMNS",
    "RefinedOutline",
    "RefinementResult",
    "detection_objects",
    "interval_label",
    "objects_table",
    "refine_object",
    "refine_objects",
]
