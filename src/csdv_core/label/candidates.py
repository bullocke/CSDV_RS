"""csdv_core.label.candidates: the inclusive candidate set of a unit.

Inputs: a :class:`~csdv_core.label.inputs.UnitInputs`.
Outputs: a GeoDataFrame of candidates in the unit's UTM CRS, from three
sources. ``di`` objects are the detections of the disturbance run, with the
refined 0.6 m outline where the run has one. ``weak`` objects are connected
cells whose disturbance probability is raised but which were not detected.
``chm`` objects are connected cells that left the forest mask with a real
drop in canopy cover, which catches loss the detection normalises away (a
fire that affects a whole neighbourhood). Each candidate is later judged by
the vision models, so false ones cost a call and nothing else.

A unit whose run holds a map of the pair model gets a fourth source.
``model`` objects are connected map cells the model gives a raised loss
probability that no candidate of the interval touches: loss the canopy
height missed, and the model's false alarms, which become judged negatives.
Every candidate then also carries ``model_share``, the share of it that the
model flags, so that a sample can favour the places where the model and
the detection disagree.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from shapely.ops import unary_union

from csdv_core.config._models import LabelCandidates
from csdv_core.label.inputs import UnitInputs, fill_forest_dips
from csdv_core.label.lattice import Bounds

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = [
    "cand_id",
    "unit",
    "source",
    "interval",
    "area_ha",
    "object_id",
    "geom_src",
    "mean_prob",
    "model_share",
    "geometry",
]
_EIGHT = np.ones((3, 3), dtype=bool)

__all__ = [
    "CANDIDATE_COLUMNS",
    "cell_objects",
    "di_candidates",
    "detected_cells",
    "model_candidates",
    "model_shares",
    "other_candidates",
    "unit_candidates",
    "window_candidates",
]


def di_candidates(unit: UnitInputs) -> gpd.GeoDataFrame:
    """Every detected object of the run, with the refined outline where it exists.

    Objects below the minimum size are kept. They are not judged, but the
    label rules need to know where they are.
    """
    objects = gpd.read_file(unit.objects_path, layer="objects")
    geom = objects.set_index("object_id").geometry
    src = pd.Series("cells", index=geom.index)
    layers = set(gpd.list_layers(unit.objects_path)["name"])
    if "refined_outlines" in layers:
        refined = gpd.read_file(unit.objects_path, layer="refined_outlines")
        refined = refined[refined.geometry.notna() & ~refined.geometry.is_empty]
        refined = refined.set_index("object_id").geometry
        geom.loc[refined.index] = refined.values
        src.loc[refined.index] = "refined"
    ids = objects["object_id"].astype(int)
    onset = objects["onset_year"].astype(int)
    return gpd.GeoDataFrame(
        {
            "cand_id": [
                f"{unit.unit}-di-{y}-{i}" for i, y in zip(ids, onset, strict=True)
            ],
            "unit": unit.unit,
            "source": "di",
            "interval": objects["interval"].astype(str),
            "area_ha": geom.loc[ids].area.to_numpy() / 1e4,
            "object_id": ids,
            "geom_src": src.loc[ids].to_numpy(),
            "mean_prob": objects["mean_prob"].astype(float),
        },
        geometry=geom.loc[ids].make_valid().to_numpy(),
        crs=objects.crs,
    )


def detected_cells(unit: UnitInputs) -> np.ndarray:
    """Cells of the detected objects, ``(years, rows, cols)`` bool.

    Band *t* holds the objects of the interval ending in ``years[t]``.
    """
    from rasterio.features import rasterize

    objects = gpd.read_file(unit.objects_path, layer="objects")
    out = np.zeros((len(unit.years), *unit.grid.cell_shape), dtype=bool)
    for t, year in enumerate(unit.years):
        geoms = objects.loc[objects["onset_year"] == year, "geometry"]
        if len(geoms):
            out[t] = rasterize(
                ((g, 1) for g in geoms),
                out_shape=unit.grid.cell_shape,
                transform=unit.grid.cell_transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
    return out


def cell_objects(
    mask: np.ndarray, unit: UnitInputs, min_cells: int
) -> tuple[list[tuple], np.ndarray]:
    """Connected groups of cells as polygons.

    Args:
        mask: Bool ``(rows, cols)`` on the cell grid.
        unit: Unit, for the cell transform.
        min_cells: Smallest group kept.

    Returns:
        A list of ``(polygon, n_cells, label)``, one per group, and the label
        array in which ``label`` marks the group's cells.
    """
    from rasterio.features import shapes

    labels, n = ndi.label(mask, structure=_EIGHT)
    if n == 0:
        return [], labels
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes >= min_cells)
    keep = keep[keep > 0]
    if not len(keep):
        return [], labels
    kept = np.where(np.isin(labels, keep), labels, 0).astype("int32")
    parts: dict[int, list] = {}
    for geom, value in shapes(
        kept, mask=kept > 0, connectivity=8, transform=unit.grid.cell_transform
    ):
        parts.setdefault(int(value), []).append(shape(geom))
    groups = [
        (unary_union(polys).buffer(0), int(sizes[value]), value)
        for value, polys in sorted(parts.items())
    ]
    return groups, labels


def other_candidates(unit: UnitInputs, cfg: LabelCandidates) -> gpd.GeoDataFrame:
    """Weak-probability and CHM-loss objects of every interval.

    A cell joins an object when it was forest at the earlier date, lies away
    from the detected objects of the interval, and either has a raised
    disturbance probability or left the forest mask with a drop in canopy
    cover. Connected cells of at least the minimum size form one object. It
    is called ``chm`` when most of its cells left the forest mask, else
    ``weak``.
    """
    forest = fill_forest_dips(unit.layer("forest"))
    cover = unit.layer("chm_cover")
    prob = unit.layer("probability")
    detected = detected_cells(unit)
    min_cells = int(np.ceil(cfg.min_ha / unit.grid.cell_ha))
    rows = []
    for t in range(1, len(unit.years)):
        with np.errstate(invalid="ignore"):
            raised = prob[t] >= cfg.weak_prob
            lost = (
                forest[t - 1]
                & ~forest[t]
                & (cover[t - 1] - cover[t] >= cfg.chm_loss_min_drop)
            )
        near = detected[t]
        if cfg.detected_buffer_cells:
            near = ndi.binary_dilation(
                near, structure=_EIGHT, iterations=cfg.detected_buffer_cells
            )
        mask = forest[t - 1] & (raised | lost) & ~near
        groups, labels = cell_objects(mask, unit, min_cells)
        for n, (poly, n_cells, value) in enumerate(groups, start=1):
            cells = labels == value
            source = "chm" if lost[cells].mean() >= 0.5 else "weak"
            year = unit.years[t]
            rows.append(
                {
                    "cand_id": f"{unit.unit}-{source}-{year}-{n}",
                    "unit": unit.unit,
                    "source": source,
                    "interval": unit.intervals[t - 1],
                    "area_ha": n_cells * unit.grid.cell_ha,
                    "object_id": -1,
                    "geom_src": "cells",
                    "mean_prob": float(np.nanmean(prob[t][cells])),
                    "geometry": poly,
                }
            )
    return gpd.GeoDataFrame(rows, columns=CANDIDATE_COLUMNS, crs=unit.grid.crs)


def _read_model_map(unit: UnitInputs, path: Path):
    """The model map: percent ``(intervals, rows, cols)`` and its transform.

    Raises:
        ValueError: If the map's CRS or number of bands is not the unit's.
    """
    import rasterio

    with rasterio.open(path) as src:
        prob = src.read()
        transform = src.transform
        epsg = src.crs.to_epsg()
    if epsg != unit.epsg:
        raise ValueError(f"{path}: EPSG {epsg}, the unit has {unit.epsg}")
    if prob.shape[0] != len(unit.intervals):
        raise ValueError(
            f"{path}: {prob.shape[0]} bands for the unit's "
            f"{len(unit.intervals)} intervals"
        )
    return prob, transform


def _on_map(geoms, shape: tuple[int, int], transform) -> np.ndarray:
    from rasterio.features import rasterize

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        ((g, 1) for g in geoms),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _split_large(poly, split_ha: float, step_m: float) -> list:
    """Cut a polygon above ``split_ha`` along lattice lines every ``step_m``."""
    if poly.area / 1e4 <= split_ha:
        return [poly]
    x0, y0, x1, y1 = poly.bounds
    parts = []
    for ix in range(int(np.floor(x0 / step_m)), int(np.ceil(x1 / step_m))):
        for iy in range(int(np.floor(y0 / step_m)), int(np.ceil(y1 / step_m))):
            tile = box(ix * step_m, iy * step_m, (ix + 1) * step_m, (iy + 1) * step_m)
            part = poly.intersection(tile)
            if not part.is_empty:
                parts.append(part)
    return parts


def model_candidates(
    unit: UnitInputs,
    cfg: LabelCandidates,
    others: gpd.GeoDataFrame,
    path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Objects of the pair model's map that no other candidate touches.

    A map cell (9.6 m) joins an object when the model's loss probability on
    forest is at least ``cfg.model_prob`` and no candidate of the same
    interval lies within ``cfg.model_buffer_cells`` cells. Connected cells
    of at least the minimum size form one object, and an object above
    ``cfg.model_split_ha`` is cut into parts. ``mean_prob`` is the model's
    mean probability over the object, not the detection's.

    Args:
        unit: The unit.
        cfg: Candidate thresholds.
        others: The unit's candidates of the other sources.
        path: The map. Default: ``unit.model_map``.

    Returns:
        The model objects, empty when the unit has no map.
    """
    from rasterio.features import shapes

    path = path or unit.model_map
    empty = gpd.GeoDataFrame([], columns=CANDIDATE_COLUMNS, crs=unit.grid.crs)
    if path is None:
        return empty
    prob, transform = _read_model_map(unit, path)
    cell_ha = abs(transform.a * transform.e) / 1e4
    min_cells = int(np.ceil(cfg.min_ha / cell_ha - 1e-9))
    rows = []
    for k, interval in enumerate(unit.intervals):
        band = prob[k]
        taken = _on_map(
            others.loc[others["interval"] == interval, "geometry"],
            band.shape,
            transform,
        )
        if cfg.model_buffer_cells:
            taken = ndi.binary_dilation(
                taken, structure=_EIGHT, iterations=cfg.model_buffer_cells
            )
        mask = (band != 255) & (band >= round(cfg.model_prob * 100)) & ~taken
        labels, n = ndi.label(mask, structure=_EIGHT)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        keep = np.flatnonzero(sizes >= min_cells)
        keep = keep[keep > 0]
        if not len(keep):
            continue
        kept = np.where(np.isin(labels, keep), labels, 0).astype("int32")
        groups: dict[int, list] = {}
        for geom, value in shapes(
            kept, mask=kept > 0, connectivity=8, transform=transform
        ):
            groups.setdefault(int(value), []).append(shape(geom))
        year = unit.years[k + 1]
        n_out = 0
        for value, polys in sorted(groups.items()):
            whole = unary_union(polys).buffer(0)
            mean = float(band[labels == value].mean()) / 100.0
            for part in _split_large(whole, cfg.model_split_ha, cfg.model_split_m):
                if part.area / 1e4 < cfg.min_ha:
                    continue
                n_out += 1
                rows.append(
                    {
                        "cand_id": f"{unit.unit}-model-{year}-{n_out}",
                        "unit": unit.unit,
                        "source": "model",
                        "interval": interval,
                        "area_ha": part.area / 1e4,
                        "object_id": -1,
                        "geom_src": "cells",
                        "mean_prob": mean,
                        "model_share": 1.0,
                        "geometry": part,
                    }
                )
    return gpd.GeoDataFrame(rows, columns=CANDIDATE_COLUMNS, crs=unit.grid.crs)


def model_shares(
    unit: UnitInputs,
    cfg: LabelCandidates,
    candidates: gpd.GeoDataFrame,
    path: Path | None = None,
) -> np.ndarray:
    """Share of each candidate that the pair model flags.

    The share is taken of the candidate's map cells at or above
    ``cfg.model_flag_prob``, in the candidate's interval and in the ones
    beside it, and the largest of the three counts. The neighbours are
    allowed because the model, like the vision models, often dates a loss
    one image from where the canopy height does. On the round-1 units a
    candidate the holdout model flagged by half or more was loss in 92% of
    cases, and one it flagged by under a tenth in 55%.

    Returns:
        One value per candidate, NaN where the unit has no map or the
        candidate lies off it.
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as window_transform

    path = path or unit.model_map
    out = np.full(len(candidates), np.nan)
    if path is None or not len(candidates):
        return out
    prob, transform = _read_model_map(unit, path)
    n_bands, height, width = prob.shape
    cut = round(cfg.model_flag_prob * 100)
    whole = Window(0, 0, width, height)
    for i, (interval, geom) in enumerate(
        zip(candidates["interval"], candidates.geometry, strict=True)
    ):
        if interval not in unit.intervals or geom is None or geom.is_empty:
            continue
        window = from_bounds(*geom.bounds, transform=transform)
        window = window.round_offsets().round_lengths().intersection(whole)
        if window.width < 1 or window.height < 1:
            continue
        rows = slice(int(window.row_off), int(window.row_off + window.height))
        cols = slice(int(window.col_off), int(window.col_off + window.width))
        inside = geometry_mask(
            [geom],
            out_shape=(rows.stop - rows.start, cols.stop - cols.start),
            transform=window_transform(window, transform),
            invert=True,
            all_touched=True,
        )
        k = unit.intervals.index(interval)
        best = np.nan
        for band in (k - 1, k, k + 1):
            if not 0 <= band < n_bands:
                continue
            values = prob[band][rows, cols]
            valid = inside & (values != 255)
            if valid.any():
                share = float((values[valid] >= cut).mean())
                best = share if np.isnan(best) else max(best, share)
        out[i] = best
    return out


def unit_candidates(unit: UnitInputs, cfg: LabelCandidates) -> gpd.GeoDataFrame:
    """All candidates of a unit: detected, weak, CHM-loss and model objects.

    Without the cell layers on this machine only the detected objects are
    returned, with a warning. Model objects and ``model_share`` need the
    pair model's map of the unit. Without it ``model_share`` is NaN.
    """
    parts = [di_candidates(unit)]
    if unit.has_layers:
        parts.append(other_candidates(unit, cfg))
    else:
        logger.warning("%s: no cell layers, detected objects only", unit.unit)
    out = pd.concat(parts, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=unit.grid.crs)
    out["model_share"] = model_shares(unit, cfg, out)
    if unit.model_map is not None:
        found = model_candidates(unit, cfg, out)
        if len(found):
            out = pd.concat([out, found], ignore_index=True)
    out = out[CANDIDATE_COLUMNS]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=unit.grid.crs)


def window_candidates(
    candidates: gpd.GeoDataFrame, bounds: Bounds, min_ha: float
) -> gpd.GeoDataFrame:
    """Candidates that reach into a window, with what is visible of each.

    Adds ``visible_ha`` (area inside the window), ``clipped`` (part of the
    candidate lies outside) and ``judged`` (large enough to send to the
    models, both as a whole and in its visible part).
    """
    area = box(*bounds)
    hits = candidates[candidates.intersects(area)].copy()
    visible = hits.geometry.intersection(area)
    hits["visible_ha"] = visible.area / 1e4
    hits = hits[hits["visible_ha"] > 0]
    hits["clipped"] = hits["visible_ha"] < 0.98 * hits["area_ha"]
    hits["judged"] = (hits["area_ha"] >= min_ha) & (hits["visible_ha"] >= min_ha)
    return hits.reset_index(drop=True)
