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
"""

from __future__ import annotations

import logging

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
    "geometry",
]
_EIGHT = np.ones((3, 3), dtype=bool)

__all__ = [
    "CANDIDATE_COLUMNS",
    "cell_objects",
    "di_candidates",
    "detected_cells",
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


def unit_candidates(unit: UnitInputs, cfg: LabelCandidates) -> gpd.GeoDataFrame:
    """All candidates of a unit: detected, weak and CHM-loss objects.

    Without the cell layers on this machine only the detected objects are
    returned, with a warning.
    """
    parts = [di_candidates(unit)]
    if unit.has_layers:
        parts.append(other_candidates(unit, cfg))
    else:
        logger.warning("%s: no cell layers, detected objects only", unit.unit)
    out = pd.concat(parts, ignore_index=True)[CANDIDATE_COLUMNS]
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
