"""csdv_core.pairmodel.compare: a model map beside the labels and the DI.

Inputs: the probability raster of a unit (:mod:`csdv_core.pairmodel.predict`),
the label chips that lie in the unit, the unit's disturbance run and,
for the panels, its NAIP mosaics.
Outputs, in the output folder:

- ``chips_agreement.csv``: cell scores of the model and of the DI probability
  against the pooled labels, per interval and overall.
- ``di_objects.csv``: the model's probability inside every detected object.
- ``model_only.csv``: patches the model flags that no detected object touches.
- ``overview.png``: model and DI probability of every interval.
- ``events/*.png``: close-ups (earlier image, later image, model, DI, label).
- ``agent_confusion.csv``: for a map with agents, the agent the model names
  on the label's loss cells against the agent of the label.
- ``compare.json``: the headline numbers.

The labels descend from the DI's own candidates, so agreement with them
favours what those candidates flag. The patches in ``model_only.csv`` are
where the map says something the labels cannot.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.config._models import LabelPooling
from csdv_core.label.lattice import GSD, Bounds
from csdv_core.label.pooling import pool_change
from csdv_core.pairmodel.data import AGENT_NAMES, pool_agents
from csdv_core.pairmodel.metrics import AgentCounts, CellCounts
from csdv_core.pairmodel.predict import PredGrid, cells_at_centres

logger = logging.getLogger(__name__)

IGNORE = 255
#: Value of label cells that no chip covers.
NO_CHIP = 254
#: Smallest patch reported in ``model_only.csv``, hectares.
MIN_PATCH_HA = 0.3

__all__ = [
    "CompareResult",
    "compare_unit",
    "label_agent_cells",
    "label_cells",
    "open_map",
]


@dataclass
class CompareResult:
    """What a comparison wrote."""

    out_dir: Path
    summary: dict[str, Any]
    panels: int = 0
    notes: list[str] = field(default_factory=list)


def open_map(path: Path) -> tuple[np.ndarray, PredGrid, list[str]]:
    """Read a probability raster: percent, its cell grid and its band names."""
    import rasterio

    with rasterio.open(path) as src:
        prob = src.read()
        cell_px = round(src.transform.a / GSD)
        grid = PredGrid(
            int(src.crs.to_epsg()),
            round(src.transform.c / GSD),
            round(src.transform.f / GSD),
            src.width,
            src.height,
            cell_px,
        )
        names = [str(d) for d in src.descriptions]
    return prob, grid, names


def label_cells(
    chips_dir: Path, unit: str, grid: PredGrid, pooling: LabelPooling
) -> tuple[np.ndarray, np.ndarray, list[Bounds]]:
    """Labels and DI probability of a unit's chips on the cells of a map.

    Returns:
        ``(target, di, footprints)``. ``target`` is ``(intervals, rows,
        cols)`` uint8 with 0 background, 1 loss, 255 ignore and 254 where no
        chip lies. ``di`` is the DI probability in percent at the centre of
        each cell (255 where not monitored or no chip). ``footprints`` are
        the chips' bounds.

    Raises:
        ValueError: If the unit has no chip, or chips differ in their number
            of intervals.
    """
    import pandas as pd
    import rasterio

    table = pd.read_parquet(Path(chips_dir) / "chips.parquet")
    table = table[table["unit"] == unit]
    if table.empty:
        raise ValueError(f"No chip of {unit} in {chips_dir}")
    target: np.ndarray | None = None
    di: np.ndarray | None = None
    footprints: list[Bounds] = []
    cell = grid.cell_px
    for chip in table.itertuples():
        bounds = tuple(json.loads(chip.bounds))
        stem = Path(chips_dir) / chip.chip_id
        with rasterio.open(f"{stem}_change.tif") as src:
            change = src.read()
        with rasterio.open(f"{stem}_aux.tif") as src:
            prob = src.read(list(range(1, change.shape[0] + 1)))
        if target is None:
            shape = (change.shape[0], grid.rows, grid.cols)
            target = np.full(shape, NO_CHIP, dtype="uint8")
            di = np.full(shape, IGNORE, dtype="uint8")
        if change.shape[0] != target.shape[0]:
            raise ValueError(
                f"{chip.chip_id}: {change.shape[0]} intervals, not {target.shape[0]}"
            )
        binary = np.where(change == IGNORE, IGNORE, change > 0).astype("uint8")
        pooled = pool_change(binary, cell, pooling)
        pooled[pooled == 2] = 1
        centre = prob[:, cell // 2 :: cell, cell // 2 :: cell]
        # Place the chip's cells on the map, clipped to it.
        row0 = (grid.row0 - round(bounds[3] / GSD)) // cell
        col0 = (round(bounds[0] / GSD) - grid.col0) // cell
        rows, cols = pooled.shape[1:]
        r_a, r_b = max(row0, 0), min(row0 + rows, grid.rows)
        c_a, c_b = max(col0, 0), min(col0 + cols, grid.cols)
        if r_a >= r_b or c_a >= c_b:
            continue
        source = (
            slice(None),
            slice(r_a - row0, r_b - row0),
            slice(c_a - col0, c_b - col0),
        )
        target[:, r_a:r_b, c_a:c_b] = pooled[source]
        di[:, r_a:r_b, c_a:c_b] = centre[source]
        footprints.append(bounds)  # type: ignore[arg-type]
    assert target is not None and di is not None
    return target, di, footprints


def label_agent_cells(
    chips_dir: Path, unit: str, grid: PredGrid, pooling: LabelPooling
) -> np.ndarray:
    """Agent class of the label's loss cells on the cells of a map.

    Returns:
        ``(intervals, rows, cols)`` uint8: the index into
        :data:`~csdv_core.pairmodel.data.AGENT_NAMES` on loss cells with an
        agreed agent, 255 everywhere else.
    """
    import pandas as pd
    import rasterio

    table = pd.read_parquet(Path(chips_dir) / "chips.parquet")
    table = table[table["unit"] == unit]
    out: np.ndarray | None = None
    cell = grid.cell_px
    for chip in table.itertuples():
        bounds = tuple(json.loads(chip.bounds))
        stem = Path(chips_dir) / chip.chip_id
        with rasterio.open(f"{stem}_change.tif") as src:
            change = src.read()
        with rasterio.open(f"{stem}_agent.tif") as src:
            agent = src.read()
        if out is None:
            out = np.full((change.shape[0], grid.rows, grid.cols), IGNORE, "uint8")
        binary = np.where(change == IGNORE, IGNORE, change > 0).astype("uint8")
        pooled = pool_change(binary, cell, pooling)
        pooled[pooled == 2] = 1
        classes = np.stack(
            [pool_agents(agent[k], pooled[k], cell) for k in range(len(pooled))]
        )
        row0 = (grid.row0 - round(bounds[3] / GSD)) // cell
        col0 = (round(bounds[0] / GSD) - grid.col0) // cell
        rows, cols = classes.shape[1:]
        r_a, r_b = max(row0, 0), min(row0 + rows, grid.rows)
        c_a, c_b = max(col0, 0), min(col0 + cols, grid.cols)
        if r_a >= r_b or c_a >= c_b:
            continue
        out[:, r_a:r_b, c_a:c_b] = classes[
            :, r_a - row0 : r_b - row0, c_a - col0 : c_b - col0
        ]
    if out is None:
        raise ValueError(f"No chip of {unit} in {chips_dir}")
    return out


def _agent_rows(
    agent_path: Path, labelled: np.ndarray, prob: np.ndarray, threshold: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Confusion of the model's agents on the label's loss cells."""
    import rasterio

    with rasterio.open(agent_path) as src:
        codes = src.read()
    named = np.where((codes >= 1) & (codes <= len(AGENT_NAMES)), codes - 1, IGNORE)
    named = named.astype("uint8")
    counts = AgentCounts(AGENT_NAMES)
    counts.update(named, labelled)
    flagged = AgentCounts(AGENT_NAMES)
    cut = int(round(threshold * 100))
    flagged.update(np.where((prob >= cut) & (prob <= 100), named, IGNORE), labelled)
    summary = {
        "cells": counts.n,
        "accuracy": counts.accuracy(),
        "balanced_accuracy": counts.balanced_accuracy(),
        "cells_where_flagged": flagged.n,
        "accuracy_where_flagged": flagged.accuracy(),
    }
    return counts.rows(), summary


def _score_rows(
    prob: np.ndarray,
    target: np.ndarray,
    di: np.ndarray,
    names: list[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, CellCounts]]:
    """Scores per interval and overall, on cells both sources can be scored on."""
    rows: list[dict[str, Any]] = []
    totals = {"model": CellCounts(), "di": CellCounts()}
    for k, name in enumerate([*names, "all"]):
        for source, values in (("model", prob), ("di", di)):
            if name == "all":
                counts = totals[source]
            else:
                common = (prob[k] <= 100) & (di[k] <= 100)
                counts = CellCounts()
                counts.update(np.where(common, values[k], IGNORE), target[k])
                totals[source] = totals[source] + counts
            best_t, best_f1 = counts.best_f1()
            row: dict[str, Any] = {
                "interval": name,
                "source": source,
                "cells": counts.n_valid,
                "loss_cells": counts.n_pos,
                "average_precision": counts.average_precision(),
                "best_f1": best_f1,
                "best_f1_threshold": best_t,
            }
            for label, value in (("050", 0.5), ("val", threshold)):
                at = counts.at(value)
                for key in ("precision", "recall", "f1", "iou"):
                    row[f"{key}_{label}"] = at[key]
            row["threshold_val"] = threshold
            rows.append(row)
    return rows, totals


def _rasterize_objects(objects: Any, grid: PredGrid) -> np.ndarray:
    """Object IDs on the cells of the map (0 where no object lies)."""
    from rasterio.features import rasterize

    if len(objects) == 0:
        return np.zeros((grid.rows, grid.cols), dtype="int32")
    shapes = [
        (geom, int(obj))
        for geom, obj in zip(objects.geometry, objects["object_id"], strict=True)
    ]
    return rasterize(
        shapes,
        out_shape=(grid.rows, grid.cols),
        transform=grid.transform,
        fill=0,
        dtype="int32",
        all_touched=True,
    )


def _object_rows(
    objects: Any,
    prob: np.ndarray,
    grid: PredGrid,
    intervals: list[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """The model's probability inside every detected object of its interval."""
    rows: list[dict[str, Any]] = []
    rasters = []
    cut = int(round(threshold * 100))
    for k, interval in enumerate(intervals):
        part = objects[objects["interval"] == interval]
        ids = _rasterize_objects(part, grid)
        rasters.append(ids)
        band = prob[k]
        for obj in part.itertuples():
            cells = (ids == obj.object_id) & (band <= 100)
            n = int(cells.sum())
            centre = obj.geometry.representative_point()
            rows.append(
                {
                    "object_id": int(obj.object_id),
                    "interval": interval,
                    "band": k + 1,
                    "area_ha": float(obj.area_ha),
                    "di_mean_prob": float(obj.mean_prob),
                    "cells": n,
                    "model_mean_percent": float(band[cells].mean()) if n else math.nan,
                    "model_share_above": (
                        float((band[cells] >= cut).mean()) if n else math.nan
                    ),
                    "x": float(centre.x),
                    "y": float(centre.y),
                }
            )
    return rows, rasters


def _patch_rows(
    prob: np.ndarray,
    object_rasters: list[np.ndarray],
    grid: PredGrid,
    intervals: list[str],
    threshold: float,
    forest: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Patches at or above the threshold that touch no detected object."""
    from scipy import ndimage

    rows: list[dict[str, Any]] = []
    cut = int(round(threshold * 100))
    cell_ha = grid.cell_m**2 / 10_000
    min_cells = math.ceil(MIN_PATCH_HA / cell_ha)
    eight = np.ones((3, 3), dtype=bool)
    left, _, _, top = grid.bounds
    for k, interval in enumerate(intervals):
        band = prob[k]
        flagged = (band >= cut) & (band <= 100)
        labels, n = ndimage.label(flagged, structure=eight)
        if n == 0:
            continue
        near_object = ndimage.binary_dilation(object_rasters[k] > 0, structure=eight)
        index = np.arange(1, n + 1)
        sizes = ndimage.sum(flagged, labels, index)
        touched = ndimage.maximum(near_object, labels, index)
        means = ndimage.mean(band, labels, index)
        centres = ndimage.center_of_mass(flagged, labels, index)
        forest_share = (
            ndimage.mean(forest[k], labels, index) if forest is not None else None
        )
        for i in np.flatnonzero((sizes >= min_cells) & (touched == 0)):
            row, col = centres[i]
            rows.append(
                {
                    "patch": f"{k + 1}-{int(index[i])}",
                    "interval": interval,
                    "band": k + 1,
                    "cells": int(sizes[i]),
                    "area_ha": float(sizes[i] * cell_ha),
                    "model_mean_percent": float(means[i]),
                    "forest_share_before": (
                        float(forest_share[i]) if forest_share is not None else math.nan
                    ),
                    "x": float(left + (col + 0.5) * grid.cell_m),
                    "y": float(top - (row + 0.5) * grid.cell_m),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        names += [key for key in row if key not in names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _label_image(target: np.ndarray) -> np.ndarray:
    """RGB image of label cells: loss red, background pale, ignore grey."""
    image = np.full((*target.shape, 3), 255, dtype="uint8")
    image[target == 0] = (214, 210, 198)
    image[target == 1] = (227, 73, 72)
    image[target == IGNORE] = (155, 154, 148)
    return image


def _overview(
    prob: np.ndarray,
    di: np.ndarray,
    names: list[str],
    grid: PredGrid,
    footprints: list[Bounds],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    left, bottom, right, top = grid.bounds
    extent = (left, right, bottom, top)
    step = max(1, math.ceil(max(grid.rows, grid.cols) / 1200))
    n = len(names)
    aspect = grid.rows / grid.cols
    fig, axes = plt.subplots(
        n, 2, figsize=(10, 4.9 * aspect * n + 0.8), squeeze=False, layout="constrained"
    )
    for k, name in enumerate(names):
        for j, (values, title) in enumerate(((prob[k], "model"), (di[k], "DI"))):
            shown = np.ma.masked_greater(values[::step, ::step], 100)
            ax = axes[k, j]
            ax.set_facecolor("0.8")
            image = ax.imshow(
                shown,
                extent=extent,
                vmin=0,
                vmax=100,
                cmap="magma",
                interpolation="nearest",
            )
            for box in footprints:
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        fill=False,
                        edgecolor="cyan",
                        linewidth=0.4,
                    )
                )
            ax.set_title(f"{title}, {name.replace('_', ' to ')}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(
        image,
        ax=axes,
        location="bottom",
        shrink=0.5,
        aspect=40,
        pad=0.01,
        label="Loss probability (%). Grey: no value. Cyan boxes: label chips.",
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _panel_bounds(x: float, y: float, side_m: float, grid: PredGrid) -> Bounds:
    """A square on the cell lattice, centred near ``(x, y)``."""
    cell = grid.cell_m
    cells = max(int(round(side_m / cell)), 8)
    col = int(round((x - grid.bounds[0]) / cell)) - cells // 2
    row = int(round((grid.bounds[3] - y) / cell)) - cells // 2
    left = grid.bounds[0] + col * cell
    top = grid.bounds[3] - row * cell
    return (left, top - cells * cell, left + cells * cell, top)


def _panel(
    title: str,
    bounds: Bounds,
    k: int,
    names: list[str],
    mosaics: list[list[str]],
    prob: np.ndarray,
    di: np.ndarray,
    target: np.ndarray,
    grid: PredGrid,
    outline: Any,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from csdv_core.label.naip import warp_sources

    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    rows, cols = grid.cell_window(bounds)
    left, _, _, top = grid.bounds
    cell_extent = (
        left + cols.start * grid.cell_m,
        left + cols.stop * grid.cell_m,
        top - rows.stop * grid.cell_m,
        top - rows.start * grid.cell_m,
    )
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    dates = names[k].split("_")
    for j in (0, 1):
        image = warp_sources(mosaics[k + j], grid.epsg, bounds)[0]
        step = max(1, image.shape[1] // 900)
        rgb = np.moveaxis(image[:3, ::step, ::step], 0, -1)
        axes[j].imshow(rgb, extent=extent, interpolation="bilinear")
        axes[j].set_title(dates[j], fontsize=9)
    for j, (values, name) in enumerate(((prob, "Model (%)"), (di, "DI (%)")), start=2):
        shown = np.ma.masked_greater(values[k][rows, cols], 100)
        axes[j].set_facecolor("0.8")
        axes[j].imshow(
            shown,
            extent=cell_extent,
            vmin=0,
            vmax=100,
            cmap="magma",
            interpolation="nearest",
        )
        axes[j].set_title(name, fontsize=9)
    axes[4].imshow(
        _label_image(target[k][rows, cols]), extent=cell_extent, interpolation="nearest"
    )
    axes[4].set_title("Label (red loss, grey ignore, white none)", fontsize=9)
    for ax in axes:
        if outline is not None:
            for part in getattr(outline, "geoms", [outline]):
                ax.plot(*part.exterior.xy, color="cyan", linewidth=0.8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_unit(
    prob_path: Path,
    chips_dir: Path,
    run_dir: Path,
    out_dir: Path,
    *,
    threshold: float = 0.5,
    data_root: Path | None = None,
    n_objects: int = 12,
    n_model_only: int = 6,
    n_di_only: int = 6,
) -> CompareResult:
    """Score a unit's map against its label chips and set it beside the DI.

    Args:
        prob_path: ``prob_<unit>.tif``.
        chips_dir: Chip folder of the label run (the label rasters suffice).
        run_dir: Disturbance run folder of the unit.
        out_dir: Folder for the tables and figures.
        threshold: Probability from which a cell counts as loss, usually the
            best threshold on the validation units.
        data_root: Root that holds ``naip/<unit>/<year>/``. Without it no
            close-up is drawn.
        n_objects: Close-ups of the largest detected objects.
        n_model_only: Close-ups of the largest model-only patches.
        n_di_only: Close-ups of the largest objects the model does not see.

    Raises:
        ValueError: If map, chips and run differ in their number of intervals.
    """
    import geopandas as gpd

    from csdv_core.config import load_label
    from csdv_core.label.inputs import fill_forest_dips, load_unit_inputs
    from csdv_core.label.naip import local_mosaics

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prob, grid, names = open_map(Path(prob_path))
    unit = load_unit_inputs(Path(run_dir))
    if len(unit.intervals) != len(names):
        raise ValueError(
            f"The map has {len(names)} intervals, the run of {unit.unit} "
            f"{len(unit.intervals)}"
        )
    target, di_chips, footprints = label_cells(
        chips_dir, unit.unit, grid, load_label().pooling
    )
    # Chips are dated per window and the map per unit, so intervals are
    # matched by their position in the series.
    if target.shape[0] != len(names):
        raise ValueError(
            f"The chips have {target.shape[0]} intervals, the map {len(names)}"
        )
    notes: list[str] = []
    score_rows, totals = _score_rows(prob, target, di_chips, names, threshold)
    _write_csv(out_dir / "chips_agreement.csv", score_rows)

    objects = gpd.read_file(unit.objects_path, layer="objects")
    object_rows, rasters = _object_rows(objects, prob, grid, unit.intervals, threshold)
    _write_csv(out_dir / "di_objects.csv", object_rows)

    forest = None
    di_full = np.full(prob.shape, IGNORE, dtype="uint8")
    if unit.has_layers:
        forest, inside = cells_at_centres(
            fill_forest_dips(unit.layer("forest")), unit.grid.cell_transform, grid
        )
        layer = unit.layer("probability")[1:]
        percent = np.where(np.isfinite(layer), np.rint(layer * 100), IGNORE)
        di_full, _ = cells_at_centres(
            percent.astype("uint8"), unit.grid.cell_transform, grid
        )
        di_full[:, ~inside] = IGNORE
    else:
        notes.append("no cell layers on this machine: DI panels show chips only")
        di_full = di_chips
    patch_rows = _patch_rows(prob, rasters, grid, unit.intervals, threshold, forest)
    _write_csv(out_dir / "model_only.csv", patch_rows)
    _overview(prob, di_full, names, grid, footprints, out_dir / "overview.png")

    agents: dict[str, Any] | None = None
    agent_path = Path(prob_path).with_name(f"agent_{unit.unit}.tif")
    if agent_path.is_file():
        labelled = label_agent_cells(chips_dir, unit.unit, grid, load_label().pooling)
        agent_rows, agents = _agent_rows(agent_path, labelled, prob, threshold)
        _write_csv(out_dir / "agent_confusion.csv", agent_rows)

    seen = [r for r in object_rows if r["cells"]]
    summary = {
        "unit": unit.unit,
        "threshold": threshold,
        "chips": len(footprints),
        "cells_scored": totals["model"].n_valid,
        "loss_cells": totals["model"].n_pos,
        "model": {
            "average_precision": totals["model"].average_precision(),
            **totals["model"].at(threshold),
        },
        "di": {
            "average_precision": totals["di"].average_precision(),
            **totals["di"].at(0.5),
        },
        "di_objects": len(seen),
        "di_objects_half_flagged": sum(r["model_share_above"] >= 0.5 for r in seen),
        "model_only_patches": len(patch_rows),
        "model_only_ha": float(sum(r["area_ha"] for r in patch_rows)),
        "agents": agents,
        "notes": notes,
    }
    (out_dir / "compare.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )

    panels = 0
    if data_root is not None:
        mosaics = [
            [str(p) for p in local_mosaics(Path(data_root), unit.unit, year)]
            for year in unit.years
        ]
        if all(mosaics):
            panels = _draw_panels(
                out_dir / "events",
                objects,
                seen,
                patch_rows,
                names,
                mosaics,
                prob,
                di_full,
                target,
                grid,
                (n_objects, n_model_only, n_di_only),
            )
        else:
            notes.append("a year has no mosaic, so no close-up was drawn")
    logger.info(
        "%s: model AP %.3f, DI AP %.3f on %d cells of %d chips. %d model-only "
        "patches. %d close-ups.",
        unit.unit,
        summary["model"]["average_precision"],
        summary["di"]["average_precision"],
        summary["cells_scored"],
        summary["chips"],
        len(patch_rows),
        panels,
    )
    return CompareResult(out_dir, summary, panels, notes)


def _draw_panels(
    folder: Path,
    objects: Any,
    object_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
    names: list[str],
    mosaics: list[list[str]],
    prob: np.ndarray,
    di: np.ndarray,
    target: np.ndarray,
    grid: PredGrid,
    counts: tuple[int, int, int],
) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    n_objects, n_model_only, n_di_only = counts
    by_area = sorted(object_rows, key=lambda r: -r["area_ha"])
    missed = [r for r in by_area if r["model_share_above"] < 0.1]
    largest = [r for r in by_area if r not in missed[:n_di_only]]
    geometry = dict(zip(objects["object_id"], objects.geometry, strict=True))
    jobs: list[tuple[str, str, dict[str, Any], Any]] = []
    for row in largest[:n_objects]:
        jobs.append(("object", f"DI object {row['object_id']}", row, geometry))
    for row in missed[:n_di_only]:
        jobs.append(
            ("di_only", f"DI object {row['object_id']}, model silent", row, geometry)
        )
    for row in sorted(patch_rows, key=lambda r: -r["area_ha"])[:n_model_only]:
        jobs.append(("model_only", f"Model-only patch {row['patch']}", row, None))
    drawn = 0
    for kind, title, row, lookup in jobs:
        k = int(row["band"]) - 1
        outline = lookup.get(row["object_id"]) if lookup else None
        side = min(max(4.0 * math.sqrt(row["area_ha"] * 10_000), 400.0), 1500.0)
        bounds = _panel_bounds(row["x"], row["y"], side, grid)
        ident = row.get("object_id", row.get("patch"))
        text = (
            f"{title}: {names[k].replace('_', ' to ')}, {row['area_ha']:.1f} ha, "
            f"model mean {row['model_mean_percent']:.0f}%"
        )
        try:
            _panel(
                text,
                bounds,
                k,
                names,
                mosaics,
                prob,
                di,
                target,
                grid,
                outline,
                folder / f"{kind}_{k + 1}_{ident}.png",
            )
            drawn += 1
        except Exception:  # noqa: BLE001 - one bad panel must not lose the tables
            logger.exception("Panel %s failed", title)
    return drawn
