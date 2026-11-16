"""csdv_core.audit.model_only: the fifth audit stratum, patches only the model flags.

Inputs: maps of the pair model (``prob_<unit>.tif``, ``prob_forest_<unit>.tif``
and ``agent_<unit>.tif`` of a run folder) on units the model never trained on,
and the candidates of the label run (``candidates/<unit>.gpkg``, every source).
Outputs: a seeded sample of patches (``patches.gpkg``), their request images
(``requests/<patch>/``), and block 3 of an existing audit.

A patch is a group of 8-connected cells at or above the run's threshold in one
interval, as in ``model_only.csv`` of ``csdv pair compare``. It enters the
population when it covers at least ``min_ha``, at least ``min_forest`` of its
cells were forest at the earlier date, and it touches no candidate of the
unit from any source and any interval. Such a patch is ground where the model
reports loss and the label run had nothing to judge, so the share of real loss
among them measures what the candidate set misses where the model sees it.

The draw takes the smallest keys of each unit, as the other strata do, so a
larger sample of the same seed holds the smaller one. Items are appended as
block 3 without touching the items already drawn.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from csdv_core.audit.draw import ITEM_COLUMNS, _key

log = logging.getLogger(__name__)

#: Audit block of the model-only patches.
BLOCK = 3
STRATUM = "model_only"
#: CRS of the drawn patches, which span several UTM zones.
SAMPLE_EPSG = 5070
#: Agent codes of ``agent_<unit>.tif`` (see csdv_core.pairmodel.predict).
AGENTS = {
    1: "harvest",
    2: "wind",
    3: "fire",
    4: "biotic_mortality",
    5: "conversion",
    6: "other",
}
NO_VALUE = 255

__all__ = [
    "BLOCK",
    "STRATUM",
    "ModelOnlyDesign",
    "append_items",
    "draw_patches",
    "patch_id",
    "patch_population",
    "render_patches",
    "shown_patches",
    "to_sample_crs",
]


@dataclass(frozen=True)
class ModelOnlyDesign:
    """Size and rules of the model-only stratum.

    Attributes:
        n_per_unit: Patches judged per unit.
        reserve: Further patches per unit, placed after every target item.
        min_ha: Smallest patch, hectares (the label run's minimum).
        min_forest: Smallest share of the patch's cells that were forest at
            the earlier date.
        seed: Seed of the keys. Keep it for every extension.
    """

    n_per_unit: int = 2
    reserve: int = 1
    min_ha: float = 0.3
    min_forest: float = 0.5
    seed: int = 0


def patch_id(unit: str, band: int, label: int) -> str:
    """Name of a patch. ``band`` and ``label`` follow ``model_only.csv``."""
    return f"{unit}_{band}-{label}"


def _threshold(run_dir: Path) -> float:
    path = run_dir / "compare" / "compare.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The threshold of the run is read from it."
        )
    return float(json.loads(path.read_text())["threshold"])


def shown_patches(run_dir: Path, unit: str) -> set[str]:
    """Patches of a run that already have a close-up in ``compare/events/``.

    The close-ups are named ``model_only_<band>_<band>-<label>.png``.
    """
    names = set()
    for path in (run_dir / "compare" / "events").glob("model_only_*.png"):
        token = path.stem.rsplit("_", 1)[-1]
        if "-" in token:
            band, label = token.split("-", 1)
            names.add(patch_id(unit, int(band), int(label)))
    return names


def _candidate_mask(
    candidates: gpd.GeoDataFrame, transform, shape_: tuple[int, int], epsg: int
) -> np.ndarray:
    """Cells that touch a candidate, of any source and interval."""
    if candidates.empty:
        return np.zeros(shape_, dtype=bool)
    if candidates.crs is not None and candidates.crs.to_epsg() != epsg:
        candidates = candidates.to_crs(epsg)
    geoms = [g for g in candidates.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape_, dtype=bool)
    mask = features.rasterize(
        ((g, 1) for g in geoms),
        out_shape=shape_,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask.astype(bool)


def patch_population(
    run_dir: Path,
    unit: str,
    candidates: gpd.GeoDataFrame,
    design: ModelOnlyDesign | None = None,
    exclude: set[str] | None = None,
) -> gpd.GeoDataFrame:
    """Every model-only patch of a unit that may enter the audit.

    Args:
        run_dir: Pair-model run folder with ``map/`` and ``compare/compare.json``.
        unit: Unit of the map.
        candidates: Candidates of the unit, every source.
        design: Rules. None: the defaults.
        exclude: Patch names the interpreter has already seen.

    Returns:
        One row per patch (UTM of the map): ``patch``, ``unit``, ``band``,
        ``interval`` (the band's dates), ``cells``, ``area_ha``,
        ``forest_share``, ``agent`` (most common model agent), ``mean_percent``.
    """
    design = design or ModelOnlyDesign()
    exclude = exclude or set()
    folder = Path(run_dir) / "map"
    threshold = _threshold(Path(run_dir))
    with rasterio.open(folder / f"prob_{unit}.tif") as src:
        prob = src.read()
        transform, epsg = src.transform, int(src.crs.to_epsg())
        names = [str(d) for d in src.descriptions]
    with rasterio.open(folder / f"prob_forest_{unit}.tif") as src:
        forest = src.read()
    agent_path = folder / f"agent_{unit}.tif"
    agent = None
    if agent_path.exists():
        with rasterio.open(agent_path) as src:
            agent = src.read()
    near = _candidate_mask(candidates, transform, prob.shape[1:], epsg)
    eight = np.ones((3, 3), dtype=bool)
    near = ndimage.binary_dilation(near, structure=eight)
    cut = int(round(threshold * 100))
    cell_ha = abs(transform.a * transform.e) / 10_000
    min_cells = math.ceil(design.min_ha / cell_ha)
    rows, geoms = [], []
    n_seen = 0
    for k in range(prob.shape[0]):
        band = prob[k]
        flagged = (band >= cut) & (band <= 100)
        labels, n = ndimage.label(flagged, structure=eight)
        if n == 0:
            continue
        index = np.arange(1, n + 1)
        sizes = ndimage.sum(flagged, labels, index)
        touched = ndimage.maximum(near, labels, index)
        was_forest = (forest[k] > 0) & (forest[k] != NO_VALUE)
        forest_share = ndimage.mean(was_forest, labels, index)
        means = ndimage.mean(band, labels, index)
        keep = (
            (sizes >= min_cells) & (touched == 0) & (forest_share >= design.min_forest)
        )
        for i in np.flatnonzero(keep):
            label = int(index[i])
            name = patch_id(unit, k + 1, label)
            if name in exclude:
                n_seen += 1
                continue
            cells = labels == label
            shapes = features.shapes(
                cells.astype("uint8"), mask=cells, transform=transform
            )
            geom = unary_union([shape(s) for s, _ in shapes])
            top_agent = ""
            if agent is not None:
                codes = agent[k][cells]
                codes = codes[(codes > 0) & (codes != NO_VALUE)]
                if codes.size:
                    top_agent = AGENTS.get(int(np.bincount(codes).argmax()), "")
            rows.append(
                {
                    "patch": name,
                    "unit": unit,
                    "band": k + 1,
                    "interval": names[k] if k < len(names) else "",
                    "cells": int(sizes[i]),
                    "area_ha": float(sizes[i] * cell_ha),
                    "forest_share": float(forest_share[i]),
                    "agent": top_agent,
                    "mean_percent": float(means[i]),
                }
            )
            geoms.append(geom)
    frame = gpd.GeoDataFrame(rows, geometry=geoms, crs=f"EPSG:{epsg}")
    log.info(
        "%s: %d model-only patches of at least %.1f ha on forest, %d excluded "
        "as seen",
        unit,
        len(frame),
        design.min_ha,
        n_seen,
    )
    return frame


def draw_patches(
    population: gpd.GeoDataFrame, design: ModelOnlyDesign | None = None
) -> gpd.GeoDataFrame:
    """The smallest keys of each unit: target patches, then reserve.

    Adds ``n_population``, ``rank`` and ``reserve``.
    """
    design = design or ModelOnlyDesign()
    parts = []
    for unit, part in population.groupby("unit", sort=True):
        part = part.copy()
        part["key"] = [_key(design.seed, f"model_only:{p}") for p in part["patch"]]
        total = min(len(part), design.n_per_unit + design.reserve)
        chosen = part.sort_values("key").head(total).copy()
        chosen["n_population"] = len(part)
        chosen["rank"] = range(1, total + 1)
        chosen["reserve"] = chosen["rank"] > design.n_per_unit
        parts.append(chosen)
        if len(part) < design.n_per_unit:
            log.warning("%s: only %d model-only patches", unit, len(part))
    if not parts:
        return population.head(0)
    return gpd.GeoDataFrame(pd.concat(parts), crs=population.crs)


def to_sample_crs(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Patches of one unit in the common CRS, with their UTM zone kept."""
    out = frame.copy()
    out["utm_epsg"] = int(frame.crs.to_epsg())
    return out.to_crs(SAMPLE_EPSG)


def render_patches(
    patches: gpd.GeoDataFrame,
    out_dir: Path,
    run,
    *,
    source: str | None = "local",
    data_root: Path | None = None,
    keep_stacks: bool = False,
) -> list[str]:
    """Write the request images of each patch, as block 1 shows a candidate.

    Every date of the unit is rendered with the patch outlined in yellow, with
    the same views and captions the vision models saw.

    Args:
        patches: Drawn patches in any CRS (``patches.gpkg`` is EPSG:5070).
        out_dir: Folder of the stratum. Writes ``requests/<patch>/`` and
            ``stacks/<patch>/meta.json``.
        run: A :class:`csdv_core.label.pipeline.LabelRun` for the unit inputs
            and the view settings.
        source: Imagery source of :func:`build_stack` (``local`` on CHPC).
        data_root: Root of the local mosaics.
        keep_stacks: Keep the yearly GeoTIFFs of each stack.

    Returns:
        Patches whose stack lacks a year, which are not rendered.
    """
    from csdv_core.label.lattice import window_for
    from csdv_core.label.pipeline import _save_requests
    from csdv_core.label.requests import verdict_request
    from csdv_core.label.stacks import build_stack, load_stack

    out_dir = Path(out_dir)
    failed = []
    for n, row in enumerate(patches.itertuples(), start=1):
        unit = run.unit(str(row.unit))
        folder = out_dir / "requests" / row.patch
        if (folder / "requests.jsonl").exists():
            continue
        geom = gpd.GeoSeries([row.geometry], crs=patches.crs).to_crs(unit.epsg).iloc[0]
        window = window_for(
            geom.bounds, window_id=row.patch, unit=unit.unit, epsg=unit.epsg
        )
        stack_dir = out_dir / "stacks" / row.patch
        meta = build_stack(
            unit,
            window,
            stack_dir,
            collar_px=run.cfg.chips.collar_px,
            source=source,
            data_root=data_root,
        )
        if meta.get("failed"):
            log.warning("%s: years without imagery: %s", row.patch, meta["failed"])
            failed.append(row.patch)
            continue
        stack = load_stack(stack_dir)
        request = verdict_request(stack, row.patch, geom, unit.unit, run.cfg.views)
        _save_requests(folder, [request], "requests.jsonl")
        if not keep_stacks:
            for path in stack_dir.glob("*.tif"):
                path.unlink()
        log.info("%d/%d %s rendered", n, len(patches), row.patch)
    return failed


def _later_image(dates: list[str], interval: str) -> str:
    """Image number (1-based) of the later date of a map interval.

    The map names an interval by the unit's dates. A window can be dated from
    its own flight, so the year is matched.
    """
    later = interval.split("_")[-1][:4]
    years = [d[:4] for d in dates]
    if later in years:
        return str(years.index(later) + 1)
    return ""


def append_items(
    items: pd.DataFrame, patches: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Add the rendered patches to the audit items as block 3.

    Existing items are returned unchanged. Patches already in the table are
    not added again. Target patches come first in a seeded order, then the
    reserve. Items are named ``M001`` onwards.

    Args:
        items: The audit items (``items.csv``).
        patches: Drawn patches (``patches.gpkg``, geometry not needed).
        out_dir: Folder of the stratum, with ``requests/<patch>/``.

    Returns:
        The items with block 3 appended.
    """
    out_dir = Path(out_dir)
    have = set(items.loc[items["kind"] == "patch", "cand_id"])
    rows = []
    for row in patches.itertuples():
        if row.patch in have:
            continue
        path = out_dir / "requests" / row.patch / "requests.jsonl"
        if not path.exists():
            log.warning("%s has no request images and is left out", row.patch)
            continue
        dates = json.loads(path.read_text().splitlines()[0])["dates"]
        rows.append(
            {
                "kind": "patch",
                "block": BLOCK,
                "stratum": STRATUM,
                "sub": row.unit,
                "window_id": row.patch,
                "cand_id": row.patch,
                "unit": row.unit,
                "status": "",
                "change_type": "",
                "agent": row.agent if isinstance(row.agent, str) else "",
                "label_images": _later_image(dates, str(row.interval)),
                "window_incl_prob": 1.0,
                "n_population": int(row.n_population),
                "rank": int(row.rank),
                "reserve": bool(row.reserve),
                "shared": False,
                "order_key": _key(0, f"order:{row.patch}"),
            }
        )
    if not rows:
        return items
    fresh = pd.DataFrame(rows).sort_values(["reserve", "order_key"])
    old = items.loc[items["block"] == BLOCK, "order"]
    start = int(old.max()) if len(old) else 0
    fresh["order"] = range(start + 1, start + len(fresh) + 1)
    fresh["item"] = [f"M{i:03d}" for i in fresh["order"]]
    result = pd.concat([items, fresh[ITEM_COLUMNS]], ignore_index=True)
    log.info("Added %d model-only items as block %d", len(fresh), BLOCK)
    return result[ITEM_COLUMNS]
