"""csdv_core.pairmodel.patches: the pair model's maps as a layer of patches.

Inputs, all from the exports of a grid run (``run_layers/<unit>/<tag>/``):
the model's map of a unit (``model/prob_forest_<unit>.tif``, percent, and
``model/agent_<unit>.tif``, the cause code of every cell), the name of its
model (``model/checkpoint.txt``) and the unit's cell layers of canopy cover
and height (``layers/chm_cover.tif``, ``layers/chm_height.tif``, 19.2 m, one
band per image year).

Output: one polygon per patch of canopy loss and interval, with the two
image dates, the area, the model's probability, the cause the model names and
the change in canopy cover and height between the two dates.

A patch is a set of connected map cells (8 neighbours) at or above ``grow``
percent that holds at least one cell at or above ``threshold`` percent, the
core. Holes under the smallest patch size are filled. The cause is counted
over the core cells. Adjoining quad maps overlap
by a strip, so a cell belongs to the quad that holds its centre. A patch that
is cut by a quad edge is joined to its other part when a block of quads is
processed (:func:`block_patches`), and the strip that two maps share is
compared per interval (:func:`seam_table`).

The layer keeps the share of every cause, so a rule that renames or merges
causes is a filter on the table, and it gives the change in cover and height
as numbers without a severity class.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from csdv_core.pairmodel.data import AGENT_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCK_CRS",
    "PatchSettings",
    "PatchesResult",
    "UnitMap",
    "block_patches",
    "grow_patches",
    "owned_cells",
    "read_unit_map",
    "seam_table",
    "unit_patches",
    "write_block",
]

#: CRS of a block's layer: CONUS Albers, so quads of two UTM zones join.
BLOCK_CRS = "EPSG:5070"
#: Smallest patch, the size used for the model-only patches of the audit.
MIN_PATCH_HA = 0.3
NODATA = 255
_EIGHT = np.ones((3, 3), dtype=bool)
#: Relative change in the canopy cover of unflagged forest between two dates
#: from which the height model of one of them is reported as off.
COVER_SHIFT = 0.2
_COUNTS = [f"n_{name}" for name in AGENT_NAMES]
_SUMS = ["s_percent", "s_cover_a", "s_cover_b", "s_height_a", "s_height_b"]
_WEIGHTS = ["w_cover", "w_height"]

#: Columns of the layer, in order.
COLUMNS = [
    "patch_id",
    "unit",
    "units",
    "interval",
    "date_a",
    "date_b",
    "area_ha",
    "cells",
    "mean_percent",
    "max_percent",
    "core_share",
    "cause",
    "cause_share",
    *[f"share_{name}" for name in AGENT_NAMES],
    "cover_before",
    "cover_after",
    "cover_loss",
    "height_before",
    "height_after",
    "height_loss",
    "bg_cover_loss",
    "bg_height_loss",
    "on_edge",
    "model",
    "threshold",
    "grow",
]


@dataclass(frozen=True)
class PatchSettings:
    """How cells become patches.

    Attributes:
        threshold: Percent from which a cell is core. A patch needs one.
        grow: Percent down to which a patch grows from its core. Equal to
            ``threshold`` for a plain cut.
        min_ha: Smallest patch.
    """

    threshold: int = 90
    grow: int = 50
    min_ha: float = MIN_PATCH_HA

    def __post_init__(self) -> None:
        if not 1 <= self.grow <= self.threshold <= 100:
            raise ValueError(
                f"Need 1 <= grow <= threshold <= 100, got grow {self.grow} "
                f"and threshold {self.threshold}"
            )


@dataclass
class UnitMap:
    """The model's map of one unit.

    Attributes:
        unit: Unit ID.
        prob: Percent ``(intervals, rows, cols)``, 255 without a value.
        agent: Cause codes of the same shape, or None for a map without.
        transform: Affine transform of the cells.
        epsg: EPSG code of the map.
        names: ``<date A>_<date B>`` of every band.
        model: The training run that made the map, or an empty string.
        bbox_4326: The unit's box in degrees, or None when every cell of the
            map is the unit's own.
        path: The probability raster, read again by :meth:`window` once
            the arrays were let go.
        shape: ``(rows, cols)`` of the map.
        background: Filled by :func:`unit_patches`, one row per interval
            with patches: the share of the unit that is flagged, and canopy
            cover and height of its unflagged forest at the two dates.
    """

    unit: str
    prob: np.ndarray | None
    agent: np.ndarray | None
    transform: Any
    epsg: int
    names: list[str]
    model: str = ""
    bbox_4326: tuple[float, float, float, float] | None = None
    path: Path | None = None
    shape: tuple[int, int] = (0, 0)
    background: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.prob is not None:
            self.shape = (self.prob.shape[1], self.prob.shape[2])

    @property
    def cell_ha(self) -> float:
        return abs(self.transform.a * self.transform.e) / 1e4

    def release(self) -> None:
        """Let the arrays go. A block of quads does not fit in memory."""
        if self.path is not None:
            self.prob = None
            self.agent = None

    def window(self, band: int, rows: slice, cols: slice) -> np.ndarray:
        """Part of one band, from memory or from the file."""
        if self.prob is not None:
            return self.prob[band][rows, cols]
        import rasterio
        from rasterio.windows import Window

        window = Window(
            cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start
        )
        with rasterio.open(self.path) as src:
            return src.read(band + 1, window=window)


@dataclass
class PatchesResult:
    """What :func:`write_block` wrote.

    Attributes:
        out_dir: Folder of the files.
        units: Units with a map.
        without_map: Units of the batch files that have none.
        patches: Number of patches.
        area_ha: Their area.
        joined: Patches that were joined from parts in two or more quads.
        seams: Rows of the seam table.
    """

    out_dir: Path
    units: list[str]
    without_map: list[str] = field(default_factory=list)
    patches: int = 0
    area_ha: float = 0.0
    joined: int = 0
    seams: int = 0


def interval_key(name: str) -> str:
    """``2012-06-07_2014-07-05`` as ``2012_2014``."""
    first, second = name.split("_")
    return f"{first[:4]}_{second[:4]}"


def read_unit_map(
    model_dir: Path,
    unit: str,
    bbox_4326: tuple[float, float, float, float] | None = None,
) -> UnitMap | None:
    """Read the map of a unit from the folder of its model.

    The probability on forest is read when the folder holds it, as the page
    of a block does.

    Returns:
        The map, or None when the folder holds no probability raster.

    Raises:
        ValueError: If a band has no pair of dates as its description, or
            the cause raster has another shape than the probability.
    """
    import rasterio

    path = model_dir / f"prob_forest_{unit}.tif"
    if not path.is_file():
        path = model_dir / f"prob_{unit}.tif"
    if not path.is_file():
        return None
    with rasterio.open(path) as src:
        prob = src.read()
        transform = src.transform
        epsg = src.crs.to_epsg()
        names = [name or "" for name in src.descriptions]
    if any("_" not in name for name in names):
        raise ValueError(f"{path}: every band needs '<date>_<date>' as description")
    agent = None
    agent_path = model_dir / f"agent_{unit}.tif"
    if agent_path.is_file():
        with rasterio.open(agent_path) as src:
            agent = src.read()
        if agent.shape != prob.shape:
            raise ValueError(f"{agent_path}: shape {agent.shape}, not {prob.shape}")
    note = model_dir / "checkpoint.txt"
    model = note.read_text(encoding="utf-8").strip() if note.is_file() else ""
    return UnitMap(
        unit, prob, agent, transform, epsg, names, model, bbox_4326, path=path
    )


def owned_cells(unit_map: UnitMap) -> np.ndarray:
    """The cells whose centre lies in the unit's box.

    The box is half open (west and south edges belong to it), so a cell on
    the lattice that two adjoining quads of one UTM zone share has one owner.

    Returns:
        bool ``(rows, cols)``. All True for a map without a box.
    """
    rows, cols = unit_map.shape
    if unit_map.bbox_4326 is None:
        return np.ones((rows, cols), dtype=bool)
    from pyproj import Transformer

    t = unit_map.transform
    xs = t.c + (np.arange(cols) + 0.5) * t.a
    ys = t.f + (np.arange(rows) + 0.5) * t.e
    grid_x, grid_y = np.meshgrid(xs, ys)
    to_degrees = Transformer.from_crs(unit_map.epsg, 4326, always_xy=True)
    lon, lat = to_degrees.transform(grid_x, grid_y)
    west, south, east, north = unit_map.bbox_4326
    return (lon >= west) & (lon < east) & (lat >= south) & (lat < north)


def grow_patches(
    band: np.ndarray, settings: PatchSettings, cell_ha: float
) -> tuple[np.ndarray, int]:
    """Label the patches of one interval.

    Args:
        band: Percent, 255 without a value.
        settings: Thresholds and the smallest patch.
        cell_ha: Area of a cell.

    Returns:
        int32 labels (0 is background, patches count from 1) and the number
        of patches.
    """
    valid = band != NODATA
    labels, n = ndi.label(valid & (band >= settings.grow), structure=_EIGHT)
    if n == 0:
        return np.zeros(band.shape, dtype=np.int32), 0
    core = valid & (band >= settings.threshold)
    cores = np.bincount(labels[core], minlength=n + 1)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    min_cells = int(np.ceil(settings.min_ha / cell_ha - 1e-9))
    keep = (cores > 0) & (sizes >= min_cells)
    keep[0] = False
    renumber = np.zeros(n + 1, dtype=np.int32)
    renumber[keep] = np.arange(1, int(keep.sum()) + 1)
    return _fill_holes(renumber[labels], min_cells), int(keep.sum())


def _fill_holes(labels: np.ndarray, min_cells: int) -> np.ndarray:
    """Give the holes of a patch that are under the smallest size to it.

    A hole is ground outside every patch that one patch encloses (it does
    not reach the edge of the map). The rule for the smallest patch applies
    to holes as well: single cells that were not forest, or that the model
    scored low, do not perforate a patch.
    """
    holes, n = ndi.label(labels == 0)
    if n == 0:
        return labels
    sizes = np.bincount(holes.ravel(), minlength=n + 1)
    rim = np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    small = sizes < min_cells
    small[np.unique(rim)] = False
    small[0] = False
    if not small.any():
        return labels
    # The patch around a hole: the label of the cells beside it.
    beside = ndi.grey_dilation(labels, footprint=_EIGHT)
    owner = ndi.maximum(beside, holes, index=np.arange(n + 1)).astype(np.int32)
    owner[~small] = 0
    return np.where(labels == 0, owner[holes], labels).astype(np.int32)


def _at_centres(layer: np.ndarray, layer_transform: Any, unit_map: UnitMap):
    """A coarser cell layer sampled at the centre of every map cell."""
    rows, cols = unit_map.shape
    t = unit_map.transform
    xs = t.c + (np.arange(cols) + 0.5) * t.a
    ys = t.f + (np.arange(rows) + 0.5) * t.e
    c = np.floor((xs - layer_transform.c) / layer_transform.a).astype(int)
    r = np.floor((ys - layer_transform.f) / layer_transform.e).astype(int)
    inside = ((r >= 0) & (r < layer.shape[0]))[:, None] & (
        (c >= 0) & (c < layer.shape[1])
    )[None, :]
    r = np.clip(r, 0, layer.shape[0] - 1)
    c = np.clip(c, 0, layer.shape[1] - 1)
    values = layer[r[:, None], c[None, :]].astype(np.float32)
    values[~inside] = np.nan
    return values


def _read_layers(layers_dir: Path | None) -> dict[str, Any]:
    """Canopy cover and height of a unit per year, when the folder has them."""
    import rasterio

    layers: dict[str, Any] = {}
    if layers_dir is None:
        return layers
    for name in ("chm_cover", "chm_height"):
        path = layers_dir / f"{name}.tif"
        if not path.is_file():
            continue
        with rasterio.open(path) as src:
            values = src.read().astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                values[values == src.nodata] = np.nan
            years = [str(d) for d in src.descriptions]
            layers[name] = (values, years, src.transform, src.crs.to_epsg())
    return layers


def _loss(before: Any, after: Any) -> Any:
    """Relative loss ``1 - after / before``, NaN without a positive start."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(before > 0, 1.0 - after / before, np.nan)


def _finish(frame: Any, settings: PatchSettings) -> Any:
    """Derive the shares and means of a table of patch sums."""
    cells = frame["cells"].to_numpy(dtype=float)
    core = frame["core_cells"].to_numpy(dtype=float)
    frame["mean_percent"] = (frame["s_percent"] / cells).round(1)
    frame["core_share"] = (core / cells).round(3)
    counts = frame[_COUNTS].to_numpy(dtype=float)
    named = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(named[:, None] > 0, counts / named[:, None], np.nan)
    for i, name in enumerate(AGENT_NAMES):
        frame[f"share_{name}"] = shares[:, i].round(3)
    top = counts.argmax(axis=1)
    frame["cause"] = [
        AGENT_NAMES[i] if n > 0 else "" for i, n in zip(top, named, strict=True)
    ]
    frame["cause_share"] = np.where(
        named > 0, shares[np.arange(len(frame)), top], np.nan
    ).round(3)
    for kind in ("cover", "height"):
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = frame[f"w_{kind}"].to_numpy(dtype=float)
            before = np.where(weight > 0, frame[f"s_{kind}_a"] / weight, np.nan)
            after = np.where(weight > 0, frame[f"s_{kind}_b"] / weight, np.nan)
        frame[f"{kind}_before"] = before.round(3)
        frame[f"{kind}_after"] = after.round(3)
        frame[f"{kind}_loss"] = _loss(before, after).round(3)
    frame["threshold"] = settings.threshold
    frame["grow"] = settings.grow
    return frame


def unit_patches(
    unit_map: UnitMap,
    settings: PatchSettings,
    layers_dir: Path | None = None,
) -> Any:
    """The patches of one unit, every interval.

    A patch is found on the whole map, so that its size and its core count
    where the map reaches past the unit's box, and is then cut to the cells
    the unit owns. A part that the cut leaves under the smallest size is
    kept when it lies on the edge, for the join with the neighbour.

    Args:
        unit_map: The map.
        settings: Thresholds and the smallest patch.
        layers_dir: Folder with ``chm_cover.tif`` and ``chm_height.tif``.
            Without it the cover and height columns are empty.

    Returns:
        GeoDataFrame in the map's CRS with the columns of :data:`COLUMNS`
        and the sums they derive from (``core_cells``, ``n_<cause>``,
        ``s_*``, ``w_*``), which :func:`block_patches` adds up.
    """
    import geopandas as gpd
    import pandas as pd

    from csdv_core.disturbance.outlines import _label_polygons

    own = owned_cells(unit_map)
    edge = own & ndi.binary_dilation(~own, structure=_EIGHT)
    border = np.zeros_like(own)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    edge |= own & border
    layers = _read_layers(layers_dir)
    min_cells = int(np.ceil(settings.min_ha / unit_map.cell_ha - 1e-9))
    frames = []
    for k, name in enumerate(unit_map.names):
        band = unit_map.prob[k]
        whole, n = grow_patches(band, settings, unit_map.cell_ha)
        if n == 0:
            continue
        labels, n = ndi.label((whole > 0) & own, structure=_EIGHT)
        if n == 0:
            continue
        flat = labels.ravel()
        size = n + 1
        cells = np.bincount(flat, minlength=size)
        on_edge = np.bincount(labels[edge], minlength=size) > 0
        keep = (cells >= min_cells) | on_edge
        keep[0] = False
        # A filled hole may hold cells without a value: they count as 0.
        values = np.where(band == NODATA, 0, band).astype(np.float64).ravel()
        core = (band >= settings.threshold) & (band != NODATA) & (labels > 0)
        date_a, date_b = name.split("_")
        table: dict[str, Any] = {
            "cells": cells,
            "core_cells": np.bincount(labels[core], minlength=size),
            "s_percent": np.bincount(flat, weights=values, minlength=size),
            "max_percent": ndi.maximum(band, labels, index=np.arange(size)),
            "on_edge": on_edge,
        }
        for code, column in enumerate(_COUNTS, start=1):
            if unit_map.agent is None:
                table[column] = np.zeros(size, dtype=int)
            else:
                hit = core & (unit_map.agent[k] == code)
                table[column] = np.bincount(labels[hit], minlength=size)
        background = own & (band != NODATA) & (band < settings.grow) & (whole == 0)
        bg: dict[str, float] = {}
        for kind in ("cover", "height"):
            sums = _layer_sums(layers.get(f"chm_{kind}"), unit_map, date_a, date_b)
            if sums is None:
                for column in (f"s_{kind}_a", f"s_{kind}_b", f"w_{kind}"):
                    table[column] = np.zeros(size)
                bg[kind] = float("nan")
                continue
            before, after = sums
            both = np.isfinite(before) & np.isfinite(after)
            picked = flat[both.ravel()]
            table[f"s_{kind}_a"] = np.bincount(
                picked, weights=before[both], minlength=size
            )
            table[f"s_{kind}_b"] = np.bincount(
                picked, weights=after[both], minlength=size
            )
            table[f"w_{kind}"] = np.bincount(picked, minlength=size).astype(float)
            # Forest that the model does not flag: what the height model alone
            # changed between the two dates.
            quiet = background & both & (before > 0)
            bg[kind] = (
                float(_loss(before[quiet].mean(), after[quiet].mean()))
                if quiet.any()
                else float("nan")
            )
            if quiet.any():
                bg[f"{kind}_a"] = float(before[quiet].mean())
                bg[f"{kind}_b"] = float(after[quiet].mean())
        mapped = int((own & (band != NODATA)).sum())
        unit_map.background.append(
            {
                "unit": unit_map.unit,
                "interval": interval_key(name),
                "date_a": date_a,
                "date_b": date_b,
                "flagged_pct": round(100 * int(cells[keep].sum()) / max(mapped, 1), 3),
                "bg_cover_before": round(bg.get("cover_a", float("nan")), 3),
                "bg_cover_after": round(bg.get("cover_b", float("nan")), 3),
                "bg_cover_loss": round(bg["cover"], 3),
                "bg_height_before": round(bg.get("height_a", float("nan")), 2),
                "bg_height_after": round(bg.get("height_b", float("nan")), 2),
                "bg_height_loss": round(bg["height"], 3),
            }
        )
        kept = np.flatnonzero(keep)
        if not len(kept):
            continue
        shapes = _label_polygons(
            np.where(keep[labels], labels, 0).astype(np.int32), unit_map.transform
        )
        key = interval_key(name)
        frame = pd.DataFrame({column: table[column][kept] for column in table})
        frame["geometry"] = [shapes[int(label)] for label in kept]
        frame["patch_id"] = [
            f"{unit_map.unit}_{key}_{i + 1:04d}" for i in range(len(kept))
        ]
        frame["unit"] = unit_map.unit
        frame["units"] = unit_map.unit
        frame["interval"] = key
        frame["date_a"] = date_a
        frame["date_b"] = date_b
        frame["area_ha"] = (frame["cells"] * unit_map.cell_ha).round(3)
        frame["bg_cover_loss"] = round(bg["cover"], 3)
        frame["bg_height_loss"] = round(bg["height"], 3)
        frame["model"] = unit_map.model
        frames.append(frame)
    crs = f"EPSG:{unit_map.epsg}"
    extra = ["core_cells", *_COUNTS, *_SUMS, *_WEIGHTS]
    if not frames:
        return gpd.GeoDataFrame(columns=[*COLUMNS, *extra], geometry=[], crs=crs)
    out = _finish(pd.concat(frames, ignore_index=True), settings)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)[
        [*COLUMNS, *extra, "geometry"]
    ]


def _layer_sums(layer: Any, unit_map: UnitMap, date_a: str, date_b: str) -> Any:
    """A cell layer at the two dates of an interval, on the map's cells."""
    if layer is None:
        return None
    values, years, transform, epsg = layer
    if epsg != unit_map.epsg or date_a[:4] not in years or date_b[:4] not in years:
        return None
    return (
        _at_centres(values[years.index(date_a[:4])], transform, unit_map),
        _at_centres(values[years.index(date_b[:4])], transform, unit_map),
    )


def block_patches(frames: Sequence[Any], settings: PatchSettings, tol_m: float = 3.0):
    """Join the patches of adjoining units into one layer.

    Parts on a unit's edge that lie within ``tol_m`` of a part of another
    unit in the same pair of years become one patch: cells and sums are
    added, the dates span both parts, and the unit of the larger part gives
    the ID and the background change. A patch that stays under the smallest
    size after the join is dropped.

    Args:
        frames: Output of :func:`unit_patches`, one per unit, in any CRS.
        settings: The settings the frames were made with.
        tol_m: Largest gap between two parts of one patch. Cells of one UTM
            zone meet exactly. Across two zones the lattices differ.

    Returns:
        GeoDataFrame in :data:`BLOCK_CRS` with the columns of
        :data:`COLUMNS` and ``parts``, the number of parts of each patch.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    frames = [f.to_crs(BLOCK_CRS) for f in frames if len(f)]
    if not frames:
        return gpd.GeoDataFrame(columns=[*COLUMNS, "parts"], geometry=[], crs=BLOCK_CRS)
    parts = pd.concat(frames, ignore_index=True)
    parts = gpd.GeoDataFrame(parts, geometry="geometry", crs=BLOCK_CRS)
    parent = np.arange(len(parts))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge = parts[parts["on_edge"].astype(bool)]
    if len(edge) > 1:
        reach = edge[["unit", "interval", "geometry"]].copy()
        reach["geometry"] = reach.geometry.buffer(tol_m)
        pairs = gpd.sjoin(
            reach, edge[["unit", "interval", "geometry"]], predicate="intersects"
        )
        pairs = pairs[
            (pairs["unit_left"] != pairs["unit_right"])
            & (pairs["interval_left"] == pairs["interval_right"])
        ]
        for left, right in zip(pairs.index, pairs["index_right"], strict=True):
            a, b = find(int(left)), find(int(right))
            if a != b:
                parent[max(a, b)] = min(a, b)
    parts["_group"] = [find(i) for i in range(len(parts))]
    parts["_cells"] = parts["cells"]
    sums = ["cells", "core_cells", *_COUNTS, *_SUMS, *_WEIGHTS]
    lead = parts.sort_values("_cells", ascending=False).groupby("_group").head(1)
    lead = lead.set_index("_group")
    grouped = parts.groupby("_group")
    out = grouped[sums].sum()
    out["area_ha"] = grouped["area_ha"].sum().round(3)
    out["max_percent"] = grouped["max_percent"].max()
    out["date_a"] = grouped["date_a"].min()
    out["date_b"] = grouped["date_b"].max()
    out["units"] = grouped["unit"].agg(lambda u: "+".join(sorted(set(u))))
    out["parts"] = grouped.size()
    out["on_edge"] = grouped["on_edge"].any()
    for column in (
        "patch_id",
        "unit",
        "interval",
        "bg_cover_loss",
        "bg_height_loss",
        "model",
    ):
        out[column] = lead[column]
    out["geometry"] = grouped["geometry"].agg(lambda g: unary_union(list(g)))
    out = out[out["area_ha"] >= settings.min_ha - 1e-9].reset_index(drop=True)
    out = _finish(out, settings)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=BLOCK_CRS)
    return out[[*COLUMNS, "parts", "geometry"]]


def _shared_window(a: UnitMap, b: UnitMap):
    """Row and column slices of the cells two maps of one lattice share."""
    if a.epsg != b.epsg:
        return None
    ta, tb = a.transform, b.transform
    if abs(ta.a - tb.a) > 1e-6 or abs(ta.e - tb.e) > 1e-6:
        return None
    dc = (tb.c - ta.c) / ta.a
    dr = (tb.f - ta.f) / ta.e
    if abs(dc - round(dc)) > 1e-3 or abs(dr - round(dr)) > 1e-3:
        return None
    dc, dr = int(round(dc)), int(round(dr))
    (rows_a, cols_a), (rows_b, cols_b) = a.shape, b.shape
    r0, r1 = max(0, dr), min(rows_a, dr + rows_b)
    c0, c1 = max(0, dc), min(cols_a, dc + cols_b)
    if r0 >= r1 or c0 >= c1:
        return None
    return (
        (slice(r0, r1), slice(c0, c1)),
        (slice(r0 - dr, r1 - dr), slice(c0 - dc, c1 - dc)),
    )


def seam_table(maps: Sequence[UnitMap], settings: PatchSettings) -> list[dict]:
    """Compare the maps of adjoining units on the cells both cover.

    Two maps of one UTM zone share cells of the same lattice in the strip
    where their mosaics overlap. Each such pair of units gives one row per
    pair of years that both maps hold.

    Returns:
        Rows with the two units, the interval, the image dates of each
        side, the number of cells with a value in both maps, the percent of
        them at or above the threshold in each map, in both, their
        agreement (cells flagged in both over cells flagged in either) and
        the mean absolute difference of the two maps in percent points.
    """
    rows: list[dict] = []
    ordered = sorted(maps, key=lambda m: m.unit)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            windows = _shared_window(a, b)
            if windows is None:
                continue
            keys_b = {interval_key(name): k for k, name in enumerate(b.names)}
            for ka, name in enumerate(a.names):
                key = interval_key(name)
                if key not in keys_b:
                    continue
                kb = keys_b[key]
                pa = a.window(ka, *windows[0])
                pb = b.window(kb, *windows[1])
                both = (pa != NODATA) & (pb != NODATA)
                n = int(both.sum())
                if n == 0:
                    continue
                fa = both & (pa >= settings.threshold)
                fb = both & (pb >= settings.threshold)
                either = int((fa | fb).sum())
                diff = np.abs(pa[both].astype(float) - pb[both].astype(float))
                rows.append(
                    {
                        "unit_a": a.unit,
                        "unit_b": b.unit,
                        "interval": key,
                        "dates_a": name,
                        "dates_b": b.names[kb],
                        "same_dates": name == b.names[kb],
                        "cells": n,
                        "flagged_a_pct": round(100 * fa.sum() / n, 3),
                        "flagged_b_pct": round(100 * fb.sum() / n, 3),
                        "flagged_both_pct": round(100 * (fa & fb).sum() / n, 3),
                        "agreement": (
                            round(float((fa & fb).sum()) / either, 3)
                            if either
                            else float("nan")
                        ),
                        "mean_abs_diff": round(float(diff.mean()), 2),
                    }
                )
    return rows


def write_block(
    units: Sequence[tuple[str, tuple[float, float, float, float] | None]],
    export_dir: Path,
    out_dir: Path,
    settings: PatchSettings | None = None,
    *,
    run_tag: str = "default",
    model_dir: Path | None = None,
    layers_dir: Path | None = None,
) -> PatchesResult:
    """Write the patch layer of a set of units.

    Args:
        units: ``(unit ID, box in degrees or None)`` of every unit.
        export_dir: Exports folder with ``run_layers/<unit>/<tag>/``.
        out_dir: Folder for ``patches.gpkg``, ``patches.csv``,
            ``summary.json`` and ``seams.csv``.
        settings: Thresholds and the smallest patch.
        run_tag: Tag of the disturbance run.
        model_dir: Folder of the map, in place of the export's. For one unit.
        layers_dir: Folder of the cell layers, in place of the export's.

    Raises:
        ValueError: If no unit has a map, or ``model_dir`` is given with
            more than one unit.
    """
    import pandas as pd

    settings = settings or PatchSettings()
    if (model_dir or layers_dir) and len(units) != 1:
        raise ValueError("A folder of the map or the layers is for one unit")
    maps: list[UnitMap] = []
    frames = []
    without: list[str] = []
    for unit, bbox in units:
        base = Path(export_dir) / "run_layers" / unit / run_tag
        unit_map = read_unit_map(model_dir or base / "model", unit, bbox)
        if unit_map is None:
            without.append(unit)
            continue
        frame = unit_patches(unit_map, settings, layers_dir or base / "layers")
        logger.info("%s: %d patches, %.1f ha", unit, len(frame), frame["area_ha"].sum())
        unit_map.release()
        maps.append(unit_map)
        frames.append(frame)
    if not maps:
        raise ValueError("None of the units has a model map in the exports")
    layer = block_patches(frames, settings)
    seams = seam_table(maps, settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "patches.gpkg"
    gpkg.unlink(missing_ok=True)
    layer.to_file(gpkg, layer="patches", driver="GPKG")
    table = pd.DataFrame(layer.drop(columns="geometry"))
    if len(layer):
        centres = layer.geometry.representative_point().to_crs(4326)
        table["lon"] = centres.x.round(5).to_numpy()
        table["lat"] = centres.y.round(5).to_numpy()
    table.to_csv(out_dir / "patches.csv", index=False)
    pd.DataFrame(seams).to_csv(out_dir / "seams.csv", index=False)
    background = pd.DataFrame([row for m in maps for row in m.background])
    background.to_csv(out_dir / "background.csv", index=False)
    shifted = (
        background[background["bg_cover_loss"].abs() >= COVER_SHIFT]
        if len(background)
        else background
    )
    if len(shifted):
        logger.warning(
            "%d of %d unit intervals: the canopy cover of unflagged forest "
            "changes by %.0f%% or more between the two dates (background.csv). "
            "The height model of one of the dates is off.",
            len(shifted),
            len(background),
            100 * COVER_SHIFT,
        )
    by = (
        table.groupby(["interval", "cause"])["area_ha"].agg(["count", "sum"])
        if len(table)
        else None
    )
    summary = {
        "settings": {
            "threshold": settings.threshold,
            "grow": settings.grow,
            "min_ha": settings.min_ha,
        },
        "units": [m.unit for m in maps],
        "without_map": without,
        "models": sorted({m.model for m in maps if m.model}),
        "patches": int(len(layer)),
        "area_ha": round(float(layer["area_ha"].sum()), 1) if len(layer) else 0.0,
        "joined": int((layer["parts"] > 1).sum()) if len(layer) else 0,
        "by_interval_cause": (
            [
                {
                    "interval": interval,
                    "cause": cause,
                    "patches": int(row["count"]),
                    "area_ha": round(float(row["sum"]), 1),
                }
                for (interval, cause), row in by.iterrows()
            ]
            if by is not None
            else []
        ),
        "seam_pairs": len({(r["unit_a"], r["unit_b"]) for r in seams}),
        "cover_shifts": int(len(shifted)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return PatchesResult(
        out_dir=out_dir,
        units=summary["units"],
        without_map=without,
        patches=summary["patches"],
        area_ha=summary["area_ha"],
        joined=summary["joined"],
        seams=len(seams),
    )
