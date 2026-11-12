"""csdv_core.label.pipeline: the labelling steps over a folder of windows.

A label run is one folder:

    windows.csv              the windows (written by the sampler, editable)
    candidates/<unit>.gpkg   the unit's candidates, in its UTM CRS
    stacks/<window_id>/      four-band NAIP of each window (where rendered)
    requests/<window_id>/    request JPEGs and ``requests.jsonl``
    answers.jsonl            one record per model answer
    ledger.jsonl, responses/ cost ledger and response cache
    chips/                   training chips
    labels/                  vector labels and tables

Rendering needs the imagery and no API key. Asking the models needs the
request files and the keys, but no imagery. Building the labels needs
neither. The three can therefore run on different machines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from csdv_core.config import load_label
from csdv_core.config._models import LabelConfig
from csdv_core.label import candidates as C
from csdv_core.label import judge as J
from csdv_core.label.inputs import UnitInputs, load_unit_inputs
from csdv_core.label.labels import JudgedCandidate, WindowLabels, window_labels
from csdv_core.label.lattice import GSD, Window
from csdv_core.label.requests import Request
from csdv_core.label.stacks import (
    CELL_NDVI_FILE,
    CellNdvi,
    WindowStack,
    build_stack,
    load_cell_ndvi,
    load_stack,
    unit_cell_ndvi,
    write_cell_ndvi,
)
from csdv_core.label.views import View

logger = logging.getLogger(__name__)

WINDOW_COLUMNS = ["window_id", "unit", "epsg", "ix0", "iy0", "nx", "ny"]

__all__ = [
    "WINDOW_COLUMNS",
    "LabelRun",
    "WindowParts",
    "build_window",
    "load_requests",
    "render_window",
    "window_parts",
]


@dataclass
class LabelRun:
    """One label run: its folder, the disturbance runs it draws on, its thresholds.

    Attributes:
        root: Folder of the label run.
        runs_dir: Folder holding one disturbance run folder per unit. For
            exported runs that is ``<export_dir>/runs/<run_tag>``. On CHPC
            pass ``<results>/disturbance`` and ``run_subdir=<run_tag>``.
        run_tag: Run tag of the disturbance runs.
        run_subdir: Sub-folder of each unit folder that holds the run.
        layers_dir: Folder with pulled cell layers (``<unit>/<tag>/``), when
            they are not inside the run folders.
        models: The two models whose verdicts are combined.
    """

    root: Path
    runs_dir: Path
    run_tag: str = "default"
    run_subdir: str = ""
    layers_dir: Path | None = None
    models: tuple[str, str] = ("gem37", "terra")
    cfg: LabelConfig = field(default_factory=load_label)
    _units: dict[str, UnitInputs] = field(default_factory=dict, repr=False)
    _cands: dict[str, gpd.GeoDataFrame] = field(default_factory=dict, repr=False)

    @property
    def answers(self) -> Path:
        return self.root / "answers.jsonl"

    def unit(self, unit_id: str) -> UnitInputs:
        if unit_id not in self._units:
            run_dir = self.runs_dir / unit_id
            if self.run_subdir:
                run_dir = run_dir / self.run_subdir
            layers = (
                self.layers_dir / unit_id / self.run_tag if self.layers_dir else None
            )
            self._units[unit_id] = load_unit_inputs(
                run_dir, layers_root=layers, run_tag=self.run_tag
            )
        return self._units[unit_id]

    def candidates(self, unit_id: str) -> gpd.GeoDataFrame:
        """Candidates of a unit: the saved file, else computed and saved."""
        if unit_id not in self._cands:
            path = self.root / "candidates" / f"{unit_id}.gpkg"
            if path.exists():
                frame = gpd.read_file(path)
            else:
                unit = self.unit(unit_id)
                frame = C.unit_candidates(unit, self.cfg.candidates)
                # Without the cell layers the set is incomplete (detected
                # objects only), so it is not saved as the unit's candidates.
                if unit.has_layers:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_file(path, layer="candidates")
            self._cands[unit_id] = frame
        return self._cands[unit_id]

    def windows(self) -> pd.DataFrame:
        table = pd.read_csv(self.root / "windows.csv", keep_default_na=False)
        missing = [c for c in WINDOW_COLUMNS if c not in table]
        if missing:
            raise ValueError(f"windows.csv lacks the columns {missing}")
        return table

    def window(self, row: Any) -> Window:
        return Window(
            str(row.window_id),
            str(row.unit),
            int(row.epsg),
            int(row.ix0),
            int(row.iy0),
            int(row.nx),
            int(row.ny),
            self.cfg.chips.chip_px,
        )


@dataclass
class WindowParts:
    """A window with what the steps need of it."""

    unit: UnitInputs
    window: Window
    candidates: gpd.GeoDataFrame
    stack: WindowStack | None = None


def window_parts(
    run: LabelRun, row: Any, *, with_stack: bool = True
) -> WindowParts | None:
    """Unit, window, candidates and (optionally) imagery of one window.

    Candidates are those that reach into the window or its collar. Only
    those with enough area inside the window are marked ``judged``. The
    others still keep their ring of unlabelled ground.

    Returns:
        None when the imagery is wanted and lacks a year of the unit, since
        the cell layers and the verdict states are indexed by those years.
    """
    cfg = run.cfg
    unit = run.unit(str(row.unit))
    window = run.window(row)
    stack = None
    if with_stack:
        stack = load_stack(run.root / "stacks" / window.window_id)
        if stack.years != unit.years:
            logger.warning(
                "%s: stack years %s differ from the unit's %s, skipped",
                window.window_id,
                stack.years,
                unit.years,
            )
            return None
    pad = cfg.chips.collar_px * GSD
    x0, y0, x1, y1 = window.bounds
    every = run.candidates(unit.unit)
    near = C.window_candidates(
        every, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), cfg.candidates.min_ha
    )
    inside = C.window_candidates(every, window.bounds, cfg.candidates.min_ha)
    visible = inside.set_index("cand_id")["visible_ha"]
    near["visible_ha"] = near["cand_id"].map(visible).fillna(0.0)
    near["judged"] = (near["area_ha"] >= cfg.candidates.min_ha) & (
        near["visible_ha"] >= cfg.candidates.min_ha
    )
    near["clipped"] = near["visible_ha"] < 0.98 * near["area_ha"]
    return WindowParts(unit, window, near, stack)


def _save_requests(folder: Path, requests: list[Request], name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for q in requests:
        files = []
        for i, img in enumerate(q.images):
            file = f"{q.cand_id}_{q.kind[0]}{i}.jpg"
            (folder / file).write_bytes(img)
            files.append(file)
        lines.append(
            json.dumps(
                {
                    "cand_id": q.cand_id,
                    "kind": q.kind,
                    "prompt": q.prompt,
                    "images": files,
                    "dates": q.dates,
                    "view": [*q.view.bounds, q.view.px],
                    "extra": q.extra,
                }
            )
        )
    (folder / name).write_text("\n".join(lines) + ("\n" if lines else ""))


def load_requests(folder: Path, name: str = "requests.jsonl") -> list[Request]:
    """Requests written by :func:`render_window`, with their image bytes."""
    path = folder / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        *bounds, px = rec["view"]
        out.append(
            Request(
                rec["cand_id"],
                rec["kind"],
                rec["prompt"],
                [(folder / f).read_bytes() for f in rec["images"]],
                rec["dates"],
                View(tuple(bounds), int(px)),
                rec.get("extra", {}),
            )
        )
    return out


def render_window(
    run: LabelRun,
    row: Any,
    *,
    kind: str = "verdict",
    source: str | None = None,
    data_root: Path | None = None,
) -> int:
    """Fetch a window's imagery and write its requests. Returns their number.

    ``kind="verdict"`` writes one request per judged candidate
    (``requests.jsonl``). ``kind="polygon"`` reads the answers so far and
    writes a polygon request for every accepted candidate whose outline the
    models doubt (``polygons.jsonl``).
    """
    unit = run.unit(str(row.unit))
    window = run.window(row)
    meta = build_stack(
        unit,
        window,
        run.root / "stacks" / window.window_id,
        collar_px=run.cfg.chips.collar_px,
        source=source,
        data_root=data_root,
    )
    if meta["failed"]:
        logger.warning(
            "%s: years without imagery: %s", window.window_id, meta["failed"]
        )
    parts = window_parts(run, row)
    if parts is None or parts.stack is None:
        return 0
    stack_folder = run.root / "stacks" / window.window_id
    if not (stack_folder / CELL_NDVI_FILE).exists():
        write_cell_ndvi(parts.unit, parts.stack, stack_folder)
    folder = run.root / "requests" / window.window_id
    if kind == "verdict":
        requests = J.verdict_requests(
            parts.unit, parts.stack, parts.candidates, run.cfg
        )
        _save_requests(folder, requests, "requests.jsonl")
    elif kind == "polygon":
        verdicts = J.verdicts_of(J.read_records(run.answers), window.window_id)
        routed = J.route_candidates(
            parts.unit, parts.candidates, verdicts, run.models, run.cfg
        )
        requests = J.polygon_requests(
            parts.unit, parts.stack, parts.candidates, routed, run.cfg
        )
        _save_requests(folder, requests, "polygons.jsonl")
    else:
        raise ValueError(f"Unknown request kind {kind!r}")
    return len(requests)


@dataclass
class BuiltWindow:
    """Labels of a window with what led to them."""

    parts: WindowParts
    judged: list[JudgedCandidate]
    labels: WindowLabels
    verdicts: dict[str, dict]


def stack_complete(run: LabelRun, row: Any) -> bool:
    """True when the rendered stack of a window holds every year of its unit.

    A window whose stack lacks a year cannot be judged or labelled, because
    the cell layers and the verdict states are indexed by the unit's years.
    """
    meta = run.root / "stacks" / str(row.window_id) / "meta.json"
    if not meta.exists():
        return False
    years = sorted(int(y) for y in json.loads(meta.read_text())["years"])
    return years == list(run.unit(str(row.unit)).years)


def window_dates(run: LabelRun, window_id: str) -> list[str] | None:
    """Acquisition dates recorded when a window was rendered, or None."""
    meta = run.root / "stacks" / window_id / "meta.json"
    if not meta.exists():
        return None
    years = json.loads(meta.read_text())["years"]
    return [years[y]["date"] for y in sorted(years, key=int)]


def window_cell_ndvi(run: LabelRun, parts: WindowParts) -> CellNdvi | None:
    """Cell NDVI of a window: the stored file, else from the stack if loaded."""
    stored = load_cell_ndvi(run.root / "stacks" / parts.window.window_id)
    if stored is not None:
        return stored
    if parts.stack is not None:
        return unit_cell_ndvi(parts.unit.grid, parts.stack)
    return None


def build_window(
    run: LabelRun, row: Any, records: list[dict[str, Any]], *, with_stack: bool = True
) -> BuiltWindow | None:
    """Route the answers of a window and build its label rasters."""
    parts = window_parts(run, row, with_stack=with_stack)
    if parts is None:
        return None
    wid = parts.window.window_id
    verdicts = J.verdicts_of(records, wid)
    routed = J.route_candidates(
        parts.unit, parts.candidates, verdicts, run.models, run.cfg
    )
    judged = J.judged_candidates(
        parts.unit, parts.candidates, routed, records, wid, run.cfg
    )
    dates = None
    if parts.stack is None:
        dates = window_dates(run, wid) or [
            parts.unit.dates[y] for y in parts.unit.years
        ]
    cell_ndvi = None
    if run.cfg.rules.never_green_ndvi is not None:
        cell_ndvi = window_cell_ndvi(run, parts)
        if cell_ndvi is None:
            logger.warning(
                "%s: no cell NDVI, so the never-green rule is not applied", wid
            )
    labels = window_labels(
        parts.unit,
        parts.window,
        parts.stack,
        judged,
        run.cfg,
        dates=dates,
        cell_ndvi=cell_ndvi,
    )
    return BuiltWindow(parts, judged, labels, verdicts)
