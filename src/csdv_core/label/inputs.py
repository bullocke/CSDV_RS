"""csdv_core.label.inputs: what the labelling pipeline reads about one unit.

Inputs: a disturbance run folder. On CHPC that is the run itself
(``<results>/disturbance/<unit>/<tag>/``). On another machine it is the
exported copy (``<export_dir>/runs/<tag>/<unit>/``), with the cell layers
pulled by ``scripts/chpc/pull_label_layers.sh`` into
``<export_dir>/run_layers/<unit>/<tag>/``.
Outputs: :class:`UnitInputs`, with the cell grid, the years and dates, the
candidate GeoPackage and the cell layers (probability, forest, cover, height).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from csdv_core.disturbance.inputs import CellGrid, grid_from_dict
from csdv_core.disturbance.store import read_year_stack

logger = logging.getLogger(__name__)

#: Cell layers and their place inside a run folder.
LAYER_FILES: dict[str, str] = {
    "probability": "detection/probability.tif",
    "forest": "layers/forest.tif",
    "chm_cover": "layers/chm_cover.tif",
    "chm_height": "layers/chm_height.tif",
}

__all__ = [
    "LAYER_FILES",
    "UnitInputs",
    "fill_forest_dips",
    "list_units",
    "load_unit_inputs",
    "unstable_intervals",
]


@dataclass
class UnitInputs:
    """One unit's run, as far as labelling needs it.

    Attributes:
        unit: Unit ID (``q4407152``).
        run_tag: Run tag of the disturbance run.
        label: Short site label (``CO_EastTroublesome``), may be empty.
        grid: The 0.6 m pixel grid and the 19.2 m cell grid of the run.
        years: NAIP years of the run, ascending.
        dates: Acquisition date of each year's mosaic (``YYYY-MM-DD``).
        sources: Where each year's NAIP comes from (``pc``, ``rest`` or
            ``local``).
        objects_path: ``disturbances.gpkg`` of the run.
        layers_root: Folder holding ``detection/`` and ``layers/``, or None
            when the cell layers are not on this machine.
    """

    unit: str
    run_tag: str
    label: str
    grid: CellGrid
    years: list[int]
    dates: dict[int, str]
    sources: dict[int, str]
    objects_path: Path
    layers_root: Path | None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def epsg(self) -> int:
        return int(self.grid.crs.to_epsg())

    @property
    def intervals(self) -> list[str]:
        """Interval labels ``"2019-2021"``, one per pair of consecutive years."""
        return [
            f"{a}-{b}" for a, b in zip(self.years[:-1], self.years[1:], strict=True)
        ]

    @property
    def has_layers(self) -> bool:
        return self.layers_root is not None and all(
            (self.layers_root / rel).is_file() for rel in LAYER_FILES.values()
        )

    def layer(self, name: str) -> np.ndarray:
        """A cell layer as ``(years, rows, cols)``.

        ``forest`` is bool. The others are float32 with NaN where a cell has
        no value. Band *t* of ``probability`` belongs to the interval ending
        in ``years[t]`` (band 0 is empty).

        Raises:
            FileNotFoundError: If the layer is not on this machine.
            ValueError: If its years or shape differ from the run's.
        """
        if name in self._cache:
            return self._cache[name]
        if name not in LAYER_FILES:
            raise KeyError(f"Unknown cell layer {name!r}")
        if self.layers_root is None:
            raise FileNotFoundError(
                f"{self.unit}: cell layers are not on this machine. Run "
                "scripts/chpc/pull_label_layers.sh."
            )
        stack = read_year_stack(self.layers_root / LAYER_FILES[name])
        if list(stack.years) != self.years:
            raise ValueError(
                f"{self.unit} {name}: years {list(stack.years)} differ from the "
                f"run's {self.years}"
            )
        if stack.values.shape[1:] != self.grid.cell_shape:
            raise ValueError(
                f"{self.unit} {name}: shape {stack.values.shape[1:]} differs from "
                f"the grid's {self.grid.cell_shape}"
            )
        self._cache[name] = stack.values
        return stack.values


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _find_layers_root(run_dir: Path, unit: str, run_tag: str) -> Path | None:
    if (run_dir / LAYER_FILES["probability"]).is_file():
        return run_dir
    # An exported run sits at <export_dir>/runs/<tag>/<unit>.
    pulled = run_dir.parents[2] / "run_layers" / unit / run_tag
    return pulled if pulled.is_dir() else None


def load_unit_inputs(
    run_dir: Path, *, layers_root: Path | None = None, run_tag: str | None = None
) -> UnitInputs:
    """Read a run folder (the run itself or its exported copy).

    Args:
        run_dir: Folder with ``run_params.json`` and the candidate GeoPackage.
        layers_root: Folder holding ``detection/`` and ``layers/``. Found
            automatically when omitted.
        run_tag: Run tag. Default: the parent folder's name for an exported
            run, else the folder's own name.

    Raises:
        FileNotFoundError: If ``run_params.json`` or the GeoPackage is missing.
    """
    run_dir = Path(run_dir)
    params_path = run_dir / "run_params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"Not a disturbance run folder: {run_dir}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    unit = str(params["site"])
    exported = (run_dir / "disturbances.gpkg").is_file()
    gpkg = run_dir / (
        "disturbances.gpkg" if exported else "detection/disturbances.gpkg"
    )
    if not gpkg.is_file():
        raise FileNotFoundError(f"No disturbances.gpkg in {run_dir}")
    if run_tag is None:
        run_tag = run_dir.parent.name if exported else run_dir.name
    grid = grid_from_dict(params["grid"])
    summary = _read_json(run_dir / "summary.json")
    unit_rec = _read_json(run_dir / "unit.json")
    years = [int(y) for y in summary.get("years") or []] or [
        int(y) for y in params.get("params", {}).get("inputs", {}).get("years") or []
    ]
    if not years:
        raise ValueError(f"{unit}: the run records no years")
    dates = {
        int(row["year"]): str(row["acquisition_date"])
        for row in summary.get("inputs", [])
        if row.get("acquisition_date")
    }
    sources = {
        int(y): str(v.get("source", "local"))
        for y, v in unit_rec.get("years", {}).items()
    }
    return UnitInputs(
        unit=unit,
        run_tag=run_tag,
        label=str(unit_rec.get("label") or ""),
        grid=grid,
        years=years,
        dates={y: dates.get(y, f"{y}-07-01") for y in years},
        sources={y: sources.get(y, "local") for y in years},
        objects_path=gpkg,
        layers_root=layers_root or _find_layers_root(run_dir, unit, run_tag),
    )


def list_units(export_dir: Path, run_tag: str = "default") -> list[UnitInputs]:
    """Every exported unit of ``run_tag``, sorted by unit ID."""
    tag_dir = Path(export_dir) / "runs" / run_tag
    units = []
    for run_dir in sorted(p for p in tag_dir.glob("*") if p.is_dir()):
        if not (run_dir / "run_params.json").is_file():
            logger.warning("%s: no run_params.json, skipped", run_dir)
            continue
        units.append(load_unit_inputs(run_dir, run_tag=run_tag))
    return units


def fill_forest_dips(forest: np.ndarray) -> np.ndarray:
    """Forest mask with single-date dips filled.

    The CHM of one date can sit low over a whole unit, which drops cells below
    the forest rule for that date only. A cell that is forest on the dates
    before and after is treated as forest in between.

    Args:
        forest: Bool ``(years, rows, cols)``.
    """
    out = forest.copy()
    if len(forest) >= 3:
        out[1:-1] |= forest[:-2] & forest[2:]
    return out


def unstable_intervals(forest: np.ndarray, threshold: float) -> list[bool]:
    """Whether the unit's forest area moves too much in each interval.

    Args:
        forest: Bool ``(years, rows, cols)``, dips already filled.
        threshold: Largest relative change of forest area that still counts
            as a reliable mask.

    Returns:
        One flag per interval (``len(forest) - 1``).
    """
    area = forest.reshape(len(forest), -1).sum(axis=1).astype(float)
    flags = []
    for before, after in zip(area[:-1], area[1:], strict=True):
        change = abs(after - before) / before if before > 0 else 1.0
        flags.append(bool(change > threshold))
    return flags
