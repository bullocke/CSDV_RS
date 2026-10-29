"""csdv_core.grid.units — processing units, their records and their folders.

A unit is the area one chain of jobs processes. It is either a grid cell
(:mod:`csdv_core.grid.cells`, ID ``q4407159``) or a box around a point (ID
``p_<label>``). The unit ID is used as the site name everywhere downstream:
``naip/<unit>/``, ``naip_chm/<unit>/`` and ``disturbance/<unit>/<run_tag>/``.

Each unit has a record, ``<grid dir>/units/<unit>/unit.json``, holding its box,
label, notes and the NAIP years planned for it with their source. Batch files
list units as CSV rows (see :func:`read_batch`).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csdv_core.grid.cells import point_bounds, quad_bounds, quad_id_at

logger = logging.getLogger(__name__)

KIND_CELL = "cell"
KIND_POINT = "point"

#: Columns of a batch file. Only ``lon``/``lat`` or ``cell`` are required.
BATCH_COLUMNS = ("label", "lon", "lat", "buffer_km", "cell", "notes")

_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")

__all__ = [
    "BATCH_COLUMNS",
    "GridPaths",
    "KIND_CELL",
    "KIND_POINT",
    "Unit",
    "grid_paths",
    "load_unit",
    "read_batch",
    "save_unit",
]


@dataclass(frozen=True)
class GridPaths:
    """Where grid records, run folders and exports live.

    Attributes:
        grid_dir: Unit records, the submission ledger and ``status.csv``.
            ``$CSDV_GRID_DIR``, default ``<results root>/grid``.
        export_dir: Small files for download. ``$CSDV_EXPORT_DIR``, default
            ``<results root>/exports``.
        data_root: ``$CSDV_DATA_ROOT``.
        results_root: ``$CSDV_RESULTS_ROOT``.
    """

    grid_dir: Path
    export_dir: Path
    data_root: Path
    results_root: Path

    @property
    def ledger(self) -> Path:
        return self.grid_dir / "submissions.csv"

    @property
    def status_csv(self) -> Path:
        return self.grid_dir / "status.csv"

    def unit_dir(self, unit_id: str) -> Path:
        return self.grid_dir / "units" / unit_id

    def unit_json(self, unit_id: str) -> Path:
        return self.unit_dir(unit_id) / "unit.json"

    def naip_year_dir(self, unit_id: str, year: int) -> Path:
        return self.data_root / "naip" / unit_id / str(year)

    def chm_year_dir(self, unit_id: str, year: int) -> Path:
        return self.data_root / "naip_chm" / unit_id / str(year)

    def run_dir(self, unit_id: str, run_tag: str) -> Path:
        return self.results_root / "disturbance" / unit_id / run_tag

    def export_run_dir(self, unit_id: str, run_tag: str) -> Path:
        return self.export_dir / "runs" / run_tag / unit_id

    def quicklook_dir(self, unit_id: str) -> Path:
        return self.export_dir / "quicklooks" / unit_id


def grid_paths() -> GridPaths:
    """Resolve :class:`GridPaths` from the environment."""
    from csdv_core.io.paths import project_paths

    p = project_paths()
    grid_dir = os.environ.get("CSDV_GRID_DIR")
    export_dir = os.environ.get("CSDV_EXPORT_DIR")
    return GridPaths(
        grid_dir=Path(grid_dir) if grid_dir else p.results_root / "grid",
        export_dir=Path(export_dir) if export_dir else p.results_root / "exports",
        data_root=p.data_root,
        results_root=p.results_root,
    )


@dataclass
class Unit:
    """One processing unit.

    Attributes:
        unit_id: ``q<quad code>`` for a grid cell, ``p_<label>`` for a point.
        kind: ``"cell"`` or ``"point"``.
        bbox_4326: ``(west, south, east, north)`` in degrees.
        label: Short human name (a place, a disturbance type).
        center_lonlat: The point a point unit was built from.
        buffer_km: Half-width of a point unit.
        notes: Free text from the batch file.
        years: Planned NAIP years, ``{year: {"source": ..., "coverage": ...}}``.
        skipped_years: Years left out at planning, with the reason.
    """

    unit_id: str
    kind: str
    bbox_4326: tuple[float, float, float, float]
    label: str = ""
    center_lonlat: tuple[float, float] | None = None
    buffer_km: float | None = None
    notes: str = ""
    years: dict[int, dict[str, Any]] = field(default_factory=dict)
    skipped_years: dict[int, str] = field(default_factory=dict)

    @classmethod
    def cell(cls, quad_id: str, *, label: str = "", notes: str = "") -> Unit:
        """The grid cell ``quad_id`` (with or without the ``q``)."""
        unit_id = quad_id if quad_id.startswith("q") else f"q{quad_id}"
        return cls(unit_id, KIND_CELL, quad_bounds(unit_id), label=label, notes=notes)

    @classmethod
    def cell_at(
        cls, lon: float, lat: float, *, label: str = "", notes: str = ""
    ) -> Unit:
        """The grid cell that contains ``(lon, lat)``."""
        unit = cls.cell(quad_id_at(lon, lat), label=label, notes=notes)
        unit.center_lonlat = (lon, lat)
        return unit

    @classmethod
    def point(
        cls, lon: float, lat: float, buffer_km: float, *, label: str, notes: str = ""
    ) -> Unit:
        """A box of half-width ``buffer_km`` around ``(lon, lat)``."""
        if not _LABEL_RE.match(label or ""):
            raise ValueError(
                f"A point unit needs a label of letters, digits, _ or -: {label!r}"
            )
        return cls(
            f"p_{label}",
            KIND_POINT,
            point_bounds(lon, lat, buffer_km),
            label=label,
            center_lonlat=(lon, lat),
            buffer_km=buffer_km,
            notes=notes,
        )

    @property
    def year_list(self) -> list[int]:
        return sorted(self.years)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["years"] = {str(y): v for y, v in sorted(self.years.items())}
        out["skipped_years"] = {
            str(y): v for y, v in sorted(self.skipped_years.items())
        }
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Unit:
        center = data.get("center_lonlat")
        return cls(
            unit_id=data["unit_id"],
            kind=data["kind"],
            bbox_4326=tuple(data["bbox_4326"]),  # type: ignore[arg-type]
            label=data.get("label", ""),
            center_lonlat=tuple(center) if center else None,  # type: ignore[arg-type]
            buffer_km=data.get("buffer_km"),
            notes=data.get("notes", ""),
            years={int(y): v for y, v in data.get("years", {}).items()},
            skipped_years={int(y): v for y, v in data.get("skipped_years", {}).items()},
        )


def load_unit(paths: GridPaths, unit_id: str) -> Unit | None:
    """Read a unit record, or None if the unit has none yet."""
    path = paths.unit_json(unit_id)
    if not path.exists():
        return None
    return Unit.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_unit(paths: GridPaths, unit: Unit) -> Path:
    """Write a unit record, refusing to change the box of an existing unit.

    A unit's box fixes its NAIP grid, so data already downloaded for one box
    must not be reused for another under the same ID.
    """
    path = paths.unit_json(unit.unit_id)
    record = unit.to_json()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if [round(v, 6) for v in old["bbox_4326"]] != [
            round(v, 6) for v in unit.bbox_4326
        ]:
            raise ValueError(
                f"{unit.unit_id} already exists with box {old['bbox_4326']}, "
                f"not {list(unit.bbox_4326)}. Use another label."
            )
        record["created"] = old.get("created", now)
    else:
        record["created"] = now
    record["updated"] = now
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _float(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def read_batch(path: Path | str) -> list[Unit]:
    """Read the units of a batch file.

    The file is CSV with the columns in :data:`BATCH_COLUMNS`; lines starting
    with ``#`` are comments. Each row becomes:

    - the cell named in ``cell``, when given;
    - a point unit ``p_<label>`` when ``buffer_km`` is given;
    - otherwise the cell that contains ``lon``, ``lat``.

    Raises:
        ValueError: On a row that names no area, or on two rows with one unit.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [
            line for line in fh if line.strip() and not line.lstrip().startswith("#")
        ]
    units: list[Unit] = []
    seen: dict[str, int] = {}
    for n, row in enumerate(csv.DictReader(lines), start=1):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        label, notes, cell = (
            row.get("label", ""),
            row.get("notes", ""),
            row.get("cell", ""),
        )
        lon, lat = _float(row.get("lon")), _float(row.get("lat"))
        buffer_km = _float(row.get("buffer_km"))
        if cell:
            unit = Unit.cell(cell, label=label, notes=notes)
        elif lon is None or lat is None:
            raise ValueError(f"{path}, row {n}: give a cell or lon and lat")
        elif buffer_km:
            unit = Unit.point(lon, lat, buffer_km, label=label, notes=notes)
        else:
            unit = Unit.cell_at(lon, lat, label=label, notes=notes)
        if unit.unit_id in seen:
            raise ValueError(
                f"{path}: rows {seen[unit.unit_id]} and {n} are both {unit.unit_id}"
            )
        seen[unit.unit_id] = n
        units.append(unit)
    return units
