"""csdv_core.grid.check — technical checks of an exported grid run.

The export of a run (``runs/<run_tag>/<unit>/`` and ``quicklooks/<unit>/``
under the export folder, see :mod:`csdv_core.grid.export`) holds everything
needed to tell whether the chain worked: the unit record, the run's summary,
parameters and log, the byte rasters, the objects and the inspection sample.
:func:`check_run` compares these with each other and returns one
:class:`CheckResult` per check, each ``pass``, ``warn`` or ``fail``.

A ``fail`` means the outputs are missing or disagree with each other, so the
run should not be used. A ``warn`` points at something to look at that is not
an error by itself (a season gap between images, a CHM year that is shorter
than the others, invalid object polygons).

Only the export is read, so the checks run on a laptop after
``scripts/chpc/pull_exports.sh`` as well as on CHPC.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.grid.export import (
    CHM_STEP_M,
    NODATA_U8,
    ONSET_YEAR_BASE,
    RUN_DIRS,
    RUN_FILES,
)

logger = logging.getLogger(__name__)

PASS, WARN, FAIL = "pass", "warn", "fail"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

#: Byte rasters written by the export next to the copied run files.
EXPORT_RASTERS = ("onset_year_u8.tif", "max_probability_u8.tif")

#: Layers of ``disturbances.gpkg``.
GPKG_LAYERS = ("objects", "refined_outlines", "refined_openings")

#: Block size of the exported COGs. GDAL adds no overviews to a raster that
#: fits in one block.
COG_BLOCK_PX = 256

#: Season gap (days of year) above which an image pair gets a warning.
SEASON_GAP_DAYS = 42

#: Departure of a year's median quicklook CHM height from the median of all
#: years (m) above which the year gets a warning.
CHM_MEDIAN_TOL_M = 0.5

#: Accepted range of the null spread sigma0 of raw z.
SIGMA0_RANGE = (0.8, 1.2)

__all__ = [
    "FAIL",
    "PASS",
    "WARN",
    "CheckResult",
    "RunCheck",
    "check_run",
    "find_runs",
    "write_check",
]


@dataclass
class CheckResult:
    """Outcome of one check."""

    name: str
    status: str
    detail: str


@dataclass
class RunCheck:
    """All checks of one exported run."""

    unit: str
    run_tag: str
    run_dir: Path
    results: list[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.results:
            return FAIL
        return max((r.status for r in self.results), key=_RANK.__getitem__)

    def add(self, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(name, status, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "run_tag": self.run_tag,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Export check: {self.unit} ({self.run_tag})",
            "",
            f"Overall: **{self.status}**.",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for r in self.results:
            detail = r.detail.replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {r.name} | {r.status} | {detail} |")
        return "\n".join(lines) + "\n"


def find_runs(
    export_dir: Path, unit: str | None = None, run_tag: str | None = None
) -> list[tuple[str, str]]:
    """``(unit, run_tag)`` of every exported run, optionally filtered."""
    runs = []
    for tag_dir in sorted((export_dir / "runs").glob("*")):
        if not tag_dir.is_dir() or (run_tag and tag_dir.name != run_tag):
            continue
        for unit_dir in sorted(tag_dir.iterdir()):
            if unit_dir.is_dir() and (unit is None or unit_dir.name == unit):
                runs.append((unit_dir.name, tag_dir.name))
    return runs


def write_check(check: RunCheck) -> list[Path]:
    """Write ``check.md`` and ``check.json`` into the run's export folder."""
    md = check.run_dir / "check.md"
    js = check.run_dir / "check.json"
    md.write_text(check.to_markdown(), encoding="utf-8")
    js.write_text(json.dumps(check.to_dict(), indent=2), encoding="utf-8")
    return [md, js]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _years(values: Any) -> list[int]:
    return sorted(int(y) for y in values or [])


def check_run(export_dir: Path, unit: str, run_tag: str) -> RunCheck:
    """Run every check on one exported run."""
    run_dir = export_dir / "runs" / run_tag / unit
    looks = export_dir / "quicklooks" / unit
    check = RunCheck(unit, run_tag, run_dir)

    summary = _read_json(run_dir / "summary.json")
    if summary is None:
        check.add("completeness", FAIL, "summary.json missing or unreadable")
        return check
    years = _years(summary.get("years"))

    _check_completeness(check, run_dir, looks, years)
    _check_status(check, export_dir, unit, run_tag)
    _check_metadata(check, run_dir, summary, looks, years)
    _check_inputs(check, summary)
    _check_rasters(check, run_dir, summary)
    _check_vectors(check, run_dir, summary)
    _check_forest(check, summary, looks, years)
    _check_null_fits(check, summary)
    _check_log(check, run_dir, summary)
    _check_inspection(check, run_dir)
    return check


def _check_completeness(
    check: RunCheck, run_dir: Path, looks: Path, years: list[int]
) -> None:
    expected = [
        *RUN_FILES.values(),
        *EXPORT_RASTERS,
        "unit.json",
        "export.json",
    ]
    missing = [n for n in expected if not (run_dir / n).is_file()]
    missing += [f"{d}/" for d in RUN_DIRS if not (run_dir / d).is_dir()]
    for year in years:
        for name in (f"chm_{year}_3m.tif", f"naip_{year}_3m_rgb.tif"):
            if not (looks / name).is_file():
                missing.append(f"quicklooks/{name}")
    if missing:
        check.add("completeness", FAIL, "missing: " + ", ".join(missing))
    else:
        check.add(
            "completeness",
            PASS,
            f"{len(expected)} run files, {len(RUN_DIRS)} folders and "
            f"{2 * len(years)} quicklooks present",
        )


def _check_status(check: RunCheck, export_dir: Path, unit: str, tag: str) -> None:
    path = export_dir / "status.csv"
    if not path.is_file():
        check.add("status", WARN, "status.csv not in the export folder")
        return
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["unit"] == unit]
    row = next((r for r in rows if r.get("run_tag") == tag), None)
    if row is None:
        check.add("status", WARN, f"no status.csv row for {unit} {tag}")
    elif row.get("state") != "done":
        check.add("status", FAIL, f"state {row.get('state')!r}")
    else:
        steps = ", ".join(
            f"{k} {row[k]}" for k in ("naip", "chm", "detect", "chips", "export")
        )
        check.add("status", PASS, f"done ({steps})")


def _check_metadata(
    check: RunCheck,
    run_dir: Path,
    summary: dict[str, Any],
    looks: Path,
    years: list[int],
) -> None:
    problems: list[str] = []
    notes: list[str] = []
    unit_rec = _read_json(run_dir / "unit.json") or {}
    params = _read_json(run_dir / "run_params.json") or {}
    export = _read_json(run_dir / "export.json") or {}

    sources = {
        "unit.json": _years(unit_rec.get("years", {}).keys()),
        "run_params": _years(params.get("params", {}).get("inputs", {}).get("years")),
        "summary": years,
        "chm quicklooks": _years(
            re.match(r"chm_(\d{4})_", p.name).group(1) for p in looks.glob("chm_*.tif")
        ),
    }
    for name, got in sources.items():
        if got != years:
            problems.append(f"{name} years {got} differ from summary {years}")
    if export.get("unit") != check.unit or export.get("run_tag") != check.run_tag:
        problems.append(
            f"export.json names {export.get('unit')} {export.get('run_tag')}"
        )
    if unit_rec.get("unit_id") not in (None, check.unit):
        problems.append(f"unit.json is for {unit_rec.get('unit_id')}")

    software = summary.get("software", {})
    if str(software.get("git_dirty", "")).lower() != "false":
        problems.append(f"git_dirty={software.get('git_dirty')}")
    notes.append(f"commit {software.get('git_commit', '?')}")
    srcs = {y: v.get("source") for y, v in unit_rec.get("years", {}).items()}
    if srcs:
        notes.append(
            "sources " + ", ".join(f"{y} {s}" for y, s in sorted(srcs.items()))
        )
    skipped = unit_rec.get("skipped_years") or {}
    if skipped:
        notes.append("skipped " + "; ".join(f"{y}: {r}" for y, r in skipped.items()))
    if not unit_rec.get("label"):
        notes.append("unit has no label")

    if problems:
        check.add("metadata", FAIL, "; ".join(problems + notes))
    elif not unit_rec.get("label") or skipped:
        check.add("metadata", WARN, "; ".join(notes))
    else:
        check.add("metadata", PASS, "; ".join(notes))


def _resolution_tag(path: str) -> str:
    m = re.search(r"_(\d+p?\d*m)_chm", Path(path).name)
    return m.group(1) if m else "?"


def _check_inputs(check: RunCheck, summary: dict[str, Any]) -> None:
    inputs = summary.get("inputs", [])
    per_year: dict[int, list[dict[str, Any]]] = {}
    for item in inputs:
        per_year.setdefault(int(item["year"]), []).append(item)
    detail = ", ".join(
        f"{y} {'+'.join(_resolution_tag(i['path']) for i in items)}"
        for y, items in sorted(per_year.items())
    )
    multi = [y for y, items in per_year.items() if len(items) > 1]
    check.add(
        "inputs",
        WARN if multi else PASS,
        f"source resolution {detail}"
        + (f"; several CHM files (mosaicked) in {multi}" if multi else ""),
    )

    dates = {}
    for y, items in per_year.items():
        acquired = items[0].get("acquisition_date")
        if acquired:
            dates[y] = date.fromisoformat(acquired)
    gaps = []
    ordered = sorted(dates)
    for before, after in pairwise(ordered):
        days = abs(dates[after].timetuple().tm_yday - dates[before].timetuple().tm_yday)
        if days > SEASON_GAP_DAYS:
            gaps.append(f"{before}-{after} {days} d")
    if gaps:
        check.add("season gaps", WARN, ", ".join(gaps))
    else:
        check.add("season gaps", PASS, f"all pairs within {SEASON_GAP_DAYS} days")


def _check_rasters(check: RunCheck, run_dir: Path, summary: dict[str, Any]) -> None:
    import rasterio
    from rasterio.crs import CRS

    grid = summary.get("grid", {})
    cell = float(grid.get("cell_m", 0))
    left, _, _, top = grid.get("bounds", [np.nan] * 4)
    shape = (grid.get("cell_rows"), grid.get("cell_cols"))
    problems: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for name in EXPORT_RASTERS:
        path = run_dir / name
        if not path.is_file():
            continue
        with rasterio.open(path) as src:
            t = src.transform
            if src.crs != CRS.from_user_input(grid.get("crs")):
                problems.append(f"{name} CRS {src.crs}")
            if src.shape != shape:
                problems.append(f"{name} shape {src.shape}, summary {shape}")
            if not np.allclose([t.a, -t.e, t.c, t.f], [cell, cell, left, top]):
                problems.append(f"{name} transform {tuple(t)[:6]}")
            if src.nodata != NODATA_U8:
                problems.append(f"{name} nodata {src.nodata}")
            if src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
                problems.append(f"{name} is not a COG")
            if max(src.shape) > COG_BLOCK_PX and not src.overviews(1):
                problems.append(f"{name} has no overviews")
            arrays[name] = src.read(1)
    if problems:
        check.add("raster grid", FAIL, "; ".join(problems))
    else:
        check.add(
            "raster grid",
            PASS,
            f"{len(arrays)} COGs on {grid.get('crs')}, {shape[0]} x {shape[1]} "
            f"cells of {cell:g} m",
        )

    onset = arrays.get("onset_year_u8.tif")
    if onset is None:
        return
    intervals = summary.get("intervals", [])
    allowed = {0, NODATA_U8} | {int(i["year"]) - ONSET_YEAR_BASE for i in intervals}
    codes, counts = np.unique(onset, return_counts=True)
    stray = sorted(set(codes.tolist()) - allowed)
    cell_ha = cell * cell / 1e4
    by_code = dict(zip(codes.tolist(), counts.tolist(), strict=True))
    bad = []
    for item in intervals:
        if item.get("skipped"):
            continue
        got = by_code.get(int(item["year"]) - ONSET_YEAR_BASE, 0) * cell_ha
        want = item["detected_ha"] - item.get("repeat_ha", 0.0)
        if abs(got - want) > cell_ha + 1e-6:
            bad.append(f"{item['label']} map {got:.2f} ha, summary {want:.2f} ha")
    ever = sum(n for c, n in by_code.items() if c not in (0, NODATA_U8)) * cell_ha
    if abs(ever - summary.get("ever_detected_ha", ever)) > cell_ha:
        bad.append(
            f"ever detected map {ever:.2f}, summary {summary['ever_detected_ha']:.2f}"
        )
    if stray:
        bad.append(f"unexpected onset codes {stray}")
    prob = arrays.get("max_probability_u8.tif")
    if prob is not None:
        differ = int(((prob == NODATA_U8) != (onset == NODATA_U8)).sum())
        if differ:
            bad.append(f"{differ} cells differ in nodata between onset and probability")
    if bad:
        check.add("onset raster", FAIL, "; ".join(bad))
    else:
        check.add(
            "onset raster",
            PASS,
            f"interval areas match the summary; {ever:.2f} ha ever detected; "
            f"{by_code.get(NODATA_U8, 0)} cells never monitored",
        )


def _check_vectors(check: RunCheck, run_dir: Path, summary: dict[str, Any]) -> None:
    import geopandas as gpd
    import pandas as pd

    gpkg = run_dir / "disturbances.gpkg"
    csv_path = run_dir / "objects.csv"
    if not gpkg.is_file() or not csv_path.is_file():
        return
    layers = set(gpd.list_layers(gpkg)["name"])
    problems: list[str] = []
    missing = [n for n in GPKG_LAYERS if n not in layers]
    if missing:
        problems.append(f"gpkg layers missing: {missing}")
    table = pd.read_csv(csv_path)
    n_objects = int(summary.get("n_objects", -1))
    invalid: dict[str, int] = {}
    frames = {n: gpd.read_file(gpkg, layer=n) for n in GPKG_LAYERS if n in layers}
    for name, frame in frames.items():
        n_bad = int((~frame.geometry.is_valid).sum())
        if n_bad:
            invalid[name] = n_bad
    objects = frames.get("objects")
    if objects is not None:
        if len(objects) != n_objects or len(table) != n_objects:
            problems.append(
                f"objects: gpkg {len(objects)}, csv {len(table)}, summary {n_objects}"
            )
        elif set(objects["object_id"]) != set(table["object_id"]):
            problems.append("object IDs differ between gpkg and csv")
        if objects.crs is None or str(objects.crs.to_epsg()) not in str(
            summary.get("grid", {}).get("crs")
        ):
            problems.append(f"objects CRS {objects.crs}")
    refined = frames.get("refined_outlines")
    n_refined = int(summary.get("n_refined", -1))
    if refined is not None and len(refined) != n_refined:
        problems.append(f"refined outlines {len(refined)}, summary {n_refined}")
    sums = table.groupby("interval")["area_ha"].sum()
    for item in summary.get("intervals", []):
        got = float(sums.get(item["label"], 0.0))
        if abs(got - item["detected_ha"]) > 1e-3:
            problems.append(
                f"{item['label']} objects {got:.2f} ha, summary {item['detected_ha']:.2f}"
            )
    if problems:
        check.add("vectors", FAIL, "; ".join(problems))
    else:
        check.add(
            "vectors",
            PASS,
            f"{n_objects} objects and {n_refined} refined outlines agree with "
            "objects.csv and the summary",
        )
    if invalid:
        check.add(
            "geometry validity",
            WARN,
            "invalid polygons: " + ", ".join(f"{k} {v}" for k, v in invalid.items()),
        )
    else:
        check.add("geometry validity", PASS, "all polygons valid")


def _check_forest(
    check: RunCheck, summary: dict[str, Any], looks: Path, years: list[int]
) -> None:
    import rasterio

    forest = {int(y): ha for y, ha in summary.get("forest_ha_by_year", {}).items()}
    detected = float(summary.get("detected_ha", 0.0))
    ordered = sorted(forest)
    swings = [(a, b, forest[b] - forest[a]) for a, b in pairwise(ordered)]
    largest = max(swings, key=lambda s: abs(s[2]), default=None)
    area = float(summary.get("grid", {}).get("area_ha", 0.0)) or 1.0
    shares = ", ".join(f"{y} {100 * forest[y] / area:.1f}%" for y in ordered)
    if largest and abs(largest[2]) > detected:
        check.add(
            "forest stability",
            WARN,
            f"forest share {shares}; largest swing {largest[0]}-{largest[1]} "
            f"{largest[2]:+.0f} ha exceeds all detections ({detected:.0f} ha)",
        )
    else:
        check.add("forest stability", PASS, f"forest share {shares}")

    medians = {}
    for year in years:
        path = looks / f"chm_{year}_3m.tif"
        if not path.is_file():
            continue
        with rasterio.open(path) as src:
            codes = src.read(1)
        valid = codes[codes != NODATA_U8]
        if valid.size:
            medians[year] = float(np.median(valid)) * CHM_STEP_M
    if not medians:
        return
    centre = float(np.median(list(medians.values())))
    off = {
        y: m - centre for y, m in medians.items() if abs(m - centre) > CHM_MEDIAN_TOL_M
    }
    text = ", ".join(f"{y} {m:.2f}" for y, m in sorted(medians.items()))
    if off:
        check.add(
            "CHM level",
            WARN,
            f"median height (m) {text}; off by more than {CHM_MEDIAN_TOL_M} m: "
            + ", ".join(f"{y} {d:+.2f}" for y, d in sorted(off.items())),
        )
    else:
        check.add("CHM level", PASS, f"median height (m) {text}")


def _check_null_fits(check: RunCheck, summary: dict[str, Any]) -> None:
    lo, hi = SIGMA0_RANGE
    problems, notes = [], []
    for item in summary.get("intervals", []):
        fit = item.get("fit") or {}
        if item.get("skipped") or fit.get("skipped"):
            problems.append(f"{item['label']} skipped")
            continue
        sigma0, p0 = fit.get("sigma0", np.nan), fit.get("p0", np.nan)
        if not lo <= sigma0 <= hi:
            problems.append(f"{item['label']} sigma0 {sigma0:.2f}")
        if p0 < 1.0:
            notes.append(f"{item['label']} p0 {p0:.3f}")
    status = WARN if problems else PASS
    detail = "; ".join(problems) or f"every sigma0 within {lo}-{hi}"
    if notes:
        detail += "; " + ", ".join(notes)
    check.add("null fits", status, detail)


def _check_log(check: RunCheck, run_dir: Path, summary: dict[str, Any]) -> None:
    path = run_dir / "detect_disturbance.log"
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    flagged = [ln for ln in lines if re.search(r"\b(WARNING|ERROR|CRITICAL)\b", ln)]
    finished = any("Run finished in" in ln for ln in lines)
    stages = summary.get("stage_seconds", {})
    total = sum(stages.values())
    slowest = max(stages, key=stages.get) if stages else "?"
    detail = f"{total:.0f} s in total, slowest stage {slowest}"
    if not finished:
        check.add("log", FAIL, "log does not end with 'Run finished'; " + detail)
    elif flagged:
        check.add("log", WARN, f"{len(flagged)} warning lines, first: {flagged[0]}")
    else:
        check.add("log", PASS, "no warnings; " + detail)


def _check_inspection(check: RunCheck, run_dir: Path) -> None:
    folder = run_dir / "inspection"
    samples = folder / "samples.csv"
    if not samples.is_file():
        check.add("inspection", WARN, "no inspection sample (chips step skipped?)")
        return
    with samples.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["chip"] for r in rows if not (folder / r["chip"]).is_file()]
    if not (folder / "contact_sheet.png").is_file():
        missing.append("contact_sheet.png")
    labelled = sum(1 for r in rows if (r.get("interp_label") or "").strip())
    if missing:
        check.add("inspection", FAIL, "missing: " + ", ".join(missing))
    else:
        check.add(
            "inspection",
            PASS,
            f"{len(rows)} chips and the contact sheet present; {labelled} labelled",
        )
