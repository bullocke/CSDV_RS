"""csdv_core.view.catalog: the runs and sites the viewer can show.

A run is one folder of ``<export_dir>/pairmodel`` with a ``map`` subfolder,
and it maps one unit (site). The Disturbance Index run of a unit
(``<export_dir>/runs/default/<unit>``) is listed as one more source of that
site, so it can sit beside a pair-model map.

The catalogue is rebuilt on every page load, so a run pulled from CHPC shows
up without a restart.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAIR_DIR = "pairmodel"
DI_TAG = "default"
DI_PREFIX = "di_"
#: What each run family is, for the model picker.
FAMILY_LABELS: dict[str, str] = {
    "look": "Non-holdout model",
    "holdout": "Holdout",
    "holdout_sf0": "Holdout, stable-forest control (0%)",
    "holdout_sf": "Holdout with stable forest (15%)",
    "holdout_r1": "Holdout, round-2 control (round 1 alone)",
    "holdout_r2": "Holdout with round 2",
    "holdout_r2sf": "Holdout with round 2 and stable forest (15%)",
    "di": "DI baseline",
}
_FAMILY_ORDER = [
    "look",
    "holdout",
    "holdout_sf0",
    "holdout_sf",
    "holdout_r1",
    "holdout_r2",
    "holdout_r2sf",
]
_INTERVAL = re.compile(r"^(\d{4})-\d{2}-\d{2}_(\d{4})-\d{2}-\d{2}$")
_THRESHOLD = re.compile(r"at or above (\d(?:\.\d+)?)")
_SCORE_KEYS = ("average_precision", "precision", "recall", "f1")

__all__ = [
    "Catalog",
    "RunInfo",
    "SiteInfo",
    "build_catalog",
    "interval_years",
    "run_family",
]


def run_family(run: str, unit: str) -> str:
    """Family of a run name: ``holdout_sf_q3708104`` is ``holdout_sf``.

    A run that is not named after its unit (``first_look_blocks2``) is its own
    family.
    """
    suffix = f"_{unit}"
    return run[: -len(suffix)] if run.endswith(suffix) else run


def interval_years(interval: str) -> tuple[int, int]:
    """Earlier and later year of an interval named ``<dateA>_<dateB>``."""
    match = _INTERVAL.match(interval)
    if match is None:
        raise ValueError(f"not an interval: {interval!r}")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class RunInfo:
    """One source of maps at one site."""

    run: str
    unit: str
    family: str
    label: str
    intervals: tuple[str, ...]
    layers: tuple[str, ...]
    threshold: float | None = None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    model_run: str | None = None
    has_forest: bool = False
    has_skip: bool = False
    version: int = 0
    map_dir: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "unit": self.unit,
            "family": self.family,
            "label": self.label,
            "intervals": list(self.intervals),
            "layers": list(self.layers),
            "threshold": self.threshold,
            "scores": self.scores,
            "model_run": self.model_run,
            "has_forest": self.has_forest,
            "has_skip": self.has_skip,
            "version": self.version,
        }


@dataclass(frozen=True)
class SiteInfo:
    """One unit and what there is to show for it."""

    unit: str
    place: str
    notes: str
    bbox: tuple[float, float, float, float] | None
    runs: tuple[str, ...]
    intervals: tuple[str, ...]
    #: NAIP years with the acquisition date where an interval names it.
    years: tuple[tuple[int, str | None], ...]
    quicklook_years: tuple[int, ...]
    chm_years: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "place": self.place,
            "notes": self.notes,
            "bbox": list(self.bbox) if self.bbox else None,
            "runs": list(self.runs),
            "intervals": list(self.intervals),
            "years": [{"year": y, "date": d} for y, d in self.years],
            "quicklook_years": list(self.quicklook_years),
            "chm_years": list(self.chm_years),
        }


@dataclass(frozen=True)
class Catalog:
    """Every run and site under an export folder."""

    export_dir: Path
    runs: dict[str, RunInfo]
    sites: dict[str, SiteInfo]

    def run(self, name: str) -> RunInfo:
        try:
            return self.runs[name]
        except KeyError:
            raise FileNotFoundError(f"no run {name}") from None

    def site(self, unit: str) -> SiteInfo:
        try:
            return self.sites[unit]
        except KeyError:
            raise FileNotFoundError(f"no site {unit}") from None

    def unit_dir(self, unit: str) -> Path:
        """Folder of the unit's Disturbance Index run."""
        return self.export_dir / "runs" / DI_TAG / unit

    def to_json(self) -> dict[str, Any]:
        return {
            "sites": [s.to_json() for s in self.sites.values()],
            "runs": {name: r.to_json() for name, r in self.runs.items()},
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _band_names(path: Path) -> tuple[str, ...]:
    import rasterio

    with rasterio.open(path) as src:
        return tuple(d or "" for d in src.descriptions)


def _threshold(run_dir: Path, unit: str, compare: dict[str, Any]) -> float:
    """Threshold of a run: the comparison, else a type raster, else 0.5."""
    if isinstance(compare.get("threshold"), int | float):
        return float(compare["threshold"])
    for path in sorted((run_dir / "map").glob(f"type_{unit}_*.tif")):
        match = _THRESHOLD.search(_band_names(path)[0])
        if match:
            return float(match.group(1))
        break
    logger.warning("%s: no threshold found, using 0.5", run_dir.name)
    return 0.5


def _scores(compare: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for source in ("model", "di"):
        values = compare.get(source)
        if isinstance(values, dict):
            out[source] = {
                key: round(float(values[key]), 3)
                for key in _SCORE_KEYS
                if isinstance(values.get(key), int | float)
            }
    return out


def _pair_run(run_dir: Path) -> RunInfo | None:
    probs = sorted((run_dir / "map").glob("prob_q*.tif"))
    probs = [p for p in probs if re.fullmatch(r"prob_q\d+", p.stem)]
    if not probs:
        return None
    prob = probs[0]
    unit = prob.stem.removeprefix("prob_")
    intervals = tuple(n for n in _band_names(prob) if _INTERVAL.match(n))
    if not intervals:
        logger.warning("%s: the bands of %s name no intervals", run_dir.name, prob.name)
        return None
    family = run_family(run_dir.name, unit)
    compare = _read_json(run_dir / "compare" / "compare.json")
    has_agent = (prob.parent / f"agent_{unit}.tif").exists()
    return RunInfo(
        run=run_dir.name,
        unit=unit,
        family=family,
        label=FAMILY_LABELS.get(family, family),
        intervals=intervals,
        layers=("date", "type", "prob") if has_agent else ("date", "prob"),
        threshold=_threshold(run_dir, unit, compare),
        scores=_scores(compare),
        model_run=_read_json(run_dir / "site.json").get("model_run"),
        has_forest=(prob.parent / f"prob_forest_{unit}.tif").exists(),
        has_skip=(prob.parent / f"prob_skip_{unit}.tif").exists(),
        version=int(prob.stat().st_mtime),
        map_dir=prob.parent,
    )


def _di_run(unit_dir: Path, unit: str, intervals: tuple[str, ...]) -> RunInfo | None:
    onset = unit_dir / "onset_year_u8.tif"
    if not onset.exists():
        return None
    layers = ["date"]
    if (unit_dir / "max_probability_u8.tif").exists():
        layers.append("prob")
    return RunInfo(
        run=f"{DI_PREFIX}{unit}",
        unit=unit,
        family="di",
        label=FAMILY_LABELS["di"],
        intervals=intervals,
        layers=tuple(layers),
        version=int(onset.stat().st_mtime),
        map_dir=unit_dir,
    )


def _read_places(sites_csv: Path | None) -> dict[str, tuple[str, str]]:
    if sites_csv is None or not sites_csv.exists():
        return {}
    with sites_csv.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        return {r["unit"]: (r.get("place") or "", r.get("notes") or "") for r in rows}


def _years_of(pattern: str, folder: Path) -> tuple[int, ...]:
    regex = re.compile(pattern)
    found = (regex.fullmatch(p.name) for p in folder.glob("*.tif"))
    return tuple(sorted(int(m.group(1)) for m in found if m))


def _run_order(run: RunInfo) -> tuple[int, str]:
    if run.family in _FAMILY_ORDER:
        return _FAMILY_ORDER.index(run.family), run.run
    return len(_FAMILY_ORDER), run.run


def build_catalog(export_dir: Path, sites_csv: Path | None = None) -> Catalog:
    """Read the runs and sites of an export folder.

    Args:
        export_dir: Folder with ``pairmodel``, ``runs`` and ``quicklooks``.
        sites_csv: Table with ``unit``, ``place`` and ``notes`` columns
            (config/pair/sites.csv). It names the sites and orders them.
    """
    export_dir = Path(export_dir)
    by_unit: dict[str, list[RunInfo]] = {}
    for run_dir in sorted((export_dir / PAIR_DIR).glob("*")):
        if not (run_dir / "map").is_dir():
            continue
        try:
            info = _pair_run(run_dir)
        except Exception:  # one unreadable run must not hide the others
            logger.exception("%s could not be read", run_dir.name)
            continue
        if info is not None:
            by_unit.setdefault(info.unit, []).append(info)

    places = _read_places(sites_csv)
    order = {unit: i for i, unit in enumerate(places)}
    runs: dict[str, RunInfo] = {}
    sites: dict[str, SiteInfo] = {}
    for unit in sorted(by_unit, key=lambda u: (order.get(u, len(order)), u)):
        pair_runs = sorted(by_unit[unit], key=_run_order)
        intervals = tuple(
            sorted({i for r in pair_runs for i in r.intervals}, key=interval_years)
        )
        unit_dir = export_dir / "runs" / DI_TAG / unit
        record = _read_json(unit_dir / "unit.json")
        dates: dict[int, str] = {}
        for interval in intervals:
            for date in interval.split("_"):
                dates.setdefault(int(date[:4]), date)
        quicklooks = export_dir / "quicklooks" / unit
        ql_years = _years_of(r"naip_(\d{4})_3m_rgb\.tif", quicklooks)
        years = sorted({*dates, *ql_years, *(int(y) for y in record.get("years", {}))})
        site_runs = [*pair_runs]
        di = _di_run(unit_dir, unit, intervals)
        if di is not None:
            site_runs.append(di)
        for run in site_runs:
            runs[run.run] = run
        place, notes = places.get(unit, (record.get("label") or unit, ""))
        bbox = record.get("bbox_4326")
        sites[unit] = SiteInfo(
            unit=unit,
            place=place,
            notes=notes or record.get("notes") or "",
            bbox=tuple(bbox) if bbox else None,
            runs=tuple(r.run for r in site_runs),
            intervals=intervals,
            years=tuple((y, dates.get(y)) for y in years),
            quicklook_years=ql_years,
            chm_years=_years_of(r"chm_(\d{4})_3m\.tif", quicklooks),
        )
    logger.info("Catalogue: %d runs at %d sites", len(runs), len(sites))
    return Catalog(export_dir=export_dir, runs=runs, sites=sites)
