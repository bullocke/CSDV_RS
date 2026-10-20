"""csdv_core.disturbance.report: run summary, ``summary.json`` and ``report.md``.

The summary records what a reader needs to trust and reuse a run: the
parameters, the input inventory (files, acquisition dates, dtype and scale),
the grid (CRS, extent, cell size, window and smoothing scales as actually
used), forest area per year, per-interval detected area, object count and
share of monitored forest, the null fits per date and skipped dates, runtime
per stage, software versions and the known caveats. ``summary.json`` is the
machine-readable form, ``report.md`` the readable one (with links to the
figures).

Inputs: results of the pipeline stages. Outputs: :class:`RunSummary`,
``summary.json`` and ``report.md``.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import json
import logging
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio

import csdv_core
from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.decide import DateFit, DecisionResult
from csdv_core.disturbance.inputs import CellGrid, SiteInputs, parse_acquisition_date
from csdv_core.disturbance.layers import CellLayers
from csdv_core.disturbance.outlines import RefinementResult, interval_label
from csdv_core.disturbance.params import params_to_dict
from csdv_core.disturbance.report_caveats import follow_up_caveat, season_caveats
from csdv_core.disturbance.report_markdown import render_report_lines
from csdv_core.disturbance.standardize import (
    LOCAL_METHODS,
    sigmas_in_cells,
    window_cells,
)
from csdv_core.disturbance.store import RunPaths

logger = logging.getLogger(__name__)

#: Version of the ``summary.json`` layout.
SUMMARY_SCHEMA_VERSION = 1
#: Packages whose versions go into :func:`software_info`.
REPORTED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "rasterio",
    "geopandas",
    "shapely",
    "pydantic",
    "matplotlib",
)
#: Caveats printed in every report.
DEFAULT_CAVEATS: tuple[str, ...] = (
    "The last interval has no later image, so rules D6 and D6s cannot check "
    "it for a rebound and it keeps the D5 probability. Rules D7 and D7s check "
    "it against the image two dates back only.",
    "Outline refinement always uses the 0.6 m CHM, whatever the detection input.",
    "Detections are not scored against reference data in this run.",
)
#: Value of ``git_commit`` and ``git_dirty`` when git gives no answer.
UNKNOWN = "unknown"
#: Seconds before a git call is abandoned.
GIT_TIMEOUT_S = 10


@dataclass(frozen=True)
class InputFileInfo:
    """One input file of the inventory.

    Attributes:
        year: NAIP year.
        kind: ``"chm"`` or ``"naip"``.
        path: Absolute path as a string.
        acquisition_date: ISO date parsed from the name, or None.
        dtype: Stored data type.
        scale: Stored-to-metres multiplier (1.0 for NAIP).
        nodata: Stored nodata value, or None.
    """

    year: int
    kind: str
    path: str
    acquisition_date: str | None
    dtype: str
    scale: float
    nodata: float | None


@dataclass(frozen=True)
class GridSummary:
    """The grid and the scales actually used.

    Attributes:
        crs: CRS as a string (EPSG code when known, else WKT).
        bounds: ``(left, bottom, right, top)`` in CRS units.
        pixel_m: Source pixel size (m).
        block_px: Pixels per cell side.
        cell_m: Cell side (m).
        cell_rows: Cell rows.
        cell_cols: Cell columns.
        area_ha: Grid area (ha).
        window_cells: Local window side in cells (None for scene methods).
        window_m_effective: ``window_cells * cell_m`` (None for scene methods).
        sigma_cells: ``decide.spatial_sigma_m / cell_m``.
    """

    crs: str
    bounds: tuple[float, float, float, float]
    pixel_m: float
    block_px: int
    cell_m: float
    cell_rows: int
    cell_cols: int
    area_ha: float
    window_cells: int | None
    window_m_effective: float | None
    sigma_cells: tuple[float, ...]


@dataclass(frozen=True)
class IntervalSummary:
    """Detection results for one NAIP interval (date index ``t >= 1``).

    Attributes:
        year_before: Earlier image year, ``years[t - 1]``.
        year: Later image year, ``years[t]`` (the onset year label).
        label: ``"<year_before>-<year>"``.
        monitored_ha: Area monitored at ``t`` (forest at ``t - 1``).
        detected_ha: Detected area after the MMU.
        detected_ha_nommu: Detected area before the MMU.
        n_objects: Objects with this onset after the MMU.
        share_of_monitored: ``detected_ha / monitored_ha`` (0 when nothing is
            monitored).
        fit: The date's null fit.
        skipped: Same as ``fit.skipped``.
        n_refined: Objects of this interval with a refined envelope.
        refined_ha: Refined envelope area of this interval (ha).
    """

    year_before: int
    year: int
    label: str
    monitored_ha: float
    detected_ha: float
    detected_ha_nommu: float
    n_objects: int
    share_of_monitored: float
    fit: DateFit
    skipped: bool
    n_refined: int
    refined_ha: float


@dataclass
class RunSummary:
    """Everything ``summary.json`` and ``report.md`` hold.

    Attributes:
        schema_version: :data:`SUMMARY_SCHEMA_VERSION`.
        site: Site name.
        created: ISO 8601 local time of the summary.
        output_dir: Run folder (string).
        command_line: Command line that started the run, or None.
        config_path: User YAML path, or None.
        from_run: Source run folder for ``--from-run``, or None.
        params: Final parameters as a nested dict.
        input_name: Detection input (``params.inputs.input``).
        method: Standardization method.
        rule: Decision rule.
        years: NAIP years.
        inputs: Input inventory.
        grid: Grid summary.
        forest_ha_by_year: Forest area per year (ha), keys are years.
        intervals: One entry per interval (``T - 1`` entries).
        skipped_years: Years whose null fit was skipped (index 0 excluded).
        n_objects: Objects after the MMU.
        detected_ha: Detected area after the MMU (ha).
        n_refined: Objects with a refined envelope (0 when refinement is off).
        n_refine_skipped: Objects refinement could not redraw.
        refined_ha: Total refined envelope area (ha).
        stage_seconds: Wall time per pipeline stage (s), in stage order.
        software: Output of :func:`software_info`.
        caveats: Caveats for the reader.
        outputs: Output files relative to ``output_dir`` keyed by short name.
    """

    schema_version: int
    site: str
    created: str
    output_dir: str
    command_line: str | None
    config_path: str | None
    from_run: str | None
    params: dict[str, Any]
    input_name: str
    method: str
    rule: str
    years: list[int]
    inputs: list[InputFileInfo]
    grid: GridSummary
    forest_ha_by_year: dict[int, float]
    intervals: list[IntervalSummary]
    skipped_years: list[int]
    n_objects: int
    detected_ha: float
    n_refined: int
    n_refine_skipped: int
    refined_ha: float
    stage_seconds: dict[str, float]
    software: dict[str, str]
    caveats: list[str] = field(default_factory=lambda: list(DEFAULT_CAVEATS))
    outputs: dict[str, str] = field(default_factory=dict)


def _run_git(arguments: Sequence[str], repo_root: Path) -> str | None:
    """Run a git command and return its stripped output.

    Args:
        arguments: Arguments after ``git``.
        repo_root: Folder to run git in.

    Returns:
        The standard output, or None when git is missing or fails.
    """
    command = ["git", "-C", str(repo_root), *arguments]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def software_info(repo_root: Path | None = None) -> dict[str, str]:
    """Versions of the code that produced a run.

    Keys: ``csdv_core`` (package version), ``git_commit`` (short hash or
    ``"unknown"``), ``git_dirty`` (``"true"``/``"false"``/``"unknown"``),
    ``python`` and one key per :data:`REPORTED_PACKAGES` entry. Any git
    failure gives ``"unknown"`` instead of an error.

    Args:
        repo_root: Folder to run git in (None = the folder of this module).

    Returns:
        Component name to version text.
    """
    git_root = Path(__file__).resolve().parent if repo_root is None else repo_root
    commit = _run_git(["rev-parse", "--short", "HEAD"], git_root)
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], git_root)
    if commit is None or status is None:
        dirty = UNKNOWN
    else:
        dirty = "true" if status else "false"
    info = {
        "csdv_core": csdv_core.__version__,
        "git_commit": commit or UNKNOWN,
        "git_dirty": dirty,
        "python": platform.python_version(),
    }
    for package_name in REPORTED_PACKAGES:
        try:
            info[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            info[package_name] = "not installed"
    return info


def input_inventory(site_inputs: SiteInputs) -> list[InputFileInfo]:
    """List every CHM (and NAIP, when present) file of the run, by year.

    Args:
        site_inputs: Discovered inputs.

    Returns:
        One entry per file.
    """
    inventory: list[InputFileInfo] = []
    for year_input in site_inputs.years:
        chm_files = zip(year_input.chm_paths, year_input.acquisition_dates, strict=True)
        for chm_path, acquisition_date in chm_files:
            date_text = None if acquisition_date is None else str(acquisition_date)
            resolved = chm_path.resolve()
            inventory.append(
                InputFileInfo(
                    year=year_input.year,
                    kind="chm",
                    path=str(resolved),
                    acquisition_date=date_text,
                    dtype=year_input.chm_dtype,
                    scale=year_input.chm_scale,
                    nodata=year_input.chm_nodata,
                )
            )
        for naip_path in year_input.naip_paths:
            with rasterio.open(naip_path) as dataset:
                naip_dtype = dataset.dtypes[0]
                naip_nodata = dataset.nodata
            naip_date = parse_acquisition_date(naip_path)
            date_text = None if naip_date is None else str(naip_date)
            resolved = naip_path.resolve()
            inventory.append(
                InputFileInfo(
                    year=year_input.year,
                    kind="naip",
                    path=str(resolved),
                    acquisition_date=date_text,
                    dtype=naip_dtype,
                    scale=1.0,
                    nodata=naip_nodata,
                )
            )
    return inventory


def grid_summary(grid: CellGrid, params: DisturbanceConfig) -> GridSummary:
    """Summarize the grid and the window and sigma sizes in cells.

    Args:
        grid: Common grid.
        params: Run parameters.

    Returns:
        The grid summary.
    """
    epsg_code = grid.crs.to_epsg()
    crs_text = grid.crs.to_wkt() if epsg_code is None else f"EPSG:{epsg_code}"
    cell_m = grid.cell_m
    n_cells = grid.cell_rows * grid.cell_cols
    window_size: int | None = None
    window_m_effective: float | None = None
    if params.standardize.method in LOCAL_METHODS:
        window_size = window_cells(params.standardize.window_m, cell_m)
        window_m_effective = window_size * cell_m
    sigmas = sigmas_in_cells(params, cell_m)
    return GridSummary(
        crs=crs_text,
        bounds=grid.bounds,
        pixel_m=grid.pixel_m,
        block_px=grid.block_px,
        cell_m=cell_m,
        cell_rows=grid.cell_rows,
        cell_cols=grid.cell_cols,
        area_ha=n_cells * grid.cell_ha,
        window_cells=window_size,
        window_m_effective=window_m_effective,
        sigma_cells=tuple(sigmas),
    )


def _existing_outputs(run_paths: RunPaths) -> dict[str, str]:
    """Files under the run folder, keyed by relative path without suffix.

    Args:
        run_paths: Paths of the run.

    Returns:
        ``{"detection/onset_year": "detection/onset_year.tif", ...}`` in
        sorted path order.
    """
    outputs: dict[str, str] = {}
    all_paths = sorted(run_paths.root.rglob("*"))
    for file_path in all_paths:
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(run_paths.root)
        short_name = relative.with_suffix("").as_posix()
        outputs[short_name] = relative.as_posix()
    return outputs


def build_summary(
    *,
    site: str,
    params: DisturbanceConfig,
    inputs: Sequence[InputFileInfo],
    grid: CellGrid,
    layers: CellLayers,
    monitored: np.ndarray,
    decision: DecisionResult,
    objects: gpd.GeoDataFrame,
    refinement: RefinementResult | None,
    stage_seconds: Mapping[str, float],
    run_paths: RunPaths,
    command_line: str | None,
    config_path: Path | None,
    from_run: Path | None,
) -> RunSummary:
    """Assemble the run summary from the stage results.

    Areas are cell counts times ``grid.cell_ha``. Monitored and detected
    areas per interval use ``monitored[t]`` and ``decision.onset_index`` /
    ``onset_index_mmu == t``. ``outputs`` lists the files that exist in the
    run folder at call time, relative to ``run_paths.root``. The caveats are
    :data:`DEFAULT_CAVEATS`, the count of objects without a follow-up image
    and one line per interval with a season mismatch (see
    :mod:`csdv_core.disturbance.report_caveats`).

    Args:
        site: Site name.
        params: Run parameters.
        inputs: Input inventory (from :func:`input_inventory`, or reloaded from
            the source run's ``summary.json`` for ``--from-run``).
        grid: Common grid.
        layers: Cell layers (forest area per year).
        monitored: bool ``(T, rows, cols)``.
        decision: Decision result.
        objects: Objects GeoDataFrame.
        refinement: Refinement result, or None when refinement is off.
        stage_seconds: Wall time per stage.
        run_paths: Paths of the run.
        command_line: Command line, or None.
        config_path: User YAML path, or None.
        from_run: Source run folder, or None.

    Returns:
        The run summary.
    """
    years = list(layers.years)
    cell_ha = grid.cell_ha
    forest_ha_by_year: dict[int, float] = {}
    for year_index, year in enumerate(years):
        n_forest = np.count_nonzero(layers.forest[year_index])
        forest_ha_by_year[year] = int(n_forest) * cell_ha
    refined_outlines = None if refinement is None else refinement.outlines
    intervals: list[IntervalSummary] = []
    for year_index in range(1, len(years)):
        year_before = years[year_index - 1]
        year = years[year_index]
        n_monitored = np.count_nonzero(monitored[year_index])
        n_detected = np.count_nonzero(decision.onset_index_mmu == year_index)
        n_detected_nommu = np.count_nonzero(decision.onset_index == year_index)
        monitored_ha = int(n_monitored) * cell_ha
        detected_ha = int(n_detected) * cell_ha
        detected_ha_nommu = int(n_detected_nommu) * cell_ha
        share = detected_ha / monitored_ha if monitored_ha > 0 else 0.0
        interval_objects = objects["onset_index"] == year_index
        n_objects_found = interval_objects.sum()
        n_objects = int(n_objects_found)
        n_refined = 0
        refined_ha = 0.0
        if refined_outlines is not None:
            interval_refined = refined_outlines["onset_year"] == year
            n_refined_found = interval_refined.sum()
            n_refined = int(n_refined_found)
            refined_area = refined_outlines.loc[interval_refined, "envelope_ha"]
            refined_sum = refined_area.sum()
            refined_ha = float(refined_sum)
        fit = decision.fits[year_index]
        label = interval_label(year_before, year)
        intervals.append(
            IntervalSummary(
                year_before=year_before,
                year=year,
                label=label,
                monitored_ha=monitored_ha,
                detected_ha=detected_ha,
                detected_ha_nommu=detected_ha_nommu,
                n_objects=n_objects,
                share_of_monitored=share,
                fit=fit,
                skipped=fit.skipped,
                n_refined=n_refined,
                refined_ha=refined_ha,
            )
        )
    skipped_years = [fit.year for fit in decision.fits[1:] if fit.skipped]
    total_detected_ha = sum(interval.detected_ha for interval in intervals)
    total_refined_ha = sum(interval.refined_ha for interval in intervals)
    n_refined_total = 0
    n_refine_skipped = 0
    if refinement is not None:
        n_refined_total = refinement.n_refined
        n_refine_skipped = len(refinement.skipped_object_ids)
    now = datetime.datetime.now()
    created = now.astimezone().isoformat(timespec="seconds")
    seconds_by_stage = {name: float(seconds) for name, seconds in stage_seconds.items()}
    config_text = None if config_path is None else str(config_path)
    from_run_text = None if from_run is None else str(from_run)
    params_dict = params_to_dict(params)
    grid_info = grid_summary(grid, params)
    software = software_info()
    outputs = _existing_outputs(run_paths)
    follow_up_text = follow_up_caveat(objects, years)
    season_texts = season_caveats(inputs, years)
    caveats = [*DEFAULT_CAVEATS, follow_up_text, *season_texts]
    return RunSummary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        site=site,
        created=created,
        output_dir=str(run_paths.root),
        command_line=command_line,
        config_path=config_text,
        from_run=from_run_text,
        params=params_dict,
        input_name=params.inputs.input,
        method=params.standardize.method,
        rule=params.decide.rule,
        years=years,
        inputs=list(inputs),
        grid=grid_info,
        forest_ha_by_year=forest_ha_by_year,
        intervals=intervals,
        skipped_years=skipped_years,
        n_objects=len(objects),
        detected_ha=total_detected_ha,
        n_refined=n_refined_total,
        n_refine_skipped=n_refine_skipped,
        refined_ha=total_refined_ha,
        stage_seconds=seconds_by_stage,
        software=software,
        caveats=caveats,
        outputs=outputs,
    )


def refresh_summary(
    summary: RunSummary,
    run_paths: RunPaths,
    stage_seconds: Mapping[str, float] | None = None,
) -> RunSummary:
    """Return a copy of ``summary`` with current outputs and stage times.

    The pipeline builds the summary before the figures exist. This refreshes
    ``outputs`` from the files now in the run folder and, when given,
    replaces ``stage_seconds``.

    Args:
        summary: Summary from :func:`build_summary`.
        run_paths: Paths of the run.
        stage_seconds: Wall time per stage, or None to keep the saved times.

    Returns:
        The updated summary (``summary`` itself is not modified).
    """
    outputs = _existing_outputs(run_paths)
    seconds_by_stage = dict(summary.stage_seconds)
    if stage_seconds is not None:
        seconds_by_stage = {
            name: float(seconds) for name, seconds in stage_seconds.items()
        }
    return dataclasses.replace(summary, outputs=outputs, stage_seconds=seconds_by_stage)


def _json_ready(value: Any) -> Any:
    """Convert a value to JSON-ready types (NaN floats become None).

    Args:
        value: Nested dicts, lists, tuples and scalars (NumPy scalars allowed).

    Returns:
        The same structure with dicts keyed by strings, tuples as lists and
        non-finite floats as None.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summary_to_dict(summary: RunSummary) -> dict[str, Any]:
    """JSON-ready dict of a summary (nested dataclasses become dicts).

    Args:
        summary: The run summary.

    Returns:
        Nested dict with string year keys and NaN floats as None.
    """
    raw = dataclasses.asdict(summary)
    return _json_ready(raw)


def _date_fit_from_dict(data: Mapping[str, Any]) -> DateFit:
    """Rebuild a :class:`DateFit` from its JSON form (null floats become NaN).

    Args:
        data: One ``fit`` entry of ``summary.json``.

    Returns:
        The fit.
    """
    values = dict(data)
    for key in ("mu0", "sigma0", "p0"):
        if values[key] is None:
            values[key] = math.nan
    return DateFit(**values)


def summary_from_dict(data: Mapping[str, Any]) -> RunSummary:
    """Inverse of :func:`summary_to_dict`.

    Args:
        data: Parsed ``summary.json``.

    Returns:
        The run summary.

    Raises:
        KeyError: If a field is missing.
        ValueError: If ``schema_version`` is newer than
            :data:`SUMMARY_SCHEMA_VERSION`.
    """
    schema_version = int(data["schema_version"])
    if schema_version > SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"summary schema {schema_version} is newer than this code "
            f"({SUMMARY_SCHEMA_VERSION})"
        )
    intervals = []
    for interval_data in data["intervals"]:
        interval_values = dict(interval_data)
        interval_values["fit"] = _date_fit_from_dict(interval_data["fit"])
        intervals.append(IntervalSummary(**interval_values))
    grid_values = dict(data["grid"])
    grid_values["bounds"] = tuple(grid_values["bounds"])
    grid_values["sigma_cells"] = tuple(grid_values["sigma_cells"])
    forest_ha_by_year = {
        int(year): area for year, area in data["forest_ha_by_year"].items()
    }
    summary_values = dict(data)
    summary_values["inputs"] = [InputFileInfo(**item) for item in data["inputs"]]
    summary_values["intervals"] = intervals
    summary_values["grid"] = GridSummary(**grid_values)
    summary_values["forest_ha_by_year"] = forest_ha_by_year
    return RunSummary(**summary_values)


def write_summary_json(summary: RunSummary, path: Path) -> Path:
    """Write ``summary.json`` with indent 2.

    Keys keep the field order of :class:`RunSummary`, so ``stage_seconds``
    stays in stage order.

    Args:
        summary: The run summary.
        path: Output file.

    Returns:
        ``path``.
    """
    content = summary_to_dict(summary)
    text = json.dumps(content, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def read_summary_json(path: Path) -> RunSummary:
    """Read ``summary.json`` back into a :class:`RunSummary`.

    Args:
        path: ``summary.json``.

    Returns:
        The run summary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Summary not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return summary_from_dict(data)


def write_report_markdown(
    summary: RunSummary,
    path: Path,
    *,
    figure_paths: Mapping[str, Path] | None = None,
) -> Path:
    """Write the readable report.

    Sections: run (site, time, command line, output folder), parameters
    (table of every section with the effective window and sigma sizes),
    inputs (inventory table), grid, forest area by year, detections by
    interval (table), null fits by date (with skipped dates flagged),
    refinement, runtime by stage, software, caveats, figures (relative links
    to ``figure_paths`` that exist) and output files. Prose follows the
    repository writing rules.

    Args:
        summary: The run summary.
        path: Output ``report.md``.
        figure_paths: Figure name to PNG path (linked relative to ``path``).
            None links every PNG in the ``figures/`` folder next to ``path``.

    Returns:
        ``path``.
    """
    if figure_paths is None:
        figures_dir = path.parent / "figures"
        found_figures = sorted(figures_dir.glob("*.png"))
        figure_paths = {figure.stem: figure for figure in found_figures}
    lines = render_report_lines(summary, report_dir=path.parent, figures=figure_paths)
    text = "\n".join(lines)
    path.write_text(text + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


__all__ = [
    "DEFAULT_CAVEATS",
    "GridSummary",
    "InputFileInfo",
    "IntervalSummary",
    "RunSummary",
    "build_summary",
    "grid_summary",
    "input_inventory",
    "read_summary_json",
    "refresh_summary",
    "software_info",
    "summary_from_dict",
    "summary_to_dict",
    "write_report_markdown",
    "write_summary_json",
]
