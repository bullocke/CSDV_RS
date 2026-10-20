"""csdv_core.disturbance.report_markdown: render a run summary as Markdown.

Builds the lines of ``report.md`` from a
:class:`~csdv_core.disturbance.report.RunSummary`: a run summary paragraph,
then tables of parameters, inputs, grid, forest area, detections and null
fits, the figures, the output files, runtime, software and caveats.
:func:`csdv_core.disturbance.report.write_report_markdown` writes the file.

Inputs: a run summary and figure paths. Outputs: a list of Markdown lines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from csdv_core.disturbance.params import flatten_mapping

if TYPE_CHECKING:
    from csdv_core.disturbance.report import RunSummary

#: Placeholder for missing values in report tables.
MISSING_TEXT = "n/a"
#: Longest CRS text shown inline in the report (longer WKT stays in JSON).
MAX_INLINE_CRS = 40
#: Plain-language names of the detection inputs.
INPUT_LABELS: dict[str, str] = {
    "d_chm_h": "the change in mean CHM height",
    "d_chm_cov": "the change in CHM canopy cover",
    "d_ndvi": "the change in NAIP NDVI",
}
#: Plain-language names of the standardization methods.
METHOD_LABELS: dict[str, str] = {
    "scene": "scene median and MAD",
    "scenemean": "scene mean and standard deviation",
    "local": "a local moving window",
    "local_twopass": "a two-pass local moving window",
}
#: Plain-language names of the decision rules.
RULE_LABELS: dict[str, str] = {
    "D1": "fixed z threshold",
    "D2": "local false discovery rate",
    "D5": "spatial support",
    "D5s": "spatial support with a strict cell test",
    "D6": "spatial support and persistence",
    "D6s": "spatial support and persistence with a strict cell test",
}


def _format_number(value: float | int | None, digits: int = 2) -> str:
    """Format a number for a Markdown table.

    Args:
        value: Number to format, or None.
        digits: Decimal places for floats.

    Returns:
        The formatted number, or ``"n/a"`` for None and NaN.
    """
    if value is None:
        return MISSING_TEXT
    if isinstance(value, (bool, int, np.integer)):
        return str(value)
    if not math.isfinite(value):
        return MISSING_TEXT
    return f"{value:,.{digits}f}"


def _count_text(count: int, noun: str) -> str:
    """Return ``"<count> <noun>"`` with a plural ``s`` unless the count is one."""
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _format_value(value: Any) -> str:
    """Format a parameter value for a Markdown table cell.

    Args:
        value: A JSON-ready parameter value.

    Returns:
        ``null`` for None, comma-separated lists, else ``str(value)``.
    """
    if value is None:
        return "null"
    if isinstance(value, list):
        parts = [str(item) for item in value]
        joined = ", ".join(parts)
        return f"[{joined}]"
    return str(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Build the lines of a Markdown table.

    Args:
        header: Column titles.
        rows: Cell text per row, same length as ``header``.

    Returns:
        Header line, separator line and one line per row.
    """
    header_line = "| " + " | ".join(header) + " |"
    separators = ["---"] * len(header)
    separator_line = "| " + " | ".join(separators) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        row_line = "| " + " | ".join(row) + " |"
        lines.append(row_line)
    return lines


def _relative_link(target: Path, report_dir: Path) -> str:
    """Path of ``target`` relative to the report folder, as POSIX text.

    Args:
        target: File to link.
        report_dir: Folder of ``report.md``.

    Returns:
        The relative path, or the absolute path when ``target`` lies outside
        ``report_dir``.
    """
    resolved_target = target.resolve()
    resolved_dir = report_dir.resolve()
    try:
        relative = resolved_target.relative_to(resolved_dir)
    except ValueError:
        return resolved_target.as_posix()
    return relative.as_posix()


def _summary_section(summary: RunSummary) -> list[str]:
    """Lines of the run summary section."""
    input_label = INPUT_LABELS.get(summary.input_name, summary.input_name)
    method_label = METHOD_LABELS.get(summary.method, summary.method)
    rule_label = RULE_LABELS.get(summary.rule, summary.rule)
    first_year = summary.years[0]
    last_year = summary.years[-1]
    n_years = len(summary.years)
    mmu_ha = summary.params["mmu"]["mmu_ha"]
    detected_text = _format_number(summary.detected_ha)
    objects_text = _count_text(summary.n_objects, "object")
    paragraph = (
        f"This run used {n_years} NAIP-CHM years from {first_year} to "
        f"{last_year}. The detection input was {input_label} "
        f"(`{summary.input_name}`), standardized with {method_label} "
        f"(`{summary.method}`). Rule `{summary.rule}` ({rule_label}) and a "
        f"minimum mapping unit of {mmu_ha} ha produced {objects_text} "
        f"covering {detected_text} ha."
    )
    lines = ["## Run summary", "", paragraph, ""]
    if summary.params["refine"]["enabled"]:
        refined_text = _format_number(summary.refined_ha)
        redrawn_text = _count_text(summary.n_refined, "object")
        not_redrawn_text = _count_text(summary.n_refine_skipped, "object")
        lines.append(
            f"Outline refinement at 0.6 m redrew {redrawn_text} "
            f"({refined_text} ha of envelopes). It could not redraw "
            f"{not_redrawn_text}."
        )
    else:
        lines.append("Outline refinement was off for this run.")
    lines.append("")
    rows = [
        ["Site", summary.site],
        ["Created", summary.created],
        ["Output folder", f"`{summary.output_dir}`"],
        [
            "Command line",
            f"`{summary.command_line}`" if summary.command_line else "none",
        ],
        ["Config file", f"`{summary.config_path}`" if summary.config_path else "none"],
        ["Reused run", f"`{summary.from_run}`" if summary.from_run else "none"],
    ]
    table_lines = _table(["Item", "Value"], rows)
    lines.extend(table_lines)
    return lines


def _parameters_section(summary: RunSummary) -> list[str]:
    """Lines of the parameter table (dotted keys)."""
    flat = flatten_mapping(summary.params)
    rows = [[f"`{key}`", _format_value(value)] for key, value in flat.items()]
    intro = (
        "Final parameters after merging the packaged defaults, the config file "
        "and the command-line options. The Grid section gives the window and "
        "smoothing sizes in cells."
    )
    table_lines = _table(["Parameter", "Value"], rows)
    return ["## Parameters", "", intro, "", *table_lines]


def _inputs_section(summary: RunSummary) -> list[str]:
    """Lines of the input inventory table."""
    rows = []
    for file_info in summary.inputs:
        file_name = Path(file_info.path).name
        rows.append(
            [
                str(file_info.year),
                file_info.kind,
                f"`{file_name}`",
                file_info.acquisition_date or MISSING_TEXT,
                file_info.dtype,
                _format_number(file_info.scale, 3),
                _format_number(file_info.nodata, 1),
            ]
        )
    folder_set: set[str] = set()
    for file_info in summary.inputs:
        input_path = Path(file_info.path)
        folder_set.add(input_path.parent.as_posix())
    folders = sorted(folder_set)
    folder_text = ", ".join(f"`{folder}`" for folder in folders)
    header = ["Year", "Kind", "File", "Acquired", "Dtype", "Scale", "Nodata"]
    table_lines = _table(header, rows)
    return ["## Inputs", "", f"Input folders: {folder_text}.", "", *table_lines]


def _grid_section(summary: RunSummary) -> list[str]:
    """Lines of the grid table."""
    grid = summary.grid
    left, bottom, right, top = grid.bounds
    crs_text = (
        grid.crs if len(grid.crs) <= MAX_INLINE_CRS else "custom (WKT in summary.json)"
    )
    if grid.window_cells is None:
        window_text = "not used (scene method)"
    else:
        effective_text = _format_number(grid.window_m_effective, 1)
        window_text = f"{grid.window_cells} cells ({effective_text} m)"
    sigma_parts = [_format_number(sigma, 2) for sigma in grid.sigma_cells]
    sigma_text = ", ".join(sigma_parts)
    pixel_text = _format_number(grid.pixel_m, 2)
    cell_text = _format_number(grid.cell_m, 2)
    area_text = _format_number(grid.area_ha, 2)
    rows = [
        ["CRS", crs_text],
        [
            "Bounds (left, bottom, right, top)",
            f"{left:.1f}, {bottom:.1f}, {right:.1f}, {top:.1f}",
        ],
        ["Source pixel", f"{pixel_text} m"],
        ["Pixels per cell side", str(grid.block_px)],
        ["Cell size", f"{cell_text} m"],
        ["Cells (rows x columns)", f"{grid.cell_rows} x {grid.cell_cols}"],
        ["Grid area", f"{area_text} ha"],
        ["Local window", window_text],
        ["Smoothing sigmas", f"{sigma_text} cells"],
    ]
    table_lines = _table(["Property", "Value"], rows)
    return ["## Grid", "", *table_lines]


def _forest_section(summary: RunSummary) -> list[str]:
    """Lines of the forest area table."""
    rows = []
    for year, forest_ha in summary.forest_ha_by_year.items():
        share = forest_ha / summary.grid.area_ha if summary.grid.area_ha > 0 else 0.0
        rows.append([str(year), _format_number(forest_ha, 2), f"{share:.1%}"])
    table_lines = _table(["Year", "Forest (ha)", "Share of grid"], rows)
    return ["## Forest area per year", "", *table_lines]


def _detections_section(summary: RunSummary) -> list[str]:
    """Lines of the per-interval detection table."""
    intro = (
        "Monitored area is the forest of the earlier image. Detected area "
        "counts cells whose first detection falls in the interval, before and "
        "after the minimum mapping unit."
    )
    rows = []
    for interval in summary.intervals:
        label = f"{interval.label} (skipped)" if interval.skipped else interval.label
        rows.append(
            [
                label,
                _format_number(interval.monitored_ha, 2),
                _format_number(interval.detected_ha, 2),
                _format_number(interval.detected_ha_nommu, 2),
                str(interval.n_objects),
                f"{interval.share_of_monitored:.2%}",
                str(interval.n_refined),
                _format_number(interval.refined_ha, 2),
            ]
        )
    header = [
        "Interval",
        "Monitored (ha)",
        "Detected (ha)",
        "Before MMU (ha)",
        "Objects",
        "Share",
        "Refined objects",
        "Refined (ha)",
    ]
    table_lines = _table(header, rows)
    return ["## Detections by interval", "", intro, "", *table_lines]


def _null_fits_section(summary: RunSummary) -> list[str]:
    """Lines of the empirical null fit table."""
    rows = []
    for interval in summary.intervals:
        fit = interval.fit
        rows.append(
            [
                str(fit.year),
                str(fit.n_cells),
                _format_number(fit.mu0, 3),
                _format_number(fit.sigma0, 3),
                _format_number(fit.p0, 3),
                fit.method,
                "yes" if fit.skipped else "no",
            ]
        )
    header = ["Year", "Cells", "mu0", "sigma0", "p0", "Method", "Skipped"]
    table_lines = _table(header, rows)
    lines = ["## Empirical null fits per date", "", *table_lines, ""]
    if summary.skipped_years:
        skipped_text = ", ".join(str(year) for year in summary.skipped_years)
        min_cells = summary.params["decide"]["min_fit_cells"]
        lines.append(
            f"Dates skipped for having fewer than {min_cells} monitored cells: "
            f"{skipped_text}. No cell can be detected on a skipped date."
        )
    else:
        lines.append("Every date had enough monitored cells for a null fit.")
    return lines


def _figures_section(figures: Mapping[str, Path], report_dir: Path) -> list[str]:
    """Lines of the figure section (only figures that exist)."""
    lines = ["## Figures", ""]
    existing = {name: figure for name, figure in figures.items() if figure.is_file()}
    if not existing:
        lines.append("No figures were written for this run.")
        return lines
    for figure_name, figure_path in existing.items():
        link = _relative_link(figure_path, report_dir)
        title = figure_name.replace("_", " ").capitalize()
        lines.extend([f"### {title}", "", f"![{title}]({link})", ""])
    return lines[:-1]


def _outputs_section(summary: RunSummary) -> list[str]:
    """Lines of the output file list."""
    listed = set(summary.outputs.values())
    for own_file in ("summary.json", "report.md"):
        listed.add(own_file)
    lines = ["## Outputs", "", "Paths are relative to the output folder.", ""]
    for relative_path in sorted(listed):
        lines.append(f"- `{relative_path}`")
    return lines


def _runtime_section(summary: RunSummary) -> list[str]:
    """Lines of the runtime table."""
    rows = [
        [stage, _format_number(seconds, 1)]
        for stage, seconds in summary.stage_seconds.items()
    ]
    total_seconds = sum(summary.stage_seconds.values())
    rows.append(["**Total**", _format_number(total_seconds, 1)])
    table_lines = _table(["Stage", "Seconds"], rows)
    return ["## Runtime per stage", "", *table_lines]


def _software_section(summary: RunSummary) -> list[str]:
    """Lines of the software version table."""
    rows = [[name, version] for name, version in summary.software.items()]
    table_lines = _table(["Component", "Version"], rows)
    return ["## Software", "", *table_lines]


def _caveats_section(summary: RunSummary) -> list[str]:
    """Lines of the numbered caveat list."""
    lines = ["## Caveats", ""]
    for caveat_number, caveat in enumerate(summary.caveats, start=1):
        lines.append(f"{caveat_number}. {caveat}")
    return lines


def render_report_lines(
    summary: RunSummary, *, report_dir: Path, figures: Mapping[str, Path]
) -> list[str]:
    """Render the Markdown report as a list of lines.

    Args:
        summary: The run summary.
        report_dir: Folder of ``report.md`` (figure links are relative to it).
        figures: Figure name to PNG path.

    Returns:
        The report lines without trailing newlines.
    """
    title = f"# Disturbance detection report: {summary.site}"
    created_line = f"Site `{summary.site}`, created {summary.created}."
    sections = [
        _summary_section(summary),
        _parameters_section(summary),
        _inputs_section(summary),
        _grid_section(summary),
        _forest_section(summary),
        _detections_section(summary),
        _null_fits_section(summary),
        _figures_section(figures, report_dir),
        _outputs_section(summary),
        _runtime_section(summary),
        _software_section(summary),
        _caveats_section(summary),
    ]
    lines = [title, "", created_line]
    for section_lines in sections:
        lines.append("")
        lines.extend(section_lines)
    return lines


__all__ = ["render_report_lines"]
