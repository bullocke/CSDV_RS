"""csdv_core.disturbance.chips: NAIP RGB and CHM chips of an inspection sample.

For each sampled unit (see :mod:`csdv_core.disturbance.sampling`) one figure
shows the NAIP RGB image (top row) and the CHM (bottom row) of every year of
the run, centred on the unit. The detected cells are outlined in yellow and,
when the run refined the object at 0.6 m, the refined envelope in orange on the
CHM row. The two panels of the onset interval carry coloured frames. A contact
sheet shows the before and after RGB pair of every unit on one page.

The chips read the source rasters of the run's site under the current
``CSDV_DATA_ROOT`` (``naip/<site>/<year>/`` and ``naip_chm/<site>/<year>/``)
through :func:`csdv_core.disturbance.windows.read_bounds`, on the run's saved
0.6 m grid. Only small windows are read, so the NAIP mosaics do not need to be
copied anywhere first.

Inputs: a disturbance run folder and the site's NAIP and CHM GeoTIFFs.
Outputs, in a new folder (default ``<run>/inspection/``):

- ``samples.csv`` and ``samples.gpkg``: the sample, with empty
  ``interp_label`` and ``interp_notes`` columns for notes
- ``chips/<sample_id>_<stratum>_<interval>.png``: one figure per unit
- ``contact_sheet.png``: before and after RGB of every unit
- ``index.md``: the contact sheet and a table linking every chip
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from csdv_core.disturbance import _chip_panels as chip_panels
from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance import windows
from csdv_core.disturbance._chip_panels import RunRasters, load_run_rasters
from csdv_core.disturbance._pipeline_inputs import discover_with_saved_grid
from csdv_core.disturbance.inputs import SiteInputs
from csdv_core.disturbance.sampling import (
    STRATUM_DETECTED,
    SampleDesign,
    draw_samples,
)
from csdv_core.disturbance.store import REFINED_OUTLINES_LAYER, load_run
from csdv_core.viz.maps import CHM_VMAX_M, stretch_rgb
from csdv_core.viz.relief import canopy_cmap
from csdv_core.viz.style import ACCENT, HIGHLIGHT, INK, MUTED, save_fig, setup_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

#: Default output folder name inside the run folder.
DEFAULT_FOLDER_NAME = "inspection"
#: Outline colour of the detected cells.
CELL_OUTLINE_COLOR = "#ffd400"
#: Line width of the outlines, in points.
CELL_OUTLINE_WIDTH = 0.7
#: Outline colour of the refined 0.6 m envelope.
ENVELOPE_COLOR = "#eb6834"
#: Frame colour of the panel before the onset interval.
BEFORE_FRAME_COLOR = ACCENT
#: Frame colour of the panel after the onset interval.
AFTER_FRAME_COLOR = HIGHLIGHT
#: Width of each chip panel in inches.
CHIP_PANEL_IN = 2.2
#: Width of the gap column before the context maps, relative to a chip panel.
SPACER_RATIO = 0.22
#: Height of the time-series row, relative to a chip panel.
SERIES_ROW_RATIO = 0.95
#: Share of the year columns taken by the DI series (the height series
#: takes the rest of the row).
DI_SERIES_SHARE = 0.6
#: Colour bar placement next to a panel, in axes fractions (x, y, w, h).
COLORBAR_INSET: tuple[float, float, float, float] = (1.04, 0.0, 0.05, 1.0)
#: Sample units per row of the contact sheet (two panels each).
CONTACT_UNITS_PER_ROW = 4
#: Percentile stretch applied to each RGB chip.
RGB_PERCENTILES: tuple[float, float] = (2.0, 98.0)
#: Empty interpretation columns added to ``samples.csv``.
INTERPRETATION_COLUMNS: tuple[str, ...] = ("interp_label", "interp_notes")


@dataclass(frozen=True)
class ChipSettings:
    """Size and look of the chips.

    Attributes:
        pad_m: Margin around the unit on every side, in metres.
        min_chip_m: Smallest chip side, in metres.
        show_chm: Draw the CHM row (and the probability map) under the RGB row.
        context_factor: Side of the context maps as a multiple of the chip side.
        min_context_m: Smallest side of the context maps, in metres.
        dpi: Resolution of the PNG files.
    """

    pad_m: float = 40.0
    min_chip_m: float = 150.0
    show_chm: bool = True
    context_factor: float = 4.0
    min_context_m: float = 600.0
    dpi: int = 150


@dataclass
class InspectionResult:
    """Files written by :func:`make_inspection_chips`.

    Attributes:
        output_dir: The inspection folder.
        samples: The drawn sample (grid CRS).
        table_path: ``samples.csv``.
        vector_path: ``samples.gpkg``.
        chip_paths: One PNG per sample unit (empty with ``sample_only``).
        contact_sheet: ``contact_sheet.png``, or None with ``sample_only``.
        index_path: ``index.md``, or None with ``sample_only``.
    """

    output_dir: Path
    samples: gpd.GeoDataFrame
    table_path: Path
    vector_path: Path
    chip_paths: list[Path]
    contact_sheet: Path | None
    index_path: Path | None


def chip_bounds(
    geometry: BaseGeometry, settings: ChipSettings
) -> tuple[float, float, float, float]:
    """Square chip bounds around a unit, at least ``min_chip_m`` wide.

    Args:
        geometry: The unit's polygon.
        settings: Chip settings.

    Returns:
        ``(left, bottom, right, top)`` in grid CRS units.
    """
    left, bottom, right, top = helpers.square_bounds(geometry, settings.pad_m)
    half_side = max((right - left) / 2, settings.min_chip_m / 2)
    centre_x = (left + right) / 2
    centre_y = (bottom + top) / 2
    return (
        centre_x - half_side,
        centre_y - half_side,
        centre_x + half_side,
        centre_y + half_side,
    )


def read_rgb_chip(
    site_inputs: SiteInputs, year: int, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read and stretch one year's NAIP RGB over ``bounds``.

    Bands 1, 2 and 3 are read onto the run's pixel grid (nearest neighbour)
    and stretched together with :func:`csdv_core.viz.maps.stretch_rgb`.

    Args:
        site_inputs: Inputs of the run, with NAIP paths.
        year: NAIP year.
        bounds: ``(left, bottom, right, top)``.

    Returns:
        ``(rgb, extent)``: float32 ``(rows, cols, 3)`` in [0, 1] with NaN
        where NAIP has no data, and the ``imshow`` extent.
    """
    year_input = site_inputs.year_input(year)
    bands = []
    transform = None
    for band_number in (1, 2, 3):
        band_values, transform = windows.read_bounds(
            year_input.naip_paths, site_inputs.grid, bounds, band=band_number
        )
        bands.append(band_values)
    rgb = np.stack(bands, axis=-1)
    stretched = stretch_rgb(rgb, percentiles=RGB_PERCENTILES)
    extent = helpers.transform_extent(transform, rgb.shape)
    return stretched, extent


def _frame_panel(ax: Axes, color: str | None) -> None:
    """Hide ticks and give a panel a thick coloured frame (or a thin grey one).

    Args:
        ax: Panel axis.
        color: Frame colour, or None for the default thin frame.
    """
    helpers.bare(ax, frame=True)
    if color is None:
        return
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)


def _year_title(site_inputs: SiteInputs, year: int) -> str:
    """Panel title: the year and the NAIP acquisition date when known.

    Args:
        site_inputs: Inputs of the run.
        year: NAIP year.

    Returns:
        For example ``"2016 (2016-08-24)"``.
    """
    year_input = site_inputs.year_input(year)
    known_dates = [date for date in year_input.acquisition_dates if date]
    if not known_dates:
        return str(year)
    first_date = min(known_dates)
    return f"{year} ({first_date.isoformat()})"


def _unit_title(sample: pd.Series) -> str:
    """Figure title of one sample unit.

    Args:
        sample: One row of the sample table.

    Returns:
        Sample id, stratum, interval, area, probability, location and, when
        the sample has a False ``follow_up`` value, "no follow-up image".
    """
    parts = [
        str(sample["sample_id"]),
        str(sample["stratum"]),
        f"interval {sample['interval']}",
        f"{sample['area_ha']:.2f} ha",
    ]
    if np.isfinite(sample["max_prob"]):
        parts.append(f"max P {sample['max_prob']:.2f}")
    if pd.notna(sample["object_id"]):
        parts.append(f"object {int(sample['object_id'])}")
    parts.append(f"{sample['centroid_lat']:.5f}, {sample['centroid_lon']:.5f}")
    follow_up = sample.get("follow_up")
    if follow_up is not None and pd.notna(follow_up) and not follow_up:
        parts.append("no follow-up image")
    return "  |  ".join(parts)


def _draw_outlines(
    ax: Axes, geometry: BaseGeometry, envelope: BaseGeometry | None
) -> None:
    """Outline the unit's cells and, when given, its refined envelope.

    The RGB panels get only a thin cell outline, so the canopy inside stays
    visible. The CHM panels also get the refined envelope.

    Args:
        ax: Map axis in grid CRS units.
        geometry: Unit polygon (19.2 m cells).
        envelope: Refined 0.6 m envelope, or None.
    """
    helpers.outline(
        ax, [geometry], CELL_OUTLINE_COLOR, linewidth=CELL_OUTLINE_WIDTH, halo=None
    )
    if envelope is not None:
        helpers.outline(
            ax, [envelope], ENVELOPE_COLOR, linewidth=CELL_OUTLINE_WIDTH, halo=None
        )


def _frame_color(year: int, sample: pd.Series) -> str | None:
    """Frame colour of a year's panel: before or after the onset interval.

    Args:
        year: Panel year.
        sample: One row of the sample table.

    Returns:
        :data:`BEFORE_FRAME_COLOR`, :data:`AFTER_FRAME_COLOR` or None.
    """
    if year == sample["year_before"]:
        return BEFORE_FRAME_COLOR
    if year == sample["onset_year"]:
        return AFTER_FRAME_COLOR
    return None


def plot_sample_chips(
    sample: pd.Series,
    site_inputs: SiteInputs,
    rasters: RunRasters,
    envelope: BaseGeometry | None,
    out_path: Path,
    settings: ChipSettings,
) -> Path:
    """Draw the chips, context maps and time series of one sample unit.

    Layout: the NAIP RGB of every year (top row) and the CHM (middle row,
    unless ``show_chm`` is off) centred on the unit. The right column holds
    the run's onset map (top) and the decision probability of the unit's
    interval (middle) over a wider area, with the chip extent boxed. The
    bottom row holds the DI tested by the run's rule (with each date's
    decision cutoff) and the cell mean CHM height of the unit's cells as
    median lines with percentile bands.

    Args:
        sample: One row of the sample table, with ``geometry``.
        site_inputs: Inputs of the run, with NAIP paths.
        rasters: Cell rasters of the run.
        envelope: Refined envelope of the object, or None.
        out_path: PNG path.
        settings: Chip settings.

    Returns:
        ``out_path``.
    """
    setup_style()
    years = site_inputs.year_list
    n_years = len(years)
    n_image_rows = 2 if settings.show_chm else 1
    width_ratios = [1.0] * n_years + [SPACER_RATIO, 1.0]
    height_ratios = [1.0] * n_image_rows + [SERIES_ROW_RATIO]
    n_grid_cols = len(width_ratios)
    figure_width = CHIP_PANEL_IN * sum(width_ratios) + 0.8
    figure_height = CHIP_PANEL_IN * sum(height_ratios) + 0.9
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid_spec = fig.add_gridspec(
        len(height_ratios),
        n_grid_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.18,
        wspace=0.06,
    )
    geometry = sample["geometry"]
    bounds = chip_bounds(geometry, settings)
    chm_image = None
    last_chm_ax = None
    for col, year in enumerate(years):
        frame_color = _frame_color(year, sample)
        rgb_ax = fig.add_subplot(grid_spec[0, col])
        rgb, rgb_extent = read_rgb_chip(site_inputs, year, bounds)
        rgb_ax.imshow(rgb, extent=rgb_extent, interpolation="nearest")
        _draw_outlines(rgb_ax, geometry, None)
        rgb_ax.set_xlim(bounds[0], bounds[2])
        rgb_ax.set_ylim(bounds[1], bounds[3])
        title = _year_title(site_inputs, year)
        title_weight = "normal" if frame_color is None else "bold"
        rgb_ax.set_title(title, fontsize=8, color=INK, fontweight=title_weight)
        _frame_panel(rgb_ax, frame_color)
        if col == 0:
            helpers.scale_bar(rgb_ax, bounds)
        if not settings.show_chm:
            continue
        chm_ax = fig.add_subplot(grid_spec[1, col])
        chm, chm_transform = windows.read_chm_bounds(site_inputs, year, bounds)
        chm_extent = helpers.transform_extent(chm_transform, chm.shape)
        chm_image = chm_ax.imshow(
            chm,
            extent=chm_extent,
            cmap=canopy_cmap(),
            vmin=0.0,
            vmax=CHM_VMAX_M,
            interpolation="nearest",
        )
        _draw_outlines(chm_ax, geometry, envelope)
        chm_ax.set_xlim(bounds[0], bounds[2])
        chm_ax.set_ylim(bounds[1], bounds[3])
        _frame_panel(chm_ax, frame_color)
        last_chm_ax = chm_ax
    if chm_image is not None:
        chm_colorbar_ax = last_chm_ax.inset_axes(COLORBAR_INSET)
        chm_colorbar = fig.colorbar(chm_image, cax=chm_colorbar_ax)
        chm_colorbar.set_label("CHM height (m)", fontsize=7)
        chm_colorbar.ax.tick_params(labelsize=6)
    context_col = n_grid_cols - 1
    view = chip_panels.context_bounds(
        bounds, settings.context_factor, settings.min_context_m
    )
    onset_ax = fig.add_subplot(grid_spec[0, context_col])
    chip_panels.draw_onset_context(onset_ax, rasters, geometry, bounds, view)
    onset_year = int(sample["onset_year"])
    if settings.show_chm:
        probability_ax = fig.add_subplot(grid_spec[1, context_col])
        probability_image = chip_panels.draw_probability_context(
            probability_ax, rasters, onset_year, geometry, bounds, view
        )
        probability_colorbar_ax = probability_ax.inset_axes(COLORBAR_INSET)
        probability_colorbar = fig.colorbar(
            probability_image, cax=probability_colorbar_ax
        )
        probability_colorbar.ax.tick_params(labelsize=6)
    series_polygon = geometry if envelope is None else envelope
    series_cells = chip_panels.cells_in_polygon(series_polygon, rasters.grid)
    di_cols = int(np.ceil(n_years * DI_SERIES_SHARE))
    di_ax = fig.add_subplot(grid_spec[-1, :di_cols])
    chip_panels.draw_di_series(di_ax, rasters, series_cells, onset_year)
    height_ax = fig.add_subplot(grid_spec[-1, di_cols + 1 :])
    sample_years = (int(sample["year_before"]), onset_year)
    chip_panels.draw_height_series(height_ax, rasters, series_cells, sample_years)
    unit_title = _unit_title(sample)
    fig.suptitle(unit_title, fontsize=9, color=INK)
    series_source = "refined envelope" if envelope is not None else "detected cells"
    caption = (
        "Yellow: detected 19.2 m cells. Orange (CHM row): refined 0.6 m envelope. "
        "Blue and red frames: images before and after the onset interval. "
        "Right column: run maps over a wider area, dashed box = chip extent.\n"
        f"Bottom: 19.2 m cells inside the {series_source}, median with 25-75 % "
        "and 10-90 % bands, onset interval shaded."
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=7, color=MUTED)
    return save_fig(fig, out_path, dpi=settings.dpi)


def plot_contact_sheet(
    samples: gpd.GeoDataFrame,
    site_inputs: SiteInputs,
    out_path: Path,
    settings: ChipSettings,
) -> Path:
    """Draw the before and after RGB chip of every unit on one page.

    Args:
        samples: The sample, with ``geometry``.
        site_inputs: Inputs of the run, with NAIP paths.
        out_path: PNG path.
        settings: Chip settings.

    Returns:
        ``out_path``.
    """
    setup_style()
    n_units = len(samples)
    n_rows = int(np.ceil(n_units / CONTACT_UNITS_PER_ROW))
    n_cols = 2 * CONTACT_UNITS_PER_ROW
    panel_in = CHIP_PANEL_IN * 0.75
    figure_size = (panel_in * n_cols, (panel_in + 0.35) * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size, squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for position, (_, sample) in enumerate(samples.iterrows()):
        row = position // CONTACT_UNITS_PER_ROW
        first_col = 2 * (position % CONTACT_UNITS_PER_ROW)
        geometry = sample["geometry"]
        bounds = chip_bounds(geometry, settings)
        pair_years = (int(sample["year_before"]), int(sample["onset_year"]))
        for offset, year in enumerate(pair_years):
            ax = axes[row, first_col + offset]
            ax.set_axis_on()
            rgb, extent = read_rgb_chip(site_inputs, year, bounds)
            ax.imshow(rgb, extent=extent, interpolation="nearest")
            helpers.outline(
                ax,
                [geometry],
                CELL_OUTLINE_COLOR,
                linewidth=CELL_OUTLINE_WIDTH,
                halo=None,
            )
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            frame_color = _frame_color(year, sample)
            _frame_panel(ax, frame_color)
            ax.set_xlabel(str(year), fontsize=7, labelpad=1)
        label = f"{sample['sample_id']} {sample['stratum']}, {sample['area_ha']:.1f} ha"
        axes[row, first_col].set_title(label, fontsize=7, loc="left", color=INK)
    return save_fig(fig, out_path, dpi=settings.dpi)


def chip_filename(sample: pd.Series) -> str:
    """File name of a unit's chip figure.

    Args:
        sample: One row of the sample table.

    Returns:
        For example ``"S03_detected_2016-2018.png"``.
    """
    return f"{sample['sample_id']}_{sample['stratum']}_{sample['interval']}.png"


def write_index_markdown(
    samples: gpd.GeoDataFrame,
    run_dir: Path,
    design: SampleDesign,
    out_path: Path,
) -> Path:
    """Write ``index.md``: the contact sheet and a table linking every chip.

    Args:
        samples: The sample.
        run_dir: The disturbance run folder.
        design: The sample design.
        out_path: Markdown path (links are relative to its folder).

    Returns:
        ``out_path``.
    """
    stratum_counts = samples["stratum"].value_counts()
    count_parts = [f"{count} {name}" for name, count in stratum_counts.items()]
    count_text = ", ".join(count_parts)
    lines = [
        f"# Inspection chips: {run_dir.name}",
        "",
        f"Run folder: `{run_dir}`. Sample seed {design.seed}, units: {count_text}. "
        "Detected objects are stratified by NAIP interval with "
        f"at least {design.min_per_interval} per interval. Record a label and notes "
        "for each unit in `samples.csv`.",
        "",
        "![Contact sheet](contact_sheet.png)",
        "",
        "| Sample | Stratum | Interval | Area (ha) | Max P | Object | Lat, lon |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, sample in samples.iterrows():
        chip_name = chip_filename(sample)
        chip_link = f"chips/{chip_name}"
        max_prob_text = (
            "" if np.isnan(sample["max_prob"]) else f"{sample['max_prob']:.2f}"
        )
        object_text = (
            "" if pd.isna(sample["object_id"]) else str(int(sample["object_id"]))
        )
        location_text = f"{sample['centroid_lat']:.5f}, {sample['centroid_lon']:.5f}"
        lines.append(
            f"| [{sample['sample_id']}]({chip_link}) | {sample['stratum']} | "
            f"{sample['interval']} | {sample['area_ha']:.2f} | {max_prob_text} | "
            f"{object_text} | {location_text} |"
        )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    logger.info("Wrote %s", out_path)
    return out_path


def _envelopes_by_object(run_dir: Path) -> dict[int, BaseGeometry]:
    """Refined envelopes of a run keyed by object id (empty when absent).

    Args:
        run_dir: The disturbance run folder.

    Returns:
        ``{object_id: envelope}``.
    """
    gpkg_path = run_dir / "detection" / "disturbances.gpkg"
    try:
        refined = gpd.read_file(gpkg_path, layer=REFINED_OUTLINES_LAYER)
    except Exception:  # noqa: BLE001 - layer missing when refinement was off
        logger.info("No refined outlines in %s", gpkg_path)
        return {}
    object_ids = refined["object_id"].astype(int)
    return dict(zip(object_ids, refined.geometry, strict=True))


def make_inspection_chips(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    design: SampleDesign | None = None,
    settings: ChipSettings | None = None,
    sample_only: bool = False,
) -> InspectionResult:
    """Draw an inspection sample from a run and render its chips.

    Args:
        run_dir: The disturbance run folder.
        output_dir: New folder for the outputs (None = ``<run_dir>/inspection``).
        design: Sample design (None = :class:`SampleDesign` defaults).
        settings: Chip settings (None = :class:`ChipSettings` defaults).
        sample_only: Write the sample table only, without reading imagery.

    Returns:
        The written files.

    Raises:
        FileExistsError: If ``output_dir`` already exists.
        FileNotFoundError: If the run folder or a year's NAIP files are missing.
    """
    if design is None:
        design = SampleDesign()
    if settings is None:
        settings = ChipSettings()
    if output_dir is None:
        output_dir = run_dir / DEFAULT_FOLDER_NAME
    saved = load_run(run_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output folder {output_dir} already exists. Choose a new folder "
            "with --output-dir."
        )
    output_dir.mkdir(parents=True)
    samples = draw_samples(saved, design)
    vector_path = output_dir / "samples.gpkg"
    samples.to_file(vector_path, layer="samples", driver="GPKG")
    attributes = samples.drop(columns="geometry")
    table = pd.DataFrame(attributes)
    chip_names = [chip_filename(row) for _, row in samples.iterrows()]
    table["chip"] = [f"chips/{name}" for name in chip_names]
    for column in INTERPRETATION_COLUMNS:
        table[column] = ""
    table_path = output_dir / "samples.csv"
    table.to_csv(table_path, index=False)
    logger.info("Wrote %s (%d units)", table_path, len(table))
    if sample_only:
        return InspectionResult(
            output_dir, samples, table_path, vector_path, [], None, None
        )
    site_inputs = discover_with_saved_grid(
        saved.site, saved.params, saved.grid, need_naip=True
    )
    envelopes = _envelopes_by_object(run_dir)
    rasters = load_run_rasters(saved)
    chips_dir = output_dir / "chips"
    chip_paths = []
    for _, sample in samples.iterrows():
        envelope = None
        is_detected = sample["stratum"] == STRATUM_DETECTED
        if is_detected:
            object_id = int(sample["object_id"])
            envelope = envelopes.get(object_id)
        chip_path = chips_dir / chip_filename(sample)
        plot_sample_chips(sample, site_inputs, rasters, envelope, chip_path, settings)
        chip_paths.append(chip_path)
    contact_sheet_path = output_dir / "contact_sheet.png"
    contact_sheet = plot_contact_sheet(
        samples, site_inputs, contact_sheet_path, settings
    )
    index_md_path = output_dir / "index.md"
    index_path = write_index_markdown(samples, run_dir, design, index_md_path)
    return InspectionResult(
        output_dir,
        samples,
        table_path,
        vector_path,
        chip_paths,
        contact_sheet,
        index_path,
    )


__all__ = [
    "DEFAULT_FOLDER_NAME",
    "ChipSettings",
    "InspectionResult",
    "chip_bounds",
    "chip_filename",
    "make_inspection_chips",
    "plot_contact_sheet",
    "plot_sample_chips",
    "read_rgb_chip",
    "write_index_markdown",
]
