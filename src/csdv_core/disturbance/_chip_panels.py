"""csdv_core.disturbance._chip_panels: context maps and time series of a chip figure.

Private helpers of :mod:`csdv_core.disturbance.chips`. They draw, for one
sampled unit:

- the onset map of the run over an area wider than the chips, with the unit
  outlined and the chip extent boxed
- the decision probability of the unit's interval over the same area (the z
  of that interval for rule D1, which has no probability)
- the DI tested by the run's rule, for the 19.2 m cells inside the unit and
  every interval, as a median line with 25-75 % and 10-90 % bands and the
  date's decision cutoff. For the spatial rules (D5, D6, D7 and their strict
  variants) this is the z smoothed at the broadest spatial scale, with the
  smoothed z where the calibrated probability reaches ``p_boundary``. For D2
  it is the raw z with its own probability cutoff, and for D1 the raw z with
  the fixed ``-baseline_z`` cut
- the mean CHM height of the same cells for every year, drawn the same way

The cells are those whose centres fall inside the refined 0.6 m envelope, or
inside the detected cells when the object has no envelope. A small envelope
that covers no cell centre falls back to every cell it touches.

Inputs: a saved run and one unit. Outputs: drawings on matplotlib axes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

from csdv_core.change import di_decide
from csdv_core.change.di_decide import LfdrModel
from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance._figure_maps import (
    Z_CMAP,
    Z_COLOR_LIMIT,
    probability_colormap,
)
from csdv_core.disturbance.decide import (
    ONSET_YEAR_NONE,
    ONSET_YEAR_UNMONITORED,
    fit_date_models,
)
from csdv_core.disturbance.inputs import CellGrid
from csdv_core.disturbance.layers import monitored_mask
from csdv_core.disturbance.outlines import interval_label
from csdv_core.disturbance.standardize import sigmas_in_cells
from csdv_core.disturbance.store import SavedRun, read_year_stack
from csdv_core.viz.style import INK, MUTED

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable

#: Percentile bands of the time series, outer band first.
SERIES_BANDS: tuple[tuple[float, float], ...] = ((10.0, 90.0), (25.0, 75.0))
#: Opacity of each percentile band, matching :data:`SERIES_BANDS`.
SERIES_BAND_ALPHAS: tuple[float, ...] = (0.18, 0.35)
#: Colour of the DI series.
DI_SERIES_COLOR = "#2a6f7f"
#: Colour of the CHM height series.
HEIGHT_SERIES_COLOR = "#2f7d4f"
#: Fill of the onset interval in the time series.
ONSET_SPAN_COLOR = "#b4553a"
#: Colour of the per-date decision cutoff on the DI series.
CUTOFF_COLOR = "#b4553a"
#: Half width of the cutoff tick at each date, in date positions.
CUTOFF_HALF_WIDTH = 0.35
#: Rules whose decision tests the spatially smoothed z.
SMOOTHED_RULES: frozenset[str] = frozenset({"D5", "D5s", "D6", "D6s", "D7", "D7s"})
#: Spacing of the z grid searched for the probability cutoff.
CUTOFF_GRID_STEP = 0.01
#: Lowest z searched for the probability cutoff.
CUTOFF_GRID_MIN = -15.0
#: Outline colour of the unit on the context maps.
CONTEXT_OUTLINE_COLOR = "black"


@dataclass(frozen=True)
class DecisionSeries:
    """The DI a rule tests, with the cutoff of every date.

    Attributes:
        values: float32 ``(T, rows, cols)`` tested DI, NaN where undefined.
        cutoffs: Per-date DI at or below which the rule flags a cell on its
            own, NaN for dates without a fit (index 0, skipped dates).
        title: Chart title fragment naming the DI.
        cutoff_label: Legend label of the cutoff.
    """

    values: np.ndarray
    cutoffs: tuple[float, ...]
    title: str
    cutoff_label: str


@dataclass(frozen=True)
class RunRasters:
    """Cell-level rasters of a run, loaded once for all chip figures.

    Attributes:
        grid: Common grid of the run.
        years: NAIP years, length ``T``.
        z: float32 ``(T, rows, cols)`` DI, NaN where undefined.
        chm_height: float32 ``(T, rows, cols)`` cell mean CHM height (m).
        onset_year: int16 ``(rows, cols)`` onset year, 0 none, -1 never
            monitored.
        probability: float32 ``(T, rows, cols)`` decision probability, or
            None for rule D1.
        p_boundary: Decision boundary of the run.
        series: DI tested by the run's rule, for the DI time series.
    """

    grid: CellGrid
    years: tuple[int, ...]
    z: np.ndarray
    chm_height: np.ndarray
    onset_year: np.ndarray
    probability: np.ndarray | None
    p_boundary: float
    series: DecisionSeries


def probability_cutoff(model: LfdrModel, p_boundary: float) -> float:
    """Highest z whose calibrated loss probability exceeds the boundary.

    The probability can only rise as z moves below the null centre, so every
    z at or below the cutoff passes and every z above it does not.

    Args:
        model: Fitted lfdr model of one date.
        p_boundary: Decision boundary.

    Returns:
        The cutoff z, or NaN when no z on the grid passes.
    """
    z_grid = np.arange(CUTOFF_GRID_MIN, model.null.mu0, CUTOFF_GRID_STEP)
    probability = model.prob(z_grid)
    passing = z_grid[probability > p_boundary]
    if passing.size == 0:
        return float("nan")
    return float(passing.max())


def _smoothed_series(
    saved: SavedRun, monitored: np.ndarray, skipped: Sequence[bool]
) -> DecisionSeries:
    """z smoothed at the broadest spatial scale, with its probability cutoffs.

    Each date is smoothed and fitted as in
    :func:`csdv_core.change.di_decide.spatial_prob`, so the cutoff is the one
    that scale applies in the run.

    Args:
        saved: The saved run.
        monitored: bool ``(T, rows, cols)``.
        skipped: Per date, True when the run skipped the date.

    Returns:
        The smoothed series.
    """
    params = saved.params
    sigmas_cells = sigmas_in_cells(params, saved.grid.cell_m)
    sigma_cells = max(sigmas_cells)
    sigma_m = max(params.decide.spatial_sigma_m)
    values = np.full(saved.z.shape, np.nan, dtype="float32")
    cutoffs = []
    for date_index, date_skipped in enumerate(skipped):
        if date_skipped:
            cutoffs.append(float("nan"))
            continue
        date_z = saved.z[date_index]
        finite = np.isfinite(date_z)
        where = monitored[date_index] & finite
        smoothed = di_decide.smooth_z(date_z, where, sigma_cells)
        model = di_decide.lfdr_model(
            smoothed,
            where,
            bin_w=params.decide.bin_w,
            smooth_bins=params.decide.smooth_bins,
        )
        values[date_index] = smoothed
        cutoff = probability_cutoff(model, params.decide.p_boundary)
        cutoffs.append(cutoff)
    return DecisionSeries(
        values=values,
        cutoffs=tuple(cutoffs),
        title=f"DI smoothed at {sigma_m:g} m (z)",
        cutoff_label=f"P = {params.decide.p_boundary:g} cutoff",
    )


def decision_series(saved: SavedRun) -> DecisionSeries:
    """The DI the run's rule tests, with the cutoff of every date.

    Args:
        saved: The saved run.

    Returns:
        Smoothed z for the spatial rules, raw z for D1 and D2.
    """
    params = saved.params
    rule = params.decide.rule
    monitored = monitored_mask(saved.layers.forest)
    models, _ = fit_date_models(
        saved.z, monitored, params, saved.years, warn_skipped=False
    )
    skipped = [model is None for model in models]
    if rule in SMOOTHED_RULES:
        return _smoothed_series(saved, monitored, skipped)
    if rule == "D1":
        cut = -params.decide.baseline_z
        cutoffs = [float("nan") if is_skipped else cut for is_skipped in skipped]
        cutoff_label = f"z = {cut:g} (rule D1)"
    else:
        cutoffs = []
        for model in models:
            if model is None:
                cutoffs.append(float("nan"))
                continue
            cutoff = probability_cutoff(model, params.decide.p_boundary)
            cutoffs.append(cutoff)
        cutoff_label = f"P = {params.decide.p_boundary:g} cutoff"
    return DecisionSeries(
        values=saved.z,
        cutoffs=tuple(cutoffs),
        title="DI (z)",
        cutoff_label=cutoff_label,
    )


def load_run_rasters(saved: SavedRun) -> RunRasters:
    """Collect the cell rasters of a saved run for the chip figures.

    Args:
        saved: The saved run.

    Returns:
        The run's rasters.
    """
    run_paths = saved.run_paths
    with rasterio.open(run_paths.onset_year) as src:
        onset_year = src.read(1)
    probability = None
    if run_paths.probability.exists():
        probability_stack = read_year_stack(run_paths.probability)
        probability = probability_stack.values
    series = decision_series(saved)
    return RunRasters(
        grid=saved.grid,
        years=saved.years,
        z=saved.z,
        chm_height=saved.layers.chm_height,
        onset_year=onset_year,
        probability=probability,
        p_boundary=saved.params.decide.p_boundary,
        series=series,
    )


def cells_in_polygon(geometry: BaseGeometry, grid: CellGrid) -> np.ndarray:
    """Cells of the grid whose centres fall inside a polygon.

    Args:
        geometry: Polygon in grid CRS units.
        grid: Common grid.

    Returns:
        Boolean ``(rows, cols)``. When no cell centre is inside, every cell
        the polygon touches.
    """
    shapes = [geometry]
    inside = geometry_mask(
        shapes,
        out_shape=grid.cell_shape,
        transform=grid.cell_transform,
        invert=True,
    )
    if inside.any():
        return inside
    return geometry_mask(
        shapes,
        out_shape=grid.cell_shape,
        transform=grid.cell_transform,
        invert=True,
        all_touched=True,
    )


def context_bounds(
    chip_box: tuple[float, float, float, float], factor: float, min_side_m: float
) -> tuple[float, float, float, float]:
    """Square bounds of the context maps, centred on the chip.

    Args:
        chip_box: ``(left, bottom, right, top)`` of the chips.
        factor: Context side as a multiple of the chip side.
        min_side_m: Smallest context side, in metres.

    Returns:
        ``(left, bottom, right, top)``.
    """
    left, bottom, right, top = chip_box
    chip_side = right - left
    half_side = max(factor * chip_side, min_side_m) / 2
    centre_x = (left + right) / 2
    centre_y = (bottom + top) / 2
    return (
        centre_x - half_side,
        centre_y - half_side,
        centre_x + half_side,
        centre_y + half_side,
    )


def _finish_context(
    ax: Axes,
    geometry: BaseGeometry,
    chip_box: tuple[float, float, float, float],
    view: tuple[float, float, float, float],
) -> None:
    """Outline the unit, box the chip extent and zoom a context map.

    Args:
        ax: Map axis in grid CRS units.
        geometry: Unit polygon.
        chip_box: ``(left, bottom, right, top)`` of the chips.
        view: ``(left, bottom, right, top)`` shown.
    """
    helpers.outline(ax, [geometry], CONTEXT_OUTLINE_COLOR, linewidth=0.9, halo="white")
    left, bottom, right, top = chip_box
    chip_rectangle = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=INK,
        linestyle="--",
        linewidth=0.8,
        zorder=8,
    )
    ax.add_patch(chip_rectangle)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    helpers.bare(ax, frame=True)
    helpers.scale_bar(ax, view)


def draw_onset_context(
    ax: Axes,
    rasters: RunRasters,
    geometry: BaseGeometry,
    chip_box: tuple[float, float, float, float],
    view: tuple[float, float, float, float],
) -> None:
    """Onset map of the run around the unit, coloured by interval.

    Args:
        ax: Map axis.
        rasters: Run rasters.
        geometry: Unit polygon.
        chip_box: ``(left, bottom, right, top)`` of the chips.
        view: ``(left, bottom, right, top)`` shown.
    """
    years = rasters.years
    n_intervals = len(years) - 1
    colors = helpers.interval_colors(n_intervals)
    code_colors: dict[int, object] = {
        ONSET_YEAR_NONE: helpers.MONITORED_FILL,
        ONSET_YEAR_UNMONITORED: helpers.NO_DATA_COLOR,
    }
    legend_handles = []
    for interval_index in range(n_intervals):
        year_before = years[interval_index]
        year = years[interval_index + 1]
        color = colors[interval_index]
        code_colors[year] = color
        label = interval_label(year_before, year)
        legend_handles.append(Patch(facecolor=color, label=label))
    rgba = helpers.category_rgba(rasters.onset_year, code_colors)
    extent = rasters.grid.extent()
    ax.imshow(rgba, extent=extent, interpolation="nearest")
    _finish_context(ax, geometry, chip_box, view)
    ax.set_title("Detected onset (run map)", fontsize=8, color=INK)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=6,
        frameon=False,
        borderaxespad=0.0,
    )


def draw_probability_context(
    ax: Axes,
    rasters: RunRasters,
    onset_year: int,
    geometry: BaseGeometry,
    chip_box: tuple[float, float, float, float],
    view: tuple[float, float, float, float],
) -> ScalarMappable:
    """Decision probability (or z for rule D1) of the unit's interval.

    Args:
        ax: Map axis.
        rasters: Run rasters.
        onset_year: Later year of the unit's interval.
        geometry: Unit polygon.
        chip_box: ``(left, bottom, right, top)`` of the chips.
        view: ``(left, bottom, right, top)`` shown.

    Returns:
        The image, for a colour bar.
    """
    date_index = rasters.years.index(onset_year)
    year_before = rasters.years[date_index - 1]
    label = interval_label(year_before, onset_year)
    extent = rasters.grid.extent()
    if rasters.probability is None:
        colormap = colormaps[Z_CMAP].copy()
        colormap.set_bad(helpers.NO_DATA_COLOR)
        values = rasters.z[date_index]
        image = ax.imshow(
            values,
            extent=extent,
            cmap=colormap,
            vmin=-Z_COLOR_LIMIT,
            vmax=Z_COLOR_LIMIT,
            interpolation="nearest",
        )
        title = f"DI (z), {label}"
    else:
        colormap = probability_colormap()
        colormap.set_bad(helpers.NO_DATA_COLOR)
        norm = TwoSlopeNorm(vcenter=rasters.p_boundary, vmin=0.0, vmax=1.0)
        values = rasters.probability[date_index]
        image = ax.imshow(
            values,
            extent=extent,
            cmap=colormap,
            norm=norm,
            interpolation="nearest",
        )
        title = f"Decision probability, {label}"
    _finish_context(ax, geometry, chip_box, view)
    ax.set_title(title, fontsize=8, color=INK)
    return image


def _percentile_series(
    stack: np.ndarray, cells: np.ndarray, percentiles: Sequence[float]
) -> np.ndarray:
    """Percentiles over the selected cells for every date of a stack.

    Args:
        stack: ``(T, rows, cols)`` values with NaN gaps.
        cells: Boolean ``(rows, cols)`` cell selection.
        percentiles: Percentiles in [0, 100].

    Returns:
        ``(len(percentiles), T)`` float array, NaN for dates without values.
    """
    cell_values = stack[:, cells]
    n_dates = stack.shape[0]
    result = np.full((len(percentiles), n_dates), np.nan)
    for date_index in range(n_dates):
        date_values = cell_values[date_index]
        finite_values = date_values[np.isfinite(date_values)]
        if finite_values.size == 0:
            continue
        result[:, date_index] = np.percentile(finite_values, percentiles)
    return result


def _draw_bands(
    ax: Axes, positions: np.ndarray, stack: np.ndarray, cells: np.ndarray, color: str
) -> None:
    """Draw the percentile bands and median line of a stack over ``cells``.

    Args:
        ax: Chart axis.
        positions: x position of every date, length ``T``.
        stack: ``(T, rows, cols)`` values.
        cells: Boolean ``(rows, cols)`` cell selection.
        color: Line and band colour.
    """
    for (low, high), alpha in zip(SERIES_BANDS, SERIES_BAND_ALPHAS, strict=True):
        band = _percentile_series(stack, cells, (low, high))
        ax.fill_between(
            positions,
            band[0],
            band[1],
            color=color,
            alpha=alpha,
            linewidth=0,
            label=f"{low:.0f}-{high:.0f} %",
        )
    median = _percentile_series(stack, cells, (50.0,))
    ax.plot(positions, median[0], color=color, marker="o", markersize=3, label="Median")


def draw_di_series(
    ax: Axes, rasters: RunRasters, cells: np.ndarray, onset_year: int
) -> None:
    """Tested DI of the unit's cells for every interval, with the date cutoffs.

    The cutoff of each date is drawn as a short dashed tick. A cell whose DI
    lies below the tick passes the rule on its own. Cells above it can still
    be flagged through the plain rules' lean cell test.

    Args:
        ax: Chart axis.
        rasters: Run rasters.
        cells: Boolean ``(rows, cols)`` cells of the unit.
        onset_year: Later year of the unit's interval.
    """
    years = rasters.years
    positions = np.arange(len(years), dtype=float)
    series = rasters.series
    _draw_bands(ax, positions, series.values, cells, DI_SERIES_COLOR)
    cutoffs = np.asarray(series.cutoffs, dtype=float)
    ax.hlines(
        cutoffs,
        positions - CUTOFF_HALF_WIDTH,
        positions + CUTOFF_HALF_WIDTH,
        colors=CUTOFF_COLOR,
        linewidth=1.0,
        linestyles="--",
        label=series.cutoff_label,
    )
    onset_index = years.index(onset_year)
    ax.axvspan(onset_index - 0.4, onset_index + 0.4, color=ONSET_SPAN_COLOR, alpha=0.12)
    ax.axhline(0.0, color=MUTED, linewidth=0.6)
    labels = [""]
    for date_index in range(1, len(years)):
        label = interval_label(years[date_index - 1], years[date_index])
        labels.append(label)
    ax.set_xticks(positions[1:])
    ax.set_xticklabels(labels[1:], fontsize=7)
    ax.set_xlim(0.5, positions[-1] + 0.5)
    n_cells = int(cells.sum())
    cell_word = "cell" if n_cells == 1 else "cells"
    ax.set_title(
        f"{series.title} of the {n_cells} {cell_word} inside the unit", fontsize=8
    )
    ax.set_ylabel("DI (z)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=4)


def draw_height_series(
    ax: Axes, rasters: RunRasters, cells: np.ndarray, sample_years: tuple[int, int]
) -> None:
    """Cell mean CHM height of the unit's cells for every year.

    Args:
        ax: Chart axis.
        rasters: Run rasters.
        cells: Boolean ``(rows, cols)`` cells of the unit.
        sample_years: ``(year_before, onset_year)`` of the unit.
    """
    years = rasters.years
    positions = np.arange(len(years), dtype=float)
    _draw_bands(ax, positions, rasters.chm_height, cells, HEIGHT_SERIES_COLOR)
    before_index = years.index(sample_years[0])
    after_index = years.index(sample_years[1])
    ax.axvspan(before_index, after_index, color=ONSET_SPAN_COLOR, alpha=0.12)
    year_labels = [str(year) for year in years]
    ax.set_xticks(positions)
    ax.set_xticklabels(year_labels, fontsize=7)
    ax.set_title("Cell mean CHM height", fontsize=8)
    ax.set_ylabel("Height (m)", fontsize=7)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=7)


__all__ = [
    "DecisionSeries",
    "RunRasters",
    "cells_in_polygon",
    "context_bounds",
    "decision_series",
    "draw_di_series",
    "draw_height_series",
    "draw_onset_context",
    "draw_probability_context",
    "load_run_rasters",
    "probability_cutoff",
]
