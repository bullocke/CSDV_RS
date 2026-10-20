"""csdv_core.disturbance._chip_panels: context maps and time series of a chip figure.

Private helpers of :mod:`csdv_core.disturbance.chips`. They draw, for one
sampled unit:

- the onset map of the run over an area wider than the chips, with the unit
  outlined and the chip extent boxed
- the decision probability of the unit's interval over the same area (the z
  of that interval for rule D1, which has no probability)
- the DI (z) of the 19.2 m cells inside the unit for every interval, as a
  median line with 25-75 % and 10-90 % bands
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

from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance._figure_maps import (
    Z_CMAP,
    Z_COLOR_LIMIT,
    probability_colormap,
)
from csdv_core.disturbance.decide import ONSET_YEAR_NONE, ONSET_YEAR_UNMONITORED
from csdv_core.disturbance.inputs import CellGrid
from csdv_core.disturbance.outlines import interval_label
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
#: DI level marked on the series (the per-cell rule D1 cut).
DI_REFERENCE_Z = -3.0
#: Outline colour of the unit on the context maps.
CONTEXT_OUTLINE_COLOR = "black"


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
    """

    grid: CellGrid
    years: tuple[int, ...]
    z: np.ndarray
    chm_height: np.ndarray
    onset_year: np.ndarray
    probability: np.ndarray | None
    p_boundary: float


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
    return RunRasters(
        grid=saved.grid,
        years=saved.years,
        z=saved.z,
        chm_height=saved.layers.chm_height,
        onset_year=onset_year,
        probability=probability,
        p_boundary=saved.params.decide.p_boundary,
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
    """DI of the unit's cells for every interval, with the onset interval shaded.

    Args:
        ax: Chart axis.
        rasters: Run rasters.
        cells: Boolean ``(rows, cols)`` cells of the unit.
        onset_year: Later year of the unit's interval.
    """
    years = rasters.years
    positions = np.arange(len(years), dtype=float)
    _draw_bands(ax, positions, rasters.z, cells, DI_SERIES_COLOR)
    onset_index = years.index(onset_year)
    ax.axvspan(onset_index - 0.4, onset_index + 0.4, color=ONSET_SPAN_COLOR, alpha=0.12)
    ax.axhline(0.0, color=MUTED, linewidth=0.6)
    ax.axhline(DI_REFERENCE_Z, color=MUTED, linewidth=0.6, linestyle="--")
    ax.text(
        positions[-1] + 0.45,
        DI_REFERENCE_Z,
        f"z = {DI_REFERENCE_Z:g}",
        fontsize=6,
        color=MUTED,
        va="center",
    )
    labels = [""]
    for date_index in range(1, len(years)):
        label = interval_label(years[date_index - 1], years[date_index])
        labels.append(label)
    ax.set_xticks(positions[1:])
    ax.set_xticklabels(labels[1:], fontsize=7)
    ax.set_xlim(0.5, positions[-1] + 0.5)
    n_cells = int(cells.sum())
    cell_word = "cell" if n_cells == 1 else "cells"
    ax.set_title(f"DI (z) of the {n_cells} {cell_word} inside the unit", fontsize=8)
    ax.set_ylabel("DI (z)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=3)


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
    "RunRasters",
    "cells_in_polygon",
    "context_bounds",
    "draw_di_series",
    "draw_height_series",
    "draw_onset_context",
    "draw_probability_context",
    "load_run_rasters",
]
