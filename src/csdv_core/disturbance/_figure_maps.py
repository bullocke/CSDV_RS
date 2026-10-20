"""csdv_core.disturbance._figure_maps: scene maps of a disturbance run.

Forest, onset, z and probability maps behind the public functions of
:mod:`csdv_core.disturbance.figures`. Cell rasters are drawn in grid CRS
coordinates through ``imshow(extent=...)`` with nearest-neighbour display,
over a plain background or a grey backdrop (NAIP, or cell CHM height on the
onset maps).

Inputs: cell rasters and year stacks on the common grid. Outputs: PNG files.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.patches import Patch

from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance._figure_helpers import (
    DEFAULT_DPI,
    NO_DATA_COLOR,
    NaipBackdrop,
)
from csdv_core.disturbance.decide import ONSET_YEAR_NONE, ONSET_YEAR_UNMONITORED
from csdv_core.disturbance.inputs import CellGrid
from csdv_core.disturbance.outlines import interval_label
from csdv_core.viz.style import INK, MUTED, save_fig

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

logger = logging.getLogger(__name__)

#: Diverging colormap for z (red = height loss).
Z_CMAP = "RdBu"
#: Symmetric colour limit for z maps.
Z_COLOR_LIMIT = 4.0
#: Probability ramp below the decision boundary (neutral, pale to warm grey).
PROBABILITY_BELOW: tuple[str, ...] = ("#f7f6f2", "#d8d3c6")
#: Probability ramp above the decision boundary (orange to dark red).
PROBABILITY_ABOVE: tuple[str, ...] = ("#f6a54c", "#e0522f", "#8f1d2c")
#: Legend swatch of never-monitored ground washed white over a backdrop.
BACKDROP_UNMONITORED_SWATCH = "#f2f2f2"
#: Width of the single-map figures, inches.
SINGLE_MAP_WIDTH_IN = 6.5


def _start_map(
    ax: Axes, grid: CellGrid, backdrop: NaipBackdrop | None
) -> tuple[float, float, float, float]:
    """Prepare a map axis: background, optional grey backdrop, limits, no ticks.

    Args:
        ax: Axis to draw on.
        grid: Common grid (sets the limits).
        backdrop: Backdrop drawn in grey, or None for a plain background.

    Returns:
        The grid extent ``(left, right, bottom, top)``.
    """
    extent = grid.extent()
    ax.set_facecolor(helpers.MAP_BACKGROUND)
    if backdrop is not None:
        gray = helpers.gray_image(backdrop.rgb)
        ax.imshow(
            gray,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=backdrop.extent,
            interpolation="nearest",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    helpers.bare(ax, frame=True)
    return extent


def _show_cells(
    ax: Axes,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    **kwargs: Any,
) -> AxesImage:
    """Draw a cell raster in map coordinates without smoothing.

    Args:
        ax: Map axis.
        image: ``(rows, cols)`` or RGBA ``(rows, cols, 4)`` cell array.
        extent: ``(left, right, bottom, top)``.
        **kwargs: Passed on to ``imshow``.

    Returns:
        The image artist.
    """
    return ax.imshow(image, extent=extent, interpolation="nearest", **kwargs)


def draw_forest(
    ax: Axes,
    forest: np.ndarray,
    grid: CellGrid,
    *,
    backdrop: NaipBackdrop | None = None,
) -> None:
    """Draw a bool ``(rows, cols)`` forest mask (forest green, rest clear).

    Args:
        ax: Map axis.
        forest: bool ``(rows, cols)``.
        grid: Common grid.
        backdrop: Grey backdrop drawn underneath, or None.
    """
    extent = _start_map(ax, grid, backdrop)
    forest_codes = forest.astype(np.int8)
    alpha = 0.7 if backdrop is not None else 1.0
    forest_rgba = helpers.category_rgba(forest_codes, {1: helpers.FOREST_COLOR})
    forest_rgba[..., 3] *= alpha
    _show_cells(ax, forest_rgba, extent)


def _forest_ha(forest: np.ndarray, grid: CellGrid) -> float:
    """Forest area of a bool mask in hectares.

    Args:
        forest: bool ``(rows, cols)``.
        grid: Common grid.

    Returns:
        Area in hectares.
    """
    n_forest_cells = np.count_nonzero(forest)
    return n_forest_cells * grid.cell_ha


def plot_forest_initial(
    forest: np.ndarray,
    grid: CellGrid,
    year: int,
    out_path: Path,
    *,
    backdrop: NaipBackdrop | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Forest at the first year (``forest[0]``, bool ``(rows, cols)``).

    Args:
        forest: bool ``(rows, cols)`` forest mask of ``year``.
        grid: Common grid.
        year: NAIP year of the mask.
        out_path: PNG to write.
        backdrop: Grey backdrop drawn underneath, or None.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    extent = grid.extent()
    figure, axes = helpers.scene_axes(1, extent, panel_width=SINGLE_MAP_WIDTH_IN)
    ax = axes[0]
    draw_forest(ax, forest, grid, backdrop=backdrop)
    helpers.scale_bar(ax, grid.bounds)
    forest_ha = _forest_ha(forest, grid)
    ax.set_title(f"Forest in {year}: {forest_ha:,.0f} ha ({grid.cell_m:g} m cells)")
    return save_fig(figure, out_path, dpi=dpi)


def plot_forest_by_year(
    forest: np.ndarray,
    grid: CellGrid,
    years: Sequence[int],
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One forest panel per year, bool ``(T, rows, cols)``, titled with area (ha).

    Args:
        forest: bool ``(T, rows, cols)``.
        grid: Common grid.
        years: NAIP years, length ``T``.
        out_path: PNG to write.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    extent = grid.extent()
    n_years = len(years)
    figure, axes = helpers.scene_axes(n_years, extent)
    for date_index, year in enumerate(years):
        ax = axes[date_index]
        draw_forest(ax, forest[date_index], grid)
        forest_ha = _forest_ha(forest[date_index], grid)
        ax.set_title(f"{year}: {forest_ha:,.0f} ha")
    helpers.scale_bar(axes[0], grid.bounds)
    return save_fig(figure, out_path, dpi=dpi)


def _neutral_colors(backdrop: NaipBackdrop | None) -> dict[int, object]:
    """Colours of the onset codes without a detection.

    Args:
        backdrop: Backdrop of the map, or None.

    Returns:
        Code to colour. Without a backdrop, monitored forest without detection
        is grey and never-monitored ground stays clear. On a backdrop,
        monitored forest stays clear so the backdrop shows, and never-monitored
        ground is washed white.
    """
    if backdrop is None:
        return {ONSET_YEAR_NONE: helpers.MONITORED_FILL}
    return {ONSET_YEAR_UNMONITORED: helpers.UNMONITORED_WASH}


def draw_onset(
    ax: Axes,
    onset_year: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    *,
    backdrop: NaipBackdrop | None = None,
) -> None:
    """Draw onset years (int16 ``(rows, cols)``) with one colour per year.

    Codes 0 (monitored, no detection) and -1 (never monitored) follow
    :func:`_neutral_colors`.

    Args:
        ax: Map axis.
        onset_year: int16 ``(rows, cols)``, onset year, 0 or -1.
        years: NAIP years.
        grid: Common grid.
        backdrop: Grey backdrop drawn underneath, or None.
    """
    extent = _start_map(ax, grid, backdrop)
    onset_years = years[1:]
    n_intervals = len(onset_years)
    year_colors = helpers.interval_colors(n_intervals)
    year_color_pairs = zip(onset_years, year_colors, strict=True)
    colors: dict[int, object] = dict(year_color_pairs)
    neutral_colors = _neutral_colors(backdrop)
    colors.update(neutral_colors)
    onset_rgba = helpers.category_rgba(onset_year, colors)
    _show_cells(ax, onset_rgba, extent)


def _interval_area_label(
    onset_year: np.ndarray, year_before: int, year: int, grid: CellGrid
) -> str:
    """Label an interval with its detected area, e.g. ``"2016-2018: 12.3 ha"``.

    Args:
        onset_year: int16 ``(rows, cols)``.
        year_before: Earlier image year.
        year: Later image year (the onset year).
        grid: Common grid.

    Returns:
        The label.
    """
    detected = onset_year == year
    n_cells = np.count_nonzero(detected)
    area_ha = n_cells * grid.cell_ha
    label = interval_label(year_before, year)
    return f"{label}: {area_ha:,.1f} ha"


def _onset_legend_handles(
    onset_year: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    backdrop: NaipBackdrop | None,
) -> list[Patch]:
    """Legend patches: one per interval with its area, plus the neutral classes.

    Args:
        onset_year: int16 ``(rows, cols)``.
        years: NAIP years.
        grid: Common grid.
        backdrop: Backdrop or None (changes the neutral swatches).

    Returns:
        Legend handles.
    """
    n_intervals = len(years) - 1
    year_colors = helpers.interval_colors(n_intervals)
    handles = []
    for year_before, year, color in zip(
        years[:-1], years[1:], year_colors, strict=True
    ):
        label = _interval_area_label(onset_year, year_before, year, grid)
        patch = Patch(facecolor=color, label=label)
        handles.append(patch)
    if backdrop is None:
        monitored_patch = Patch(
            facecolor=helpers.MONITORED_FILL, label="Monitored forest, no detection"
        )
        unmonitored_color = helpers.MAP_BACKGROUND
    else:
        monitored_patch = Patch(
            facecolor="none",
            edgecolor=MUTED,
            linewidth=0.5,
            hatch="////",
            label="Monitored forest, no detection (backdrop shows)",
        )
        unmonitored_color = BACKDROP_UNMONITORED_SWATCH
    unmonitored_patch = Patch(
        facecolor=unmonitored_color,
        edgecolor=MUTED,
        linewidth=0.5,
        label="Never monitored",
    )
    handles.extend([monitored_patch, unmonitored_patch])
    return handles


def plot_disturbance_onset(
    onset_year: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    out_path: Path,
    *,
    backdrop: NaipBackdrop | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Single map of onset years with a categorical legend.

    Args:
        onset_year: int16 ``(rows, cols)`` after the MMU.
        years: NAIP years.
        grid: Common grid.
        out_path: PNG to write.
        backdrop: Grey backdrop drawn underneath, or None.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    extent = grid.extent()
    figure, axes = helpers.scene_axes(1, extent, panel_width=SINGLE_MAP_WIDTH_IN)
    ax = axes[0]
    draw_onset(ax, onset_year, years, grid, backdrop=backdrop)
    helpers.scale_bar(ax, grid.bounds)
    handles = _onset_legend_handles(onset_year, years, grid, backdrop)
    ax.legend(
        handles=handles,
        title="Disturbance onset",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        alignment="left",
    )
    ax.set_title("Disturbance onset by NAIP interval")
    return save_fig(figure, out_path, dpi=dpi)


def plot_onset_by_interval(
    onset_year: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    out_path: Path,
    *,
    backdrop: NaipBackdrop | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One panel per interval (``years[1:]``) showing that interval's detections.

    Args:
        onset_year: int16 ``(rows, cols)`` after the MMU.
        years: NAIP years.
        grid: Common grid.
        out_path: PNG to write.
        backdrop: Grey backdrop drawn underneath, or None.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    extent = grid.extent()
    n_intervals = len(years) - 1
    year_colors = helpers.interval_colors(n_intervals)
    figure, axes = helpers.scene_axes(n_intervals, extent)
    neutral_colors = _neutral_colors(backdrop)
    for ax, year_before, year, color in zip(
        axes, years[:-1], years[1:], year_colors, strict=True
    ):
        _start_map(ax, grid, backdrop)
        interval_codes = np.where(onset_year == year, 1, onset_year)
        panel_colors = dict(neutral_colors)
        panel_colors[1] = color
        panel_rgba = helpers.category_rgba(interval_codes, panel_colors)
        _show_cells(ax, panel_rgba, extent)
        title = _interval_area_label(onset_year, year_before, year, grid)
        ax.set_title(title)
    helpers.scale_bar(axes[0], grid.bounds)
    return save_fig(figure, out_path, dpi=dpi)


def _interval_maps(
    stack: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    *,
    colormap: Colormap,
    norm: Normalize,
) -> tuple[Figure, list[Axes], AxesImage]:
    """Draw ``stack[1:]`` as one continuous map panel per interval.

    Args:
        stack: float ``(T, rows, cols)``. NaN cells take :data:`NO_DATA_COLOR`.
        years: NAIP years.
        grid: Common grid.
        colormap: Colormap of the values.
        norm: Value-to-colour normalization.

    Returns:
        ``(figure, axes, last image)`` for adding a colour bar.
    """
    extent = grid.extent()
    n_intervals = len(years) - 1
    figure, axes = helpers.scene_axes(n_intervals, extent)
    masked_colormap = colormap.copy()
    masked_colormap.set_bad(NO_DATA_COLOR)
    image = None
    for date_index, ax in enumerate(axes, start=1):
        _start_map(ax, grid, None)
        image = _show_cells(
            ax, stack[date_index], extent, cmap=masked_colormap, norm=norm
        )
        title = interval_label(years[date_index - 1], years[date_index])
        ax.set_title(title)
    helpers.scale_bar(axes[0], grid.bounds)
    return figure, axes, image


def plot_z_maps(
    z: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    out_path: Path,
    *,
    color_limit: float = Z_COLOR_LIMIT,
    monitored: np.ndarray | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One diverging z panel per interval (``z[1:]``), limits +/- ``color_limit``.

    NaN cells, and cells outside ``monitored`` when it is given, are drawn in
    :data:`NO_DATA_COLOR`.

    Args:
        z: float32 ``(T, rows, cols)``.
        years: NAIP years.
        grid: Common grid.
        out_path: PNG to write.
        color_limit: Symmetric colour limit.
        monitored: bool ``(T, rows, cols)`` used to mask z, or None.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    shown_z = z
    if monitored is not None:
        shown_z = np.where(monitored, z, np.nan)
    colormap = plt.get_cmap(Z_CMAP)
    norm = Normalize(vmin=-color_limit, vmax=color_limit)
    figure, axes, image = _interval_maps(
        shown_z, years, grid, colormap=colormap, norm=norm
    )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.7, pad=0.02, extend="both")
    colorbar.set_label("Standardized change (z), negative = loss")
    return save_fig(figure, out_path, dpi=dpi)


def probability_colormap() -> ListedColormap:
    """Return the two-part probability ramp, split at its midpoint.

    The lower half runs from pale to warm grey, the upper half from orange to
    dark red. With a :class:`~matplotlib.colors.TwoSlopeNorm` centred on the
    decision boundary, every cell above the boundary is plainly coloured and
    every cell below it stays neutral.

    Returns:
        A 256-colour colormap.
    """
    below_ramp = LinearSegmentedColormap.from_list("below", PROBABILITY_BELOW)
    above_ramp = LinearSegmentedColormap.from_list("above", PROBABILITY_ABOVE)
    half_positions = np.linspace(0.0, 1.0, 128)
    below_colors = below_ramp(half_positions)
    above_colors = above_ramp(half_positions)
    all_colors = np.vstack([below_colors, above_colors])
    return ListedColormap(all_colors, name="csdv_probability")


def plot_probability_maps(
    probability: np.ndarray,
    years: Sequence[int],
    grid: CellGrid,
    out_path: Path,
    *,
    p_boundary: float,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One probability panel per interval (``probability[1:]``) in [0, 1].

    The ``p_boundary`` level is marked on the colour bar. The colour ramp
    changes from neutral to warm at ``p_boundary`` (see
    :func:`probability_colormap`), which marks the boundary on the maps
    without contour lines around single cells.

    Args:
        probability: float32 ``(T, rows, cols)``.
        years: NAIP years.
        grid: Common grid.
        out_path: PNG to write.
        p_boundary: Decision boundary, strictly between 0 and 1.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    colormap = probability_colormap()
    norm = TwoSlopeNorm(vcenter=p_boundary, vmin=0.0, vmax=1.0)
    figure, axes, image = _interval_maps(
        probability, years, grid, colormap=colormap, norm=norm
    )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.7, pad=0.02)
    colorbar.set_label(f"Decision probability (line: boundary {p_boundary:g})")
    colorbar.ax.axhline(p_boundary, color=INK, linewidth=1.2)
    tick_values = [0.0, p_boundary, 1.0]
    colorbar.set_ticks(tick_values)
    return save_fig(figure, out_path, dpi=dpi)
