"""csdv_core.disturbance._figure_helpers: drawing helpers for run figures.

Private helpers shared by the figure modules: panel layouts, a map-unit scale
bar, polygon outlines with a halo, a grey backdrop, the interval colours, the
shared figure constants and the :class:`NaipBackdrop` value object
(re-exported by ``figures``). Every map is drawn in grid CRS
coordinates through ``imshow(extent=...)``, so cell rasters, 0.6 m chips and
polygons overlay without resampling. Ported from the research helpers in
``analysis/NAIP_disturbance_detection/2_Disturbance_Index/_figs.py``.

Inputs: matplotlib axes, grid extents, geometries. Outputs: drawn artists.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import to_rgba
from shapely.geometry.base import BaseGeometry

from csdv_core.viz.style import INK

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Default figure resolution (dpi). A run uses ``figures.dpi``.
DEFAULT_DPI = 200
#: Colour of cells without a value (NaN) on maps and chips.
NO_DATA_COLOR = "#e6e6e3"
#: Categorical colours for onset intervals, in fixed order (validated palette).
INTERVAL_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
#: Colormap sampled when there are more intervals than categorical colours.
MANY_INTERVALS_CMAP = "viridis"
#: Map background where nothing is drawn (non-forest without a backdrop).
MAP_BACKGROUND = "#f4f3ef"
#: Fill of monitored forest without a detection, without a backdrop.
MONITORED_FILL = "#c9cdc6"
#: White wash laid over never-monitored ground on a NAIP backdrop.
UNMONITORED_WASH = (1.0, 1.0, 1.0, 0.6)
#: Forest colour on forest maps.
FOREST_COLOR = "#2f7d4f"
#: Lightness of the darkest pixel of the grey NAIP backdrop.
GRAY_OFFSET = 0.35
#: Lightness range of the grey NAIP backdrop.
GRAY_RANGE = 0.6
#: Candidate scale bar lengths in metres.
NICE_SCALE_LENGTHS_M: tuple[float, ...] = (
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
)
#: Colour of the thin frame round map panels.
FRAME_COLOR = "#9aa3a8"
#: Width of one small-multiple panel, inches.
PANEL_WIDTH_IN = 2.6
#: Most panels in one row of a small-multiple figure.
MAX_PANEL_COLUMNS = 4


@dataclass(frozen=True)
class NaipBackdrop:
    """An RGB image for map backgrounds (stretched NAIP or grey CHM height).

    Attributes:
        rgb: uint8 ``(rows, cols, 3)``.
        extent: ``(left, right, bottom, top)`` for ``imshow``.
    """

    rgb: np.ndarray
    extent: tuple[float, float, float, float]


def interval_colors(n_intervals: int) -> list[str | tuple[float, ...]]:
    """Return one colour per interval, in interval order.

    Args:
        n_intervals: Number of intervals.

    Returns:
        The categorical colours in fixed order, or samples of
        :data:`MANY_INTERVALS_CMAP` when there are more intervals than
        categorical colours (never cycled).
    """
    n_categorical = len(INTERVAL_COLORS)
    if n_intervals <= n_categorical:
        return list(INTERVAL_COLORS[:n_intervals])
    colormap = plt.get_cmap(MANY_INTERVALS_CMAP)
    positions = np.linspace(0.0, 0.9, n_intervals)
    return [colormap(position) for position in positions]


def bare(ax: Axes, *, frame: bool = False) -> None:
    """Remove ticks from a map axis and hide or thin its spines.

    Args:
        ax: Axis to clean.
        frame: Keep a thin grey frame instead of hiding the spines.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
        spine.set_linewidth(0.5)
        spine.set_color(FRAME_COLOR)


def panel_grid_shape(
    n_panels: int, max_columns: int = MAX_PANEL_COLUMNS
) -> tuple[int, int]:
    """Rows and columns for ``n_panels`` small multiples, rows filled evenly.

    Args:
        n_panels: Number of panels (at least 1).
        max_columns: Most panels per row.

    Returns:
        ``(n_rows, n_columns)``. Five panels give two rows of three rather
        than one row of four and one of one.
    """
    n_rows = math.ceil(n_panels / max_columns)
    n_columns = math.ceil(n_panels / n_rows)
    return n_rows, n_columns


def panel_axes(
    n_panels: int, panel_width: float, panel_height: float, *, extra_height: float
) -> tuple[Figure, list[Axes]]:
    """Create a grid of equal panels, rows filled by :func:`panel_grid_shape`.

    Args:
        n_panels: Number of panels needed.
        panel_width: Width of one panel in inches.
        panel_height: Height of one panel in inches.
        extra_height: Inches added to the figure height for titles and legends.

    Returns:
        ``(figure, axes)`` with exactly ``n_panels`` visible axes (spare grid
        cells are hidden).
    """
    n_rows, n_columns = panel_grid_shape(n_panels)
    figure_size = (panel_width * n_columns, panel_height * n_rows + extra_height)
    figure, axes_grid = plt.subplots(
        n_rows, n_columns, figsize=figure_size, squeeze=False, layout="constrained"
    )
    flat_axes = axes_grid.ravel().tolist()
    for spare_ax in flat_axes[n_panels:]:
        spare_ax.set_visible(False)
    return figure, flat_axes[:n_panels]


def scene_axes(
    n_panels: int,
    extent: tuple[float, float, float, float],
    *,
    panel_width: float = PANEL_WIDTH_IN,
) -> tuple[Figure, list[Axes]]:
    """Create a grid of map panels shaped like the scene.

    Args:
        n_panels: Number of panels needed.
        extent: Scene extent ``(left, right, bottom, top)``.
        panel_width: Width of one panel in inches.

    Returns:
        ``(figure, axes)`` from :func:`panel_axes`.
    """
    left, right, bottom, top = extent
    aspect = (top - bottom) / (right - left)
    panel_height = panel_width * aspect
    return panel_axes(n_panels, panel_width, panel_height, extra_height=0.5)


def scale_bar(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    *,
    length_m: float | None = None,
    color: str = INK,
    halo: str = "white",
) -> None:
    """Draw a scale bar in the lower left of a map axis, in map units.

    Args:
        ax: Map axis drawn in CRS metres.
        bounds: ``(left, bottom, right, top)`` of the shown area.
        length_m: Bar length (None picks a round length under a quarter of
            the width).
        color: Bar and label colour.
        halo: Outline colour that keeps the bar legible on any background.
    """
    left, bottom, right, top = bounds
    width = right - left
    height = top - bottom
    if length_m is None:
        fitting_lengths = [
            length for length in NICE_SCALE_LENGTHS_M if length <= width / 4
        ]
        if not fitting_lengths:
            logger.debug("Scene of %.0f m is too narrow for a scale bar", width)
            return
        length_m = fitting_lengths[-1]
    x_start = left + 0.05 * width
    y_start = bottom + 0.06 * height
    effects = [path_effects.withStroke(linewidth=3.5, foreground=halo)]
    ax.plot(
        [x_start, x_start + length_m],
        [y_start, y_start],
        color=color,
        lw=2,
        solid_capstyle="butt",
        path_effects=effects,
        zorder=8,
    )
    if length_m >= 1000:
        label = f"{length_m / 1000:g} km"
    else:
        label = f"{length_m:g} m"
    ax.text(
        x_start + length_m / 2,
        y_start + 0.02 * height,
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=7,
        path_effects=effects,
        zorder=8,
    )


def outline(
    ax: Axes,
    geometries: Iterable[BaseGeometry | None],
    color: str,
    *,
    linewidth: float = 1.0,
    linestyle: str | tuple = "-",
    halo: str | None = "white",
    halo_width: float = 1.4,
) -> None:
    """Draw polygon rings (exteriors and holes) on a map axis.

    Args:
        ax: Map axis drawn in CRS units.
        geometries: Polygons or multipolygons (None or empty ones skipped).
        color: Line colour.
        linewidth: Line width in points.
        linestyle: Matplotlib line style.
        halo: Colour of a wider stroke drawn under the line, or None.
        halo_width: Extra width of the halo stroke, in points.
    """
    effects = None
    if halo is not None:
        halo_stroke = path_effects.withStroke(
            linewidth=linewidth + halo_width, foreground=halo
        )
        effects = [halo_stroke]
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        parts = getattr(geometry, "geoms", [geometry])
        for part in parts:
            rings = [part.exterior, *part.interiors]
            for ring in rings:
                xs, ys = ring.xy
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    lw=linewidth,
                    ls=linestyle,
                    path_effects=effects,
                    zorder=7,
                )


def gray_image(rgb: np.ndarray) -> np.ndarray:
    """Desaturate and lighten a uint8 RGB image for use behind map layers.

    Args:
        rgb: uint8 ``(rows, cols, 3)``.

    Returns:
        float32 ``(rows, cols)`` lightness in [0.35, 0.95].
    """
    scaled = rgb.astype(np.float32) / 255.0
    mean_band = scaled.mean(axis=-1)
    return GRAY_OFFSET + GRAY_RANGE * mean_band


def category_rgba(codes: np.ndarray, colors: dict[int, object]) -> np.ndarray:
    """Paint integer codes as an RGBA image (codes without a colour stay clear).

    Args:
        codes: Integer ``(rows, cols)``.
        colors: Code to any matplotlib colour or RGBA tuple.

    Returns:
        float32 ``(rows, cols, 4)``.
    """
    image_shape = codes.shape + (4,)
    rgba = np.zeros(image_shape, dtype=np.float32)
    for code, color in colors.items():
        color_rgba = to_rgba(color)
        rgba[codes == code] = color_rgba
    return rgba


def square_bounds(
    geometry: BaseGeometry, pad_m: float
) -> tuple[float, float, float, float]:
    """Square bounds centred on a geometry, its longer side plus ``pad_m`` each way.

    Args:
        geometry: Shape to frame.
        pad_m: Margin added on every side, in CRS units.

    Returns:
        ``(left, bottom, right, top)``.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    half_side = max(max_x - min_x, max_y - min_y) / 2 + pad_m
    centre_x = (min_x + max_x) / 2
    centre_y = (min_y + max_y) / 2
    return (
        centre_x - half_side,
        centre_y - half_side,
        centre_x + half_side,
        centre_y + half_side,
    )


def transform_extent(
    transform: Affine, shape: Sequence[int]
) -> tuple[float, float, float, float]:
    """Return the ``imshow`` extent of an array with an affine transform.

    Args:
        transform: Affine transform of the array's top-left pixel corner.
        shape: Array shape, rows and columns first.

    Returns:
        ``(left, right, bottom, top)``.
    """
    n_rows, n_columns = shape[0], shape[1]
    left = transform.c
    top = transform.f
    right = left + transform.a * n_columns
    bottom = top + transform.e * n_rows
    return (left, right, bottom, top)
