"""csdv_core.viz.style — shared figure styling.

One place for the palette, typography and export settings, so that figures made
in a notebook and figures made by a regeneration script look the same and sit
beside the existing schematic in the classification document without a visible
break. The palette matches ``Planning/Classification_Alt/make_schematic.py``.

Figures in this project carry as little text as they can. The date, the axis
label and the scale go on the figure; the class assignment, the caveats and the
reasoning go in the caption.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

DPI = 200
FONT_SIZE = 9

INK = "#1b2a33"
MUTED = "#5c6b75"
GRID = "#d7dee2"
ACCENT = "#2a6f7f"
HIGHLIGHT = "#b4553a"
INPUT_FILL = "#e8edf0"
METRIC_FILL = "#dce8ea"
LAYER_FILL = "#e4ece2"
CONTEXT_FILL = "#f2ece0"

#: Colours for the seven developmental stages, running from open ground through
#: canopy closure to old growth. Ordered light to dark so the sequence reads as
#: a progression and stays legible in greyscale.
STAGE_COLORS: dict[str, str] = {
    "ESI": "#f0dcae",
    "LSI": "#cfd694",
    "ESE": "#9bc08a",
    "LSE": "#5fa383",
    "UR": "#3d8686",
    "MA_OW": "#2f6580",
    "OG": "#2b4368",
}
UNCLASSIFIED_COLOR = "#e3e6e8"

#: Colours for the four trajectory groups.
TRAJECTORY_GROUP_COLORS: dict[str, str] = {
    "DS": "#b4553a",
    "EF": "#c98a3f",
    "LC": "#6f5a86",
    "FC": "#3d7a8c",
}

#: Colours for the reasons a trajectory rule cannot fire.
BLOCKING_COLORS: dict[str, str] = {
    "threshold not set": "#c98a3f",
    "metric not available": "#b4553a",
    "evaluated, conditions not met": "#7d8f99",
    "rule not defined": "#5c6b75",
}

_PANEL_LABEL_KW = {
    "fontsize": 10,
    "fontweight": "bold",
    "va": "top",
    "ha": "left",
    "color": "white",
    "bbox": {"boxstyle": "square,pad=0.16", "fc": INK, "alpha": 0.72, "lw": 0},
}

__all__ = [
    "ACCENT",
    "BLOCKING_COLORS",
    "DPI",
    "FONT_SIZE",
    "GRID",
    "HIGHLIGHT",
    "INK",
    "MUTED",
    "STAGE_COLORS",
    "TRAJECTORY_GROUP_COLORS",
    "UNCLASSIFIED_COLOR",
    "add_scale_bar",
    "panel_label",
    "save_fig",
    "setup_style",
    "stage_color",
]


def setup_style() -> None:
    """Apply the project's rcParams. Call once before building a figure."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "sans-serif",
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "legend.frameon": False,
            "grid.color": GRID,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def stage_color(stage: str | None) -> str:
    """Return the colour for a stage code, or the unclassified grey."""
    if stage is None:
        return UNCLASSIFIED_COLOR
    return STAGE_COLORS.get(stage, UNCLASSIFIED_COLOR)


def panel_label(ax: Axes, letter: str, *, x: float = 0.025, y: float = 0.975) -> None:
    """Put a bold panel label in the upper left of ``ax``."""
    text = letter if letter.startswith("(") else f"({letter})"
    ax.text(x, y, text, transform=ax.transAxes, zorder=6, **_PANEL_LABEL_KW)


def add_scale_bar(
    ax: Axes,
    pixel_size_m: float,
    *,
    bar_m: float = 50.0,
    color: str = "white",
) -> None:
    """Draw a scale bar in the lower left of an image axis.

    ``ax`` is expected to be showing an array in pixel coordinates, as
    :func:`matplotlib.axes.Axes.imshow` produces. That inverts the y axis, so
    the bar is positioned from the axis limits by screen direction rather than
    by value; reading the smaller limit as "the bottom" would put it in the top
    corner, over the image.
    """
    from matplotlib.patches import Rectangle

    if pixel_size_m <= 0:
        return
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    x_range, y_range = abs(x_right - x_left), abs(y_top - y_bottom)
    bar_px = bar_m / pixel_size_m
    if bar_px > 0.7 * x_range:
        logger.debug("Scale bar of %.0f m is too wide for this panel", bar_m)
        return
    # Positive is up the screen, whichever way the axis runs.
    up = -1.0 if y_bottom > y_top else 1.0
    x0 = min(x_left, x_right) + 0.05 * x_range
    y0 = y_bottom + up * 0.06 * y_range
    height = up * 0.02 * y_range
    ax.add_patch(
        Rectangle(
            (x0, y0),
            bar_px,
            height,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.4,
            zorder=6,
        )
    )
    ax.text(
        x0 + bar_px / 2,
        y0 + height * 2.4,
        f"{bar_m:g} m",
        ha="center",
        va="bottom" if up > 0 else "top",
        fontsize=FONT_SIZE - 2,
        color=color,
        zorder=6,
    )


def save_fig(fig: Figure, out_path: Path | str, *, dpi: int = DPI) -> Path:
    """Write a figure, creating the parent directory if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path
