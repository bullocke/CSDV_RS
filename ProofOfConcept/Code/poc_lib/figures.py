"""
poc_lib/figures.py — Shared figure utilities for CSDV proof-of-concept analyses.

Provides consistent panel labeling, scale bars, colorbars, and RGB display
helpers used across all analysis scripts. All functions operate on matplotlib
axes objects and follow the style conventions set in 08_compare_chm_sources.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

logger = logging.getLogger(__name__)

# Default figure export settings
_DPI = 200
_FONT_SIZE = 9
_PANEL_LABEL_KW = dict(
    fontsize=11,
    fontweight="bold",
    va="top",
    ha="left",
    color="white",
    bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.6, lw=0),
)


def setup_style() -> None:
    """Apply consistent rcParams for all figures in this project."""
    plt.rcParams.update(
        {
            "font.size": _FONT_SIZE,
            "font.family": "sans-serif",
            "axes.titlesize": _FONT_SIZE,
            "figure.dpi": _DPI,
            "savefig.dpi": _DPI,
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax: Axes, letter: str, x: float = 0.02, y: float = 0.97) -> None:
    """Add a bold panel label (e.g. '(a)') in the upper-left corner of *ax*.

    Parameters
    ----------
    ax : Axes
        Target axes.
    letter : str
        Label text, e.g. 'a', 'b', '(a)'.
    x, y : float
        Axes-fraction position. Default (0.02, 0.97) = upper left.
    """
    label = f"({letter})" if not letter.startswith("(") else letter
    ax.text(x, y, label, transform=ax.transAxes, **_PANEL_LABEL_KW)


def add_scale_bar(
    ax: Axes,
    pixel_size_m: float,
    bar_m: float = 100.0,
    loc: str = "lower left",
    color: str = "white",
) -> None:
    """Draw a scale bar on *ax*.

    The scale bar is drawn as a filled rectangle whose width corresponds to
    *bar_m* meters at the raster's pixel size.

    Parameters
    ----------
    ax : Axes
        Target axes (should have extent set in pixel coordinates).
    pixel_size_m : float
        Spatial resolution of the displayed raster in meters per pixel.
    bar_m : float
        Desired scale bar length in meters. Default 100 m.
    loc : str
        Location string. Only "lower left" is implemented; ignored otherwise.
    color : str
        Bar and label color. Default "white".
    """
    bar_px = bar_m / pixel_size_m
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])

    pad_x = 0.04 * x_range
    pad_y = 0.06 * y_range
    bar_height = 0.015 * y_range

    x0 = min(xlim) + pad_x
    y0 = min(ylim) + pad_y

    from matplotlib.patches import Rectangle

    rect = Rectangle((x0, y0), bar_px, bar_height, color=color, zorder=5)
    ax.add_patch(rect)
    ax.text(
        x0 + bar_px / 2,
        y0 + bar_height * 2.5,
        f"{int(bar_m)} m",
        ha="center",
        va="bottom",
        fontsize=7,
        color=color,
        zorder=5,
    )


def shared_cbar(
    fig: Figure,
    axes: list[Axes],
    im: AxesImage,
    label: str,
    fraction: float = 0.015,
    pad: float = 0.02,
) -> None:
    """Add a single colorbar to the right of a row or column of axes.

    Parameters
    ----------
    fig : Figure
    axes : list of Axes
        The axes to which the colorbar should be attached (rightmost used).
    im : AxesImage
        The imshow handle used to derive the colorbar range.
    label : str
        Colorbar axis label.
    fraction : float
        Fraction of the axes width donated to the colorbar. Default 0.015.
    pad : float
        Padding between axes and colorbar. Default 0.02.
    """
    cbar = fig.colorbar(im, ax=axes, fraction=fraction, pad=pad)
    cbar.set_label(label, fontsize=_FONT_SIZE)
    cbar.ax.tick_params(labelsize=_FONT_SIZE - 1)


def rgb_display(
    naip_path: Path,
    bbox: Optional[tuple[float, float, float, float]] = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Read NAIP RGBN and return a uint8 RGB array suitable for imshow.

    Applies a per-channel percentile stretch to improve visual contrast.

    Parameters
    ----------
    naip_path : Path
        NAIP GeoTIFF with bands ordered R, G, B, N (1-based).
    bbox : tuple or None
        If provided, clip to (west, south, east, north) before reading.
        Must be in the same CRS as the raster.
    percentile_stretch : (p_low, p_high)
        Lower and upper percentiles for the contrast stretch. Default (2, 98).

    Returns
    -------
    rgb : np.ndarray
        uint8 array of shape (H, W, 3).
    """
    from rasterio.mask import mask as rasterio_mask
    from shapely.geometry import box

    with rasterio.open(naip_path) as src:
        if bbox is not None:
            west, south, east, north = bbox
            geom = box(west, south, east, north)
            data, _ = rasterio_mask(src, [geom], crop=True, all_touched=True)
        else:
            data = src.read()  # (bands, H, W)

    rgb = data[:3].astype(np.float32)  # R, G, B
    out = np.zeros_like(rgb)
    p_low, p_high = percentile_stretch
    for i in range(3):
        band = rgb[i]
        lo = float(np.percentile(band[band > 0], p_low)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], p_high)) if np.any(band > 0) else 1.0
        stretched = (band - lo) / (hi - lo + 1e-8)
        out[i] = np.clip(stretched * 255, 0, 255)

    return out.transpose(1, 2, 0).astype(np.uint8)  # (H, W, 3)


def save_fig(fig: Figure, out_path: Path, dpi: int = _DPI) -> None:
    """Save *fig* with tight layout and close it.

    Parameters
    ----------
    fig : Figure
    out_path : Path
        Output file path. Parent directory is created if needed.
    dpi : int
        Output resolution. Default 200.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path.name)
