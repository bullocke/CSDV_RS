"""Shaded canopy height map of the Santa Clara River riparian corridor.

Draws the northern half of the 2024 NAIP canopy height model, where the
corridor runs from the west edge toward the northeast corner. Height is drawn
with a dark-to-bright ramp and multiplied by a hillshade of the canopy surface
so that individual crowns read as texture. Ground with almost no canopy takes a
faint terrain hillshade from the 3DEP DEM, which shows the channel and terraces.

Example::

    python scripts/figures/santaclara_riparian_map.py --theme dark
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import reproject  # noqa: E402

from csdv_core.viz.maps import read_band_window  # noqa: E402
from csdv_core.viz.relief import (  # noqa: E402
    CANOPY_VMAX_M,
    canopy_cmap,
    shaded_canopy_rgb,
)
from csdv_core.viz.style import setup_style  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CHM = REPO / "data/naip_chm/SantaClara/m_3411948_ne_11_060_20240728_chm.tif"
DEM = REPO / "data/topo/SantaClara/dem.tif"
OUT_DIR = REPO / "results/figures/santaclara_riparian"
# The northern half of the tile plus 440 m, so the channel is not clipped
# where it meets the west edge.
NORTH_HALF = (310014.0, 3801700.0, 316308.0, 3805866.0)

THEMES = {
    "dark": {"face": "#0d1015", "ink": "#e8ecef", "muted": "#9aa6af"},
    "light": {"face": "white", "ink": "#1b2a33", "muted": "#5c6b75"},
}


def read_to_grid(
    path: Path, transform, shape: tuple[int, int], crs, *, resampling: str = "cubic"
) -> np.ndarray:
    """Resample band 1 of ``path`` onto a target grid, nodata as NaN."""
    out = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            rasterio.band(src, 1),
            out,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling[resampling],
        )
    return out


def draw_furniture(ax, pixel_m: float, shape: tuple[int, int], theme: dict) -> None:
    """Add a kilometre scale bar and a north arrow in pixel coordinates."""
    height, width = shape
    bar_px = 1000.0 / pixel_m
    x0, y0 = 0.035 * width, 0.94 * height
    for k, colour in enumerate((theme["ink"], theme["face"])):
        ax.add_patch(
            Rectangle(
                (x0 + k * bar_px / 2, y0),
                bar_px / 2,
                0.012 * height,
                facecolor=colour,
                edgecolor=theme["ink"],
                linewidth=0.6,
                zorder=6,
            )
        )
    ax.text(
        x0 + bar_px / 2,
        y0 - 0.012 * height,
        "1 km",
        color=theme["ink"],
        ha="center",
        va="bottom",
        fontsize=9,
        zorder=6,
    )
    ax.annotate(
        "N",
        xy=(0.965 * width, 0.05 * height),
        xytext=(0.965 * width, 0.15 * height),
        color=theme["ink"],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": theme["ink"], "linewidth": 1.4},
        zorder=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chm", type=Path, default=CHM)
    parser.add_argument("--dem", type=Path, default=DEM)
    parser.add_argument("--bounds", type=float, nargs=4, default=NORTH_HALF)
    parser.add_argument("--max-px", type=int, default=5300)
    parser.add_argument("--vmax", type=float, default=CANOPY_VMAX_M)
    parser.add_argument("--theme", choices=sorted(THEMES), default="dark")
    parser.add_argument("--width-in", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_style()
    theme = THEMES[args.theme]

    chm, transform = read_band_window(
        args.chm, tuple(args.bounds), scale=0.01, max_px=args.max_px
    )
    pixel_m = abs(transform.a)
    logger.info("CHM window %s at %.2f m", chm.shape, pixel_m)
    dem = None
    if args.dem.exists():
        with rasterio.open(args.chm) as src:
            crs = src.crs
        dem = read_to_grid(args.dem, transform, chm.shape, crs)
    else:
        logger.warning("%s not found, drawing without terrain relief", args.dem)
    rgb = shaded_canopy_rgb(chm, pixel_m, dem=dem, vmax=args.vmax)

    height, width = chm.shape
    fig_h = args.width_in * height / width
    fig = plt.figure(figsize=(args.width_in, fig_h + 0.9), facecolor=theme["face"])
    ax = fig.add_axes((0.0, 0.9 / (fig_h + 0.9), 1.0, fig_h / (fig_h + 0.9)))
    ax.imshow(rgb, interpolation="lanczos")
    ax.set_axis_off()
    draw_furniture(ax, pixel_m, chm.shape, theme)

    fig.text(
        0.012,
        0.45 / (fig_h + 0.9),
        "Santa Clara River riparian corridor, Ventura County, California",
        color=theme["ink"],
        fontsize=15,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.012,
        0.16 / (fig_h + 0.9),
        "Canopy height modelled from 0.6 m NAIP imagery acquired 28 July 2024",
        color=theme["muted"],
        fontsize=10.5,
        va="center",
    )
    cax = fig.add_axes((0.70, 0.42 / (fig_h + 0.9), 0.285, 0.16 / (fig_h + 0.9)))
    bar = fig.colorbar(
        ScalarMappable(Normalize(0, args.vmax), canopy_cmap()),
        cax=cax,
        orientation="horizontal",
        extend="max",
    )
    bar.set_label("Canopy height (m)", color=theme["ink"], fontsize=10)
    bar.ax.tick_params(colors=theme["muted"], labelsize=9)
    bar.outline.set_edgecolor(theme["muted"])

    out = args.out or OUT_DIR / f"fig1_chm_map_{args.theme}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=theme["face"], bbox_inches=None)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
