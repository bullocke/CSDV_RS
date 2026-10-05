"""Landsat, Sentinel-2 and NAIP over one harvested stand (slides C2 to C4).

The figure argues that satellite time series date a harvest well but say little
about recovery. Three rows of image chips (Landsat, Sentinel-2, NAIP) sit above
two time series. The first shows growing-season NDVI from all three sensors and
recovers within a year or two. The second shows NAIP canopy structure (median
height, trees per hectare, crown width CV and gap fraction), which is still far
from the pre-harvest stand years later.

Variants
--------
``stacked``
    Portrait. Chip rows, then the NDVI panel, then the structure panel.
``pixel_grid``
    As ``stacked``, but satellite pixels whose centres fall outside the stand
    are faded, the NAIP row carries the 30 m and 10 m grids, and the row labels
    give the number of pixels inside the stand. Ties into the mixed-pixel slide.
``slide_wide``
    16:9. Chips in a grid on the left, the two time series stacked on the right.

Usage::

    python scripts/figures/lab_meeting_2026_10/sensor_comparison.py --variant all
    python scripts/figures/lab_meeting_2026_10/sensor_comparison.py \\
        --stand ELKNE-U13-0-0 --variant pixel_grid --refresh
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import _sensors as S  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _lib import elkinsville, load_example_context, stand_mask  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from csdv_core.viz.style import INK, MUTED, setup_style  # noqa: E402

logger = logging.getLogger(__name__)

OUT_ROOT = S.REPO / "Presentations" / "Lab_Meeting_2026-10" / "figures"
STANDS = ("ELKNE-U9-0-0", "ELKNE-U13-0-0")
VARIANTS = ("stacked", "pixel_grid", "slide_wide")
XLIM = (2011.6, 2025.9)

#: Categorical slots, validated together per panel (dataviz validate_palette).
SENSOR_COLORS = {"landsat": "#2a78d6", "s2": "#eb6834", "naip": "#1baf7a"}
SENSOR_MARKERS = {"landsat": "o", "s2": "s", "naip": "D"}
SENSOR_NAMES = {"landsat": "Landsat", "s2": "Sentinel-2", "naip": "NAIP"}
SENSOR_RES = {"landsat": "30 m", "s2": "10 m", "naip": "0.6 m"}

#: (column, label, colour, marker, axis limits).
STRUCTURE = (
    ("height_median", "Median height (m)", "#4a3aa7", "o", (0, 32)),
    ("trees_per_ha", "Trees per ha", "#eda100", "^", None),
    ("crown_cv", "Crown width CV", "#e87ba4", "s", (0, 0.8)),
    ("gap_fraction", "Gap fraction", "#008300", "D", (0, 1.0)),
)
HARVEST_FILL = "#ececec"


@dataclass
class StandData:
    stand_id: str
    info: dict
    view: tuple[float, float, float, float]
    chips: dict[str, dict[int, S.Chip]]
    naip_rgb: dict[int, np.ndarray]
    naip_extent: dict[int, tuple[float, float, float, float]]
    naip_ndvi: pd.DataFrame
    landsat_obs: pd.DataFrame
    s2_obs: pd.DataFrame
    structure: pd.DataFrame
    n_pixels: dict[str, int]

    @property
    def short(self) -> str:
        return self.stand_id.split("-")[1]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _harmonize_naip(
    blocks: dict[int, np.ndarray],
    outside: dict[int, np.ndarray],
    *,
    ref_year: int = 2016,
    gamma: float = 1.15,
) -> dict[int, np.ndarray]:
    """Match each flight to ``ref_year`` on the forest outside the stand.

    Same rule as ``_flipbook.harmonize_rgb``, but each date keeps its own mask,
    since the 1 m and 0.6 m flights read to different shapes.
    """
    ref = blocks[ref_year]
    ref_sel = outside[ref_year]
    out = {}
    for year, block in blocks.items():
        sel = outside[year]
        matched = np.empty_like(block[..., :3])
        for b in range(3):
            src = block[..., b]
            mu, sd = src[sel].mean(), src[sel].std()
            rmu, rsd = ref[..., b][ref_sel].mean(), ref[..., b][ref_sel].std()
            matched[..., b] = (src - mu) / max(sd, 1e-6) * rsd + rmu
        out[year] = matched
    lo, hi = np.nanpercentile(out[ref_year], [0.5, 99.5])
    return {
        y: np.power(np.clip((b - lo) / (hi - lo), 0, 1), 1 / gamma)
        for y, b in out.items()
    }


def load_stand(site, context, stand_id: str, s2_all: pd.DataFrame, refresh: bool):
    info = S.stand_info(context, stand_id)
    geom = info["geometry"]
    view = S.square_view(geom)
    chips = {
        sensor: {
            y: S.fetch_chip(sensor, stand_id, y, view, refresh=refresh) for y in S.YEARS
        }
        for sensor in ("landsat", "s2")
    }

    blocks, outside, extents, ndvi_rows = {}, {}, {}, []
    dates = context.metrics.loc[context.metrics["stand_id"] == stand_id]
    dates = dates.set_index("year")["date"]
    for year in S.YEARS:
        block, transform = S.read_naip(site, year, view)
        inside = stand_mask(geom, block.shape[:2], transform)
        blocks[year] = block
        outside[year] = ~inside
        rows, cols = block.shape[:2]
        left, top = transform.c, transform.f
        extents[year] = (
            left,
            left + cols * transform.a,
            top + rows * transform.e,
            top,
        )
        value = S.naip_ndvi(block, inside)
        ndvi_rows.append(
            {"year": year, "t": S.decimal_year([dates[year]])[0], "ndvi": value}
        )
        logger.info("%s NAIP %d NDVI %.3f", stand_id, year, value)

    structure = S.structure_series(context, site, stand_id)
    n_pixels = {s: S.pixels_in_stand(geom, view, s) for s in ("landsat", "s2")}
    return StandData(
        stand_id=stand_id,
        info=info,
        view=view,
        chips=chips,
        naip_rgb=_harmonize_naip(blocks, outside),
        naip_extent=extents,
        naip_ndvi=pd.DataFrame(ndvi_rows),
        landsat_obs=S.landsat_observations(site, stand_id),
        s2_obs=s2_all[s2_all["stand_id"] == stand_id],
        structure=structure,
        n_pixels=n_pixels,
    )


# --------------------------------------------------------------------------
# Chips
# --------------------------------------------------------------------------


def _sensor_stretch(chips: dict[int, S.Chip]) -> tuple[float, float]:
    """One stretch shared by all bands and all years of a sensor.

    A single scalar range keeps the band ratios, so forest stays green and a
    clearcut stays tan. Stretching each band separately exaggerated the small
    colour differences inside closed forest into false purples and pinks.
    """
    pooled = np.concatenate(
        [c.rgb.ravel() for c in chips.values() if c.rgb is not None]
    )
    pooled = pooled[np.isfinite(pooled)]
    return float(np.percentile(pooled, 0.5)), float(np.percentile(pooled, 99.5))


def _display_rgb(
    chip: S.Chip, stretch, *, gamma: float = 1.3, saturation: float = 0.65
) -> np.ndarray:
    """Stretch, then pull toward grey so satellite greens read like NAIP's."""
    lo, hi = stretch
    rgb = np.clip((chip.rgb - lo) / (hi - lo), 0, 1)
    rgb = np.power(rgb, 1 / gamma)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    rgb = np.clip(lum[..., None] + saturation * (rgb - lum[..., None]), 0, 1)
    return np.where(np.isfinite(rgb), rgb, 0.85)


def _outline(ax: Axes, geometry, *, lw: float = 1.3) -> None:
    halo = [pe.Stroke(linewidth=lw + 1.6, foreground="black"), pe.Normal()]
    polys = getattr(geometry, "geoms", [geometry])
    for poly in polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="white", lw=lw, path_effects=halo, zorder=6)


def _frame(ax: Axes, view) -> None:
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.6)


def _grid_lines(ax: Axes, view, res: float, offset: float, color: str) -> None:
    start = np.floor((view[0] - offset) / res) * res + offset
    xs = np.arange(start, view[2] + res, res)
    start = np.floor((view[1] - offset) / res) * res + offset
    ys = np.arange(start, view[3] + res, res)
    kw = dict(color=color, lw=0.5, alpha=0.85, zorder=5)
    for x in xs:
        ax.axvline(x, **kw)
    for y in ys:
        ax.axhline(y, **kw)


def draw_chip_rows(fig, axes_rows, data: StandData, *, variant: str) -> None:
    """Fill a 3 x 6 grid of axes: Landsat, Sentinel-2, NAIP by year."""
    geom = data.info["geometry"]
    fade = variant == "pixel_grid"
    for row, sensor in zip(axes_rows[:2], ("landsat", "s2"), strict=True):
        stretch = _sensor_stretch(data.chips[sensor])
        for ax, year in zip(row, S.YEARS, strict=True):
            chip = data.chips[sensor][year]
            _frame(ax, data.view)
            if chip.rgb is None:
                ax.set_facecolor("#f2f2f2")
                ax.text(
                    0.5,
                    0.5,
                    f"No {SENSOR_NAMES[sensor]}\nbefore 2015",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=8,
                )
                continue
            rgb = _display_rgb(chip, stretch)
            if fade:
                inside = S.native_inside(geom, chip)
                rgb = np.where(inside[..., None], rgb, rgb * 0.35 + 0.65)
            ax.imshow(rgb, extent=chip.extent, interpolation="nearest", zorder=1)
            _outline(ax, geom)
            note = f"{chip.n_scenes} scene" + ("" if chip.n_scenes == 1 else "s")
            ax.text(
                0.97,
                0.04,
                note,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="white",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                zorder=7,
            )

    pre = int(data.info["last_pre"] or 2016)
    post = min(y for y in S.YEARS if y > pre)
    for ax, year in zip(axes_rows[2], S.YEARS, strict=True):
        _frame(ax, data.view)
        ax.imshow(
            data.naip_rgb[year],
            extent=data.naip_extent[year],
            interpolation="antialiased",
            zorder=1,
        )
        _outline(ax, geom)
        if fade and year == pre:
            _grid_lines(ax, data.view, *S.GRID["landsat"], "white")
            _chip_tag(ax, "30 m grid")
        if fade and year == post:
            _grid_lines(ax, data.view, *S.GRID["s2"], "white")
            _chip_tag(ax, "10 m grid")

    for row, sensor in zip(axes_rows, ("landsat", "s2", "naip"), strict=True):
        label = f"{SENSOR_NAMES[sensor]}\n{SENSOR_RES[sensor]}"
        if fade and sensor in data.n_pixels:
            label += f"\n{data.n_pixels[sensor]} px in stand"
        row[0].set_ylabel(label, fontsize=9, color=INK, labelpad=6)
    for ax, year in zip(axes_rows[0], S.YEARS, strict=True):
        ax.set_title(str(year), fontsize=10, color=INK, pad=4)


def _chip_tag(ax: Axes, text: str) -> None:
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        zorder=8,
    )


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------


def _harvest_span(ax: Axes, data: StandData, *, label: bool) -> None:
    pre, post = data.info["last_pre"], data.info["first_post"]
    if pre is None or post is None:
        return
    ax.axvspan(pre + 0.5, post + 0.5, color=HARVEST_FILL, zorder=0, lw=0)
    if label:
        ax.text(
            (pre + post) / 2 + 0.5,
            0.97,
            "harvest",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color=MUTED,
        )


def _naip_ticks(ax: Axes, data: StandData) -> None:
    for t in data.naip_ndvi["t"]:
        ax.axvline(t, color="#d6d6d6", lw=0.6, ls=(0, (2, 2)), zorder=0)


def _style_time_axis(ax: Axes, *, show_labels: bool) -> None:
    ax.set_xlim(*XLIM)
    ax.set_xticks(range(2012, 2026, 2))
    ax.tick_params(axis="x", labelbottom=show_labels, length=3)
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _legend_above(ax: Axes, handles, title: str, *, ncol: int) -> None:
    """Legend in a band above the axes, titled, so it never covers data."""
    leg = ax.legend(
        handles=handles,
        title=title,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=ncol,
        fontsize=7.5,
        title_fontsize=9.5,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.4,
    )
    leg._legend_box.align = "left"


def ndvi_panel(ax: Axes, data: StandData, *, legend_ncol: int = 5) -> None:
    """Satellite NDVI on the left axis, NAIP NDVI on its own right axis.

    NAIP NDVI comes from uncalibrated digital numbers and sits near 0.3 to 0.45
    where the satellites read 0.85 to 0.9. A shared axis would flatten both, so
    NAIP gets its own scale and only its shape is compared.
    """
    _harvest_span(ax, data, label=True)
    _naip_ticks(ax, data)
    satellite_values = []
    for sensor, obs in (("landsat", data.landsat_obs), ("s2", data.s2_obs)):
        color, marker = SENSOR_COLORS[sensor], SENSOR_MARKERS[sensor]
        season = S.season_obs(obs)
        season = season[season["t"] >= XLIM[0]]
        ax.scatter(
            season["t"], season["ndvi"], s=7, color=color, alpha=0.3, lw=0, zorder=2
        )
        annual = S.annual_ndvi(obs)
        annual = annual[annual["t"] >= XLIM[0]]
        satellite_values.append(annual["ndvi"].to_numpy(float))
        ax.plot(annual["t"], annual["ndvi"], color=color, lw=2, zorder=3)
        ok, weak = annual[annual["enough"]], annual[~annual["enough"]]
        ax.plot(ok["t"], ok["ndvi"], marker, color=color, ms=5, mec="white", zorder=4)
        ax.plot(
            weak["t"],
            weak["ndvi"],
            marker,
            mfc="white",
            mec=color,
            mew=1.3,
            ms=5,
            zorder=4,
        )
    lo = np.nanmin(np.concatenate(satellite_values))
    ax.set_ylim(np.floor(lo * 10) / 10 - 0.05, 1.0)
    ax.set_ylabel("Satellite NDVI", color=INK)
    ax.spines["left"].set_color(INK)

    naip = data.naip_ndvi
    color = SENSOR_COLORS["naip"]
    twin = ax.twinx()
    twin.plot(naip["t"], naip["ndvi"], color=color, lw=2, ls=(0, (4, 2)), zorder=3)
    twin.plot(
        naip["t"],
        naip["ndvi"],
        SENSOR_MARKERS["naip"],
        color=color,
        ms=5.5,
        mec="white",
        zorder=4,
    )
    span = naip["ndvi"].max() - naip["ndvi"].min()
    twin.set_ylim(naip["ndvi"].min() - 0.6 * span, naip["ndvi"].max() + 0.25 * span)
    twin.set_ylabel("NAIP NDVI (uncalibrated)", color=INK)
    twin.spines[["top", "left", "bottom"]].set_visible(False)
    twin.spines["right"].set_color(color)
    twin.spines["right"].set_linewidth(1.6)
    twin.tick_params(axis="y", colors=INK, color=color, length=3)
    _style_time_axis(ax, show_labels=True)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            color=SENSOR_COLORS[s],
            marker=SENSOR_MARKERS[s],
            mec="white",
            lw=2,
            ls=(0, (4, 2)) if s == "naip" else "-",
            label=f"{SENSOR_NAMES[s]} ({SENSOR_RES[s]})"
            + (", right axis" if s == "naip" else ""),
        )
        for s in ("landsat", "s2", "naip")
    ]
    handles += [
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            ls="none",
            alpha=0.4,
            ms=3,
            label="single scene",
        ),
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            mfc="white",
            ls="none",
            ms=5,
            label=f"< {S.MIN_OBS} scenes",
        ),
    ]
    _legend_above(ax, handles, "Growing-season NDVI over the stand", ncol=legend_ncol)


def structure_panel(
    ax: Axes, data: StandData, *, offset_pt: float = 46, legend_ncol: int | None = None
) -> None:
    """Four NAIP structure metrics, one y axis each, over the NAIP dates."""
    _harvest_span(ax, data, label=False)
    _naip_ticks(ax, data)
    frame = data.structure
    axes = [ax, ax.twinx(), ax.twinx(), ax.twinx()]
    handles = []
    for i, (col, label, color, marker, ylim) in enumerate(STRUCTURE):
        a = axes[i]
        values = frame[col].to_numpy(float)
        a.plot(frame["t"], values, color=color, lw=2, zorder=3)
        weak = (
            frame["cv_low_support"].to_numpy(bool)
            if col == "crown_cv"
            else np.zeros(len(frame), bool)
        )
        a.plot(
            frame["t"][~weak],
            values[~weak],
            marker,
            color=color,
            ms=6,
            mec="white",
            zorder=4,
        )
        if weak.any():
            a.plot(
                frame["t"][weak],
                values[weak],
                marker,
                mfc="white",
                mec=color,
                mew=1.4,
                ms=6,
                zorder=4,
            )
        if ylim is None:
            ylim = (0, np.nanmax(values) * 1.1)
        a.set_ylim(*ylim)
        a.set_ylabel(label, color=INK)
        side = "left" if i == 0 else "right"
        a.spines[side].set_color(color)
        a.spines[side].set_linewidth(1.6)
        a.tick_params(axis="y", colors=INK, color=color, length=3)
        if i > 0:
            a.spines["right"].set_position(("outward", offset_pt * (i - 1)))
            a.spines[["top", "left", "bottom"]].set_visible(False)
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                mec="white",
                lw=2,
                ms=6,
                label=label,
            )
        )
    _style_time_axis(ax, show_labels=True)
    ax.spines["right"].set_visible(False)
    if frame["cv_low_support"].any():
        handles.append(
            Line2D(
                [],
                [],
                color=STRUCTURE[2][2],
                marker="s",
                mfc="white",
                mew=1.4,
                ls="none",
                label="CV from < 75 crowns",
            )
        )
    handles.append(Patch(color=HARVEST_FILL, label="harvest"))
    _legend_above(
        ax, handles, "NAIP canopy structure", ncol=legend_ncol or len(handles)
    )


def _caption(data: StandData) -> str:
    info = data.info
    return (
        f"{data.short}, {info['label'].lower()}, {info['area_ha']:.1f} ha. "
        f"Harvested between the {int(info['last_pre'])} and "
        f"{int(info['first_post'])} imagery."
    )


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


def _chip_grid(fig, left, right, top, chip_in, gap_in=0.06):
    """3 x 6 square chips, sized in inches so rows sit close together."""
    width_in, height_in = fig.get_size_inches()
    col_w = (right - left) * width_in
    chip_in = min(chip_in, (col_w - 5 * gap_in) / 6)
    rows = []
    for r in range(3):
        y1 = top - r * (chip_in + gap_in) / height_in
        row = []
        for c in range(6):
            x0 = left + c * (chip_in + gap_in) / width_in
            row.append(
                fig.add_axes(
                    [
                        x0,
                        y1 - chip_in / height_in,
                        chip_in / width_in,
                        chip_in / height_in,
                    ]
                )
            )
        rows.append(row)
    bottom = top - (3 * chip_in + 2 * gap_in) / height_in
    return rows, bottom


def figure_stacked(data: StandData, *, variant: str):
    fig = plt.figure(figsize=(12, 12.4))
    left, right = 0.085, 0.79
    rows, chips_bottom = _chip_grid(fig, left, right, 0.935, chip_in=1.6)
    draw_chip_rows(fig, rows, data, variant=variant)
    gs = fig.add_gridspec(
        2,
        1,
        left=left,
        right=right,
        top=chips_bottom - 0.075,
        bottom=0.04,
        hspace=0.42,
        height_ratios=[1, 1.15],
    )
    ax_ndvi = fig.add_subplot(gs[0])
    ax_struct = fig.add_subplot(gs[1], sharex=ax_ndvi)
    ndvi_panel(ax_ndvi, data)
    structure_panel(ax_struct, data)
    fig.text(left, 0.975, _caption(data), fontsize=11, color=INK, va="center")
    return fig


def figure_slide_wide(data: StandData, *, variant: str):
    fig = plt.figure(figsize=(13.333, 7.5))
    left, right = 0.05, 0.47
    chip_in = ((right - left) * 13.333 - 5 * 0.05) / 6
    block = (3 * chip_in + 2 * 0.05) / 7.5
    top = 0.5 + block / 2 - 0.02
    rows, _ = _chip_grid(fig, left, right, top, chip_in=chip_in, gap_in=0.05)
    draw_chip_rows(fig, rows, data, variant=variant)
    gs = fig.add_gridspec(
        2,
        1,
        left=0.56,
        right=0.80,
        top=0.86,
        bottom=0.07,
        hspace=0.62,
    )
    ax_ndvi = fig.add_subplot(gs[0])
    ax_struct = fig.add_subplot(gs[1], sharex=ax_ndvi)
    ndvi_panel(ax_ndvi, data, legend_ncol=3)
    structure_panel(ax_struct, data, offset_pt=40, legend_ncol=3)
    fig.text(left, 0.955, _caption(data), fontsize=12, color=INK, va="center")
    return fig


def render(data: StandData, variant: str, out_dir: Path, dpi: int) -> Path:
    if variant == "slide_wide":
        fig = figure_slide_wide(data, variant=variant)
    else:
        fig = figure_stacked(data, variant=variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant}_{data.short}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def contact_sheet(paths: list[Path], out: Path, *, cols: int = 3) -> None:
    from PIL import Image

    images = [Image.open(p) for p in paths]
    width = 900
    thumbs = [im.resize((width, int(im.height * width / im.width))) for im in images]
    rows = int(np.ceil(len(thumbs) / cols))
    row_h = [
        max(t.height for t in thumbs[r * cols : (r + 1) * cols]) for r in range(rows)
    ]
    sheet = Image.new("RGB", (cols * width, sum(row_h)), "white")
    y = 0
    for r in range(rows):
        for c, thumb in enumerate(thumbs[r * cols : (r + 1) * cols]):
            sheet.paste(thumb, (c * width, y))
        y += row_h[r]
    sheet.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stand", nargs="+", default=list(STANDS))
    parser.add_argument(
        "--variant", choices=(*VARIANTS, "all"), default="all", nargs="+"
    )
    parser.add_argument("--refresh", action="store_true", help="refetch from GEE")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("rasterio").setLevel(logging.ERROR)

    variants = VARIANTS if "all" in args.variant else tuple(args.variant)
    setup_style()
    site = elkinsville()
    context = load_example_context(site)
    s2_all = S.s2_observations(context.stands, list(args.stand), refresh=args.refresh)
    out_dir = args.out_root / "sensor_comparison"
    paths = []
    for stand_id in args.stand:
        data = load_stand(site, context, stand_id, s2_all, args.refresh)
        for variant in variants:
            paths.append(render(data, variant, out_dir, args.dpi))
    if len(paths) > 1:
        contact_sheet(paths, out_dir / "_contact_sheet.png", cols=len(variants))


if __name__ == "__main__":
    main()
