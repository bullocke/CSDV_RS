"""Slide flipbooks that step one stand through its developmental stages.

Each run writes a series of 16:9 frames for one stand and one or more figure
variants. Every frame shows all NAIP dates, and one date at a time is
highlighted with the same colour on the chip frames and on the vertical line in
the time panels. Frame 00 is an overview with nothing highlighted.

Variants:

``filmstrip``
    NAIP, NAIP with segmented crowns, and median canopy height in the
    disturbance gaps with its 10th to 90th percentile spread.
``spotlight``
    A small strip of dates, the current date's NAIP and crowns shown large,
    and four small time panels: gap height, crown count, crown diameter and
    crown width CV.
``gap_closure``
    NAIP, a two-class map split at 2 m, and gap fraction through time.
``two_stands``
    NAIP, a map of residual canopy and gap regrowth height, and the median
    height of each component through time.

Run:
    .micromamba/envs/CSDV/bin/python scripts/figures/lab_meeting_2026_10/stage_flipbook.py \
        --stand ELKNE-U3-0-0 --variant filmstrip
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _flipbook import (  # noqa: E402
    HEIGHT_STAGES,
    NOW_COLOR,
    REGROWTH_COLOR,
    REPO,
    RESIDUAL,
    RESIDUAL_COLOR,
    YEARS,
    StandSeries,
    bare_axis,
    crown_paths,
    draw_outline,
    harmonize_rgb,
    highlight,
    image_extent,
    load_stand_series,
    sqrt_axis,
    stage_tag,
)

from csdv_core.viz.style import (
    GRID,
    INK,
    MUTED,
    STAGE_COLORS,
    setup_style,
)  # noqa: E402

logger = logging.getLogger(__name__)

OUT_ROOT = REPO / "Presentations" / "Lab_Meeting_2026-10" / "figures"
FIG_W, FIG_H = 13.333, 7.5
HEIGHT_VMAX = 25.0
HEIGHT_CMAP = "viridis"
#: Crowns sit on green imagery, so their ramp avoids green entirely. Set from
#: the command line with --crown-cmap and --crown-edge.
CROWN_STYLE = {"cmap": "plasma", "edge": "#1b1030", "lw_scale": 1.0}
CROWN_RAMPS = {
    "plasma": lambda: plt.get_cmap("plasma"),
    "magma": lambda: ListedColormap(plt.get_cmap("magma")(np.linspace(0.25, 1.0, 256))),
    "pinkpurple": lambda: LinearSegmentedColormap.from_list(
        "pinkpurple", ["#49006a", "#ae017e", "#f768a1", "#fde0dd"]
    ),
}


def crown_cmap():
    return CROWN_RAMPS[CROWN_STYLE["cmap"]]()


HEIGHT_YLIM = (0.0, 27.0)
HEIGHT_TICKS = (0.0, 1.5, 3.0, 10.0, 20.0)
BAND_ALPHA = 0.45
TOP_BAND_LABEL = "Late stem exclusion\nor older"


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------


def chip_row(fig, n: int, *, left: float, right: float, top: float, gap_in: float):
    """Place ``n`` square axes in a row, returning axes and the row geometry."""
    width_in = (right - left) * FIG_W
    chip_in = (width_in - gap_in * (n - 1)) / n
    w = chip_in / FIG_W
    h = chip_in / FIG_H
    g = gap_in / FIG_W
    axes = [fig.add_axes((left + i * (w + g), top - h, w, h)) for i in range(n)]
    return axes, w, h, g


def aligned_xlim(years, w: float, g: float) -> tuple[float, float]:
    """x limits that put each year under the centre of its chip column."""
    step = (years[1] - years[0]) / (w + g)
    return (years[0] - step * w / 2, years[-1] + step * w / 2)


def year_labels(axes, years, current: int | None, *, fontsize: float = 12) -> None:
    for ax, year in zip(axes, years, strict=True):
        now = year == current
        ax.set_title(
            str(year),
            fontsize=fontsize,
            fontweight="bold" if now else "normal",
            color=NOW_COLOR if now else INK,
            pad=4,
        )


def row_label(ax, text: str, *, fontsize: float = 11) -> None:
    ax.text(
        -0.06,
        0.5,
        text,
        transform=ax.transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=fontsize,
        color=MUTED,
    )


def show_rgb(ax, series: StandSeries, rgb: np.ndarray, *, interp="nearest") -> None:
    ax.imshow(rgb, extent=image_extent(series), interpolation=interp)
    bare_axis(ax)
    ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
    ax.set_ylim(rgb.shape[0] - 0.5, -0.5)


def draw_crowns(ax, series: StandSeries, year: int, *, lw: float = 0.45):
    """Every in-stand crown, filled by its maximum height.

    Height alone separates the tall reserve trees from the regrowth, so the
    two need no separate symbol.
    """
    crowns = series.crowns[year]
    collection = PathCollection(
        crown_paths(series, crowns),
        array=crowns["max_h_m"].to_numpy(dtype=float),
        cmap=crown_cmap(),
        norm=Normalize(0.0, HEIGHT_VMAX),
        alpha=0.85,
        edgecolors=CROWN_STYLE["edge"],
        linewidths=lw * CROWN_STYLE["lw_scale"],
        zorder=4,
    )
    ax.add_collection(collection)
    return collection


def mark_event(ax, series: StandSeries) -> None:
    """Shade the disturbance interval and name the event above it."""
    ax.axvspan(series.last_pre, series.first_post, color=GRID, alpha=0.9, lw=0)
    label = series.label.lower()
    text = "windthrow" if "wind" in label else "harvest"
    ax.text(
        (series.last_pre + series.first_post) / 2,
        0.98,
        text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        rotation=90,
        fontsize=9,
        color=MUTED,
        zorder=1,
    )


def now_line(ax, current: int | None) -> None:
    if current is not None:
        ax.axvline(current, color=NOW_COLOR, lw=2.6, alpha=0.9, zorder=2)


def height_top(series: StandSeries) -> float:
    """Top of the height axis, so that no stand's 90th percentile is clipped."""
    p90 = [
        q["p90"]
        for st in series.stats.values()
        for q in (st.gap_q, st.residual_q)
        if np.isfinite(q["p90"])
    ]
    return max(HEIGHT_YLIM[1], 1.06 * max(p90))


def stage_bands(ax, *, labels: bool, top: float, fontsize: float = 9.5) -> None:
    """Height-defined stage bands on a square root axis, labelled at the right."""
    lowers = [lower for _, _, lower in HEIGHT_STAGES]
    uppers = [*lowers[1:], top]
    for (code, name, lower), upper in zip(HEIGHT_STAGES, uppers, strict=True):
        ax.axhspan(
            lower, upper, color=STAGE_COLORS[code], alpha=BAND_ALPHA, lw=0, zorder=0
        )
        if not labels:
            continue
        mid = float(np.square((np.sqrt(lower) + np.sqrt(upper)) / 2))
        ax.text(
            1.012,
            mid,
            TOP_BAND_LABEL if code == "LSE" else name,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=INK,
        )
    ax.set_ylim(0.0, top)
    ax.set_yticks(HEIGHT_TICKS)
    ax.set_yticklabels([f"{t:g}" for t in HEIGHT_TICKS])


def visible_years(series: StandSeries, current: int | None, reveal: bool):
    if not reveal or current is None:
        return list(series.years)
    return [y for y in series.years if y <= current]


def split_pre_post(series: StandSeries, years):
    pre = [y for y in years if y <= series.last_pre]
    post = [y for y in years if y > series.last_pre]
    return pre, post


def quantile_series(
    ax,
    series: StandSeries,
    years,
    getter: Callable[[int], dict[str, float]],
    *,
    color: str,
    current: int | None,
    lo: str = "p10",
    hi: str = "p90",
    band_alpha: float = 0.22,
    lw: float = 2.4,
    ms: float = 6.5,
    label: str | None = None,
) -> None:
    """Median line with a spread band. Pre and post dates are drawn apart."""
    pre, post = split_pre_post(series, years)
    for group in (pre, post):
        if not group:
            continue
        x = np.asarray(group, dtype=float)
        q = [getter(y) for y in group]
        med = np.array([v["p50"] for v in q])
        low = np.array([v[lo] for v in q])
        high = np.array([v[hi] for v in q])
        if len(group) > 1:
            ax.fill_between(x, low, high, color=color, alpha=band_alpha, lw=0, zorder=2)
        else:
            ax.vlines(x, low, high, color=color, alpha=band_alpha * 2, lw=7, zorder=2)
        ax.plot(x, med, "-o", color=color, lw=lw, ms=ms, zorder=4, label=label)
        label = None
    if pre and post:
        a, b = pre[-1], post[0]
        ax.plot(
            [a, b],
            [getter(a)["p50"], getter(b)["p50"]],
            ":",
            color=color,
            lw=1.4,
            zorder=3,
        )
    if current is not None and current in years:
        ax.plot(
            current,
            getter(current)["p50"],
            "o",
            ms=ms * 2.1,
            mfc="none",
            mec=NOW_COLOR,
            mew=2.4,
            zorder=5,
        )


def time_axis(ax, series: StandSeries, xlim, *, fontsize: float = 11) -> None:
    ax.set_xlim(*xlim)
    ax.set_xticks(list(series.years))
    ax.tick_params(labelsize=fontsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------
# Variant A: filmstrip
# --------------------------------------------------------------------------


def frame_filmstrip(series: StandSeries, rgb, current: int | None, reveal: bool):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left, right = 0.06, 0.865
    years = list(series.years)
    naip_axes, w, h, g = chip_row(
        fig, len(years), left=left, right=right, top=0.935, gap_in=0.09
    )
    crown_axes, *_ = chip_row(
        fig, len(years), left=left, right=right, top=0.925 - h, gap_in=0.09
    )
    collection = None
    for ax_n, ax_c, year in zip(naip_axes, crown_axes, years, strict=True):
        show_rgb(ax_n, series, rgb[year])
        draw_outline(ax_n, series)
        stage_tag(ax_n, series.stats[year].stage)
        show_rgb(ax_c, series, rgb[year])
        collection = draw_crowns(ax_c, series, year)
        draw_outline(ax_c, series, lw=1.0)
        for ax in (ax_n, ax_c):
            highlight(ax, on=year == current, dim=current is not None)
    year_labels(naip_axes, years, current)
    row_label(naip_axes[0], "NAIP")
    row_label(crown_axes[0], "Tree crowns")

    # Colour key and residual swatch beside the crown row.
    crown_bottom = 0.925 - 2 * h
    cax = fig.add_axes((right + 0.012, crown_bottom + h * 0.08, 0.011, h * 0.84))
    cb = fig.colorbar(collection, cax=cax)
    cb.set_label("Crown height (m)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax = fig.add_axes((left, 0.075, right - left, crown_bottom - 0.075 - 0.07))
    sqrt_axis(ax)
    stage_bands(ax, labels=True, top=height_top(series))
    time_axis(ax, series, aligned_xlim(years, w, g))
    mark_event(ax, series)
    shown = visible_years(series, current, reveal)
    quantile_series(
        ax,
        series,
        shown,
        lambda y: series.stats[y].gap_q,
        color=INK,
        current=current,
        band_alpha=0.16,
    )
    now_line(ax, current)
    ax.set_ylabel("Height in disturbed area (m)", fontsize=11)
    ax.legend(
        handles=[
            Line2D([], [], color=INK, lw=2.4, marker="o", label="median"),
            Patch(color=INK, alpha=0.16, label="10th to 90th percentile"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.99),
        ncols=2,
        fontsize=10,
        frameon=False,
    )
    return fig


# --------------------------------------------------------------------------
# Shared map composites
# --------------------------------------------------------------------------


def washed(series: StandSeries, rgb: np.ndarray, inside_rgb: np.ndarray) -> np.ndarray:
    """Class colours inside the stand, imagery faded toward white outside."""
    out = rgb * 0.45 + 0.55
    mask = series.inside[..., None]
    return np.where(mask, inside_rgb, out)


def height_rgb(values: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap(HEIGHT_CMAP)
    scaled = np.clip(np.nan_to_num(values, nan=0.0) / HEIGHT_VMAX, 0, 1)
    return cmap(scaled)[..., :3]


def hex_rgb(color: str) -> np.ndarray:
    return np.asarray(matplotlib.colors.to_rgb(color))


OPEN_COLOR = STAGE_COLORS["ESI"]
CANOPY_COLOR = "#4f8f5b"


def two_class_rgb(series: StandSeries, year: int) -> np.ndarray:
    heights = series.chm[year]
    open_ = np.nan_to_num(heights, nan=0.0) < 2.0
    return np.where(open_[..., None], hex_rgb(OPEN_COLOR), hex_rgb(CANOPY_COLOR))


def component_rgb(series: StandSeries, year: int) -> np.ndarray:
    base = height_rgb(series.chm[year])
    residual = series.components == RESIDUAL
    return np.where(residual[..., None], hex_rgb(RESIDUAL_COLOR), base)


def map_rows(fig, series, rgb, current, second_row: Callable, label2: str):
    """NAIP row and a second map row, both highlighted. Returns layout values."""
    left, right = 0.06, 0.865
    years = list(series.years)
    naip_axes, w, h, g = chip_row(
        fig, len(years), left=left, right=right, top=0.935, gap_in=0.09
    )
    second_axes, *_ = chip_row(
        fig, len(years), left=left, right=right, top=0.925 - h, gap_in=0.09
    )
    for ax_n, ax_2, year in zip(naip_axes, second_axes, years, strict=True):
        show_rgb(ax_n, series, rgb[year])
        draw_outline(ax_n, series)
        stage_tag(ax_n, series.stats[year].stage)
        show_rgb(ax_2, series, washed(series, rgb[year], second_row(series, year)))
        draw_outline(ax_2, series, lw=1.0)
        for ax in (ax_n, ax_2):
            highlight(ax, on=year == current, dim=current is not None)
    year_labels(naip_axes, years, current)
    row_label(naip_axes[0], "NAIP")
    row_label(second_axes[0], label2)
    return left, right, w, h, g, 0.925 - 2 * h


# --------------------------------------------------------------------------
# Variant C: gap closure
# --------------------------------------------------------------------------


def frame_gap_closure(series: StandSeries, rgb, current: int | None, reveal: bool):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left, right, w, h, g, bottom = map_rows(
        fig, series, rgb, current, two_class_rgb, "Open or canopy"
    )
    fig.legend(
        handles=[
            Patch(fc=CANOPY_COLOR, label="Canopy\n(2 m or taller)"),
            Patch(fc=OPEN_COLOR, label="Open\n(under 2 m)"),
        ],
        loc="center left",
        bbox_to_anchor=(right + 0.004, bottom + h / 2),
        fontsize=10,
        frameon=False,
        handlelength=1.4,
        labelspacing=1.2,
    )
    ax = fig.add_axes((left, 0.075, right - left, bottom - 0.075 - 0.05))
    time_axis(ax, series, aligned_xlim(list(series.years), w, g))
    mark_event(ax, series)
    shown = visible_years(series, current, reveal)
    pre, post = split_pre_post(series, shown)
    gf = {y: 100 * series.stats[y].gap_fraction for y in series.years}
    for group in (pre, post):
        if group:
            ax.plot(group, [gf[y] for y in group], "-o", color=INK, lw=2.4, ms=6.5)
    if pre and post:
        ax.plot([pre[-1], post[0]], [gf[pre[-1]], gf[post[0]]], ":", color=INK, lw=1.4)
    if current is not None:
        ax.plot(current, gf[current], "o", ms=14, mfc="none", mec=NOW_COLOR, mew=2.4)
    now_line(ax, current)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Open area (%)", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    return fig


# --------------------------------------------------------------------------
# Variant D: residual canopy and regrowth as two stands
# --------------------------------------------------------------------------


def frame_two_stands(series: StandSeries, rgb, current: int | None, reveal: bool):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left, right, w, h, g, bottom = map_rows(
        fig, series, rgb, current, component_rgb, "Residual and gaps"
    )
    cax = fig.add_axes((right + 0.012, bottom + h * 0.30, 0.011, h * 0.62))
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(Normalize(0, HEIGHT_VMAX), HEIGHT_CMAP), cax=cax
    )
    cb.set_label("Gap height (m)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.legend(
        handles=[Patch(fc=RESIDUAL_COLOR, label="Residual canopy")],
        loc="lower left",
        bbox_to_anchor=(right + 0.004, bottom - 0.01),
        fontsize=9.5,
        frameon=False,
        handlelength=1.2,
    )
    ax = fig.add_axes((left, 0.075, right - left, bottom - 0.075 - 0.07))
    sqrt_axis(ax)
    stage_bands(ax, labels=True, top=height_top(series))
    time_axis(ax, series, aligned_xlim(list(series.years), w, g))
    mark_event(ax, series)
    shown = visible_years(series, current, reveal)
    quantile_series(
        ax,
        series,
        shown,
        lambda y: series.stats[y].residual_q,
        color=RESIDUAL_COLOR,
        current=current,
        band_alpha=0.16,
    )
    quantile_series(
        ax,
        series,
        shown,
        lambda y: series.stats[y].gap_q,
        color=REGROWTH_COLOR,
        current=current,
        band_alpha=0.22,
    )
    now_line(ax, current)
    ax.set_ylabel("Canopy height (m)", fontsize=11)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=RESIDUAL_COLOR,
                lw=2.4,
                marker="o",
                label="Residual canopy",
            ),
            Line2D(
                [],
                [],
                color=REGROWTH_COLOR,
                lw=2.4,
                marker="o",
                label="Disturbance gaps",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.99),
        ncols=2,
        fontsize=10,
        frameon=False,
    )
    return fig


# --------------------------------------------------------------------------
# Variant B: spotlight
# --------------------------------------------------------------------------


def small_panel(ax, series, xlim, *, ylabel: str, last: bool) -> None:
    time_axis(ax, series, xlim, fontsize=9.5)
    mark_event_quiet(ax, series)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9)
    if not last:
        ax.tick_params(axis="x", labelbottom=False)


def mark_event_quiet(ax, series: StandSeries) -> None:
    ax.axvspan(series.last_pre, series.first_post, color=GRID, alpha=0.9, lw=0)


def simple_line(ax, series, years, values: dict[int, float], current, *, color=INK):
    pre, post = split_pre_post(series, years)
    for group in (pre, post):
        group = [y for y in group if np.isfinite(values[y])]
        if group:
            ax.plot(group, [values[y] for y in group], "-o", color=color, lw=2.0, ms=5)
    if current is not None and np.isfinite(values.get(current, np.nan)):
        ax.plot(
            current, values[current], "o", ms=11, mfc="none", mec=NOW_COLOR, mew=2.2
        )


def frame_spotlight(series: StandSeries, rgb, current: int | None, reveal: bool):
    if current is None:
        return None
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    years = list(series.years)
    strip, *_ = chip_row(fig, len(years), left=0.03, right=0.70, top=0.94, gap_in=0.07)
    for ax, year in zip(strip, years, strict=True):
        show_rgb(ax, series, rgb[year])
        draw_outline(ax, series, lw=0.9)
        highlight(ax, on=year == current, dim=True, lw=3.2)
    year_labels(strip, years, current, fontsize=11)

    side = (0.70 - 0.03 - 0.012) / 2
    side_h = side * FIG_W / FIG_H
    top = 0.94 - strip[0].get_position().height - 0.03
    big_naip = fig.add_axes((0.03, top - side_h, side, side_h))
    big_crown = fig.add_axes((0.03 + side + 0.012, top - side_h, side, side_h))
    show_rgb(big_naip, series, rgb[current], interp="bilinear")
    draw_outline(big_naip, series, lw=2.0)
    stage_tag(big_naip, series.stats[current].stage, fontsize=14, y=0.03)
    show_rgb(big_crown, series, rgb[current], interp="bilinear")
    collection = draw_crowns(big_crown, series, current, lw=0.8)
    draw_outline(big_crown, series, lw=1.6)
    for ax in (big_naip, big_crown):
        highlight(ax, on=True, dim=False, lw=4.5)
    bottom = top - side_h
    cax = fig.add_axes(
        (0.03 + side + 0.012 + side * 0.2, bottom - 0.045, side * 0.6, 0.016)
    )
    cb = fig.colorbar(collection, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=9)
    cax.text(
        -0.03,
        0.5,
        "Crown height (m)",
        transform=cax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )

    xlim = (years[0] - 1, years[-1] + 1)
    shown = visible_years(series, current, reveal)
    x0, wid = 0.775, 0.205
    heights = (0.215, 0.185, 0.185, 0.185)
    gap_y = 0.025
    y = 0.95
    axes = []
    for hgt in heights:
        y -= hgt
        axes.append(fig.add_axes((x0, y, wid, hgt)))
        y -= gap_y

    ax = axes[0]
    sqrt_axis(ax)
    stage_bands(ax, labels=False, top=height_top(series))
    small_panel(ax, series, xlim, ylabel="Height (m)", last=False)
    quantile_series(
        ax,
        series,
        shown,
        lambda yr: series.stats[yr].gap_q,
        color=INK,
        current=current,
        band_alpha=0.16,
        lw=2.0,
        ms=5,
    )
    ax.set_yticks((0, 3, 10, 20))
    ax.set_yticklabels(["0", "3", "10", "20"])

    st = series.stats
    post = [yr for yr in years if yr > series.last_pre]
    counts = {
        yr: (st[yr].n_gap_crowns / series.area_ha if yr in post else np.nan)
        for yr in years
    }
    diam = {
        yr: (st[yr].gap_crown_diam_median if yr in post else np.nan) for yr in years
    }
    cv = {yr: (st[yr].gap_crown_cv if yr in post else np.nan) for yr in years}

    small_panel(axes[1], series, xlim, ylabel="New crowns\nper ha", last=False)
    simple_line(axes[1], series, shown, counts, current)
    axes[1].set_ylim(0, np.nanmax(list(counts.values())) * 1.2)

    small_panel(axes[2], series, xlim, ylabel="Crown\nwidth (m)", last=False)
    simple_line(axes[2], series, shown, diam, current)
    axes[2].set_ylim(0, np.nanmax(list(diam.values())) * 1.25)

    small_panel(axes[3], series, xlim, ylabel="Crown width\nCV", last=True)
    simple_line(axes[3], series, shown, cv, current)
    axes[3].set_ylim(0, 0.6)
    axes[3].set_xticks(years)
    axes[3].set_xticklabels([f"'{yr % 100:02d}" for yr in years])
    for a in axes:
        now_line(a, current)
        a.yaxis.set_label_coords(-0.2, 0.5)
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

VARIANTS: dict[str, Callable] = {
    "filmstrip": frame_filmstrip,
    "spotlight": frame_spotlight,
    "gap_closure": frame_gap_closure,
    "two_stands": frame_two_stands,
}
#: Variants whose time panels change between the full and reveal versions.
REVEALABLE = {"filmstrip", "spotlight", "gap_closure", "two_stands"}


def short_id(stand_id: str) -> str:
    return stand_id.replace("ELKNE-", "").replace("-0-0", "")


def save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def contact_sheet(paths: list[Path], out: Path, *, cols: int = 3) -> None:
    """All frames of one series on one page, for quick review."""
    thumbs = [Image.open(p).convert("RGB") for p in paths]
    tw = 800
    th = int(thumbs[0].height * tw / thumbs[0].width)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 10, rows * th + (rows + 1) * 10))
    sheet.paste((200, 200, 200), (0, 0, *sheet.size))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(
            im.resize((tw, th), Image.LANCZOS), (10 + c * (tw + 10), 10 + r * (th + 10))
        )
    sheet.save(out)


def render(
    series: StandSeries, variant: str, *, reveal: bool, out_root: Path, dpi: int
):
    rgb = harmonize_rgb(series)
    folder = out_root / variant / short_id(series.stand_id)
    if variant in REVEALABLE:
        folder = folder / ("reveal" if reveal else "full")
    draw = VARIANTS[variant]
    written = []
    overview = draw(series, rgb, None, reveal)
    if overview is not None:
        written.append(save(overview, folder / "frame_00_overview.png", dpi))
    for i, year in enumerate(series.years, start=1):
        fig = draw(series, rgb, year, reveal)
        written.append(save(fig, folder / f"frame_{i:02d}_{year}.png", dpi))
    contact_sheet(written, folder / "contact_sheet.png")
    logger.info("Wrote %d frames to %s", len(written), folder)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("fiona", "pyogrio", "rasterio", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stand", default="ELKNE-U3-0-0")
    parser.add_argument(
        "--variant", nargs="+", default=["filmstrip"], choices=[*VARIANTS, "all"]
    )
    parser.add_argument("--years", default=",".join(map(str, YEARS)))
    parser.add_argument(
        "--reveal",
        choices=("both", "full", "reveal"),
        default="both",
        help="Draw the whole time series, or only dates up to the current one",
    )
    parser.add_argument("--pad-m", type=float, default=25.0)
    parser.add_argument("--min-side-m", type=float, default=150.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--crown-cmap", choices=list(CROWN_RAMPS), default="plasma")
    parser.add_argument("--crown-edge", default="#1b1030", help="Crown border colour")
    parser.add_argument("--crown-lw-scale", type=float, default=1.0)
    args = parser.parse_args()

    setup_style()
    CROWN_STYLE.update(
        cmap=args.crown_cmap, edge=args.crown_edge, lw_scale=args.crown_lw_scale
    )
    years = tuple(int(y) for y in args.years.split(","))
    series = load_stand_series(
        args.stand,
        years=years,
        pad_m=args.pad_m,
        min_side_m=args.min_side_m,
        use_cache=not args.no_cache,
    )
    variants = list(VARIANTS) if "all" in args.variant else args.variant
    modes = {"both": (False, True), "full": (False,), "reveal": (True,)}[args.reveal]
    for variant in variants:
        for reveal in modes if variant in REVEALABLE else (False,):
            render(series, variant, reveal=reveal, out_root=args.out_root, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
