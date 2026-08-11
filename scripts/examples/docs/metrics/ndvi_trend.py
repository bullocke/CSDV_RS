"""Build the ``ndvi_trend`` example figure for ``docs/metrics.md``.

The top row is one trailing five-year window per panel. The dots are the yearly
``ndvi_mean`` values the slope is fitted to, the solid line is the Theil-Sen
slope reported at the last year of the window, and the two dashed lines are the
low and high ends of its confidence interval drawn through the same point.

Those dashed lines are the figure. A window whose interval sits entirely below
zero is a decline. A window whose interval straddles zero is a flat series with
noise in it. The slope alone reads the same in both cases.

The lower panel carries the interval as a band, with the two trajectory
thresholds shaded as a strip. Both sit within 0.006 of zero, which is a good
deal narrower than the interval on any window here.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/ndvi_trend.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    ExamplePreset,
    build_parser,
    check_value,
    column_year_labels,
    configure_logging,
    disturbance_years,
    elkinsville,
    finish,
    load_example_context,
    load_satellite_annual,
    load_satellite_observations,
    resolve_preset,
    row_label,
    stand_metric_series,
    stand_row,
)

from csdv_core.satellite.annual import ndvi_trend, theil_sen_slope  # noqa: E402
from csdv_core.viz.panels import satellite_panel  # noqa: E402
from csdv_core.viz.style import (  # noqa: E402
    ACCENT,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    setup_style,
)

logger = logging.getLogger(__name__)

METRIC = "ndvi_trend"

#: One figure, on the same stand as the other two satellite examples. The three
#: windows are the steepest decline in the record, a window that reads as flat,
#: and the recovery.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U16-0-0a", years=(2010, 2016, 2020), slug="ndvi_trend"
    ),
}

PANEL_IN = 2.3
SERIES_IN = 2.8
TREND_YLIM = (-0.15, 0.15)
LEVEL_YLIM = (0.40, 1.0)

#: Trailing window length, from ``config/satellite.yaml``.
WINDOW_YEARS = 5

#: The two trajectory rules that read this metric. DS2 fires at or below the
#: first, DS3a and EF3 at or below the second. Shaded as one strip, because at
#: this scale they are one strip.
DS2_THRESHOLD = -0.004
DS3A_THRESHOLD = 0.002


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(__doc__ or "", PRESETS, default="clearcut")
    args = parser.parse_args()
    preset = resolve_preset(args, PRESETS)
    years = list(preset.years)
    out_path = args.out or preset.out_path

    setup_style()
    site = elkinsville()
    context = load_example_context(site)
    stand = stand_row(context.stands, preset.stand)
    last_pre, first_post = disturbance_years(stand)

    annual = load_satellite_annual(site, preset.stand)
    observations = load_satellite_observations(site, preset.stand)
    naip_years = [
        int(y) for y in stand_metric_series(context.metrics, preset.stand)["year"]
    ]

    logger.info(
        "Stand %s, %s, %.1f acres, disturbance between %s and %s",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        last_pre,
        first_post,
    )
    _verify(observations, annual)

    n = len(years)
    fig = plt.figure(
        figsize=(PANEL_IN * n + 0.4, PANEL_IN + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04)
    grid = fig.add_gridspec(2, n, height_ratios=[1.0, SERIES_IN / PANEL_IN])

    window_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    for column, year in enumerate(years):
        _window_panel(window_axes[column], annual, year, first=column == 0)

    column_year_labels(window_axes, years)
    row_label(window_axes[0], "Growing-season NDVI")

    _series_panel(
        fig.add_subplot(grid[1, 0:n]),
        annual,
        naip_years=naip_years,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _window_panel(ax, annual: pd.DataFrame, year: int, *, first: bool) -> None:
    """One trailing window of yearly levels with the fitted slope through it."""
    span = list(range(int(year) - WINDOW_YEARS + 1, int(year) + 1))
    window = annual[annual["year"].isin(span)]
    x = window["year"].to_numpy(dtype=float)
    y = window["ndvi_mean"].to_numpy(dtype=float)
    usable = np.isfinite(y)
    slope, lo, hi = theil_sen_slope(x[usable], y[usable])

    # Theil-Sen with scipy's default method pins the line at the median of both
    # axes, so the interval ends are drawn through that same point. The three
    # lines then differ only in slope, which is the quantity being compared.
    x_mid, y_mid = float(np.median(x[usable])), float(np.median(y[usable]))
    edge = np.array([x.min() - 0.3, x.max() + 0.3])
    ax.axvspan(x.min() - 0.3, x.max() + 0.3, color=GRID, alpha=0.4, lw=0, zorder=0)
    for value, style, width in ((lo, "--", 0.9), (hi, "--", 0.9), (slope, "-", 1.5)):
        ax.plot(
            edge,
            y_mid + value * (edge - x_mid),
            style,
            color=INK if style == "-" else MUTED,
            lw=width,
            zorder=3,
        )
    ax.plot(x[usable], y[usable], "o", color=ACCENT, ms=4.5, zorder=4)
    if not usable.all():
        # A year the level metric withheld is marked along the floor rather than
        # left out, so the reader can see the window rested on four points and
        # not five. It is a marker with no other meaning on this figure, so it
        # carries its own legend.
        ax.plot(
            x[~usable],
            np.full(int((~usable).sum()), LEVEL_YLIM[0] + 0.03),
            "x",
            color=MUTED,
            ms=5,
            zorder=4,
            label="no value that year",
        )
        ax.legend(fontsize=6.0, loc="lower left")

    ax.set_ylim(*LEVEL_YLIM)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    ax.set_xticks(span[::2])
    ax.grid(alpha=0.3)
    if not first:
        ax.set_yticklabels([])
    logger.info(
        "%d window %d to %d: %d usable years, slope %.4f, interval %.4f to %.4f",
        year,
        span[0],
        span[-1],
        int(usable.sum()),
        slope,
        lo,
        hi,
    )


def _verify(observations: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Recompute the metric from the observations and check it against the table."""
    for _, row in annual.iterrows():
        year = int(row["year"])
        value = ndvi_trend(observations, year=year).value
        check_value(f"{METRIC} {year}", value, float(row[METRIC]), tolerance=1e-12)


def _series_panel(ax, annual, *, naip_years, years, last_pre, first_post) -> None:
    """Draw the trend series with its confidence interval behind it.

    No stage envelope constrains this metric, on purpose. A rate has no meaning
    in a single-date envelope, so it lives at the trajectory layer, and the two
    trajectory thresholds are what the backdrop shows instead.
    """
    ax.set_ylim(*TREND_YLIM)
    record_years = annual["year"].to_numpy(dtype=float)

    ax.axhspan(
        DS2_THRESHOLD,
        DS3A_THRESHOLD,
        color=HIGHLIGHT,
        alpha=0.35,
        lw=0,
        zorder=0,
        label="DS2 and DS3a thresholds",
    )
    ax.fill_between(
        record_years,
        annual["sat_trend_slope_lo"].to_numpy(dtype=float),
        annual["sat_trend_slope_hi"].to_numpy(dtype=float),
        color=ACCENT,
        alpha=0.18,
        lw=0,
        zorder=1,
        label="confidence interval",
    )
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=2)

    satellite_panel(
        ax,
        record_years,
        annual[METRIC],
        label="NDVI trend (index per year)",
        ylim=TREND_YLIM,
        n_obs=annual["sat_trend_n_years"],
        last_pre=last_pre,
        first_post=first_post,
        imagery_years=naip_years,
        xlim=(record_years.min() - 1, record_years.max() + 1),
        series_label="Theil-Sen slope",
        # Every window the metric reports rests on at least four usable years,
        # which is its own floor, so nothing here is drawn hollow. Passing the
        # floor keeps the rule the same as on the other satellite figures.
        marginal_at_or_below=3,
    )
    mapped = annual[annual["year"].isin(years)]
    ax.plot(
        mapped["year"].to_numpy(dtype=float),
        mapped[METRIC].to_numpy(dtype=float),
        "o",
        mfc="none",
        mec=INK,
        mew=1.3,
        ms=11.0,
        zorder=5,
        clip_on=False,
    )
    ax.set_xlabel("Year")
    # Upper left. The NAIP ticks sit along the bottom edge from 2012 onward and
    # the series runs high on the right, so the early record is the only clear
    # corner.
    ax.legend(fontsize=6.5, loc="upper left")


if __name__ == "__main__":
    raise SystemExit(main())
