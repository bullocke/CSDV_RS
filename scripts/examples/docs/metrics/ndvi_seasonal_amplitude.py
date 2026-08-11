"""Build the ``ndvi_seasonal_amplitude`` example figure for ``docs/metrics.md``.

The top row is the metric before it is reduced. Each panel is one year of clear
Landsat observations plotted against day of year, with the single harmonic
fitted through them. The two dotted lines are the modelled peak and trough, so
the distance between them is the value reported below.

The curve is refitted here with ``fit_single_harmonic`` rather than
reconstructed from the stored coefficients. That call doubles as the check the
procedure asks for, since the amplitude it returns has to equal the one in
``satellite_annual.parquet``.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/ndvi_seasonal_amplitude.py
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
    stage_envelope_bands,
    stand_metric_series,
    stand_row,
)

from csdv_core.satellite.annual import (  # noqa: E402
    fit_single_harmonic,
    ndvi_seasonal_amplitude,
)
from csdv_core.viz.panels import satellite_panel  # noqa: E402
from csdv_core.viz.style import ACCENT, GRID, INK, MUTED, setup_style  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "ndvi_seasonal_amplitude"

#: One figure, on the same stand as the other two satellite examples. The three
#: years are the stand before the cut, at the bottom of the decline, and after
#: two decades of regrowth.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U16-0-0a",
        years=(2005, 2016, 2025),
        slug="ndvi_seasonal_amplitude",
    ),
}

PANEL_IN = 2.3
SERIES_IN = 2.8
AMPLITUDE_YLIM = (0.20, 1.0)
SCATTER_YLIM = (0.0, 1.0)

#: The growing-season window ``ndvi_mean`` averages over. Shaded on the day-of-
#: year panels to show how much of the cycle the level metric does not see.
GROWING_SEASON = (152, 258)
MIN_OBS = 6


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

    fit_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    for column, year in enumerate(years):
        _fit_panel(fit_axes[column], observations, year, first=column == 0)

    column_year_labels(fit_axes, years)
    row_label(fit_axes[0], "NDVI by day of year")

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


def _fit_panel(ax, observations: pd.DataFrame, year: int, *, first: bool) -> None:
    """One year of observations with the harmonic fitted through them."""
    window = observations.loc[
        pd.to_numeric(observations["year"], errors="coerce") == int(year)
    ]
    doy = pd.to_numeric(window["doy"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(window["ndvi"], errors="coerce").to_numpy(dtype=float)
    fit = fit_single_harmonic(doy, values, year=year)

    ax.axvspan(*GROWING_SEASON, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.plot(doy, values, "o", color=ACCENT, ms=3.4, alpha=0.85, zorder=3)

    period = 366.0 if year % 4 == 0 else 365.0
    grid_doy = np.linspace(1.0, period, 400)
    t = 2.0 * np.pi * grid_doy / period
    curve = fit.offset + fit.cos_coef * np.cos(t) + fit.sin_coef * np.sin(t)
    ax.plot(grid_doy, curve, "-", color=INK, lw=1.3, zorder=4)

    # The two dotted lines are the modelled peak and trough. Their separation is
    # the metric, which is why it is drawn rather than written.
    semi = float(np.hypot(fit.cos_coef, fit.sin_coef))
    for level in (fit.offset - semi, fit.offset + semi):
        ax.axhline(level, color=MUTED, lw=0.9, ls=":", zorder=2)

    ax.set_ylim(*SCATTER_YLIM)
    ax.set_xlim(0, 370)
    ax.set_xticks([1, 100, 200, 300])
    ax.set_xlabel("Day of year")
    ax.grid(alpha=0.3)
    if not first:
        ax.set_yticklabels([])
    logger.info(
        "%d: %d observations spanning %.0f days, amplitude %.3f, r2 %.2f, peak at "
        "day %.0f",
        year,
        fit.n_obs,
        fit.doy_span,
        fit.amplitude,
        fit.r2,
        fit.phase_doy,
    )


def _verify(observations: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Recompute the metric from the observations and check it against the table."""
    for _, row in annual.iterrows():
        year = int(row["year"])
        value = ndvi_seasonal_amplitude(observations, year=year).value
        check_value(f"{METRIC} {year}", value, float(row[METRIC]), tolerance=1e-12)


def _series_panel(ax, annual, *, naip_years, years, last_pre, first_post) -> None:
    """Draw the amplitude series with the stage envelopes behind it."""
    ax.set_ylim(*AMPLITUDE_YLIM)
    # Four of the seven envelopes here are identical and merge into one band, so
    # unlike ndvi_mean the four that remain are far enough apart to take a stage
    # code each. The narrowest gap between two midpoints is 0.05, which is about
    # twice the label height on this axis.
    stage_envelope_bands(ax, METRIC, alpha=0.20)

    record_years = annual["year"].to_numpy(dtype=float)
    satellite_panel(
        ax,
        record_years,
        annual[METRIC],
        label="NDVI seasonal amplitude",
        ylim=AMPLITUDE_YLIM,
        n_obs=annual["sat_amplitude_n_obs"],
        last_pre=last_pre,
        first_post=first_post,
        imagery_years=naip_years,
        xlim=(record_years.min() - 1, record_years.max() + 1),
        marginal_at_or_below=MIN_OBS,
    )
    # Ring the three years drawn above, which is what ties the two rows
    # together without putting any text on the figure.
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
    withheld = int(np.isnan(annual[METRIC].to_numpy(dtype=float)).sum())
    logger.info("%d of %d years report no amplitude", withheld, len(annual))


if __name__ == "__main__":
    raise SystemExit(main())
