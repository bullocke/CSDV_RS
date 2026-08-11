"""Build the ``ndvi_mean`` example figure for ``docs/metrics.md``.

The satellite metrics run on their own time base. The canopy height metrics have
six NAIP dates over ten years, this has one value a year back to 1985, which
reaches disturbances that happened before the aerial record starts. The stand
here was cut in 2007, so the top row cannot show it happening. Those three
panels are context, and no number on the figure is computed from them.

The lower panel is the metric with its evidence behind it. Each faint dot is one
Landsat scene that survived quality control inside the growing-season window,
and the line is their yearly mean. A year drawn hollow rests on the fewest
observations the metric accepts.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/ndvi_mean.py
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
    decimal_year,
    disturbance_years,
    elkinsville,
    finish,
    load_example_context,
    load_satellite_annual,
    load_satellite_observations,
    resolve_preset,
    row_label,
    stage_envelope_bands,
    stage_envelope_legend_handles,
    stand_metric_series,
    stand_row,
)

from csdv_core.satellite.annual import ndvi_mean  # noqa: E402
from csdv_core.viz.maps import padded_bounds, rgb_panel  # noqa: E402
from csdv_core.viz.panels import satellite_panel  # noqa: E402
from csdv_core.viz.style import setup_style  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "ndvi_mean"

#: One figure, on the stand ``stages.yaml`` already cites as its worked example.
#: ``ELKNE-U16-0-0a`` is 2.8 acres clearcut between the 2007 and 2008 imagery,
#: which is the only stand in the module with a large NDVI signal that runs in
#: both directions inside the Landsat record.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U16-0-0a", years=(2012, 2016, 2022), slug="ndvi_mean"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 3.0
NDVI_YLIM = (0.30, 1.0)

#: The growing-season window the metric averages over, 1 June to 15 September,
#: and the observation floor below which it reports nothing. Both come from
#: ``config/satellite.yaml`` and are read from the registry rather than typed
#: again here.
GROWING_SEASON = (152, 258)
MIN_OBS = 3


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

    bounds = padded_bounds(
        stand.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )
    logger.info(
        "Stand %s, %s, %.1f acres, %.0f m2, disturbance between %s and %s",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        float(stand.get("area_m2", float("nan"))),
        last_pre,
        first_post,
    )
    _verify(observations, annual)

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 0.6, CHIP_IN + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(2, n, height_ratios=[1.0, SERIES_IN / CHIP_IN])

    naip_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
    column_year_labels(naip_axes, years)
    row_label(naip_axes[0], "NAIP, context only")

    _series_panel(
        fig.add_subplot(grid[1, 0:n]),
        annual,
        observations,
        naip_years=naip_years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _growing_season(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep the observations the metric actually averages.

    Drawing the whole calendar year behind the line would put the winter trough
    far below every annual value and make the mean look biased. The dots have to
    be the evidence for the number above them.
    """
    doy = pd.to_numeric(observations["doy"], errors="coerce")
    lo, hi = GROWING_SEASON
    return observations.loc[(doy >= lo) & (doy <= hi)]


def _verify(observations: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Recompute the metric from the observations and check it against the table."""
    for _, row in annual.iterrows():
        year = int(row["year"])
        value = ndvi_mean(observations, year=year).value
        check_value(f"{METRIC} {year}", value, float(row[METRIC]), tolerance=1e-12)


def _series_panel(
    ax, annual, observations, *, naip_years, last_pre, first_post
) -> None:
    """Draw the whole Landsat record with the stage envelopes behind it."""
    ax.set_ylim(*NDVI_YLIM)
    # Seven stages carry an ndvi_mean envelope, and six of them sit between 0.70
    # and 0.93. Their midpoints fall within 0.02 of each other, so no font size
    # fits a stage code beside each band the way the other figures do. The bands
    # are named in the legend instead, with their ranges, which is the only
    # placement that survives the overlap. The pile-up is the point rather than
    # a drawing problem: NDVI saturates over closed canopy and cannot separate
    # the stages it is asked to separate.
    stage_envelope_bands(ax, METRIC, alpha=0.20, label=False)

    window = _growing_season(observations)
    logger.info(
        "%d of %d quality-controlled observations fall in DOY %d to %d",
        len(window),
        len(observations),
        *GROWING_SEASON,
    )
    years = annual["year"].to_numpy(dtype=float)
    satellite_panel(
        ax,
        years,
        annual[METRIC],
        label="Growing-season NDVI",
        ylim=NDVI_YLIM,
        n_obs=annual["sat_ndvi_mean_n_obs"],
        observations=(
            decimal_year(window),
            pd.to_numeric(window["ndvi"], errors="coerce").to_numpy(dtype=float),
        ),
        last_pre=last_pre,
        first_post=first_post,
        imagery_years=naip_years,
        xlim=(years.min() - 1, years.max() + 1),
        marginal_at_or_below=MIN_OBS,
    )
    withheld = int(np.isnan(annual[METRIC].to_numpy(dtype=float)).sum())
    logger.info("%d of %d years report no value", withheld, len(annual))
    ax.set_xlabel("Year")

    # Lower left. Every value before the harvest sits above 0.80 and the axis
    # runs down to 0.30, so the early record below the series is the one large
    # clear area on the panel.
    series_handles, series_labels = ax.get_legend_handles_labels()
    bands = stage_envelope_legend_handles(METRIC, low=NDVI_YLIM[0], high=NDVI_YLIM[1])
    ax.legend(
        handles=bands + series_handles,
        labels=[patch.get_label() for patch in bands] + series_labels,
        fontsize=5.8,
        loc="lower left",
        ncols=2,
        handlelength=1.4,
        columnspacing=1.2,
        labelspacing=0.35,
    )


if __name__ == "__main__":
    raise SystemExit(main())
