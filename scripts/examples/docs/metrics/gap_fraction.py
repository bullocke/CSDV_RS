"""Build the ``gap_fraction`` example figure for ``docs/metrics.md``.

Renders one stand at three dates: imagery, canopy height, and the canopy height
model split at the 2 m threshold, over the full gap fraction series with the
stage envelopes shaded behind it.

The default stand is a clearcut with reserves at Elkinsville. Gap fraction sits
near zero through 2016, rises above 0.8 once the stand is cut, and falls back to
about 0.16 as the regrowth crosses 2 m.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/gap_fraction.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CANOPY_THRESHOLD_M,
    DOC_FIGURE_DIR,
    build_parser,
    column_year_labels,
    configure_logging,
    disturbance_years,
    elkinsville,
    finish,
    height_class_key,
    height_class_panel,
    load_example_context,
    mark_mapped_dates,
    parse_years,
    row_label,
    stage_envelope_bands,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import chm_panel, padded_bounds, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import setup_style  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "gap_fraction"
DEFAULT_STAND = "ELKNE-U9-0-0"
DEFAULT_YEARS = (2016, 2018, 2022)

CHIP_IN = 2.1
SERIES_IN = 2.5
ROW_LABELS = ("NAIP", "Canopy height", f"Split at {CANOPY_THRESHOLD_M:g} m")


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(
        __doc__ or "",
        default_stand=DEFAULT_STAND,
        default_years=DEFAULT_YEARS,
        default_out=DOC_FIGURE_DIR / f"{METRIC}.png",
    )
    args = parser.parse_args()
    years = parse_years(args.years)

    setup_style()
    site = elkinsville()
    context = load_example_context(site)
    stand = stand_row(context.stands, args.stand_id)
    series = stand_metric_series(context.metrics, args.stand_id)
    last_pre, first_post = disturbance_years(stand)

    bounds = padded_bounds(
        stand.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )
    logger.info(
        "Stand %s, %s, %.1f acres, disturbance between %s and %s",
        args.stand_id,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        last_pre,
        first_post,
    )

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.3, CHIP_IN * 3 + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        4,
        n + 1,
        width_ratios=[1.0] * n + [0.055],
        height_ratios=[1.0, 1.0, 1.0, SERIES_IN / CHIP_IN],
    )

    naip_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    chm_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    mask_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    chm_image = None
    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        chm_image, _ = chm_panel(
            chm_axes[column],
            site.chm(year),
            bounds,
            geometry=stand.geometry,
            scale=site.chm_scale,
            max_px=args.max_px,
        )
        inside = height_class_panel(
            mask_axes[column],
            site.chm(year),
            bounds,
            geometry=stand.geometry,
            threshold_m=CANOPY_THRESHOLD_M,
            scale=site.chm_scale,
            max_px=args.max_px,
        )
        logger.info("%d: %d in-stand pixels drawn", year, int(inside.sum()))

    column_year_labels(naip_axes, years)
    for axes, label in zip((naip_axes, chm_axes, mask_axes), ROW_LABELS, strict=True):
        row_label(axes[0], label)

    height_bar = fig.add_subplot(grid[1, n])
    fig.colorbar(chm_image, cax=height_bar, label="Canopy height (m)")
    height_class_key(fig.add_subplot(grid[2, n]))

    _series_panel(
        fig.add_subplot(grid[3, 0:n]),
        series,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, args.out, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _series_panel(ax, series, *, years, last_pre, first_post) -> None:
    """Draw the full gap fraction series with the stage envelopes behind it."""
    ax.set_ylim(0.0, 1.0)
    stage_envelope_bands(ax, METRIC)
    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Gap fraction",
        ylim=(0.0, 1.0),
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))


if __name__ == "__main__":
    raise SystemExit(main())
