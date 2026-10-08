"""Build the ``crown_count`` example figures for ``docs/metrics.md``.

Renders one stand at three dates with the segmented crowns drawn over the
imagery, one dot per counted crown per date, and the count through time. The
layout follows ``crown_cv.py`` so the two read as a pair.

``crown_count`` is not part of any stage envelope, so the series panel carries
no stage bands. It is also exempt from the ``MIN_CROWNS`` support floor, so
every date is reported and none is drawn as withheld. The right-hand axis
restates the count as crowns per hectare, which is the only form that compares
across stands of different size.

Two presets, matching ``crown_cv.py``. ``stable`` is a 191 acre selection
harvest with several thousand crowns. ``sparse`` is a 3.3 acre clearcut where
the count collapses with the harvest and recovers as regrowth segments.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/crown_count.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/crown_count.py --preset sparse
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CROWN_DIAM_MAX_M,
    CROWN_DIAM_MIN_M,
    ExamplePreset,
    build_parser,
    check_value,
    column_year_labels,
    configure_logging,
    crown_overlay,
    detail_bounds,
    diameter_strip_panel,
    disturbance_years,
    elkinsville,
    finish,
    load_example_context,
    load_stand_crowns,
    mark_mapped_dates,
    resolve_preset,
    row_label,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import padded_bounds, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import INK, MUTED, setup_style  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "crown_count"

#: Same stands, dates and crop as ``crown_cv.py``. ``stable`` crops the image
#: row to a 250 m box because a 10 m segment is unreadable at the full 191 acre
#: extent. The count labels and both lower panels still cover the whole stand.
PRESETS = {
    "stable": ExamplePreset(
        stand="ELKNE-U44-0-0",
        years=(2016, 2018, 2022),
        slug="crown_count_stable",
        detail_m=250.0,
    ),
    "sparse": ExamplePreset(
        stand="ELKNE-U13-0-0",
        years=(2016, 2018, 2022),
        slug="crown_count_sparse",
    ),
}

CHIP_IN = 2.1
STRIP_IN = 2.0
SERIES_IN = 2.3

M2_PER_HA = 10_000.0


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(__doc__ or "", PRESETS, default="stable")
    args = parser.parse_args()
    preset = resolve_preset(args, PRESETS)
    years = list(preset.years)
    out_path = args.out or preset.out_path

    setup_style()
    site = elkinsville()
    context = load_example_context(site)
    stand = stand_row(context.stands, preset.stand)
    series = stand_metric_series(context.metrics, preset.stand)
    last_pre, first_post = disturbance_years(stand)
    all_years = [int(y) for y in series["year"]]
    area_ha = float(stand.geometry.area) / M2_PER_HA

    logger.info(
        "Stand %s, %s, %.1f acres (%.2f ha), disturbance between %s and %s",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        area_ha,
        last_pre,
        first_post,
    )

    crowns_by_year = {}
    diameters_by_year = {}
    for year in all_years:
        crowns = load_stand_crowns(site, year, stand.geometry)
        crowns_by_year[year] = crowns
        diameters_by_year[year] = crowns["crown_diam_m"].to_numpy(dtype=float)
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year}", float(len(crowns)), expected, tolerance=1e-6)
        logger.info(
            "%d: %d in-stand crowns, %.1f per ha",
            year,
            len(crowns),
            len(crowns) / area_ha,
        )

    view = (
        detail_bounds(stand.geometry, preset.detail_m)
        if preset.detail_m
        else padded_bounds(
            stand.geometry,
            pad_fraction=args.pad_fraction,
            min_pad_m=args.min_pad_m,
            square=True,
        )
    )

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.4, CHIP_IN + STRIP_IN + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        3,
        n + 1,
        width_ratios=[1.0] * n + [0.055],
        height_ratios=[1.0, STRIP_IN / CHIP_IN, SERIES_IN / CHIP_IN],
    )

    image_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    collection = None
    for ax, year in zip(image_axes, years, strict=True):
        transform = rgb_panel(
            ax,
            site.naip(year),
            view,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        drawn = crowns_by_year[year].cx[view[0] : view[2], view[1] : view[3]]
        collection = crown_overlay(ax, drawn, transform)
        _count_label(ax, len(crowns_by_year[year]), cropped=bool(preset.detail_m))
        logger.info("%d: %d crowns drawn in the view", year, len(drawn))

    column_year_labels(image_axes, years)
    row_label(image_axes[0], "NAIP and crowns")
    fig.colorbar(
        collection,
        cax=fig.add_subplot(grid[0, n]),
        label="Crown diameter (m)",
        extend="max",
    )

    strip = fig.add_subplot(grid[1, 0:n])
    diameter_strip_panel(
        strip,
        all_years,
        diameters_by_year,
        vmin=CROWN_DIAM_MIN_M,
        vmax=CROWN_DIAM_MAX_M,
    )
    strip.set_ylabel("Crown diameter (m)")
    # Headroom for the count labels, which otherwise sit on the largest crowns.
    strip.set_ylim(0.0, strip.get_ylim()[1] * 1.12)
    _strip_counts(strip, all_years, diameters_by_year)

    _series_panel(
        fig.add_subplot(grid[2, 0:n]),
        series,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
        area_ha=area_ha,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _count_label(ax, count: int, *, cropped: bool) -> None:
    """Print the whole-stand crown count in the corner of an image panel.

    On a cropped panel the drawn crowns are a subset of the counted ones, and
    the label says so. Without that a reader would count the visible segments
    and find the number wrong.
    """
    text = f"n = {count:,}" + (" in stand" if cropped else "")
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8, "lw": 0},
        zorder=6,
    )


def _strip_counts(ax, all_years, diameters_by_year) -> None:
    """Write the count above each column of dots.

    ``diameter_strip_panel`` places the dates at integer positions, not at the
    year, so the labels do the same.
    """
    for position, year in enumerate(all_years):
        ax.annotate(
            f"{len(diameters_by_year[year]):,}",
            xy=(position, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.5,
            color=MUTED,
        )


def _series_panel(ax, series, *, years, last_pre, first_post, area_ha: float) -> None:
    """Draw the crown count series with a crowns per hectare axis.

    No stage envelope exists for this metric, so nothing is drawn behind the
    line. The module-wide 10th to 90th percentile band used by other figures is
    also left out. Raw counts scale with stand area, and the 40 stands span two
    orders of magnitude in area, so that band would describe stand size.
    """
    values = series[METRIC].to_numpy(dtype=float)
    top = float(np.nanmax(values)) * 1.15
    metric_panel(
        ax,
        series["year"],
        values,
        label="Crown count",
        ylim=(0.0, top),
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))

    density = ax.secondary_yaxis(
        "right",
        functions=(lambda count: count / area_ha, lambda dens: dens * area_ha),
    )
    density.set_ylabel("Crowns per hectare")


if __name__ == "__main__":
    raise SystemExit(main())
