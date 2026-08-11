"""Build the ``crown_cv`` example figures for ``docs/metrics.md``.

Renders one stand at three dates with the segmented crowns drawn over the
imagery, the full crown diameter distribution per date, and the coefficient of
variation against the stage envelopes.

Two presets, and they are meant to be read together. ``stable`` is a large
stand where several thousand segments support the statistic, and the value
holds near 0.30 through a selection harvest. ``sparse`` is a small clearcut
that falls below the support floor when it is cut, so the metric is withheld
rather than reported from too small a sample.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/crown_cv.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/crown_cv.py --preset sparse
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CROWN_DIAM_MAX_M,
    CROWN_DIAM_MIN_M,
    ExamplePreset,
    build_parser,
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
    stage_envelope_bands,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import padded_bounds, rgb_panel  # noqa: E402
from csdv_core.viz.panels import mark_disturbance  # noqa: E402
from csdv_core.viz.style import ACCENT, MUTED, setup_style  # noqa: E402
from csdv_core.zonal.crowns import MIN_CROWNS  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "crown_cv"

#: ``stable`` crops to a 250 m box because the stand runs to 191 acres. At full
#: extent a 10 m segment is a handful of pixels and the overlay is unreadable.
#: The panels below still summarise every in-stand crown.
#:
#: ``sparse`` uses 2016, 2018 and 2022 because that is the sequence where the
#: support floor bites: 106 crowns before the cut, 37 after it, and 119 once
#: the stand has regrown. The middle date is below MIN_CROWNS and is reported
#: as missing.
PRESETS = {
    "stable": ExamplePreset(
        stand="ELKNE-U44-0-0",
        years=(2016, 2018, 2022),
        slug="crown_cv_stable",
        detail_m=250.0,
    ),
    "sparse": ExamplePreset(
        stand="ELKNE-U13-0-0",
        years=(2016, 2018, 2022),
        slug="crown_cv_sparse",
    ),
}

CHIP_IN = 2.1
STRIP_IN = 2.0
SERIES_IN = 2.3

#: The crown_cv envelopes run to 1.5, but the old growth band alone is 0.6 to
#: 1.5 and every observed value sits between 0.2 and 0.6. Plotting the full
#: range spends more than half the panel on one band and squeezes the data into
#: the bottom third. The axis stops at 0.8, which still shows all four band
#: boundaries, and the caption says the old growth band continues past the top.
CV_YLIM = (0.0, 0.8)


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

    logger.info(
        "Stand %s, %s, %.1f acres, disturbance between %s and %s",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        last_pre,
        first_post,
    )

    crowns_by_year = {}
    diameters_by_year = {}
    for year in all_years:
        crowns = load_stand_crowns(site, year, stand.geometry)
        crowns_by_year[year] = crowns
        diameters_by_year[year] = crowns["crown_diam_m"].to_numpy(dtype=float)
        logger.info(
            "%d: %d in-stand crowns, mean %.1f m, cv %.3f",
            year,
            len(crowns),
            float(np.mean(diameters_by_year[year])) if len(crowns) else float("nan"),
            _cv(diameters_by_year[year]),
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
        # The outline matters here. Only in-stand crowns are drawn, so without
        # it a reader cannot tell why the segments stop part way across the
        # panel.
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

    _series_panel(
        fig.add_subplot(grid[2, 0:n]),
        series,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
        diameters_by_year=diameters_by_year,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation, matching ``zonal/crowns.py``."""
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0 else float("nan")


def _series_panel(
    ax, series, *, years, last_pre, first_post, diameters_by_year
) -> None:
    """Draw the full crown_cv series with the stage envelopes behind it.

    Dates below ``MIN_CROWNS`` are drawn, not dropped. The pipeline reports NaN
    there, and plotting only the reported dates leaves isolated points with no
    line between them, which reads as missing data rather than as a value the
    sample was too small to trust. The value is therefore computed from the
    crowns at every date and the unsupported ones are drawn hollow, so the
    reader can see both the trajectory and where it stops being reportable.
    """
    ax.set_ylim(*CV_YLIM)
    stage_envelope_bands(ax, METRIC)

    all_years = np.asarray([int(y) for y in series["year"]], dtype=float)
    counts = np.array([len(diameters_by_year[int(y)]) for y in all_years])
    values = np.array([_cv(diameters_by_year[int(y)]) for y in all_years])
    supported = counts >= MIN_CROWNS

    # The x limits come first. mark_disturbance reads them to decide whether the
    # event falls in the plotted range, so calling it any earlier silently
    # draws nothing.
    ax.set_xlim(all_years.min() - 1, all_years.max() + 1)
    mark_disturbance(ax, last_pre, first_post)
    # A solid line between two supported dates, dashed wherever an end of the
    # segment is a value the pipeline withholds.
    for i in range(len(all_years) - 1):
        both = supported[i] and supported[i + 1]
        ax.plot(
            all_years[i : i + 2],
            values[i : i + 2],
            "-" if both else "--",
            color=ACCENT,
            lw=1.5 if both else 1.0,
            alpha=1.0 if both else 0.55,
            zorder=3,
        )
    ax.plot(all_years[supported], values[supported], "o", color=ACCENT, ms=5, zorder=4)
    ax.plot(
        all_years[~supported],
        values[~supported],
        "o",
        mfc="white",
        mec=MUTED,
        mew=1.4,
        ms=5,
        zorder=4,
    )
    ax.set_ylabel("Crown width CV")

    # Ring only the dates drawn as image columns above, and only where a value
    # is reported. Ringing every date would stop the marker meaning anything.
    mapped = np.isin(all_years, np.asarray(years, dtype=float)) & supported
    mark_mapped_dates(ax, all_years[mapped], values[mapped])
    ax.set_xticks(list(all_years))
    if (~supported).any():
        ax.legend(
            handles=[
                Line2D([], [], color=ACCENT, marker="o", ms=5, label="reported"),
                Line2D(
                    [],
                    [],
                    color=MUTED,
                    ls="--",
                    marker="o",
                    ms=5,
                    mfc="white",
                    mec=MUTED,
                    label=f"below {MIN_CROWNS} crowns, withheld",
                ),
            ],
            frameon=False,
            fontsize=6.5,
            # Lower right. The band labels sit on the left edge and the series
            # runs high on the right of this stand, so it is the only free
            # corner.
            loc="lower right",
        )


if __name__ == "__main__":
    raise SystemExit(main())
