"""Build the ``gap_fraction`` example figures for ``docs/metrics.md``.

Renders one stand at three dates: imagery, canopy height, and the canopy height
model split at the 2 m threshold, over the full gap fraction series with the
stage envelopes shaded behind it.

Two presets. ``clearcut`` is a clearcut with reserves that opens above 0.8 and
only partly closes. ``closure`` is a stand cut earlier in the record that opens
to 0.73 and closes all the way back, which is the half the first figure cannot
show.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/gap_fraction.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/gap_fraction.py --preset closure
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
    ExamplePreset,
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
    resolve_preset,
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

#: The two figures in the docs. ``closure`` uses 2016, 2020 and 2022 because
#: the cut falls between 2012 and 2013, so the only pre-disturbance date is
#: 2012 at 1.0 m native resolution. Including it would put a resolution change
#: inside a figure about a change in the forest, and the ``clearcut`` figure
#: already carries the intact state.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U9-0-0", years=(2016, 2018, 2022), slug="gap_fraction"
    ),
    "closure": ExamplePreset(
        stand="ELKNE-U4-0-0", years=(2016, 2020, 2022), slug="gap_fraction_closure"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 2.5
ROW_LABELS = ("NAIP", "Canopy height", f"Split at {CANOPY_THRESHOLD_M:g} m")


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
    series = stand_metric_series(context.metrics, preset.stand)
    last_pre, first_post = disturbance_years(stand)

    bounds = padded_bounds(
        stand.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )
    logger.info(
        "Stand %s, %s, %.1f acres, disturbance between %s and %s",
        preset.stand,
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

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
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
