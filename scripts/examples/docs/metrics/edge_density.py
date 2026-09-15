"""Build the ``edge_density`` example figures for ``docs/metrics.md``.

The middle row is the metric itself. It splits the canopy height model at 2 m,
runs the same ``interior_edge_mask`` the metric calls, and paints the pixels
that mask returns. Those pixels, times the pixel size, over the in-stand area,
are the number on the axis below.

Two presets, because one stand cannot make the point. ``clearcut`` shows that
the value peaks four years after the harvest rather than at it, since a fresh
opening is one large hole with almost no internal boundary. ``selection`` shows
a light harvest returning a higher value than that clearcut on a third of the
opening. Edge density reads the pattern of removal where gap fraction reads the
amount.

Neither has a stage envelope, so both draw the module's own range instead.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/edge_density.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/edge_density.py --preset selection
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
    CANOPY_COLOR,
    CANOPY_THRESHOLD_M,
    GAP_COLOR,
    ExamplePreset,
    build_parser,
    check_value,
    class_key,
    class_panel,
    column_year_labels,
    configure_logging,
    disturbance_years,
    elkinsville,
    finish,
    load_example_context,
    mark_mapped_dates,
    reference_band,
    resolve_preset,
    row_label,
    stand_mask,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import padded_bounds, read_band_window, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import HIGHLIGHT, setup_style  # noqa: E402
from csdv_core.zonal.mask import read_stand_array  # noqa: E402
from csdv_core.zonal.spatial import interior_edge_mask, stand_edge_density  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "edge_density"

#: The two figures. ``ELKNE-U12-0-0`` is a small clearcut with reserves whose
#: value keeps climbing after the cut. ``ELKNE-U47-0-0`` is the selection
#: harvest from the texture figure, which reads higher in 2018 than the clearcut
#: does on a third of the gap fraction.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U12-0-0", years=(2016, 2018, 2022), slug="edge_density"
    ),
    "selection": ExamplePreset(
        stand="ELKNE-U47-0-0", years=(2016, 2018, 2022), slug="edge_density_selection"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 2.6
EDGE_YLIM = (0.0, 0.30)
ROW_LABELS = ("NAIP", "Canopy boundary")

#: Gap and canopy keep the colours of the two-class canopy height key, so this
#: row reads against the ``gap_fraction`` figures. The boundary takes the event
#: colour, because it is the only class the metric counts.
EDGE_COLORS = (GAP_COLOR, CANOPY_COLOR, HIGHLIGHT)
EDGE_LABELS = ("gap", "canopy", "boundary, counted")


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
    _verify(site, stand, series)

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.5, CHIP_IN * 2 + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        3,
        n + 1,
        width_ratios=[1.0] * n + [0.055],
        height_ratios=[1.0, 1.0, SERIES_IN / CHIP_IN],
    )

    naip_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    edge_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]

    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        classes, invalid, transform, inside, drawn = _edge_classes(
            site, stand.geometry, bounds, year
        )
        class_panel(
            edge_axes[column],
            classes,
            transform,
            colors=EDGE_COLORS,
            geometry=stand.geometry,
            inside=inside,
            invalid=invalid,
        )
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year} as drawn", drawn, expected, tolerance=1e-6)

    column_year_labels(naip_axes, years)
    for axes, label in zip((naip_axes, edge_axes), ROW_LABELS, strict=True):
        row_label(axes[0], label)

    class_key(fig.add_subplot(grid[1, n]), EDGE_COLORS, EDGE_LABELS, fontsize=6.5)

    _series_panel(
        fig.add_subplot(grid[2, 0:n]),
        series,
        context.metrics,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _edge_classes(site, geometry, bounds, year: int):
    """Split the displayed block into gap, canopy and counted boundary.

    The boundary comes from ``interior_edge_mask``, the function the metric
    itself calls, so the painted pixels are the counted pixels.

    This row ignores ``--max-px`` and reads at native resolution. The edge count
    of a binary mask scales with the pixel size, so a decimated read would paint
    a boundary that is genuinely a different length from the one the metric
    counted. The blocks here are around 550 pixels square, which is small enough
    that there is nothing to gain by decimating.

    Returns:
        ``(classes, invalid, transform, inside, drawn_value)``.
    """
    arr, transform = read_band_window(
        site.chm(year), bounds, scale=site.chm_scale, max_px=None
    )
    inside = stand_mask(geometry, arr.shape, transform)
    valid = inside & np.isfinite(arr)
    canopy = valid & (arr >= CANOPY_THRESHOLD_M)
    edges = interior_edge_mask(canopy, inside)

    classes = np.zeros(arr.shape, dtype=np.uint8)
    classes[arr >= CANOPY_THRESHOLD_M] = 1
    classes[edges] = 2

    pixel_size_m = float(abs(transform.a))
    area_m2 = float(inside.sum()) * pixel_size_m**2
    drawn = float(edges.sum()) * pixel_size_m / area_m2 if area_m2 > 0 else float("nan")
    logger.info(
        "%d: %d boundary pixels at %.2f m over %.0f m2 in stand",
        year,
        int(edges.sum()),
        pixel_size_m,
        area_m2,
    )
    return classes, ~np.isfinite(arr), transform, inside, drawn


def _verify(site, stand, series) -> None:
    """Recompute the metric from the source raster and check it against the table."""
    for year in (int(y) for y in series["year"]):
        arr, window = read_stand_array(
            site.chm(year),
            stand.geometry,
            scale=site.chm_scale,
            stand_id=stand["stand_id"],
        )
        valid = window.mask & np.isfinite(arr)
        canopy = valid & (arr >= CANOPY_THRESHOLD_M)
        result = stand_edge_density(
            canopy, window.mask, pixel_size_m=window.pixel_size_m
        )
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year}", result.value, expected, tolerance=1e-9)


def _series_panel(ax, series, metrics, *, years, last_pre, first_post) -> None:
    """Draw the edge density series over the module's own range at each date.

    No stage envelope constrains this metric, so the backdrop is the other
    stands. Without it the panel would be a bare grid and a reader would have
    nothing to judge 0.15 against.
    """
    ax.set_ylim(*EDGE_YLIM)
    n_stands = reference_band(
        ax, metrics, METRIC, label="10th to 90th percentile, all stands"
    )
    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Edge density (1/m)",
        ylim=EDGE_YLIM,
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
        series_label="this stand",
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))
    if n_stands:
        ax.legend(fontsize=6.5, loc="upper left", ncols=2)


if __name__ == "__main__":
    raise SystemExit(main())
