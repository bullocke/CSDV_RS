"""Build the ``gap_persistence`` example figure for ``docs/metrics.md``.

The metric pairs each canopy height model with the one before it and counts the
pixels that read as gap at both dates. Each image column is therefore a pair of
dates rather than a single one, and the class row is the full cross-tabulation
of that pair: canopy at both, gap at one end only, gap at both.

The point of the figure is the class the metric does not count. A stand cut
between two dates is almost entirely "gap at the second date only", so
``gap_persistence`` stays near zero through the event and only rises one date
later. It says the opening lasted, not that it happened.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/gap_persistence.py
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
    resolve_preset,
    row_label,
    stage_envelope_bands,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import padded_bounds, read_band_window, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import (
    HIGHLIGHT,
    TRAJECTORY_GROUP_COLORS,
    setup_style,
)  # noqa: E402
from csdv_core.zonal.mask import read_stand_array  # noqa: E402
from csdv_core.zonal.pixel import gap_persistence  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "gap_persistence"

#: One figure. ``ELKNE-U9-0-0`` is the stand behind ``gap_fraction.png``, on
#: purpose, so the two figures can be read against each other. The columns are
#: the pair that straddles the cut, the pair after it, and the pair that catches
#: the regrowth closing the opening again.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U9-0-0", years=(2018, 2020, 2022), slug="gap_persistence"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 2.5
ROW_LABELS = ("NAIP, later date", "Gap at each date")

#: The four ways a pixel can fall across a pair of dates, in class order. The
#: two unchanged classes reuse the canopy height key so the row reads against
#: the other figures. The two changed classes take trajectory-group colours,
#: because a change of state is a different kind of thing from a state.
PAIR_COLORS = (
    CANOPY_COLOR,
    TRAJECTORY_GROUP_COLORS["LC"],
    HIGHLIGHT,
    GAP_COLOR,
)
PAIR_LABELS = (
    "canopy at both",
    "gap at the earlier date",
    "gap at the later date",
    "gap at both, counted",
)


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

    all_years = [int(y) for y in series["year"]]
    pairs = _date_pairs(all_years, years)

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
    _verify(site, stand, series, all_years)

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.6, CHIP_IN * 2 + SERIES_IN),
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
    pair_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]

    for column, (earlier, later) in enumerate(pairs):
        rgb_panel(
            naip_axes[column],
            site.naip(later),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        classes, invalid, transform = _pair_classes(
            site, bounds, earlier, later, max_px=args.max_px
        )
        inside = class_panel(
            pair_axes[column],
            classes,
            transform,
            colors=PAIR_COLORS,
            geometry=stand.geometry,
            invalid=invalid,
        )
        counted = inside & ~invalid & (classes == 3)
        valid = inside & ~invalid
        logger.info(
            "%d to %d: %d of %d in-stand pixels are gap at both, %.3f in the panel",
            earlier,
            later,
            int(counted.sum()),
            int(valid.sum()),
            float(counted.sum()) / max(int(valid.sum()), 1),
        )

    column_year_labels(naip_axes, [f"{a} to {b}" for a, b in pairs], fontsize=8)
    for axes, label in zip((naip_axes, pair_axes), ROW_LABELS, strict=True):
        row_label(axes[0], label)

    class_key(fig.add_subplot(grid[1, n]), PAIR_COLORS, PAIR_LABELS, fontsize=6.0)

    _series_panel(
        fig.add_subplot(grid[2, 0:n]),
        series,
        years=[b for _, b in pairs],
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _date_pairs(all_years: list[int], years: list[int]) -> list[tuple[int, int]]:
    """Return the ``(earlier, later)`` pair the metric used at each column.

    Raises:
        ValueError: If a requested year is the first date of the series, where
            the metric has nothing to compare against.
    """
    pairs: list[tuple[int, int]] = []
    for year in years:
        index = all_years.index(year)
        if index == 0:
            raise ValueError(
                f"{year} is the first date of the series, so gap_persistence "
                "has no earlier date to pair it with"
            )
        pairs.append((all_years[index - 1], year))
    return pairs


def _pair_classes(site, bounds, earlier: int, later: int, *, max_px: int | None):
    """Cross-tabulate the gap mask at two dates into four classes.

    Returns:
        ``(classes, invalid, transform)``. Class 3 is the numerator of the
        metric, class 2 is the opening the metric cannot see yet.
    """
    a, transform = read_band_window(
        site.chm(earlier), bounds, scale=site.chm_scale, max_px=max_px
    )
    b, _ = read_band_window(
        site.chm(later), bounds, scale=site.chm_scale, max_px=max_px
    )
    if a.shape != b.shape:
        raise ValueError(f"Date pair disagrees on shape: {a.shape} against {b.shape}")

    gap_a = a < CANOPY_THRESHOLD_M
    gap_b = b < CANOPY_THRESHOLD_M
    classes = np.zeros(a.shape, dtype=np.uint8)
    classes[gap_a & ~gap_b] = 1
    classes[~gap_a & gap_b] = 2
    classes[gap_a & gap_b] = 3
    return classes, ~(np.isfinite(a) & np.isfinite(b)), transform


def _verify(site, stand, series, all_years: list[int]) -> None:
    """Recompute the metric at full resolution and check it against the table.

    The panels read a decimated window so the chips stay small, so they cannot
    reproduce the value to the last digit. This does, straight from the source
    rasters with the function the pipeline calls.
    """
    for earlier, later in zip(all_years[:-1], all_years[1:], strict=True):
        a, window = read_stand_array(
            site.chm(earlier),
            stand.geometry,
            scale=site.chm_scale,
            stand_id=stand["stand_id"],
        )
        b, _ = read_stand_array(
            site.chm(later),
            stand.geometry,
            scale=site.chm_scale,
            stand_id=stand["stand_id"],
        )
        value = gap_persistence(a, b, window.mask)
        expected = float(series.loc[series["year"] == later, METRIC].iloc[0])
        check_value(f"{METRIC} {earlier} to {later}", value, expected, tolerance=1e-6)


def _series_panel(ax, series, *, years, last_pre, first_post) -> None:
    """Draw the full gap persistence series with the stage envelopes behind it."""
    ax.set_ylim(0.0, 1.0)
    stage_envelope_bands(ax, METRIC)
    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Gap persistence",
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
