"""Build the provisional canopy height percentile example figures.

Renders one stand at three dates: imagery, the canopy height model inside the
segmented crowns, and the 10th, 50th and 90th percentile of canopy height
through time over a provisional set of stage classes.

The percentiles come from every valid canopy height pixel inside the stand. No
height threshold is applied, unlike ``height_stats``, so ground and low
regrowth stay in the sample and the median can fall to zero after a harvest.
The crowns appear only in the map row. They were segmented with a 2 m minimum
height, so a statistic taken inside them could never reach the two lowest
classes.

The stage classes here are an idea under review. They are not the envelopes in
``stages.yaml`` and nothing in the pipeline reads them. Each date is assigned a
class from the median alone.

Two presets. ``clearcut`` is the stand used by ``height_mean`` and
``crown_count``, which drops from the tallest class to the lowest in one
interval. ``regrowth`` was cut before 2014 and climbs through three classes
over the rest of the record.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/height_percentiles.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/height_percentiles.py --preset regrowth
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    ExamplePreset,
    band_panel,
    build_parser,
    column_year_labels,
    configure_logging,
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

from csdv_core.viz.maps import (  # noqa: E402
    CHM_VMAX_M,
    padded_bounds,
    read_band_window,
    rgb_panel,
)
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import (
    ACCENT,
    INK,
    MUTED,
    setup_style,
    stage_color,
)  # noqa: E402
from csdv_core.zonal.mask import read_stand_array  # noqa: E402

logger = logging.getLogger(__name__)

#: All three image dates are 0.6 m native in both presets. 2012 and 2014 still
#: appear on the series, drawn hollow.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U13-0-0",
        years=(2016, 2018, 2022),
        slug="height_percentiles_clearcut",
    ),
    "regrowth": ExamplePreset(
        stand="ELKNE-U3-0-0",
        years=(2016, 2018, 2022),
        slug="height_percentiles_regrowth",
    ),
}

#: Provisional stage classes, keyed on the unthresholded median canopy height.
#: Each entry is (stage code, label, lower bound in metres). A date belongs to
#: the last class whose lower bound its median exceeds.
PROVISIONAL_STAGES = (
    ("ESI", "Early stand initiation", 0.0),
    ("LSI", "Late stand initiation", 1.5),
    ("ESE", "Early stem exclusion", 3.0),
    ("LSE", "Late stem exclusion", 10.0),
)

CHIP_IN = 2.1
SERIES_IN = 2.9
ROW_LABELS = ("NAIP", "Height in crowns")

HEIGHT_YLIM = (0.0, 36.0)
HEIGHT_TICKS = (0.0, 1.5, 3.0, 10.0, 20.0, 30.0)
LABEL_GUTTER_YEARS = 3.7


def provisional_stage(median_m: float) -> str:
    """Return the provisional stage code for a median canopy height."""
    code = PROVISIONAL_STAGES[0][0]
    for stage, _, lower in PROVISIONAL_STAGES[1:]:
        if median_m > lower:
            code = stage
    return code


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

    stats = _height_percentiles(site, stand, series)

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.3, CHIP_IN * 2 + SERIES_IN),
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
    crown_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]

    image = None
    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        image = _crown_height_panel(
            crown_axes[column],
            site,
            bounds,
            stand=stand,
            year=year,
            max_px=args.max_px,
        )

    column_year_labels(naip_axes, years)
    for axes, label in zip((naip_axes, crown_axes), ROW_LABELS, strict=True):
        row_label(axes[0], label)

    # The grey needs a word, and a swatch key beside the NAIP row would point
    # at the wrong panels. A note inside the first height panel does it.
    crown_axes[0].text(
        0.03,
        0.04,
        "grey: not in a segmented crown",
        transform=crown_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    fig.colorbar(image, cax=fig.add_subplot(grid[1, n]), label="Canopy height (m)")

    _series_panel(
        fig.add_subplot(grid[2, 0:n]),
        series,
        stats,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _crown_height_panel(ax, site, bounds, *, stand, year: int, max_px: int | None):
    """Draw the canopy height model inside the in-stand crowns only.

    Every pixel outside a crown polygon becomes NaN, which ``band_panel`` paints
    grey. The crowns are the ones the pipeline counts for this stand, selected
    by centroid, so a crown that straddles the outline is drawn whole.
    """
    arr, transform = read_band_window(
        site.chm(year), bounds, scale=site.chm_scale, max_px=max_px
    )
    crowns = load_stand_crowns(site, year, stand.geometry)
    if len(crowns):
        in_crown = geometry_mask(
            list(crowns.geometry),
            out_shape=arr.shape,
            transform=transform,
            invert=True,
        )
    else:
        in_crown = np.zeros(arr.shape, dtype=bool)
    logger.info("%d: %d crowns rasterised onto the panel", year, len(crowns))
    mappable, _ = band_panel(
        ax,
        np.where(in_crown, arr, np.nan),
        transform,
        cmap="viridis",
        vmin=0.0,
        vmax=CHM_VMAX_M,
        geometry=stand.geometry,
    )
    return mappable


def _height_percentiles(site, stand, series) -> dict[int, dict[str, float]]:
    """Take P10, P50 and P90 of every valid canopy height pixel in the stand.

    No height threshold. ``height_stats`` drops pixels below 2 m before it
    summarises, so its ``height_median`` is logged beside the unthresholded one
    to show how far the two part after a harvest.
    """
    stats = {}
    for year in (int(y) for y in series["year"]):
        chm, window = read_stand_array(
            site.chm(year),
            stand.geometry,
            scale=site.chm_scale,
            stand_id=stand["stand_id"],
        )
        heights = chm[window.mask & np.isfinite(chm)]
        p10, p50, p90 = (float(v) for v in np.percentile(heights, [10, 50, 90]))
        stats[year] = {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "stage": provisional_stage(p50),
        }
        row = series.loc[series["year"] == year].iloc[0]
        logger.info(
            "%d: %d pixels, p10 %.2f m, median %.2f m, p90 %.2f m, stage %s "
            "(thresholded height_median %.2f m)",
            year,
            heights.size,
            p10,
            p50,
            p90,
            stats[year]["stage"],
            float(row["height_median"]),
        )
    return stats


def _series_panel(ax, series, stats, *, years, last_pre, first_post) -> None:
    """Draw the three percentiles over the provisional stage classes.

    The axis is square root scaled. Two of the four classes sit below 3 m, and
    on a linear axis that runs to 36 m they would share a twelfth of the panel.
    The ticks sit on the class boundaries, so the scale can be read off them.

    The bands classify the median only. P10 and P90 cross them freely, and the
    class written above each date is the one the median falls in.
    """
    ax.set_yscale("function", functions=(_sqrt, np.square))

    record_years = np.asarray([int(y) for y in series["year"]], dtype=float)
    p10 = np.array([stats[int(y)]["p10"] for y in record_years])
    p50 = np.array([stats[int(y)]["p50"] for y in record_years])
    p90 = np.array([stats[int(y)]["p90"] for y in record_years])

    _stage_bands(ax)
    ax.fill_between(record_years, p10, p90, color=ACCENT, alpha=0.10, lw=0, zorder=2)
    for values, style, name in (
        (p90, "--", "90th percentile"),
        (p10, ":", "10th percentile"),
    ):
        ax.plot(
            record_years,
            values,
            style,
            color=ACCENT,
            lw=1.1,
            marker="o",
            ms=2.8,
            zorder=3,
            label=name,
        )

    metric_panel(
        ax,
        record_years,
        p50,
        label="Canopy height (m), no threshold",
        ylim=HEIGHT_YLIM,
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
        series_label="median (sets the class)",
    )
    mapped = np.isin(record_years, np.asarray(years, dtype=float))
    mark_mapped_dates(ax, record_years[mapped], p50[mapped])
    ax.set_xticks(list(record_years))
    ax.set_yticks(list(HEIGHT_TICKS))
    ax.set_yticklabels([f"{tick:g}" for tick in HEIGHT_TICKS])
    # A gutter to the right of the last date holds the class labels, so they
    # never sit under a line. The legend goes below the axis for the same
    # reason: the lowest class is where the post-harvest values are.
    ax.set_xlim(record_years.min() - 1, record_years.max() + LABEL_GUTTER_YEARS)
    ax.legend(
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncols=3,
        frameon=False,
    )

    for year in record_years:
        stage = stats[int(year)]["stage"]
        ax.annotate(
            stage,
            xy=(year, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=INK,
            bbox={"boxstyle": "round,pad=0.22", "fc": stage_color(stage), "lw": 0},
            annotation_clip=False,
        )


def _stage_bands(ax) -> None:
    """Shade one band per provisional class and label it on the right edge."""
    lowers = [lower for _, _, lower in PROVISIONAL_STAGES]
    uppers = [*lowers[1:], HEIGHT_YLIM[1]]
    for (stage, label, lower), upper in zip(PROVISIONAL_STAGES, uppers, strict=True):
        ax.axhspan(lower, upper, color=stage_color(stage), alpha=0.42, lw=0, zorder=0)
        middle = float(np.square((_sqrt(lower) + _sqrt(upper)) / 2.0))
        ax.text(
            0.995,
            middle,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.5,
            color=INK,
            alpha=0.8,
            zorder=1,
        )


def _sqrt(values):
    """Square root that tolerates the small negatives matplotlib probes with."""
    return np.sqrt(np.clip(values, 0.0, None))


if __name__ == "__main__":
    raise SystemExit(main())
