"""Build the ``height_mean`` example figures for ``docs/metrics.md``.

Renders one stand at three dates: imagery, the canopy height model, and the
canopy height model with every pixel below 2 m removed, over the full series
with the within-stand spread around it and the module's own range behind that.

The third row is the point of the figure. ``height_stats`` drops every pixel
below the threshold before it takes a mean, so the sample changes size and
composition between dates, and the row shows which pixels survive.

Two presets. ``clearcut`` is a stand the metric reports plainly: one harvest,
one collapse, no recovery. ``reserves`` is the trap. The mean there falls
fastest over the interval the stand recovers, because that is when regrowth
crosses 2 m and joins the sample it had been excluded from.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/height_mean.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/height_mean.py --preset reserves
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
    CANOPY_THRESHOLD_M,
    ExamplePreset,
    band_panel,
    build_parser,
    check_value,
    class_key,
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
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import (  # noqa: E402
    CHM_VMAX_M,
    chm_panel,
    padded_bounds,
    read_band_window,
    rgb_panel,
)
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import ACCENT, MUTED, setup_style  # noqa: E402
from csdv_core.zonal.mask import read_stand_array  # noqa: E402
from csdv_core.zonal.pixel import height_stats  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "height_mean"

#: Both stands are interpreter-labelled clearcuts cut between the 2016 and 2017
#: imagery, and both are drawn at 2016, 2018 and 2022. Every one of those dates
#: is 0.6 m native, so no resolution change sits inside a figure about a change
#: in the forest. 2012 and 2014 still appear on the series, drawn hollow.
#:
#: ``reserves`` is the same stand as the ``gap_fraction`` and ``gap_persistence``
#: figures on purpose. Its caption answers the one the ``gap_fraction`` caption
#: asks when it tells the reader to pair that metric with this one.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U13-0-0", years=(2016, 2018, 2022), slug="height_mean"
    ),
    "reserves": ExamplePreset(
        stand="ELKNE-U9-0-0", years=(2016, 2018, 2022), slug="height_mean_reserves"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 2.5
ROW_LABELS = ("NAIP", "Canopy height", f"Counted, ≥ {CANOPY_THRESHOLD_M:g} m")

#: Heights, not a fraction. The top is set by ``height_p90`` on the clearcut
#: preset, which reaches 32.5 m in 2014.
HEIGHT_YLIM = (0.0, 35.0)


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(__doc__ or "", PRESETS, default="clearcut")
    # Nine raster panels, and the third row carries fine grey speckle along
    # every canopy edge that a palette re-encode cannot cheaply merge. At the
    # default 150 dpi both figures land over the 400 KB guide, at 410 and
    # 487 KB.
    parser.set_defaults(dpi=132)
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

    spread = _canopy_stats(site, stand, series)

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
    counted_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

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
        _counted_panel(
            counted_axes[column],
            site,
            bounds,
            geometry=stand.geometry,
            year=year,
            max_px=args.max_px,
        )

    column_year_labels(naip_axes, years)
    for axes, label in zip(
        (naip_axes, chm_axes, counted_axes), ROW_LABELS, strict=True
    ):
        row_label(axes[0], label)

    fig.colorbar(chm_image, cax=fig.add_subplot(grid[1, n]), label="Canopy height (m)")
    # One swatch, not a ramp. The heights in this row are already keyed by the
    # colourbar above it, so the only thing left to explain is the grey.
    class_key(
        fig.add_subplot(grid[2, n]),
        [MUTED],
        [f"below {CANOPY_THRESHOLD_M:g} m,\nnot counted"],
        fontsize=6.5,
    )

    _series_panel(
        fig.add_subplot(grid[3, 0:n]),
        series,
        context.metrics,
        spread,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _counted_panel(ax, site, bounds, *, geometry, year: int, max_px: int | None):
    """Draw the canopy height model with the uncounted pixels struck out.

    Every pixel below the threshold becomes NaN, which ``band_panel`` paints as
    grey. What is left carries the same viridis 0 to 30 m ramp as the row above,
    so the two rows share one colourbar and the difference between them is
    exactly the set of pixels the mean is taken over.

    ``height_class_panel`` cannot serve here. It collapses the array to two
    classes, which is what ``gap_fraction`` counts, and it would throw away the
    heights this metric averages.
    """
    arr, transform = read_band_window(
        site.chm(year), bounds, scale=site.chm_scale, max_px=max_px
    )
    counted = np.where(arr >= CANOPY_THRESHOLD_M, arr, np.nan)
    band_panel(
        ax,
        counted,
        transform,
        cmap="viridis",
        vmin=0.0,
        vmax=CHM_VMAX_M,
        geometry=geometry,
    )


def _canopy_stats(site, stand, series) -> dict[int, dict[str, float]]:
    """Recompute the canopy height distribution per date, straight from the CHM.

    ``stand_metrics.parquet`` carries no standard deviation and no percentile
    below the 90th, so a within-stand spread has to come from the pixels. The
    same read verifies the figure: ``height_stats`` is the function the pipeline
    calls, so if the recomputed mean does not match the table then the panel is
    wrong, not the table.
    """
    stats = {}
    for year in (int(y) for y in series["year"]):
        chm, window = read_stand_array(
            site.chm(year),
            stand.geometry,
            scale=site.chm_scale,
            stand_id=stand["stand_id"],
        )
        canopy = chm[window.mask & np.isfinite(chm) & (chm >= CANOPY_THRESHOLD_M)]
        result = height_stats(chm, window.mask)
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year}", result[METRIC], expected, tolerance=1e-6)

        row = series.loc[series["year"] == year].iloc[0]
        stats[year] = {
            "p10": float(np.percentile(canopy, 10)),
            "p50": float(np.percentile(canopy, 50)),
            "p90": float(np.percentile(canopy, 90)),
            "mean": float(np.mean(canopy)),
        }
        logger.info(
            "%d: %d canopy pixels of %d in stand, mean %.2f m, p10 %.2f, p90 %.2f, "
            "gap_fraction %.3f",
            year,
            canopy.size,
            int(window.mask.sum()),
            stats[year]["mean"],
            stats[year]["p10"],
            stats[year]["p90"],
            float(row["gap_fraction"]),
        )
    return stats


def _series_panel(ax, series, metrics, spread, *, years, last_pre, first_post) -> None:
    """Draw the height series with its own spread and the module's range.

    Two references, and they answer different questions. The teal whiskers are
    the 10th to 90th percentile of the canopy pixels inside this stand, so they
    say how much of the stand the mean is standing in for. On a post-harvest
    date they run from the threshold to the reserve trees, which is a mean
    sitting between two populations rather than inside one. The grey band is the
    same percentiles across all 40 stands at each date, and it is the only thing
    on the figure that shows the site-wide shift in 2018 and 2020.
    """
    ax.set_ylim(*HEIGHT_YLIM)
    n_stands = reference_band(
        ax, metrics, METRIC, label="10th to 90th percentile, all stands"
    )

    record_years = np.asarray([int(y) for y in series["year"]], dtype=float)
    lo = np.array([spread[int(y)]["p10"] for y in record_years])
    hi = np.array([spread[int(y)]["p90"] for y in record_years])
    # A whisker per date, not a filled band. Filled, this sits at almost the
    # same value as the grey module band behind it and the two read as one
    # shape. A whisker also refuses to interpolate a spread between two NAIP
    # dates, which a band would draw and nothing measured.
    ax.errorbar(
        record_years,
        series[METRIC].to_numpy(dtype=float),
        yerr=np.vstack(
            [series[METRIC].to_numpy(dtype=float) - lo, hi - series[METRIC]]
        ),
        fmt="none",
        ecolor=ACCENT,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        alpha=0.85,
        zorder=2,
        label="10th to 90th percentile, this stand",
    )

    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Canopy height (m)",
        ylim=HEIGHT_YLIM,
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
        series_label="mean, this stand",
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))
    if n_stands:
        ax.legend(fontsize=6.5, loc="lower left", ncols=2)


if __name__ == "__main__":
    raise SystemExit(main())
