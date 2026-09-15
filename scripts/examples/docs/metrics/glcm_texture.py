"""Build the ``glcm_texture`` example figure for ``docs/metrics.md``.

Three rows take the metric apart. The near infrared band is what it reads, the
16 grey levels are what the co-occurrence matrix counts, and the series carries
the entropy those counts produce.

The fourth row is the reason the figure exists. Behind the stand's own series
sits the range every other stand in the module occupies at the same date. The
two move together, which says the value is set by the aerial acquisition rather
than by the forest.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/glcm_texture.py
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
    ExamplePreset,
    band_panel,
    build_parser,
    check_value,
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
    stage_envelope_bands,
    stand_mask,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import padded_bounds, read_band_window, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import setup_style  # noqa: E402
from csdv_core.zonal.compute import NAIP_NIR_BAND  # noqa: E402
from csdv_core.zonal.mask import read_stand_array  # noqa: E402
from csdv_core.zonal.texture import (  # noqa: E402
    DEFAULT_LEVELS,
    quantize_masked,
    texture_entropy,
)

logger = logging.getLogger(__name__)

METRIC = "glcm_texture"

#: One figure. ``ELKNE-U47-0-0`` is an uneven-age selection harvest, which is
#: the disturbance a texture metric ought to be best at: it removes scattered
#: individual crowns and leaves the canopy otherwise intact. The columns are the
#: last date before the harvest, the first date after it, and four years on.
PRESETS = {
    "selection": ExamplePreset(
        stand="ELKNE-U47-0-0", years=(2016, 2018, 2022), slug="glcm_texture"
    ),
}

CHIP_IN = 2.1
SERIES_IN = 2.6
TEXTURE_YLIM = (0.0, 8.0)

#: Both image rows below the imagery show the same quantity, so they share one
#: ramp. The second is the first rounded to 16 steps, and nothing else changes.
NIR_CMAP = "gray"
NIR_RANGE = (0.0, 255.0)

ROW_LABELS = (
    "NAIP",
    "NAIP band 4, near infrared",
    f"{DEFAULT_LEVELS} grey levels",
)


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(__doc__ or "", PRESETS, default="selection")
    # Four rows, one of them a full-frame greyscale image, which is the worst
    # case for a palette re-encode. Lowering `--max-px` does not help, because
    # a chip is displayed at roughly 320 px and the read is already oversampled
    # at any value near the default. Dropping the output resolution slightly is
    # what keeps this figure inside the size guide.
    parser.set_defaults(dpi=142)
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
        "Stand %s, %s, %.1f acres, bbox fill %.2f",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        float(stand.get("bbox_fill", float("nan"))),
    )
    stretches = _verify(site, stand, series)

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.4, CHIP_IN * 3 + SERIES_IN),
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
    nir_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    level_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    nir_image = level_image = None
    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        nir, transform = read_band_window(
            site.naip(year), bounds, band=NAIP_NIR_BAND, max_px=args.max_px
        )
        inside = stand_mask(stand.geometry, nir.shape, transform)
        # The one panel here that is not washed out beyond the stand. It is the
        # input rather than the metric, and dimming it would make the stand look
        # darker in the near infrared than the forest around it, which is an
        # artefact of the dimming and not something in the imagery.
        nir_image, _ = band_panel(
            nir_axes[column],
            nir,
            transform,
            cmap=NIR_CMAP,
            vmin=NIR_RANGE[0],
            vmax=NIR_RANGE[1],
            geometry=stand.geometry,
            inside=inside,
            dim_outside=False,
        )
        # The stretch comes from the full-resolution read, so the levels drawn
        # here are the levels the metric quantized to rather than a second
        # stretch fitted to the decimated block.
        vmin, vmax = stretches[year]
        quant, _, _ = quantize_masked(
            nir, inside & np.isfinite(nir), levels=DEFAULT_LEVELS, vmin=vmin, vmax=vmax
        )
        # Out-of-stand pixels take grey level 0 inside `quantize_masked`, purely
        # so the array has an integer dtype. They never enter a pair count, so
        # they are drawn as nodata rather than as the darkest level.
        levels = np.where(inside, quant, np.nan).astype(np.float32)
        level_image, _ = band_panel(
            level_axes[column],
            levels,
            transform,
            cmap=NIR_CMAP,
            vmin=0.0,
            vmax=float(DEFAULT_LEVELS - 1),
            geometry=stand.geometry,
            inside=inside,
        )
        logger.info(
            "%d: in-stand near infrared stretched from %.0f to %.0f", year, vmin, vmax
        )

    column_year_labels(naip_axes, years)
    for axes, label in zip((naip_axes, nir_axes, level_axes), ROW_LABELS, strict=True):
        row_label(axes[0], label)

    fig.colorbar(nir_image, cax=fig.add_subplot(grid[1, n]), label="Reflectance (DN)")
    fig.colorbar(level_image, cax=fig.add_subplot(grid[2, n]), label="Grey level")

    _series_panel(
        fig.add_subplot(grid[3, 0:n]),
        series,
        context.metrics,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _verify(site, stand, series) -> dict[int, tuple[float, float]]:
    """Recompute the entropy at full resolution and check it against the table.

    Returns:
        The in-stand stretch the metric used at each date, so the grey level
        row can be drawn on the same bounds rather than on its own.
    """
    stretches: dict[int, tuple[float, float]] = {}
    for year in (int(y) for y in series["year"]):
        nir, window = read_stand_array(
            site.naip(year),
            stand.geometry,
            band=NAIP_NIR_BAND,
            stand_id=stand["stand_id"],
        )
        result = texture_entropy(nir, window.mask, levels=DEFAULT_LEVELS)
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year}", result.entropy_bits, expected, tolerance=1e-6)
        stretches[year] = (result.vmin, result.vmax)
    return stretches


def _series_panel(ax, series, metrics, *, years, last_pre, first_post) -> None:
    """Draw the entropy series over the module's own range at each date."""
    ax.set_ylim(*TEXTURE_YLIM)
    stage_envelope_bands(ax, METRIC)
    n_stands = reference_band(
        ax, metrics, METRIC, label="10th to 90th percentile, all stands"
    )
    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Texture entropy (bits)",
        ylim=TEXTURE_YLIM,
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
        series_label="this stand",
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))
    if n_stands:
        ax.legend(fontsize=6.5, loc="lower left", ncols=2)


if __name__ == "__main__":
    raise SystemExit(main())
