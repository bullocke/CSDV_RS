"""Build the ``row_directionality`` example figures for ``docs/metrics.md``.

The metric reduces a 2-D FFT power spectrum to ``clip(1 - mean/peak, 0, 1)``
over 36 angular bins. Nothing about that is visible in a number, so both
figures draw the transform.

Two presets, and they answer different questions. ``clearcut`` runs the
machinery on ``ELKNE-U13-0-0``, the one interpreter-labelled stand that both
clears the support gate and moves across its harvest. ``scale`` says what a
value means, by putting synthetic patterns whose answer is known next to the
stand and next to every reported value in the module.

Neither figure has a stage envelope to sit against, so ``clearcut`` draws the
module's own range instead.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/row_directionality.py
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/row_directionality.py --preset scale
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

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
    stand_metric_series,
    stand_row,
)

from csdv_core.metrics.spatial import fft_directionality  # noqa: E402
from csdv_core.viz.maps import padded_bounds, rgb_panel  # noqa: E402
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import (  # noqa: E402
    ACCENT,
    HIGHLIGHT,
    INK,
    MUTED,
    setup_style,
)
from csdv_core.zonal.mask import read_stand_array  # noqa: E402
from csdv_core.zonal.spatial import stand_row_directionality  # noqa: E402

logger = logging.getLogger(__name__)

METRIC = "row_directionality"

N_BINS = 36

#: ``1 - mean/peak`` over 36 bins reaches its maximum when one bin holds every
#: unit of power and the other 35 hold none. The metric can therefore never
#: return 1.0, which matters when reading any value against the top of the axis.
CEILING = 1.0 - 1.0 / N_BINS

#: FC2, row-structured plantation, in config/trajectories.yaml.
FC2_THRESHOLD = 0.50

#: ``clearcut`` uses 2016, 2018 and 2022, all 0.6 m native. ``ELKNE-U13-0-0`` is
#: one of only two labelled stands that clear the 0.50 support gate, and the
#: only one whose value moves on the date of its disturbance.
#:
#: ``scale`` borrows the same stand at 2016 for its right-hand column, so the
#: two figures share a reference point a reader can carry between them.
PRESETS = {
    "clearcut": ExamplePreset(
        stand="ELKNE-U13-0-0", years=(2016, 2018, 2022), slug="row_directionality"
    ),
    "scale": ExamplePreset(
        stand="ELKNE-U13-0-0", years=(2016,), slug="row_directionality_scale"
    ),
}

CHIP_IN = 2.1
PROFILE_IN = 1.7
SERIES_IN = 2.4
STRIP_IN = 1.5

#: The spectrum is drawn as log10 power relative to the strongest analysed
#: frequency. Six decades puts the low-frequency core and the noise floor on one
#: ramp without the core saturating.
SPECTRUM_DECADES = 6.0

SPECTRUM_CMAP = "magma"


@dataclass(frozen=True)
class SpectrumParts:
    """Every intermediate the metric computes and then throws away.

    ``fft_directionality`` returns a bare float, so a figure that wants to draw
    the transform has to replay it. ``value`` is checked against the real
    function on the same input, which is what makes the drawn spectrum the
    counted one.
    """

    tapered: np.ndarray
    power: np.ndarray
    inner: np.ndarray
    sums: np.ndarray
    value: float
    radius: int


def spectrum_parts(tile: np.ndarray, *, n_bins: int = N_BINS) -> SpectrumParts:
    """Replay ``metrics/spatial.py::fft_directionality`` and keep the workings.

    Mirrors that function line for line: fill the invalid pixels with the
    in-stand mean, remove the global mean, apply a separable Hanning window,
    transform, and bin the power of the annulus ``1 < r < min(cy, cx)`` into
    ``n_bins`` angular bins over ``[0, pi)``.
    """
    valid = np.isfinite(tile)
    img = np.where(valid, tile, np.nanmean(tile[valid])).astype(np.float64)
    img = img - img.mean()
    rows, cols = img.shape
    # A separable Hanning window, the outer product of two 1-D windows, exactly
    # as the metric builds it. The taper is why the panel darkens at its border.
    tapered = img * np.outer(np.hanning(rows), np.hanning(cols))
    power = np.abs(np.fft.fftshift(np.fft.fft2(tapered))) ** 2

    cy, cx = rows // 2, cols // 2
    yy, xx = np.indices(power.shape)
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx)
    radius = min(cy, cx)
    # The annulus drops the DC term and its four immediate neighbours, and stops
    # at the inscribed circle so the corners cannot over-feed the diagonal bins.
    inner = (r > 1.0) & (r < radius)

    theta = np.arctan2(dy[inner], dx[inner]) % np.pi
    bins = np.linspace(0.0, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power[inner], minlength=n_bins)
    value = float(np.clip(1.0 - sums.mean() / sums.max(), 0.0, 1.0))
    return SpectrumParts(tapered, power, inner, sums, value, radius)


def main() -> int:
    """Build the figure and write it to disk."""
    configure_logging()
    parser = build_parser(__doc__ or "", PRESETS, default="clearcut")
    # The spectrum panels are high-entropy speckle that a 256-colour palette
    # cannot merge, so the default 150 dpi puts both figures over the size guide.
    parser.set_defaults(dpi=125)
    args = parser.parse_args()
    preset = resolve_preset(args, PRESETS)
    out_path = args.out or preset.out_path

    setup_style()
    site = elkinsville()
    context = load_example_context(site)

    if preset.slug.endswith("_scale"):
        fig = _scale_figure(site, context, preset, args)
    else:
        fig = _stand_figure(site, context, preset, args)

    written = finish(fig, out_path, dpi=args.dpi, optimize=args.optimize)
    logger.info("Wrote %s", written)
    return 0


def _stand_tile(site, stand, year: int):
    """Read one stand-date and return what the metric sees.

    ``tile`` is the array the kernel receives, out-of-stand pixels set to NaN,
    which is what ``stand_row_directionality`` builds internally. ``arr`` is the
    raw read the wrapper itself expects. The canopy height model is passed
    whole, never thresholded at 2 m, unlike the two sibling metrics in the same
    module.
    """
    arr, window = read_stand_array(
        site.chm(year),
        stand.geometry,
        scale=site.chm_scale,
        stand_id=stand["stand_id"],
    )
    tile = np.where(window.mask, arr, np.nan)
    return tile, arr, window.mask, window.transform


def _stand_figure(site, context, preset, args):
    """The machinery, on one stand at three dates."""
    stand = stand_row(context.stands, preset.stand)
    series = stand_metric_series(context.metrics, preset.stand)
    last_pre, first_post = disturbance_years(stand)
    years = list(preset.years)

    logger.info(
        "Stand %s, %s, %.1f acres, bbox fill %.3f, disturbance between %s and %s",
        preset.stand,
        stand.get("dist_label", "unlabelled"),
        float(stand.get("Acres", float("nan"))),
        float(stand.get("bbox_fill", float("nan"))),
        last_pre,
        first_post,
    )

    parts = {}
    geometry_by_year = {}
    for year in (int(y) for y in series["year"]):
        tile, arr, mask, transform = _stand_tile(site, stand, year)
        part = spectrum_parts(tile)
        parts[year] = part
        # Kept so the drawing loop below does not read the same block again.
        geometry_by_year[year] = (mask, transform)
        # Two checks per date. The replay against the kernel it copies, and the
        # stand wrapper against the published table.
        check_value(
            f"{METRIC} {year} replayed",
            part.value,
            fft_directionality(tile),
            tolerance=1e-9,
        )
        reported = stand_row_directionality(arr, mask)
        expected = float(series.loc[series["year"] == year, METRIC].iloc[0])
        check_value(f"{METRIC} {year}", reported.value, expected, tolerance=1e-9)
        logger.info(
            "%d: box %dx%d, radius %d, %d analysed frequencies, peak/mean %.2f",
            year,
            *part.power.shape,
            part.radius,
            int(part.inner.sum()),
            part.sums.max() / part.sums.mean(),
        )

    bounds = padded_bounds(
        stand.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )

    n = len(years)
    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.35, CHIP_IN * 3 + PROFILE_IN + SERIES_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        5,
        n + 1,
        width_ratios=[1.0] * n + [0.055],
        height_ratios=[
            1.0,
            1.0,
            1.0,
            PROFILE_IN / CHIP_IN,
            SERIES_IN / CHIP_IN,
        ],
    )

    naip_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    input_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    spec_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    input_image = spec_image = None
    for column, year in enumerate(years):
        rgb_panel(
            naip_axes[column],
            site.naip(year),
            bounds,
            geometry=stand.geometry,
            gamma=1.2,
            max_px=args.max_px,
        )
        part = parts[year]
        mask, transform = geometry_by_year[year]
        input_image = _input_panel(
            input_axes[column], part, stand.geometry, mask, transform
        )
        spec_image = _spectrum_panel(spec_axes[column], part)

    column_year_labels(naip_axes, years)
    for axes, label in zip(
        (naip_axes, input_axes, spec_axes),
        ("NAIP", "FFT input", "Power spectrum"),
        strict=True,
    ):
        row_label(axes[0], label)

    fig.colorbar(
        input_image,
        cax=fig.add_subplot(grid[1, n]),
        label="Height minus mean (m)",
    )
    fig.colorbar(
        spec_image,
        cax=fig.add_subplot(grid[2, n]),
        label="Power (log10, relative)",
    )

    _profile_panel(fig.add_subplot(grid[3, 0:n]), {y: parts[y] for y in years})
    _series_panel(
        fig.add_subplot(grid[4, 0:n]),
        series,
        context.metrics,
        years=years,
        last_pre=last_pre,
        first_post=first_post,
    )
    return fig


def _input_panel(ax, part: SpectrumParts, geometry, mask, transform):
    """The array handed to the transform, after fill, mean removal and taper.

    Drawn on a symmetric diverging ramp centred on zero, because the array is
    mean-removed and the sign is the point. Everything outside the polygon was
    filled with the in-stand mean, so it sits at exactly zero and reads as flat.
    """
    limit = float(np.nanpercentile(np.abs(part.tapered), 99.0))
    image, _ = band_panel(
        ax,
        part.tapered,
        transform,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        geometry=geometry,
        inside=mask,
        dim_outside=False,
    )
    return image


def _spectrum_panel(ax, part: SpectrumParts):
    """The shifted power spectrum, cropped to the disc the metric analyses.

    Directional structure in the image appears here as a streak through the
    origin, turned 90 degrees from the structure that produced it. A radially
    symmetric blob means no preferred direction at any frequency.
    """
    rows, cols = part.power.shape
    cy, cx = rows // 2, cols // 2
    r = part.radius
    block = part.power[cy - r : cy + r, cx - r : cx + r]

    peak = float(part.power[part.inner].max())
    with np.errstate(divide="ignore"):
        shown = np.log10(np.maximum(block, 0.0) / peak)
    shown = np.clip(shown, -SPECTRUM_DECADES, 0.0)

    image = ax.imshow(
        shown,
        cmap=SPECTRUM_CMAP,
        vmin=-SPECTRUM_DECADES,
        vmax=0.0,
        interpolation="nearest",
    )
    # The circle is the outer edge of the analysed annulus. Everything drawn
    # outside it is in the panel but not in the metric.
    ax.add_patch(plt.Circle((r, r), r, fill=False, color="white", lw=0.7, alpha=0.75))
    ax.set_xticks([])
    ax.set_yticks([])
    return image


#: Angular bin centres in degrees, shared by both figures.
BIN_CENTRES = np.degrees((np.arange(N_BINS) + 0.5) * (np.pi / N_BINS))

#: Profiles are divided by their own peak, never by their own mean. Dividing by
#: the peak puts every panel on one axis that runs 0 to 1, so panels drawn side
#: by side can be compared without a per-panel scale. It also draws the metric
#: directly: with the peak pinned at 1, the dashed mean line sits at
#: ``mean/peak``, and the metric is ``1 -`` that height. A low dashed line is a
#: high value.
PROFILE_YLIM = (0.0, 1.08)


def _profile_panel(ax, parts: dict[int, SpectrumParts]) -> None:
    """Summed power per angular bin, each date divided by its own peak."""
    colors = [ACCENT, HIGHLIGHT, INK]
    for (year, part), color in zip(sorted(parts.items()), colors, strict=False):
        peak = float(part.sums.max())
        ax.plot(
            BIN_CENTRES,
            part.sums / peak,
            "-",
            color=color,
            lw=1.4,
            label=str(year),
        )
        ax.axhline(
            float(part.sums.mean()) / peak, ls="--", lw=1.0, color=color, alpha=0.9
        )
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(*PROFILE_YLIM)
    ax.set_xlabel("Angle of the spatial frequency (degrees)")
    ax.set_ylabel("Power / peak")
    ax.grid(alpha=0.35)
    # Upper centre. The tallest bin on this stand is either the first one or the
    # one near 135 degrees, so both upper corners are usually occupied.
    ax.legend(fontsize=6.5, loc="upper center", ncols=3)


def _series_panel(ax, series, metrics, *, years, last_pre, first_post) -> None:
    """The series over the range of every stand the gate lets through.

    No stage envelope constrains this metric, so the backdrop is the module. It
    is the only thing that shows the harvest moving the value less than the
    undisturbed stands move between consecutive dates.
    """
    ax.set_ylim(0.0, 1.0)
    n_stands = reference_band(
        ax, metrics, METRIC, label="10th to 90th percentile, reported stands"
    )
    ax.axhline(
        FC2_THRESHOLD,
        ls=":",
        lw=1.2,
        color=HIGHLIGHT,
        zorder=2,
        label="FC2 threshold",
    )
    metric_panel(
        ax,
        series["year"],
        series[METRIC],
        label="Row directionality",
        ylim=(0.0, 1.0),
        native_res_m=series["native_res_m"],
        last_pre=last_pre,
        first_post=first_post,
        series_label="this stand",
    )
    mapped = series[series["year"].isin(years)]
    mark_mapped_dates(ax, mapped["year"], mapped[METRIC])
    ax.set_xticks(list(series["year"]))
    if n_stands:
        ax.legend(fontsize=6.5, loc="lower left", ncols=3)


def _reference_patches(site, context, preset) -> list[tuple[str, np.ndarray]]:
    """Synthetic patterns whose answer is known, plus one real stand.

    Every value is computed by the metric's own function at draw time, so
    nothing here is asserted. The pair that carries the figure is ``noise`` and
    ``noise, smoothed``: identical random field, no direction added to either,
    only a Gaussian blur between them.
    """
    size = 256
    rng = np.random.default_rng(0)
    yy, xx = np.indices((size, size))
    white = rng.uniform(0.0, 20.0, size=(size, size)).astype(np.float32)

    def rows(period: float, angle_deg: float) -> np.ndarray:
        t = np.deg2rad(angle_deg)
        proj = xx * np.cos(t) + yy * np.sin(t)
        return (np.sin(2.0 * np.pi * proj / period) * 5.0 + 20.0).astype(np.float32)

    stand = stand_row(context.stands, preset.stand)
    real, _, _, _ = _stand_tile(site, stand, int(preset.years[0]))

    return [
        ("rows", rows(8.0, 0.0)),
        ("rows, turned 30°", rows(8.0, 30.0)),
        ("noise", white),
        ("noise, smoothed", gaussian_filter(white, 8.0)),
        (f"{preset.stand.split('-')[1]}, {preset.years[0]}", real),
    ]


def _scale_figure(site, context, preset, args):
    """What a value means: known patterns against the whole module."""
    patches = _reference_patches(site, context, preset)
    parts = []
    for name, patch in patches:
        part = spectrum_parts(patch)
        check_value(
            f"{METRIC} patch {name}",
            part.value,
            fft_directionality(patch),
            tolerance=1e-9,
        )
        parts.append(part)
        logger.info("patch %-18s -> %.4f", name, part.value)

    n = len(patches)
    fig = plt.figure(
        figsize=(1.62 * n + 0.9, 1.62 * 2 + PROFILE_IN + STRIP_IN),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.03, wspace=0.03)
    grid = fig.add_gridspec(
        4,
        n,
        height_ratios=[1.0, 1.0, PROFILE_IN / 1.62, STRIP_IN / 1.62],
    )

    patch_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    spec_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    prof_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    for column, ((_, patch), part) in enumerate(zip(patches, parts, strict=True)):
        finite = np.isfinite(patch)
        lo, hi = np.percentile(patch[finite], [2, 98])
        patch_axes[column].imshow(
            patch, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest"
        )
        patch_axes[column].set_xticks([])
        patch_axes[column].set_yticks([])
        _spectrum_panel(spec_axes[column], part)

        peak = float(part.sums.max())
        ax = prof_axes[column]
        ax.plot(BIN_CENTRES, part.sums / peak, "-", color=ACCENT, lw=1.2)
        # Dividing by the peak is what lets all five panels share one axis, so
        # hiding the repeated tick labels is honest here. The dashed line is
        # mean/peak, and the metric is one minus its height.
        ax.axhline(float(part.sums.mean()) / peak, ls="--", lw=0.9, color=HIGHLIGHT)
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks([0, 90, 180])
        ax.set_ylim(*PROFILE_YLIM)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
        if column:
            ax.set_yticklabels([])

    column_year_labels(patch_axes, [name for name, _ in patches], fontsize=7.5)
    row_label(patch_axes[0], "Input")
    row_label(spec_axes[0], "Power spectrum")
    prof_axes[0].set_ylabel("Power / peak", fontsize=8, color=MUTED)
    prof_axes[n // 2].set_xlabel(
        "Angle of the spatial frequency (degrees)", fontsize=8, color=MUTED
    )

    _value_strip(fig.add_subplot(grid[3, 0:n]), context.metrics, parts)
    return fig


def _value_strip(ax, metrics, parts) -> None:
    """Every reported value in the module, against the anchors above it.

    The point of the row is where the FC2 threshold falls. It sits below almost
    every value the module produces, so a rule meant to find plantations selects
    ordinary hardwood forest instead.
    """
    values = metrics[METRIC].dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    ax.scatter(
        values,
        rng.uniform(-0.35, 0.35, size=values.size),
        s=11.0,
        color=ACCENT,
        alpha=0.55,
        linewidths=0.0,
        label=f"{values.size} reported stand-dates",
    )
    ax.axvline(FC2_THRESHOLD, ls=":", lw=1.3, color=HIGHLIGHT, label="FC2 threshold")
    ax.axvline(CEILING, ls="--", lw=1.0, color=MUTED, label=f"ceiling, 1 - 1/{N_BINS}")
    ax.scatter(
        [p.value for p in parts],
        np.full(len(parts), 0.72),
        marker="v",
        s=34.0,
        color=INK,
        zorder=4,
        clip_on=False,
        label="patches above",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Row directionality")
    ax.grid(alpha=0.35, axis="x")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    # Left of 0.4 is empty on this axis because no stand in the module reads
    # that low, which is the only free space the legend has.
    ax.legend(fontsize=6.5, loc="lower left", ncols=1)

    above = int((values >= FC2_THRESHOLD).sum())
    logger.info(
        "%d of %d reported values clear the FC2 threshold of %.2f",
        above,
        values.size,
        FC2_THRESHOLD,
    )


if __name__ == "__main__":
    raise SystemExit(main())
