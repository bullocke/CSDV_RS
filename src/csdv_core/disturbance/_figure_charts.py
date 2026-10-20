"""csdv_core.disturbance._figure_charts: charts and CHM chips of a disturbance run.

Null-fit histograms, the detected area per interval and the refined-outline
examples behind the public functions of :mod:`csdv_core.disturbance.figures`.
CHM chips are read through :func:`csdv_core.disturbance.windows.read_chm_bounds`.

Inputs: z stacks, fits, interval summaries, object GeoDataFrames. Outputs:
PNG files.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance import windows
from csdv_core.disturbance._figure_helpers import DEFAULT_DPI, NO_DATA_COLOR
from csdv_core.disturbance.decide import DateFit
from csdv_core.disturbance.inputs import SiteInputs
from csdv_core.disturbance.report import IntervalSummary
from csdv_core.viz.maps import CHM_VMAX_M
from csdv_core.viz.relief import canopy_cmap
from csdv_core.viz.style import ACCENT, GRID, HIGHLIGHT, INK, MUTED, save_fig

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

#: Number of objects shown by :func:`plot_refined_examples`.
DEFAULT_N_EXAMPLES = 6
#: Histogram fill of the null-fit panels.
HISTOGRAM_COLOR = "#c9ccd1"
#: Number of histogram bins in the running mean used for the observed density.
DENSITY_SMOOTH_BINS = 5
#: Smallest y-axis top of the area chart, in hectares (keeps empty runs drawable).
MIN_AXIS_HA = 1.0
#: Colour of the cell object outline on refined examples.
CELL_OUTLINE_COLOR = "white"
#: Colour of the refined envelope on refined examples.
ENVELOPE_COLOR = "#eb6834"


def _null_density(z_values: np.ndarray, fit: DateFit) -> np.ndarray:
    """Scaled null density ``p0 N(z; mu0, sigma0)``.

    Args:
        z_values: z values to evaluate.
        fit: The date's null fit.

    Returns:
        Density at each value.
    """
    standard_scores = (z_values - fit.mu0) / fit.sigma0
    gaussian = np.exp(-0.5 * standard_scores**2)
    normalizer = fit.sigma0 * np.sqrt(2.0 * np.pi)
    return fit.p0 * gaussian / normalizer


def _null_boundary(
    bin_centres: np.ndarray, density: np.ndarray, fit: DateFit, p_boundary: float
) -> float | None:
    """Loss-side z where the cell posterior ``1 - lfdr`` reaches ``p_boundary``.

    The observed density is the histogram smoothed over
    :data:`DENSITY_SMOOTH_BINS` bins, and ``lfdr = p0 f0 / f`` uses the fit's
    null. Returns the largest bin centre below ``mu0`` with a posterior at or
    above ``p_boundary``.

    Args:
        bin_centres: Histogram bin centres.
        density: Histogram density per bin.
        fit: The date's null fit.
        p_boundary: Decision boundary.

    Returns:
        The boundary z, or None when no bin below ``mu0`` reaches it.
    """
    smoothed = uniform_filter1d(density, size=DENSITY_SMOOTH_BINS, mode="nearest")
    null_density = _null_density(bin_centres, fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_fdr = np.where(smoothed > 0, null_density / smoothed, 1.0)
    clipped_fdr = np.clip(local_fdr, 0.0, 1.0)
    posterior = 1.0 - clipped_fdr
    candidates = bin_centres[(bin_centres < fit.mu0) & (posterior >= p_boundary)]
    if candidates.size == 0:
        return None
    boundary_z = candidates.max()
    return float(boundary_z)


def draw_null_fit(
    ax: Axes,
    z_values: np.ndarray,
    fit: DateFit,
    *,
    bin_w: float,
    p_boundary: float | None = None,
) -> None:
    """Histogram of 1-D monitored z with the fitted null ``p0 N(mu0, sigma0)``.

    A skipped fit draws the histogram only, with "skipped" in the title.
    When ``p_boundary`` is given, the loss-side z where the cell posterior
    ``1 - p0 f0 / f`` reaches it is marked (``f`` from the histogram).

    Args:
        ax: Chart axis.
        z_values: 1-D finite z of monitored cells.
        fit: The date's null fit (its values are drawn, nothing is refitted).
        bin_w: Histogram bin width.
        p_boundary: Decision boundary to mark, or None.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("z")
    if z_values.size == 0:
        ax.set_title(f"{fit.year}: no monitored cells")
        return
    low_z, high_z = np.percentile(z_values, [0.1, 99.9])
    if not fit.skipped:
        low_z = min(low_z, fit.mu0 - 5.0 * fit.sigma0)
        high_z = max(high_z, fit.mu0 + 5.0 * fit.sigma0)
    bin_edges = np.arange(low_z, high_z + bin_w, bin_w)
    in_range = z_values[(z_values >= low_z) & (z_values <= high_z)]
    counts, bin_edges = np.histogram(in_range, bins=bin_edges)
    density = counts / (z_values.size * bin_w)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.stairs(density, bin_edges, fill=True, color=HISTOGRAM_COLOR, linewidth=0)
    ax.set_yscale("log")
    density_floor = 0.5 / (z_values.size * bin_w)
    ax.set_ylim(bottom=density_floor)
    ax.set_xlim(low_z, high_z)
    if fit.skipped:
        ax.set_title(f"{fit.year}: skipped (n = {fit.n_cells:,})")
        return
    null_density = _null_density(bin_centres, fit)
    ax.plot(bin_centres, null_density, color=ACCENT, linewidth=1.4)
    ax.axvline(fit.mu0, color=ACCENT, linewidth=0.8, linestyle=":")
    if p_boundary is not None:
        boundary_z = _null_boundary(bin_centres, density, fit, p_boundary)
        if boundary_z is not None:
            ax.axvline(boundary_z, color=HIGHLIGHT, linewidth=1.0, linestyle="--")
    ax.set_title(
        f"{fit.year}: μ0 {fit.mu0:.2f}, σ0 {fit.sigma0:.2f}, p0 {fit.p0:.2f}",
        fontsize=8,
    )


def plot_null_fits(
    z: np.ndarray,
    monitored: np.ndarray,
    fits: Sequence[DateFit],
    out_path: Path,
    *,
    bin_w: float,
    p_boundary: float | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One :func:`draw_null_fit` panel per interval (dates ``t >= 1``).

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        fits: One :class:`DateFit` per date.
        out_path: PNG to write.
        bin_w: Histogram bin width.
        p_boundary: Decision boundary to mark, or None.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    n_dates = len(fits)
    n_intervals = n_dates - 1
    figure, axes = helpers.panel_axes(n_intervals, 2.8, 2.4, extra_height=0.4)
    for date_index in range(1, n_dates):
        ax = axes[date_index - 1]
        usable = monitored[date_index] & np.isfinite(z[date_index])
        z_values = z[date_index][usable]
        draw_null_fit(
            ax, z_values, fits[date_index], bin_w=bin_w, p_boundary=p_boundary
        )
    axes[0].set_ylabel("Density (log)")
    handles = [
        Patch(color=HISTOGRAM_COLOR, label="Monitored cells"),
        Line2D([], [], color=ACCENT, label="Fitted null p0·N(μ0, σ0)"),
    ]
    if p_boundary is not None:
        boundary_label = f"Cell boundary (1 − lfdr = {p_boundary:g})"
        boundary_handle = Line2D(
            [], [], color=HIGHLIGHT, linestyle="--", label=boundary_label
        )
        handles.append(boundary_handle)
    n_handles = len(handles)
    figure.legend(handles=handles, loc="outside lower center", ncol=n_handles)
    return save_fig(figure, out_path, dpi=dpi)


def draw_area_by_interval(ax: Axes, intervals: Sequence[IntervalSummary]) -> None:
    """Bars of detected area (ha) per interval, labelled with share of monitored.

    Each interval shows the area before the MMU (pale, behind) and after the
    MMU (solid). Labels give the share of monitored forest after the MMU.
    Skipped intervals are labelled "skipped".

    Args:
        ax: Chart axis.
        intervals: One summary per interval.
    """
    n_intervals = len(intervals)
    positions = np.arange(n_intervals)
    labels = [interval.label for interval in intervals]
    detected = np.array([interval.detected_ha for interval in intervals])
    detected_nommu = np.array([interval.detected_ha_nommu for interval in intervals])
    ax.bar(positions, detected_nommu, width=0.62, color=GRID, label="Before MMU")
    ax.bar(positions, detected, width=0.42, color=ACCENT, label="After MMU")
    largest_area = detected_nommu.max(initial=0.0)
    top_value = max(largest_area, MIN_AXIS_HA)
    for position, interval in zip(positions, intervals, strict=True):
        if interval.skipped:
            text = "skipped"
        else:
            share_percent = 100.0 * interval.share_of_monitored
            text = f"{share_percent:.2f}%"
        label_height = max(interval.detected_ha, interval.detected_ha_nommu)
        ax.text(
            position,
            label_height + 0.02 * top_value,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, top_value * 1.15)
    ax.set_ylabel("Detected area (ha)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")


def plot_area_by_interval(
    intervals: Sequence[IntervalSummary], out_path: Path, *, dpi: int = DEFAULT_DPI
) -> Path:
    """Detected area per interval (after and before the MMU).

    Args:
        intervals: One summary per interval.
        out_path: PNG to write.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    n_intervals = len(intervals)
    figure_width = max(3.5, 1.1 * n_intervals + 1.5)
    figure, ax = plt.subplots(figsize=(figure_width, 3.0))
    draw_area_by_interval(ax, intervals)
    ax.set_title("Detected area (labels: share of monitored forest after MMU)")
    return save_fig(figure, out_path, dpi=dpi)


def _read_chm_chip(
    site_inputs: SiteInputs, year: int, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a 0.6 m CHM chip in metres and its ``imshow`` extent.

    Args:
        site_inputs: Inputs and grid of the run.
        year: NAIP year.
        bounds: ``(left, bottom, right, top)``.

    Returns:
        ``(chm, extent)``.
    """
    chm, transform = windows.read_chm_bounds(site_inputs, year, bounds)
    extent = helpers.transform_extent(transform, chm.shape)
    return chm, extent


def plot_refined_examples(
    objects: gpd.GeoDataFrame,
    refined: gpd.GeoDataFrame,
    site_inputs: SiteInputs,
    out_path: Path,
    *,
    n_examples: int = DEFAULT_N_EXAMPLES,
    pad_m: float = 60.0,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """CHM before and after for the largest refined objects, with outlines.

    One row per object (the ``n_examples`` largest by ``area_ha`` that have a
    refined outline): CHM at ``year_before`` and at ``onset_year`` over the
    object bounds buffered by ``pad_m``, with the cell object and the refined
    envelope drawn on both.

    Args:
        objects: Objects GeoDataFrame.
        refined: Refined outlines GeoDataFrame.
        site_inputs: Inputs and grid of the run (CHM files).
        out_path: PNG to write.
        n_examples: Number of objects shown.
        pad_m: Margin around each object, in metres.
        dpi: Output resolution.

    Returns:
        ``out_path``.

    Raises:
        ValueError: If no object has a refined outline.
    """
    refined_ids = set(refined["object_id"])
    has_outline = objects["object_id"].isin(refined_ids)
    candidates = objects[has_outline]
    if candidates.empty:
        raise ValueError("No object has a refined outline")
    examples = candidates.nlargest(n_examples, "area_ha")
    envelopes = refined.set_index("object_id")
    n_rows = len(examples)
    figure, axes_grid = plt.subplots(
        n_rows, 2, figsize=(5.6, 2.7 * n_rows), squeeze=False, layout="constrained"
    )
    colormap = canopy_cmap()
    colormap.set_bad(NO_DATA_COLOR)
    image = None
    example_rows = examples.itertuples()
    for row_axes, example in zip(axes_grid, example_rows, strict=True):
        envelope = envelopes.loc[example.object_id, "geometry"]
        bounds = helpers.square_bounds(example.geometry, pad_m)
        years = (example.year_before, example.onset_year)
        for ax, year in zip(row_axes, years, strict=True):
            chm, extent = _read_chm_chip(site_inputs, year, bounds)
            image = ax.imshow(
                chm,
                cmap=colormap,
                vmin=0,
                vmax=CHM_VMAX_M,
                extent=extent,
                interpolation="nearest",
            )
            helpers.outline(
                ax,
                [example.geometry],
                CELL_OUTLINE_COLOR,
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                halo=INK,
                halo_width=0.8,
            )
            helpers.outline(
                ax, [envelope], ENVELOPE_COLOR, linewidth=0.9, halo=INK, halo_width=0.8
            )
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            helpers.bare(ax)
            helpers.scale_bar(ax, bounds, color="white", halo=INK)
            year_label = str(year)
            ax.set_title(year_label)
        envelope_ha = envelopes.loc[example.object_id, "envelope_ha"]
        row_axes[0].set_ylabel(
            f"Object {example.object_id}\n{example.area_ha:.2f} ha cells, "
            f"{envelope_ha:.2f} ha refined",
            fontsize=7,
        )
        row_axes[0].yaxis.label.set_visible(True)
    colorbar = figure.colorbar(
        image,
        ax=axes_grid,
        location="bottom",
        shrink=0.7,
        aspect=30,
        fraction=0.02,
        pad=0.01,
        extend="max",
    )
    colorbar.set_label("Canopy height (m)")
    handles = [
        Line2D([], [], color=MUTED, linestyle=(0, (3, 2)), label="Detected cells"),
        Line2D([], [], color=ENVELOPE_COLOR, linewidth=1.2, label="Refined envelope"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=2)
    return save_fig(figure, out_path, dpi=dpi)
