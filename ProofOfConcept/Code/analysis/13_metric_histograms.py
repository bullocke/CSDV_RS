"""
13_metric_histograms.py — Distribution comparison histograms: NEON vs NAIP CHM
structural metrics at 25, 50, and 100 m window sizes.

Purpose
-------
The scatter plots and difference maps (09, 10) compare NEON and NAIP metrics
spatially, cell by cell. Histograms complement those by showing the full
distribution shape for each metric and data source. They answer a different
question: does the NAIP CHM produce the same distribution of values as NEON,
or does it systematically shift the distribution (e.g., skewing gap fraction
higher)? A distribution shift would bias stage classification thresholds even
if the spatial correlation is high.

Layout
------
Two figures, one per metric type:

  fig_hist_gapfrac_{SITE}.png   — 1 row × 3 cols (25, 50, 100 m)
  fig_hist_crownCV_{SITE}.png   — 1 row × 3 cols (25, 50, 100 m)

Each panel:
  - Overlapping histograms: NEON (blue, alpha=0.6) and NAIP (orange, alpha=0.6)
  - Vertical lines at NEON mean (blue dashed) and NAIP mean (orange dashed)
  - Annotation box: mean, median, std, N for each source
  - Shared x-axis range within each figure (across all three window sizes)
  - KDE curves overlaid for smoother shape comparison

Inputs
------
Uses mediumzoom intermediate gap fraction and crown CV rasters already
computed by 08_compare_chm_sources.py (or 09_metric_differences.py).

Usage
-----
  python 13_metric_histograms.py --site SCBI
  python 13_metric_histograms.py --site HARV
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import gaussian_kde

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from poc_lib import get_site, read_band
from poc_lib import figures as fig_utils

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures"

WINDOW_SIZES = [25.0, 50.0, 100.0]

# Colors: NEON blue, NAIP orange — deliberately distinct, colour-blind friendly
_NEON_COLOR = "#1565c0"   # deep blue
_NAIP_COLOR = "#e65100"   # deep orange


def _load_values(path: Path) -> np.ndarray:
    """Load all non-NaN values from a raster as a flat float32 array."""
    arr, _, _ = read_band(path)
    return arr[~np.isnan(arr)].ravel()


def _describe(vals: np.ndarray) -> str:
    """Return a compact stats string for annotation."""
    if vals.size == 0:
        return "No data"
    return (
        f"mean={vals.mean():.3f}\n"
        f"med={np.median(vals):.3f}\n"
        f"std={vals.std():.3f}\n"
        f"N={vals.size}"
    )


def _kde_curve(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Return KDE density evaluated on x_grid, or zeros if insufficient data."""
    if vals.size < 5:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(vals, bw_method="scott")
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def make_histogram_figure(
    neon_paths: dict[float, Path],
    naip_paths: dict[float, Path],
    out_path: Path,
    metric_name: str,
    xlabel: str,
    site_label: str,
    n_bins: int = 35,
    x_lim: tuple[float, float] | None = None,
) -> None:
    """1×3 histogram figure comparing NEON and NAIP metric distributions.

    Parameters
    ----------
    neon_paths, naip_paths : dict mapping window_m → Path
    metric_name : str  label used in suptitle
    xlabel : str  x-axis label (with units)
    site_label : str  site description for suptitle
    n_bins : int  number of histogram bins
    x_lim : (lo, hi) or None  shared x-axis range; auto if None
    """
    fig_utils.setup_style()
    window_sizes = sorted(set(neon_paths) | set(naip_paths))

    # Determine shared x-axis limits from combined data across all windows.
    # x_lim may be None (auto both), or a tuple where either bound may be None (auto that bound).
    if x_lim is None or None in x_lim:
        all_vals: list[np.ndarray] = []
        for w in window_sizes:
            if w in neon_paths and neon_paths[w].exists():
                all_vals.append(_load_values(neon_paths[w]))
            if w in naip_paths and naip_paths[w].exists():
                all_vals.append(_load_values(naip_paths[w]))
        if all_vals:
            combined = np.concatenate(all_vals)
            auto_lo = float(np.percentile(combined, 0.5))
            auto_hi = float(np.percentile(combined, 99.5))
        else:
            auto_lo, auto_hi = 0.0, 1.0
        if x_lim is None:
            x_lim = (auto_lo, auto_hi)
        else:
            x_lim = (auto_lo if x_lim[0] is None else x_lim[0],
                     auto_hi if x_lim[1] is None else x_lim[1])

    fig, axes = plt.subplots(1, len(window_sizes), figsize=(4.5 * len(window_sizes), 4.2),
                             sharey=False)
    if len(window_sizes) == 1:
        axes = [axes]

    fig.suptitle(
        f"{site_label} — {metric_name} distribution: NEON ALS vs NAIP CHM",
        fontsize=9, fontweight="bold",
    )

    bins = np.linspace(x_lim[0], x_lim[1], n_bins + 1)
    x_grid = np.linspace(x_lim[0], x_lim[1], 300)

    for ax, w in zip(axes, window_sizes):
        neon_vals = _load_values(neon_paths[w]) if w in neon_paths and neon_paths[w].exists() else np.array([])
        naip_vals = _load_values(naip_paths[w]) if w in naip_paths and naip_paths[w].exists() else np.array([])

        # Histograms (density=True so KDE is on the same scale)
        if neon_vals.size:
            ax.hist(neon_vals, bins=bins, density=True,
                    color=_NEON_COLOR, alpha=0.45, label="NEON ALS CHM", zorder=2)
            kde_neon = _kde_curve(neon_vals, x_grid)
            ax.plot(x_grid, kde_neon, color=_NEON_COLOR, lw=1.8, zorder=4)
            ax.axvline(neon_vals.mean(), color=_NEON_COLOR, lw=1.3, ls="--", zorder=5)

        if naip_vals.size:
            ax.hist(naip_vals, bins=bins, density=True,
                    color=_NAIP_COLOR, alpha=0.45, label="NAIP CHM (Morford 2025)", zorder=3)
            kde_naip = _kde_curve(naip_vals, x_grid)
            ax.plot(x_grid, kde_naip, color=_NAIP_COLOR, lw=1.8, zorder=4)
            ax.axvline(naip_vals.mean(), color=_NAIP_COLOR, lw=1.3, ls="--", zorder=5)

        ax.set_xlim(x_lim)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=8)
        ax.set_title(f"{int(w)} m window", fontsize=9)
        ax.tick_params(labelsize=7)

        # Annotation box: stats for each source
        neon_text = f"NEON:  {_describe(neon_vals)}" if neon_vals.size else "NEON: no data"
        naip_text = f"NAIP:  {_describe(naip_vals)}" if naip_vals.size else "NAIP: no data"
        ann = neon_text + "\n\n" + naip_text
        ax.text(
            0.97, 0.97, ann,
            transform=ax.transAxes, fontsize=6.5,
            va="top", ha="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, lw=0.5),
            zorder=6,
        )

        if ax is axes[0]:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    fig_utils.save_fig(fig, out_path)
    logger.info("Saved: %s", out_path.name)


@click.command()
@click.option("--site", "-s", required=True, help="Site code: SCBI or HARV")
def main(site: str) -> None:
    """Generate gap fraction and crown CV histogram comparison figures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_site(site)
    s = cfg.site_code
    intermediate = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / s
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gap fraction histograms
    # ------------------------------------------------------------------
    logger.info("=== Gap fraction histograms (%s) ===", s)
    gap_neon = {
        w: intermediate / f"gap_frac_neon_{s}_mediumzoom_{int(w)}m.tif"
        for w in WINDOW_SIZES
    }
    gap_naip = {
        w: intermediate / f"gap_frac_naip_{s}_mediumzoom_{int(w)}m.tif"
        for w in WINDOW_SIZES
    }
    # Check all required files exist
    missing = [str(p) for p in list(gap_neon.values()) + list(gap_naip.values()) if not p.exists()]
    if missing:
        logger.error("Missing gap fraction rasters:\n  %s\nRun 08_compare_chm_sources.py first.",
                     "\n  ".join(missing))
        sys.exit(1)

    make_histogram_figure(
        neon_paths=gap_neon,
        naip_paths=gap_naip,
        out_path=_FIGURES_OUT / f"fig_hist_gapfrac_{s}.png",
        metric_name="Gap Fraction",
        xlabel="Gap fraction (0–1)",
        site_label=cfg.label,
        x_lim=(0.0, 1.0),
    )

    # ------------------------------------------------------------------
    # Crown width CV histograms
    # ------------------------------------------------------------------
    logger.info("=== Crown CV histograms (%s) ===", s)
    cv_neon = {
        w: intermediate / f"crown_cv_neon_{s}_mediumzoom_{int(w)}m.tif"
        for w in WINDOW_SIZES
    }
    cv_naip = {
        w: intermediate / f"crown_cv_naip_{s}_mediumzoom_{int(w)}m.tif"
        for w in WINDOW_SIZES
    }
    missing_cv = [str(p) for p in list(cv_neon.values()) + list(cv_naip.values()) if not p.exists()]
    if missing_cv:
        logger.warning("Some crown CV rasters missing — running with available data:\n  %s",
                       "\n  ".join(missing_cv))

    make_histogram_figure(
        neon_paths=cv_neon,
        naip_paths=cv_naip,
        out_path=_FIGURES_OUT / f"fig_hist_crownCV_{s}.png",
        metric_name="Crown Width CV",
        xlabel="Crown CV (σ/μ)",
        site_label=cfg.label,
        x_lim=(0.0, None),  # auto upper bound from data
    )

    logger.info("=== Done: %s ===", s)


if __name__ == "__main__":
    main()
