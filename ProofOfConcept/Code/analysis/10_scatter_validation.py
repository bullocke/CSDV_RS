"""Scatter plots comparing NEON and NAIP CHM metrics across window sizes.

For each site, this script writes two figures and two tables:

    - Gap fraction figure: 1 × 3 scatter panels for 25, 50, and 100 m windows
    - Crown width CV figure: 1 × 3 scatter panels for 25, 50, and 100 m windows
    - Gap fraction table: regression and agreement statistics by window size
    - Crown width CV table: regression and agreement statistics by window size

Each scatter panel shows NEON on the x-axis and NAIP on the y-axis, with:
    - semi-transparent point symbols for individual windows
    - 1:1 reference line
    - OLS regression line
    - annotation box with R², RMSE, bias, and N

Usage
-----
    python 10_scatter_validation.py --site SCBI
    python 10_scatter_validation.py --site HARV
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from poc_lib import (
    crown_stats_per_window,
    figures as fig_utils,
    gap_fraction,
    get_site,
    read_band,
    save_raster,
)
from poc_lib.metrics import metric_difference

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures"
_TABLES_OUT = _POC / "Results" / "summary_document" / "tables"

WINDOW_SIZES = [25.0, 50.0, 100.0]
TABLE_FIELDNAMES = [
    "metric",
    "window_m",
    "slope",
    "intercept",
    "r2",
    "rmse",
    "bias",
    "pearson_r",
    "spearman_r",
    "n",
]


@dataclass(frozen=True)
class MetricConfig:
    """Settings for one scatter-validation metric."""

    metric_key: str
    raster_prefix: str
    title: str
    file_label: str
    figure_subdir: str
    xlabel: str
    ylabel: str
    point_color: str
    regression_color: str


METRIC_CONFIGS = (
    MetricConfig(
        metric_key="gap_fraction",
        raster_prefix="gap_frac",
        title="Gap Fraction",
        file_label="gap_fraction",
        figure_subdir="gap_fraction",
        xlabel="NEON gap fraction",
        ylabel="NAIP gap fraction",
        point_color="#1b9e77",
        regression_color="#d95f02",
    ),
    MetricConfig(
        metric_key="crown_cv",
        raster_prefix="crown_cv",
        title="Crown Width CV",
        file_label="crown_width_cv",
        figure_subdir="crownCV",
        xlabel="NEON crown CV (σ/μ)",
        ylabel="NAIP crown CV (σ/μ)",
        point_color="#1f78b4",
        regression_color="#e31a1c",
    ),
)


def _regression_stats(
    x: np.ndarray, y: np.ndarray
) -> dict[str, float | int]:
    """Compute OLS regression and agreement statistics for paired arrays."""
    from scipy.stats import linregress, spearmanr

    mask = ~np.isnan(x) & ~np.isnan(y)
    xv, yv = x[mask], y[mask]
    n = int(xv.size)

    if n < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "n": n,
        }

    slope, intercept, r_value, _, _ = linregress(xv, yv)
    rmse = float(np.sqrt(np.mean((yv - xv) ** 2)))
    bias = float(np.mean(yv - xv))
    spear_r, _ = spearmanr(xv, yv)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value**2),
        "rmse": float(rmse),
        "bias": float(bias),
        "pearson_r": float(r_value),
        "spearman_r": float(spear_r),
        "n": n,
    }


def _make_scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    window_m: float,
    stats: dict[str, float | int],
    point_color: str,
    regression_color: str,
    xlabel: str = "NEON ALS CHM",
    ylabel: str = "NAIP CHM",
) -> None:
    """Render one scatter panel with reference line, regression, and stats."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    xv, yv = x[mask], y[mask]

    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_title(f"{int(window_m)} m window", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)

    if xv.size < 4:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )
        return

    ax.scatter(
        xv,
        yv,
        s=18,
        c=point_color,
        alpha=0.32,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    lo = float(min(xv.min(), yv.min()))
    hi = float(max(xv.max(), yv.max()))
    span = hi - lo
    pad = 0.04 * span if span > 0 else 0.05
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.plot(
        [lo_plot, hi_plot],
        [lo_plot, hi_plot],
        color="#2f2f2f",
        linestyle="--",
        linewidth=1.1,
        label="1:1",
        zorder=3,
    )

    if not np.isnan(stats["slope"]):
        x_fit = np.array([lo_plot, hi_plot])
        y_fit = stats["slope"] * x_fit + stats["intercept"]
        ax.plot(
            x_fit,
            y_fit,
            color=regression_color,
            linewidth=1.4,
            label="OLS",
            zorder=4,
        )

    ann = (
        f"R²={stats['r2']:.3f}\n"
        f"RMSE={stats['rmse']:.3f}\n"
        f"bias={stats['bias']:+.3f}\n"
        f"N={stats['n']}"
    )
    ax.text(
        0.03,
        0.97,
        ann,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, lw=0.5),
    )


def _metric_raster_path(
    metric_cfg: MetricConfig,
    intermediate: Path,
    site_code: str,
    source: str,
    window_m: float,
) -> Path:
    """Return the cached raster path for one metric, source, and window size."""
    return intermediate / (
        f"{metric_cfg.raster_prefix}_{source}_{site_code}_mediumzoom_{int(window_m)}m.tif"
    )


def _ensure_metric_raster(
    metric_cfg: MetricConfig,
    source: str,
    window_m: float,
    recompute: bool,
    intermediate: Path,
    site_code: str,
    neon_chm: Path,
    naip_chm: Path,
    neon_crowns: Path,
    naip_crowns: Path,
) -> Path | None:
    """Create a cached raster if needed and return its path."""
    raster_path = _metric_raster_path(metric_cfg, intermediate, site_code, source, window_m)
    if raster_path.exists() and not recompute:
        return raster_path

    chm_path = neon_chm if source == "neon" else naip_chm
    if metric_cfg.metric_key == "gap_fraction":
        values, transform, crs = gap_fraction(
            chm_path,
            window_m=window_m,
            height_threshold_m=2.0,
        )
        save_raster(values, raster_path, transform, crs)
        return raster_path

    crowns_path = neon_crowns if source == "neon" else naip_crowns
    values, transform, crs = crown_stats_per_window(
        crowns_path,
        chm_path,
        window_m,
        stat="cv",
    )
    if values is None:
        return None
    save_raster(values, raster_path, transform, crs)
    return raster_path


def _aligned_metric_arrays(neon_path: Path, naip_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened NEON values and grid-aligned NAIP values."""
    diff, _, _ = metric_difference(neon_path, naip_path)
    neon_arr, _, _ = read_band(neon_path)
    naip_aligned = neon_arr + diff
    naip_aligned[np.isnan(diff)] = np.nan
    return neon_arr.ravel(), naip_aligned.ravel()


def _save_stats_table(
    metric_cfg: MetricConfig,
    site_code: str,
    rows: list[dict[str, Any]],
) -> None:
    """Write one per-metric statistics table for a site."""
    if not rows:
        return

    table_path = _TABLES_OUT / (
        f"table_scatter_stats_v2_{metric_cfg.file_label}_{site_code}.csv"
    )
    with open(table_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=TABLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved table: %s", table_path.name)


def _make_metric_figure(
    metric_cfg: MetricConfig,
    site_label: str,
    site_code: str,
    intermediate: Path,
    neon_chm: Path,
    naip_chm: Path,
    neon_crowns: Path,
    naip_crowns: Path,
    recompute: bool,
) -> list[dict[str, Any]]:
    """Generate and save the 1 × 3 scatter figure for one metric."""
    fig_utils.setup_style()
    fig, axes = plt.subplots(1, len(WINDOW_SIZES), figsize=(13.2, 4.2))
    fig.suptitle(
        f"{site_label} — {metric_cfg.title}: NEON vs NAIP CHM comparison",
        fontsize=10,
        fontweight="bold",
    )

    stats_rows: list[dict[str, Any]] = []

    for axis, window_m in zip(axes, WINDOW_SIZES):
        neon_path = _ensure_metric_raster(
            metric_cfg,
            source="neon",
            window_m=window_m,
            recompute=recompute,
            intermediate=intermediate,
            site_code=site_code,
            neon_chm=neon_chm,
            naip_chm=naip_chm,
            neon_crowns=neon_crowns,
            naip_crowns=naip_crowns,
        )
        naip_path = _ensure_metric_raster(
            metric_cfg,
            source="naip",
            window_m=window_m,
            recompute=recompute,
            intermediate=intermediate,
            site_code=site_code,
            neon_chm=neon_chm,
            naip_chm=naip_chm,
            neon_crowns=neon_crowns,
            naip_crowns=naip_crowns,
        )

        if neon_path is None or naip_path is None or not neon_path.exists() or not naip_path.exists():
            axis.set_facecolor("#fbfbfb")
            axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8, zorder=0)
            axis.set_title(f"{int(window_m)} m window", fontsize=9)
            axis.set_xlabel(metric_cfg.xlabel, fontsize=8)
            axis.set_ylabel(metric_cfg.ylabel if axis is axes[0] else "", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
            stats = _regression_stats(np.array([]), np.array([]))
        else:
            neon_values, naip_values = _aligned_metric_arrays(neon_path, naip_path)
            stats = _regression_stats(neon_values, naip_values)
            _make_scatter_panel(
                axis,
                neon_values,
                naip_values,
                window_m,
                stats,
                point_color=metric_cfg.point_color,
                regression_color=metric_cfg.regression_color,
                xlabel=metric_cfg.xlabel,
                ylabel=metric_cfg.ylabel if axis is axes[0] else "",
            )
            if axis is axes[0]:
                axis.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

        stats_rows.append(
            {
                "metric": metric_cfg.metric_key,
                "window_m": int(window_m),
                **stats,
            }
        )

    plt.tight_layout()
    figure_path = _FIGURES_OUT / "metrics" / metric_cfg.figure_subdir / (
        f"fig_scatter_v2_{metric_cfg.file_label}_NEON_vs_NAIP_{site_code}.png"
    )
    fig_utils.save_fig(fig, figure_path)
    return stats_rows


@click.command()
@click.option("--site", "-s", required=True, help="Site code: SCBI or HARV")
@click.option("--recompute", is_flag=True, default=False)
def main(site: str, recompute: bool) -> None:
    """Generate split scatter-validation figures and tables for one site."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_site(site)
    s = cfg.site_code
    intermediate = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / s
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    _TABLES_OUT.mkdir(parents=True, exist_ok=True)

    neon_chm = intermediate / f"neon_chm_{s}_mediumzoom.tif"
    naip_chm = intermediate / f"naip_chm_{s}_mediumzoom_meters.tif"
    neon_crowns = intermediate / f"crowns_neon_{s}_mediumzoom.gpkg"
    naip_crowns = intermediate / f"crowns_naip_{s}_mediumzoom.gpkg"

    for p in [neon_chm, naip_chm]:
        if not p.exists():
            logger.error("Required file missing: %s\nRun 08_compare_chm_sources.py first.", p)
            sys.exit(1)

    for metric_cfg in METRIC_CONFIGS:
        stats_rows = _make_metric_figure(
            metric_cfg=metric_cfg,
            site_label=cfg.label,
            site_code=s,
            intermediate=intermediate,
            neon_chm=neon_chm,
            naip_chm=naip_chm,
            neon_crowns=neon_crowns,
            naip_crowns=naip_crowns,
            recompute=recompute,
        )
        _save_stats_table(metric_cfg, s, stats_rows)

    logger.info("=== Done: %s ===", s)


if __name__ == "__main__":
    main()
