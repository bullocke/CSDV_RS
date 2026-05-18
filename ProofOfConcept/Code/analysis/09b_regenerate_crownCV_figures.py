"""
09b_regenerate_crownCV_figures.py — Regenerate crown CV difference figures
using the fixed crown_stats_per_window (floor division instead of np.ceil).

Saves figures with '_v2' suffix to avoid overwriting originals.
Also produces a diagnostic 'crown count' figure showing how many crowns
fall in each window cell.

Usage
-----
  python 09b_regenerate_crownCV_figures.py --site SCBI
  python 09b_regenerate_crownCV_figures.py --site HARV
  python 09b_regenerate_crownCV_figures.py --site all
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from poc_lib import (
    crown_stats_per_window,
    figures as fig_utils,
    gap_fraction,
    get_site,
    metric_difference,
    read_band,
    save_raster,
)

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_RESULTS = _POC / "Results" / "summary_document"
_FIGURES_OUT = _RESULTS / "figures" / "metrics" / "crownCV"

WINDOW_SIZES = [25.0, 50.0, 100.0]


def _load_metric_raster(path: Path) -> tuple[np.ndarray, float]:
    arr, transform, _ = read_band(path)
    pixel_size = abs(transform.a)
    return arr, pixel_size


def _diff_stats(diff: np.ndarray) -> tuple[float, float, int]:
    valid = diff[~np.isnan(diff)]
    if valid.size == 0:
        return np.nan, np.nan, 0
    rmse = float(np.sqrt(np.mean(valid**2)))
    bias = float(np.mean(valid))
    return rmse, bias, int(valid.size)


def make_diff_figure(
    naip_rgb_path: Path,
    neon_paths: dict[float, Path],
    naip_paths: dict[float, Path],
    diff_paths: dict[float, Path],
    out_path: Path,
    metric_name: str,
    metric_label: str,
    site_label: str,
    cmap_metric: str = "viridis",
    cmap_diff: str = "RdBu_r",
) -> None:
    """3x4 figure: NAIP RGB | NEON metric | NAIP metric | difference."""
    fig_utils.setup_style()
    window_sizes = sorted(neon_paths)
    n_rows = len(window_sizes)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, n_rows * 3.6))

    with rasterio.open(naip_rgb_path) as src:
        rgb_data = src.read()[:3].astype(np.float32)
    rgb_disp = np.zeros((*rgb_data.shape[1:], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_data[i]
        lo = float(np.percentile(band[band > 0], 2)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], 98)) if np.any(band > 0) else 1.0
        rgb_disp[..., i] = np.clip(
            (band - lo) / (hi - lo + 1e-8) * 255, 0, 255
        ).astype(np.uint8)

    # Shared vmin/vmax for metric panels
    metric_vals = []
    for w in window_sizes:
        for paths in (neon_paths, naip_paths):
            if w in paths and paths[w].exists():
                arr, _ = _load_metric_raster(paths[w])
                v = arr[~np.isnan(arr)]
                if v.size:
                    metric_vals.append(v)
    if metric_vals:
        all_metric = np.concatenate(metric_vals)
        metric_vmin = float(np.nanpercentile(all_metric, 2))
        metric_vmax = float(np.nanpercentile(all_metric, 98))
    else:
        metric_vmin, metric_vmax = 0.0, 1.0

    # Symmetric vmax for difference column
    diff_abs_max = 0.0
    for w in window_sizes:
        if w in diff_paths and diff_paths[w].exists():
            diff, _ = _load_metric_raster(diff_paths[w])
            v = diff[~np.isnan(diff)]
            if v.size:
                diff_abs_max = max(
                    diff_abs_max, float(np.nanpercentile(np.abs(v), 98))
                )
    if diff_abs_max == 0.0:
        diff_abs_max = 0.1

    last_metric_im = None
    last_diff_im = None

    for row_i, w in enumerate(window_sizes):
        ax_rgb, ax_neon, ax_naip, ax_diff = axes[row_i]

        ax_rgb.imshow(rgb_disp)
        ax_rgb.set_axis_off()
        ax_rgb.set_ylabel(f"{int(w)} m", fontsize=9, labelpad=4)
        fig_utils.panel_label(ax_rgb, chr(ord("a") + row_i * 4))

        if w in neon_paths and neon_paths[w].exists():
            neon_arr, _ = _load_metric_raster(neon_paths[w])
            last_metric_im = ax_neon.imshow(
                neon_arr,
                cmap=cmap_metric,
                vmin=metric_vmin,
                vmax=metric_vmax,
                interpolation="nearest",
            )
        ax_neon.set_axis_off()
        fig_utils.panel_label(ax_neon, chr(ord("b") + row_i * 4))

        if w in naip_paths and naip_paths[w].exists():
            naip_arr, _ = _load_metric_raster(naip_paths[w])
            ax_naip.imshow(
                naip_arr,
                cmap=cmap_metric,
                vmin=metric_vmin,
                vmax=metric_vmax,
                interpolation="nearest",
            )
        ax_naip.set_axis_off()
        fig_utils.panel_label(ax_naip, chr(ord("c") + row_i * 4))

        if w in diff_paths and diff_paths[w].exists():
            diff_arr, _ = _load_metric_raster(diff_paths[w])
            rmse, bias, n = _diff_stats(diff_arr)
            last_diff_im = ax_diff.imshow(
                diff_arr,
                cmap=cmap_diff,
                vmin=-diff_abs_max,
                vmax=diff_abs_max,
                interpolation="nearest",
            )
            stats_text = f"RMSE={rmse:.3f}\nbias={bias:+.3f}\nN={n}"
            ax_diff.text(
                0.98,
                0.02,
                stats_text,
                transform=ax_diff.transAxes,
                fontsize=7,
                ha="right",
                va="bottom",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0
                ),
            )
        ax_diff.set_axis_off()
        fig_utils.panel_label(ax_diff, chr(ord("d") + row_i * 4))

    axes[0, 0].set_title("NAIP RGB", fontsize=9)
    axes[0, 1].set_title(f"NEON {metric_name}", fontsize=9)
    axes[0, 2].set_title(f"NAIP {metric_name}", fontsize=9)
    axes[0, 3].set_title("NAIP \u2212 NEON", fontsize=9)

    if last_metric_im is not None:
        cb1 = fig.colorbar(
            last_metric_im,
            ax=axes[:, 1:3].ravel().tolist(),
            fraction=0.015,
            pad=0.03,
        )
        cb1.set_label(metric_label, fontsize=8)
        cb1.ax.tick_params(labelsize=7)

    if last_diff_im is not None:
        cb2 = fig.colorbar(
            last_diff_im, ax=axes[:, 3].tolist(), fraction=0.03, pad=0.02
        )
        cb2.set_label(f"Difference ({metric_label})", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{site_label} \u2014 {metric_name} comparison and difference",
        fontsize=10,
        y=1.01,
    )
    fig_utils.save_fig(fig, out_path)
    logger.info("Saved: %s", out_path.name)


def make_count_figure(
    naip_rgb_path: Path,
    neon_count_paths: dict[float, Path],
    naip_count_paths: dict[float, Path],
    out_path: Path,
    site_label: str,
) -> None:
    """Diagnostic figure showing crown count per window cell.

    Each row corresponds to one window size; the leftmost panel of each row
    carries a bold y-axis label naming that window size. The suptitle is
    placed close to the top row so the figure does not have a large white
    margin above the panels.
    """
    fig_utils.setup_style()
    window_sizes = sorted(neon_count_paths)
    n_rows = len(window_sizes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, n_rows * 3.4))

    with rasterio.open(naip_rgb_path) as src:
        rgb_data = src.read()[:3].astype(np.float32)
    rgb_disp = np.zeros((*rgb_data.shape[1:], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_data[i]
        lo = float(np.percentile(band[band > 0], 2)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], 98)) if np.any(band > 0) else 1.0
        rgb_disp[..., i] = np.clip(
            (band - lo) / (hi - lo + 1e-8) * 255, 0, 255
        ).astype(np.uint8)

    # Find global max for consistent color scale
    count_max = 1.0
    for paths in (neon_count_paths, naip_count_paths):
        for w, p in paths.items():
            if p.exists():
                arr, _ = _load_metric_raster(p)
                v = arr[~np.isnan(arr)]
                if v.size:
                    count_max = max(count_max, float(np.nanmax(v)))

    last_im = None
    for row_i, w in enumerate(window_sizes):
        ax_rgb, ax_neon, ax_naip = axes[row_i]

        ax_rgb.imshow(rgb_disp)
        # Suppress ticks/spines but keep the axis on so the ylabel is visible.
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        for spine in ax_rgb.spines.values():
            spine.set_visible(False)
        ax_rgb.set_ylabel(
            f"{int(w)} m window",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
        )
        fig_utils.panel_label(ax_rgb, chr(ord("a") + row_i * 3))

        if w in neon_count_paths and neon_count_paths[w].exists():
            arr, _ = _load_metric_raster(neon_count_paths[w])
            last_im = ax_neon.imshow(
                arr, cmap="YlOrRd", vmin=0, vmax=count_max,
                interpolation="nearest",
            )
            valid = arr[~np.isnan(arr)]
            n_ge3 = int(np.sum(valid >= 3))
            ax_neon.text(
                0.98, 0.02,
                f"cells\u22653: {n_ge3}/{arr.size}",
                transform=ax_neon.transAxes, fontsize=7,
                ha="right", va="bottom", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0),
            )
        ax_neon.set_axis_off()
        fig_utils.panel_label(ax_neon, chr(ord("b") + row_i * 3))

        if w in naip_count_paths and naip_count_paths[w].exists():
            arr, _ = _load_metric_raster(naip_count_paths[w])
            last_im = ax_naip.imshow(
                arr, cmap="YlOrRd", vmin=0, vmax=count_max,
                interpolation="nearest",
            )
            valid = arr[~np.isnan(arr)]
            n_ge3 = int(np.sum(valid >= 3))
            ax_naip.text(
                0.98, 0.02,
                f"cells\u22653: {n_ge3}/{arr.size}",
                transform=ax_naip.transAxes, fontsize=7,
                ha="right", va="bottom", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0),
            )
        ax_naip.set_axis_off()
        fig_utils.panel_label(ax_naip, chr(ord("c") + row_i * 3))

    axes[0, 0].set_title("NAIP RGB", fontsize=9)
    axes[0, 1].set_title("NEON crown count", fontsize=9)
    axes[0, 2].set_title("NAIP crown count", fontsize=9)

    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.05, right=0.92,
                        hspace=0.08, wspace=0.04)

    if last_im is not None:
        cb = fig.colorbar(
            last_im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.03,
        )
        cb.set_label("Crowns per window", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{site_label} \u2014 Crown count per window (diagnostic)",
        fontsize=11, y=0.985,
    )
    # Save without bbox_inches="tight" so the manual subplots_adjust top
    # margin is preserved (otherwise tight bbox re-trims and the title
    # drifts away from the panels again).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", out_path.name)


def run_site(site: str) -> None:
    """Recompute crown CV with fixed grid and save figures with _v2 suffix."""
    cfg = get_site(site)
    s = cfg.site_code
    intermediate = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / s
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    neon_chm = intermediate / f"neon_chm_{s}_mediumzoom.tif"
    naip_chm = intermediate / f"naip_chm_{s}_mediumzoom_meters.tif"
    naip_rgb = intermediate / f"naip_rgb_{s}_mediumzoom.tif"
    neon_crowns = intermediate / f"crowns_neon_{s}_mediumzoom.gpkg"
    naip_crowns = intermediate / f"crowns_naip_{s}_mediumzoom.gpkg"

    if not neon_chm.exists() or not naip_chm.exists():
        logger.error("Missing CHM files in %s", intermediate)
        return
    if not neon_crowns.exists() or not naip_crowns.exists():
        logger.error("Missing crown GeoPackages in %s", intermediate)
        return

    # Use a v2 subdirectory for recomputed intermediates
    v2_dir = intermediate / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    cv_neon: dict[float, Path] = {}
    cv_naip: dict[float, Path] = {}
    cv_diff: dict[float, Path] = {}
    count_neon: dict[float, Path] = {}
    count_naip: dict[float, Path] = {}

    for w in WINDOW_SIZES:
        wi = int(w)

        # NEON crown CV
        p_neon = v2_dir / f"crown_cv_neon_{s}_{wi}m_v2.tif"
        logger.info("Computing NEON crown CV %dm (fixed grid) ...", wi)
        cv, ct, ccrs = crown_stats_per_window(neon_crowns, neon_chm, w, stat="cv")
        if cv is not None:
            save_raster(cv, p_neon, ct, ccrs)
            cv_neon[w] = p_neon

        # NAIP crown CV
        p_naip = v2_dir / f"crown_cv_naip_{s}_{wi}m_v2.tif"
        logger.info("Computing NAIP crown CV %dm (fixed grid) ...", wi)
        cv, ct, ccrs = crown_stats_per_window(naip_crowns, neon_chm, w, stat="cv")
        if cv is not None:
            save_raster(cv, p_naip, ct, ccrs)
            cv_naip[w] = p_naip

        # Difference
        if w in cv_neon and w in cv_naip:
            p_diff = v2_dir / f"diff_crownCV_{s}_{wi}m_v2.tif"
            logger.info("Computing crown CV difference %dm ...", wi)
            diff, dt, dcrs = metric_difference(cv_neon[w], cv_naip[w])
            save_raster(diff, p_diff, dt, dcrs)
            cv_diff[w] = p_diff

        # Crown counts (diagnostic)
        p_count_neon = v2_dir / f"crown_count_neon_{s}_{wi}m_v2.tif"
        ct_arr, ct_t, ct_crs = crown_stats_per_window(
            neon_crowns, neon_chm, w, stat="count"
        )
        if ct_arr is not None:
            save_raster(ct_arr, p_count_neon, ct_t, ct_crs)
            count_neon[w] = p_count_neon

        p_count_naip = v2_dir / f"crown_count_naip_{s}_{wi}m_v2.tif"
        ct_arr, ct_t, ct_crs = crown_stats_per_window(
            naip_crowns, neon_chm, w, stat="count"
        )
        if ct_arr is not None:
            save_raster(ct_arr, p_count_naip, ct_t, ct_crs)
            count_naip[w] = p_count_naip

    # Crown CV difference figure (v2)
    if cv_neon and cv_naip and cv_diff:
        make_diff_figure(
            naip_rgb_path=naip_rgb,
            neon_paths=cv_neon,
            naip_paths=cv_naip,
            diff_paths=cv_diff,
            out_path=_FIGURES_OUT / f"fig_diff_crownCV_{s}_v2.png",
            metric_name="Crown Width CV",
            metric_label="Crown CV (\u03c3/\u03bc)",
            site_label=cfg.label,
            cmap_metric="viridis",
            cmap_diff="RdBu_r",
        )

    # Crown count diagnostic figure (improved layout: row labels, tighter
    # top margin). Saved as _v3 alongside the existing _v2 file.
    if count_neon and count_naip:
        make_count_figure(
            naip_rgb_path=naip_rgb,
            neon_count_paths=count_neon,
            naip_count_paths=count_naip,
            out_path=_FIGURES_OUT / f"fig_crowncount_{s}_v3.png",
            site_label=cfg.label,
        )

    logger.info("=== Done: %s ===", s)


@click.command()
@click.option("--site", "-s", required=True,
              help="Site code: SCBI, HARV, or 'all'")
def main(site: str) -> None:
    """Regenerate crown CV figures with fixed grid (floor division)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    sites = ["SCBI", "HARV"] if site.lower() == "all" else [site.upper()]
    for s in sites:
        run_site(s)


if __name__ == "__main__":
    main()
