"""
09_metric_differences.py — Pixel-level difference maps: NAIP CHM metric minus
NEON ALS CHM metric at 25, 50, and 100 m window sizes.

Inputs
------
Uses mediumzoom intermediate outputs already computed by 08_compare_chm_sources.py:
  intermediate/mediumzoom/{SITE}/gap_frac_{neon,naip}_{SITE}_mediumzoom_{w}m.tif
  intermediate/mediumzoom/{SITE}/crowns_{neon,naip}_{SITE}_mediumzoom.gpkg

Outputs
-------
Figures (saved to Results/summary_document/figures/):
  fig_diff_gapfrac_{SITE}.png   — 3×3: NAIP RGB | NEON GF | NAIP−NEON difference
  fig_diff_crownCV_{SITE}.png   — same layout for crown width CV

Intermediate difference rasters (saved alongside other mediumzoom outputs):
  diff_gapfrac_{SITE}_mediumzoom_{w}m.tif
  diff_crownCV_{SITE}_mediumzoom_{w}m.tif

Usage
-----
  python 09_metric_differences.py --site SCBI
  python 09_metric_differences.py --site HARV
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Add poc_lib to sys.path (parent directory of this script contains poc_lib)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent  # ProofOfConcept/Code
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
_FIGURES_OUT = _RESULTS / "figures"

WINDOW_SIZES = [25.0, 50.0, 100.0]


def _load_metric_raster(path: Path) -> tuple[np.ndarray, float]:
    """Read metric raster, return (2d array with NaN, pixel size in m)."""
    arr, transform, _ = read_band(path)
    pixel_size = abs(transform.a)
    return arr, pixel_size


def _diff_stats(diff: np.ndarray) -> tuple[float, float, int]:
    """Return (RMSE, bias, N) from a difference array."""
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
    cmap_metric: str = "YlGn",
    cmap_diff: str = "RdBu_r",
) -> None:
    """3×4 figure: NAIP RGB | NEON metric | NAIP metric | NAIP−NEON difference.

    Rows correspond to window sizes (25, 50, 100 m).
    Columns 1 and 2 (NEON and NAIP metrics) share an identical colormap range
    derived from the combined 2nd–98th percentile of both sources, so the two
    panels are directly comparable by colour. The difference column (col 3) uses
    a symmetric diverging colormap.
    """
    fig_utils.setup_style()
    window_sizes = sorted(neon_paths)
    n_rows = len(window_sizes)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, n_rows * 3.6))

    # Read NAIP RGB once
    with rasterio.open(naip_rgb_path) as src:
        rgb_data = src.read()[:3].astype(np.float32)
    rgb_disp = np.zeros((*rgb_data.shape[1:], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_data[i]
        lo = float(np.percentile(band[band > 0], 2)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], 98)) if np.any(band > 0) else 1.0
        rgb_disp[..., i] = np.clip((band - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)

    # Shared vmin/vmax for metric panels (cols 1 AND 2) from combined NEON+NAIP values
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

    # Symmetric vmax for difference column (col 3)
    diff_abs_max = 0.0
    for w in window_sizes:
        if w in diff_paths and diff_paths[w].exists():
            diff, _ = _load_metric_raster(diff_paths[w])
            v = diff[~np.isnan(diff)]
            if v.size:
                diff_abs_max = max(diff_abs_max, float(np.nanpercentile(np.abs(v), 98)))
    if diff_abs_max == 0.0:
        diff_abs_max = 0.1

    last_metric_im = None
    last_diff_im = None

    for row_i, w in enumerate(window_sizes):
        ax_rgb, ax_neon, ax_naip, ax_diff = axes[row_i]

        # Col 0: NAIP RGB
        ax_rgb.imshow(rgb_disp)
        ax_rgb.set_axis_off()
        ax_rgb.set_ylabel(f"{int(w)} m", fontsize=9, labelpad=4)
        fig_utils.panel_label(ax_rgb, chr(ord("a") + row_i * 4))

        # Col 1: NEON metric (shared scale)
        if w in neon_paths and neon_paths[w].exists():
            neon_arr, _ = _load_metric_raster(neon_paths[w])
            last_metric_im = ax_neon.imshow(
                neon_arr, cmap=cmap_metric, vmin=metric_vmin, vmax=metric_vmax,
                interpolation="nearest",
            )
        ax_neon.set_axis_off()
        fig_utils.panel_label(ax_neon, chr(ord("b") + row_i * 4))

        # Col 2: NAIP metric (same shared scale as NEON)
        if w in naip_paths and naip_paths[w].exists():
            naip_arr, _ = _load_metric_raster(naip_paths[w])
            ax_naip.imshow(
                naip_arr, cmap=cmap_metric, vmin=metric_vmin, vmax=metric_vmax,
                interpolation="nearest",
            )
        ax_naip.set_axis_off()
        fig_utils.panel_label(ax_naip, chr(ord("c") + row_i * 4))

        # Col 3: Difference with RMSE/bias annotation
        if w in diff_paths and diff_paths[w].exists():
            diff_arr, _ = _load_metric_raster(diff_paths[w])
            rmse, bias, n = _diff_stats(diff_arr)
            last_diff_im = ax_diff.imshow(
                diff_arr, cmap=cmap_diff,
                vmin=-diff_abs_max, vmax=diff_abs_max,
                interpolation="nearest",
            )
            stats_text = f"RMSE={rmse:.3f}\nbias={bias:+.3f}\nN={n}"
            ax_diff.text(
                0.98, 0.02, stats_text,
                transform=ax_diff.transAxes, fontsize=7,
                ha="right", va="bottom",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0),
            )
        ax_diff.set_axis_off()
        fig_utils.panel_label(ax_diff, chr(ord("d") + row_i * 4))

    # Column headers (top row only)
    axes[0, 0].set_title("NAIP RGB", fontsize=9)
    axes[0, 1].set_title(f"NEON {metric_name}", fontsize=9)
    axes[0, 2].set_title(f"NAIP {metric_name}", fontsize=9)
    axes[0, 3].set_title(f"NAIP − NEON", fontsize=9)

    # Shared colorbar for metric columns (cols 1–2)
    if last_metric_im is not None:
        cb1 = fig.colorbar(
            last_metric_im,
            ax=axes[:, 1:3].ravel().tolist(),
            fraction=0.015, pad=0.03,
        )
        cb1.set_label(metric_label, fontsize=8)
        cb1.ax.tick_params(labelsize=7)

    # Colorbar for difference column
    if last_diff_im is not None:
        cb2 = fig.colorbar(last_diff_im, ax=axes[:, 3].tolist(), fraction=0.03, pad=0.02)
        cb2.set_label(f"Difference ({metric_label})", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    fig.suptitle(f"{site_label} — {metric_name} comparison and difference", fontsize=10, y=1.01)
    fig_utils.save_fig(fig, out_path)
    logger.info("Saved: %s", out_path.name)


@click.command()
@click.option("--site", "-s", required=True, help="Site code: SCBI or HARV")
@click.option("--recompute", is_flag=True, default=False,
              help="Recompute crown CV even if output files exist.")
def main(site: str, recompute: bool) -> None:
    """Generate metric difference figures for one NEON site."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_site(site)
    s = cfg.site_code
    intermediate = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / s
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    # Resolve NEON CHM for the mediumzoom extent
    neon_chm = intermediate / f"neon_chm_{s}_mediumzoom.tif"
    naip_rgb = intermediate / f"naip_rgb_{s}_mediumzoom.tif"
    if not neon_chm.exists():
        logger.error(
            "Mediumzoom NEON CHM not found: %s\nRun 08_compare_chm_sources.py --site %s first.",
            neon_chm, s,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Gap fraction difference
    # ------------------------------------------------------------------
    logger.info("=== Gap fraction differences (%s) ===", s)
    gap_neon: dict[float, Path] = {}
    gap_naip: dict[float, Path] = {}
    gap_diff: dict[float, Path] = {}

    for w in WINDOW_SIZES:
        p_neon = intermediate / f"gap_frac_neon_{s}_mediumzoom_{int(w)}m.tif"
        p_naip = intermediate / f"gap_frac_naip_{s}_mediumzoom_{int(w)}m.tif"
        p_diff = intermediate / f"diff_gapfrac_{s}_mediumzoom_{int(w)}m.tif"

        if not p_neon.exists():
            logger.info("  Computing NEON gap fraction %dm ...", int(w))
            gf, gt, crs = gap_fraction(neon_chm, window_m=w, height_threshold_m=2.0)
            save_raster(gf, p_neon, gt, crs)
        if not p_naip.exists():
            naip_chm = intermediate / f"naip_chm_{s}_mediumzoom_meters.tif"
            logger.info("  Computing NAIP gap fraction %dm ...", int(w))
            gf, gt, crs = gap_fraction(naip_chm, window_m=w, height_threshold_m=2.0)
            save_raster(gf, p_naip, gt, crs)
        if not p_diff.exists() or recompute:
            logger.info("  Computing difference %dm ...", int(w))
            diff, dt, dcrs = metric_difference(p_neon, p_naip)
            save_raster(diff, p_diff, dt, dcrs)

        gap_neon[w] = p_neon
        gap_naip[w] = p_naip
        gap_diff[w] = p_diff

    make_diff_figure(
        naip_rgb_path=naip_rgb,
        neon_paths=gap_neon,
        naip_paths=gap_naip,
        diff_paths=gap_diff,
        out_path=_FIGURES_OUT / f"fig_diff_gapfrac_{s}.png",
        metric_name="Gap Fraction",
        metric_label="Gap fraction (0–1)",
        site_label=cfg.label,
        cmap_metric="YlGn_r",
        cmap_diff="RdBu_r",
    )

    # ------------------------------------------------------------------
    # Crown width CV difference
    # ------------------------------------------------------------------
    logger.info("=== Crown CV differences (%s) ===", s)
    neon_crowns = intermediate / f"crowns_neon_{s}_mediumzoom.gpkg"
    naip_crowns = intermediate / f"crowns_naip_{s}_mediumzoom.gpkg"

    if not neon_crowns.exists() or not naip_crowns.exists():
        logger.error(
            "Crown GeoPackages not found in %s\nRun 08_compare_chm_sources.py first.", intermediate
        )
        sys.exit(1)

    cv_neon: dict[float, Path] = {}
    cv_naip: dict[float, Path] = {}
    cv_diff: dict[float, Path] = {}

    for w in WINDOW_SIZES:
        p_neon = intermediate / f"crown_cv_neon_{s}_mediumzoom_{int(w)}m.tif"
        p_naip = intermediate / f"crown_cv_naip_{s}_mediumzoom_{int(w)}m.tif"
        p_diff = intermediate / f"diff_crownCV_{s}_mediumzoom_{int(w)}m.tif"

        if not p_neon.exists() or recompute:
            logger.info("  Computing NEON crown CV %dm ...", int(w))
            cv, ct, ccrs = crown_stats_per_window(neon_crowns, neon_chm, w, stat="cv")
            if cv is not None:
                save_raster(cv, p_neon, ct, ccrs)
        if not p_naip.exists() or recompute:
            naip_chm = intermediate / f"naip_chm_{s}_mediumzoom_meters.tif"
            logger.info("  Computing NAIP crown CV %dm ...", int(w))
            cv, ct, ccrs = crown_stats_per_window(naip_crowns, naip_chm, w, stat="cv")
            if cv is not None:
                save_raster(cv, p_naip, ct, ccrs)
        if p_neon.exists() and p_naip.exists() and (not p_diff.exists() or recompute):
            logger.info("  Computing crown CV difference %dm ...", int(w))
            diff, dt, dcrs = metric_difference(p_neon, p_naip)
            save_raster(diff, p_diff, dt, dcrs)

        if p_neon.exists():
            cv_neon[w] = p_neon
        if p_naip.exists():
            cv_naip[w] = p_naip
        if p_diff.exists():
            cv_diff[w] = p_diff

    if cv_neon and cv_naip and cv_diff:
        make_diff_figure(
            naip_rgb_path=naip_rgb,
            neon_paths=cv_neon,
            naip_paths=cv_naip,
            diff_paths=cv_diff,
            out_path=_FIGURES_OUT / f"fig_diff_crownCV_{s}.png",
            metric_name="Crown Width CV",
            metric_label="Crown CV (σ/μ)",
            site_label=cfg.label,
            cmap_metric="viridis",
            cmap_diff="RdBu_r",
        )
    else:
        logger.warning("Insufficient crown CV rasters to make figure — skipping.")

    logger.info("=== Done: %s ===", s)


if __name__ == "__main__":
    main()
