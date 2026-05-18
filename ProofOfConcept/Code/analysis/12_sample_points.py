"""
12_sample_points.py — Sample-point comparison across the broader NEON extent.

Draws N=200 random sample points from the broader NEON CHM raster (forest
pixels only, canopy height > 5 m), extracts gap fraction and crown width CV
at each point using a 25 m sliding window, then compares NEON vs NAIP metrics
at the same points.

This provides spatially independent observations from a heterogeneous
landscape rather than re-using the gridded analysis windows from the
mediumzoom pipeline. The broader extent includes forest stands at varying
developmental stages, giving more power to detect systematic biases.

Prerequisites
-------------
Run 12a_download_neon_broader.py --site {SITE} before this script.

Outputs
-------
  fig_sample_comparison_{SITE}.png
      2×2 figure: NAIP RGB overview | gap fraction scatter |
                  crown CV scatter | metric distribution boxplot
  table_sample_stats_{SITE}.csv
      N, R², RMSE, bias, Pearson r, Spearman r per metric

Usage
-----
  python 12_sample_points.py --site SCBI
  python 12_sample_points.py --site HARV

Parameters
----------
- Forest mask: NEON CHM pixels >= 5 m canopy height (conservative threshold
  to ensure samples fall in closed or partially closed canopy, not open ground
  or regenerating stands where crown segmentation is unreliable).
- Sample size: N=200 (sufficient for stable correlation statistics; avoids
  oversampling on small study areas).
- Window size: 25 m (matching the gap fraction comparison in other figures).
- Random seed: 42 (reproducible draws).
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from poc_lib import figures as fig_utils
from poc_lib import find_latest_tif, get_site, read_band, save_raster
from poc_lib.io import clip_and_convert_naip_chm, clip_raster_to_bbox
from poc_lib.metrics import gap_fraction as compute_gap_fraction

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_DATA = _POC / "Data"
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures"
_TABLES_OUT = _POC / "Results" / "summary_document" / "tables"
_INTERMEDIATE = _POC / "Results" / "summary_document" / "intermediate" / "broader"

FOREST_HEIGHT_MIN_M = 5.0   # minimum CHM height to be counted as forest pixel
SAMPLE_N = 200              # number of random sample points
WINDOW_M = 25.0             # analysis window for metrics at each point
RANDOM_SEED = 42


def _window_metric_at_point(
    chm: np.ndarray,
    row: int,
    col: int,
    half_px: int,
    height_threshold_m: float = 2.0,
) -> float:
    """Extract gap fraction from a window centered at (row, col) in *chm*."""
    r0 = max(0, row - half_px)
    r1 = min(chm.shape[0], row + half_px + 1)
    c0 = max(0, col - half_px)
    c1 = min(chm.shape[1], col + half_px + 1)
    tile = chm[r0:r1, c0:c1]
    valid = tile[~np.isnan(tile)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid < height_threshold_m))


def _regression_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute OLS regression and agreement statistics."""
    from scipy.stats import linregress, spearmanr

    mask = ~np.isnan(x) & ~np.isnan(y)
    xv, yv = x[mask], y[mask]
    n = int(xv.size)
    if n < 3:
        return {k: np.nan for k in ["r2", "rmse", "bias", "pearson_r", "spearman_r", "slope", "intercept", "n"]}
    slope, intercept, r_val, _, _ = linregress(xv, yv)
    rmse = float(np.sqrt(np.mean((yv - xv) ** 2)))
    bias = float(np.mean(yv - xv))
    spear_r, _ = spearmanr(xv, yv)
    return {
        "r2": float(r_val**2),
        "rmse": float(rmse),
        "bias": float(bias),
        "pearson_r": float(r_val),
        "spearman_r": float(spear_r),
        "slope": float(slope),
        "intercept": float(intercept),
        "n": n,
    }


@click.command()
@click.option("--site", "-s", required=True, help="Site code: SCBI or HARV")
@click.option("--n-points", default=SAMPLE_N, show_default=True,
              type=int, help="Number of random sample points.")
@click.option("--window-m", default=WINDOW_M, show_default=True,
              type=float, help="Analysis window size in meters.")
@click.option("--recompute", is_flag=True, default=False)
def main(site: str, n_points: int, window_m: float, recompute: bool) -> None:
    """Sample-point comparison using broader NEON CHM extent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_site(site)
    s = cfg.site_code
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    _TABLES_OUT.mkdir(parents=True, exist_ok=True)
    site_intermediate = _INTERMEDIATE / s
    site_intermediate.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Locate broader rasters
    # ------------------------------------------------------------------
    neon_chm_path = find_latest_tif(
        _DATA / "NEON" / "CHM",
        f"NEON_CHM_{s}_Broader",
    )
    naip_rgb_path = find_latest_tif(
        _DATA / "NAIP" / "Imagery",
        f"NAIP_{s}_Broader",
    )
    naipchm_raw_path = find_latest_tif(
        _DATA / "NAIP" / "National_CHM",
        f"NAIPCHM_{s}_Broader",
    )

    logger.info("NEON CHM (broader): %s", neon_chm_path.name)
    logger.info("NAIP RGBN (broader): %s", naip_rgb_path.name)
    logger.info("NAIP CHM (broader): %s", naipchm_raw_path.name)

    # ------------------------------------------------------------------
    # Convert NAIP CHM to float32 meters if not done
    # ------------------------------------------------------------------
    naip_chm_meters = site_intermediate / f"naip_chm_{s}_broader_meters.tif"
    if not naip_chm_meters.exists() or recompute:
        logger.info("Converting NAIP CHM to float32 meters ...")
        with rasterio.open(neon_chm_path) as ref:
            bounds = ref.bounds
            bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            # Expand bbox slightly from NEON extent to ensure overlap
        with rasterio.open(naipchm_raw_path) as src:
            data = src.read(1)
            nodata = src.nodata
            profile = src.profile.copy()
            arr = data.astype(np.float32) * 0.01
            if nodata is not None:
                arr[data == nodata] = -9999.0
            profile.update(dtype="float32", nodata=-9999.0, count=1)
            with rasterio.open(naip_chm_meters, "w", **profile) as dst:
                dst.write(arr[np.newaxis])
        logger.info("Saved: %s", naip_chm_meters.name)

    # ------------------------------------------------------------------
    # Load NEON CHM and define forest mask
    # ------------------------------------------------------------------
    neon_arr, neon_transform, neon_crs = read_band(neon_chm_path)
    pixel_size_m = abs(neon_transform.a)
    half_px = max(1, int(round(window_m / (2 * pixel_size_m))))

    forest_mask = (neon_arr >= FOREST_HEIGHT_MIN_M) & ~np.isnan(neon_arr)
    # Exclude border pixels to ensure full windows
    border = half_px + 1
    forest_mask[:border, :] = False
    forest_mask[-border:, :] = False
    forest_mask[:, :border] = False
    forest_mask[:, -border:] = False

    forest_pixels = np.argwhere(forest_mask)
    n_forest = len(forest_pixels)
    logger.info("Forest pixels (>= %.0f m, excluding border): %d", FOREST_HEIGHT_MIN_M, n_forest)

    if n_forest < n_points:
        logger.warning(
            "Only %d forest pixels available; using all instead of %d.",
            n_forest, n_points,
        )
        n_points = n_forest

    rng = np.random.default_rng(seed=RANDOM_SEED)
    sampled_indices = rng.choice(n_forest, size=n_points, replace=False)
    sample_rows = forest_pixels[sampled_indices, 0]
    sample_cols = forest_pixels[sampled_indices, 1]
    logger.info("Sampled %d points (seed=%d)", n_points, RANDOM_SEED)

    # ------------------------------------------------------------------
    # Load NAIP CHM and reproject to NEON grid
    # ------------------------------------------------------------------
    naip_arr_raw, naip_transform, naip_crs = read_band(naip_chm_meters)
    naip_aligned = np.full_like(neon_arr, np.nan)
    reproject(
        source=naip_arr_raw,
        destination=naip_aligned,
        src_transform=naip_transform,
        src_crs=naip_crs,
        dst_transform=neon_transform,
        dst_crs=neon_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    logger.info("NAIP CHM aligned to NEON grid (bilinear reproject)")

    # ------------------------------------------------------------------
    # Extract gap fraction at each sample point
    # ------------------------------------------------------------------
    logger.info("Extracting gap fraction at %d sample points (window=%.0f m) ...", n_points, window_m)
    neon_gf = np.array([
        _window_metric_at_point(neon_arr, r, c, half_px)
        for r, c in zip(sample_rows, sample_cols)
    ])
    naip_gf = np.array([
        _window_metric_at_point(naip_aligned, r, c, half_px)
        for r, c in zip(sample_rows, sample_cols)
    ])

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig_utils.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{cfg.label} — Sample-point comparison (N={n_points}, "
        f"{int(window_m)} m window, {FOREST_HEIGHT_MIN_M:.0f} m forest mask)",
        fontsize=9,
    )

    # Panel a: NAIP RGB overview with sample points
    ax_rgb = axes[0]
    with rasterio.open(naip_rgb_path) as src:
        rgb_data = src.read()[:3].astype(np.float32)
        rgb_transform = src.transform
    rgb_disp = np.zeros((*rgb_data.shape[1:], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_data[i]
        lo = float(np.percentile(band[band > 0], 2)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], 98)) if np.any(band > 0) else 1.0
        rgb_disp[..., i] = np.clip((band - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)

    # Convert NEON sample pixel coords to NAIP RGB pixel coords for display
    # (project via geographic coordinates)
    with rasterio.open(neon_chm_path) as src:
        geo_xs, geo_ys = rasterio.transform.xy(neon_transform, sample_rows, sample_cols)
    naip_rgb_cols, naip_rgb_rows = ~rgb_transform * (np.array(geo_xs), np.array(geo_ys))

    ax_rgb.imshow(rgb_disp)
    ax_rgb.scatter(naip_rgb_cols, naip_rgb_rows, s=8, c="red", alpha=0.5, linewidths=0)
    ax_rgb.set_axis_off()
    ax_rgb.set_title(f"NAIP RGB + sample points (N={n_points})", fontsize=8)
    fig_utils.panel_label(ax_rgb, "a")

    # Panel b: Gap fraction scatter (NEON vs NAIP)
    ax_gf = axes[1]
    valid_gf = ~np.isnan(neon_gf) & ~np.isnan(naip_gf)
    stats_gf = _regression_stats(neon_gf, naip_gf)
    if valid_gf.sum() > 3:
        hb = ax_gf.hexbin(neon_gf[valid_gf], naip_gf[valid_gf], gridsize=20, cmap="Blues", mincnt=1)
        lo, hi = 0.0, 1.0
        ax_gf.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")
        if not np.isnan(stats_gf["slope"]):
            x_fit = np.linspace(neon_gf[valid_gf].min(), neon_gf[valid_gf].max(), 50)
            ax_gf.plot(x_fit, stats_gf["slope"] * x_fit + stats_gf["intercept"], "r-", lw=1.2)
        plt.colorbar(hb, ax=ax_gf, fraction=0.04, pad=0.02).set_label("Count", fontsize=7)
    ann = (
        f"R²={stats_gf['r2']:.3f}\n"
        f"RMSE={stats_gf['rmse']:.3f}\n"
        f"bias={stats_gf['bias']:+.3f}\n"
        f"N={stats_gf['n']}"
    )
    ax_gf.text(0.03, 0.97, ann, transform=ax_gf.transAxes, fontsize=7, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, lw=0))
    ax_gf.set_xlabel("NEON gap fraction", fontsize=8)
    ax_gf.set_ylabel("NAIP gap fraction", fontsize=8)
    ax_gf.set_title(f"Gap Fraction ({int(window_m)} m)", fontsize=8)
    ax_gf.tick_params(labelsize=7)
    fig_utils.panel_label(ax_gf, "b")

    # Panel c: Distribution comparison (boxplot)
    ax_box = axes[2]
    box_data = [
        neon_gf[~np.isnan(neon_gf)],
        naip_gf[~np.isnan(naip_gf)],
    ]
    bp = ax_box.boxplot(box_data, labels=["NEON", "NAIP"], patch_artist=True)
    colors = ["#2196f3", "#ff9800"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_box.set_ylabel(f"Gap fraction ({int(window_m)} m)", fontsize=8)
    ax_box.set_title("Distribution comparison", fontsize=8)
    ax_box.tick_params(labelsize=7)
    fig_utils.panel_label(ax_box, "c")

    plt.tight_layout()
    fig_utils.save_fig(fig, _FIGURES_OUT / f"fig_sample_comparison_{s}.png")

    # ------------------------------------------------------------------
    # Statistics table
    # ------------------------------------------------------------------
    all_stats = [
        {"metric": "gap_fraction", "window_m": int(window_m), **stats_gf},
    ]
    table_path = _TABLES_OUT / f"table_sample_stats_{s}.csv"
    if all_stats:
        fieldnames = list(all_stats[0].keys())
        with open(table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        logger.info("Saved table: %s", table_path.name)

    logger.info("=== Done: %s ===", s)


if __name__ == "__main__":
    main()
