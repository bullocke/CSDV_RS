"""
11_additional_metrics.py — Crown Fraction and Crown Width P90 comparison figures.

Computes two metrics that appear in the CSDV V5 classification system but are
not shown in the existing 08_compare_chm_sources.py figures:

1. Crown Fraction (canopy cover fraction) — complement of gap fraction.
   This is the primary canopy closure indicator in the V5 stage definitions.
   Computed at 25 m window (stand-scale).

2. Crown Width P90 — 90th-percentile crown diameter per analysis window.
   Sensitive to removal of the largest-diameter trees; a diagnostic signature
   of highgrading (see summary_v1.md §4.3: "Crown Width P90 drops sharply"
   after highgrading while Crown Width Mean stays roughly the same).
   Computed at 50 m window (contains 5–30+ crowns in closed canopy).

Both metrics use the existing mediumzoom intermediate outputs from
08_compare_chm_sources.py. No new data downloads are required.

Parameter notes
---------------
- Crown fraction threshold: 2 m (height pixels >= 2 m = crown cover).
  Source: CSDV Classification V5 Full.md gap pixel definition.
- Window size (crown fraction): 25 m. NOT specified in V5 docs; chosen as a
  reasonable stand-scale window that matches the existing gap fraction figures.
- Window size (P90): 50 m. NOT specified in V5 docs; chosen because a 50 m
  window typically contains enough crowns (5–30+ in closed canopy) for a stable
  90th-percentile estimate. The 25 m window often contains fewer than 3 crowns
  in the deepzoom extent.
- "Best choice" for operational use: crown fraction at 25–50 m, P90 at 50 m.

Outputs
-------
  fig_compare_crownfrac_{SITE}.png  — 3-panel: NAIP RGB | NEON | NAIP
  fig_compare_crownP90_{SITE}.png   — 3-panel: NAIP RGB | NEON | NAIP
  (intermediate rasters saved to mediumzoom/{SITE}/)

Usage
-----
  python 11_additional_metrics.py --site SCBI
  python 11_additional_metrics.py --site HARV
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
    crown_fraction,
    crown_width_p90,
    figures as fig_utils,
    get_site,
    read_band,
    save_raster,
)

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures"

# Window sizes used for the two metrics
CROWN_FRAC_WINDOW_M = 25.0
CROWN_P90_WINDOW_M = 50.0
HEIGHT_THRESHOLD_M = 2.0  # V5-defined gap threshold


def _read_display(path: Path) -> np.ndarray:
    """Read a single-band raster, return float32 with NaN for nodata."""
    arr, _, _ = read_band(path)
    return arr


def _rgb_display(naip_path: Path) -> np.ndarray:
    """Return uint8 RGB array from a NAIP raster."""
    with rasterio.open(naip_path) as src:
        rgb = src.read()[:3].astype(np.float32)
    out = np.zeros((*rgb.shape[1:], 3), dtype=np.uint8)
    for i in range(3):
        band = rgb[i]
        lo = float(np.percentile(band[band > 0], 2)) if np.any(band > 0) else 0.0
        hi = float(np.percentile(band[band > 0], 98)) if np.any(band > 0) else 1.0
        out[..., i] = np.clip((band - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
    return out


def _make_three_panel(
    naip_rgb_path: Path,
    neon_arr: np.ndarray,
    naip_arr: np.ndarray,
    neon_label: str,
    naip_label: str,
    cmap: str,
    out_path: Path,
    site_label: str,
    metric_title: str,
    cbar_label: str,
    window_note: str,
) -> None:
    """3-panel comparison figure: NAIP RGB | NEON metric | NAIP metric."""
    fig_utils.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Shared colormap limits from combined 2nd/98th percentile
    all_vals = np.concatenate([
        neon_arr[~np.isnan(neon_arr)].ravel(),
        naip_arr[~np.isnan(naip_arr)].ravel(),
    ])
    if all_vals.size:
        vmin = float(np.nanpercentile(all_vals, 2))
        vmax = float(np.nanpercentile(all_vals, 98))
    else:
        vmin, vmax = 0.0, 1.0

    # Panel a: NAIP RGB
    rgb = _rgb_display(naip_rgb_path)
    axes[0].imshow(rgb)
    axes[0].set_axis_off()
    axes[0].set_title("NAIP RGB", fontsize=9)
    fig_utils.panel_label(axes[0], "a")

    # Panel b: NEON
    im = axes[1].imshow(neon_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_axis_off()
    axes[1].set_title(neon_label, fontsize=9)
    fig_utils.panel_label(axes[1], "b")

    # Panel c: NAIP
    axes[2].imshow(naip_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_axis_off()
    axes[2].set_title(naip_label, fontsize=9)
    fig_utils.panel_label(axes[2], "c")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{site_label} — {metric_title}  ({window_note})",
        fontsize=9, y=1.01,
    )
    fig_utils.save_fig(fig, out_path)
    logger.info("Saved: %s", out_path.name)


@click.command()
@click.option("--site", "-s", required=True, help="Site code: SCBI or HARV")
@click.option("--recompute", is_flag=True, default=False)
def main(site: str, recompute: bool) -> None:
    """Compute crown fraction and crown width P90 comparison figures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_site(site)
    s = cfg.site_code
    intermediate = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / s
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    neon_chm = intermediate / f"neon_chm_{s}_mediumzoom.tif"
    naip_chm = intermediate / f"naip_chm_{s}_mediumzoom_meters.tif"
    naip_rgb = intermediate / f"naip_rgb_{s}_mediumzoom.tif"
    neon_crowns = intermediate / f"crowns_neon_{s}_mediumzoom.gpkg"
    naip_crowns = intermediate / f"crowns_naip_{s}_mediumzoom.gpkg"

    for p in [neon_chm, naip_chm, naip_rgb]:
        if not p.exists():
            logger.error("Missing: %s\nRun 08_compare_chm_sources.py --site %s first.", p, s)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Crown Fraction (25 m)
    # ------------------------------------------------------------------
    logger.info("=== Crown fraction (%s, %.0f m) ===", s, CROWN_FRAC_WINDOW_M)
    p_cf_neon = intermediate / f"crown_frac_neon_{s}_mediumzoom_{int(CROWN_FRAC_WINDOW_M)}m.tif"
    p_cf_naip = intermediate / f"crown_frac_naip_{s}_mediumzoom_{int(CROWN_FRAC_WINDOW_M)}m.tif"

    if not p_cf_neon.exists() or recompute:
        cf, ct, crs = crown_fraction(neon_chm, CROWN_FRAC_WINDOW_M, HEIGHT_THRESHOLD_M)
        save_raster(cf, p_cf_neon, ct, crs)
    if not p_cf_naip.exists() or recompute:
        cf, ct, crs = crown_fraction(naip_chm, CROWN_FRAC_WINDOW_M, HEIGHT_THRESHOLD_M)
        save_raster(cf, p_cf_naip, ct, crs)

    neon_cf = _read_display(p_cf_neon)
    naip_cf = _read_display(p_cf_naip)

    _make_three_panel(
        naip_rgb_path=naip_rgb,
        neon_arr=neon_cf,
        naip_arr=naip_cf,
        neon_label=f"NEON crown fraction ({int(CROWN_FRAC_WINDOW_M)} m)",
        naip_label=f"NAIP crown fraction ({int(CROWN_FRAC_WINDOW_M)} m)",
        cmap="YlGn",
        out_path=_FIGURES_OUT / f"fig_compare_crownfrac_{s}.png",
        site_label=cfg.label,
        metric_title="Crown Fraction (canopy cover)",
        cbar_label=f"Crown fraction (0–1, {int(CROWN_FRAC_WINDOW_M)} m window)",
        window_note=(
            f"window = {int(CROWN_FRAC_WINDOW_M)} m, "
            f"height threshold = {HEIGHT_THRESHOLD_M} m"
        ),
    )

    # ------------------------------------------------------------------
    # Crown Width P90 (50 m)
    # ------------------------------------------------------------------
    logger.info("=== Crown Width P90 (%s, %.0f m) ===", s, CROWN_P90_WINDOW_M)
    p_p90_neon = intermediate / f"crown_p90_neon_{s}_mediumzoom_{int(CROWN_P90_WINDOW_M)}m.tif"
    p_p90_naip = intermediate / f"crown_p90_naip_{s}_mediumzoom_{int(CROWN_P90_WINDOW_M)}m.tif"

    if not p_p90_neon.exists() or recompute:
        if neon_crowns.exists():
            p90, pt, pcrs = crown_width_p90(neon_crowns, neon_chm, CROWN_P90_WINDOW_M)
            if p90 is not None:
                save_raster(p90, p_p90_neon, pt, pcrs)
        else:
            logger.warning("NEON crowns not found: %s", neon_crowns)

    if not p_p90_naip.exists() or recompute:
        if naip_crowns.exists():
            p90, pt, pcrs = crown_width_p90(naip_crowns, naip_chm, CROWN_P90_WINDOW_M)
            if p90 is not None:
                save_raster(p90, p_p90_naip, pt, pcrs)
        else:
            logger.warning("NAIP crowns not found: %s", naip_crowns)

    if p_p90_neon.exists() and p_p90_naip.exists():
        neon_p90 = _read_display(p_p90_neon)
        naip_p90 = _read_display(p_p90_naip)
        _make_three_panel(
            naip_rgb_path=naip_rgb,
            neon_arr=neon_p90,
            naip_arr=naip_p90,
            neon_label=f"NEON crown width P90 ({int(CROWN_P90_WINDOW_M)} m)",
            naip_label=f"NAIP crown width P90 ({int(CROWN_P90_WINDOW_M)} m)",
            cmap="plasma",
            out_path=_FIGURES_OUT / f"fig_compare_crownP90_{s}.png",
            site_label=cfg.label,
            metric_title="Crown Width P90 (highgrading indicator)",
            cbar_label=f"Crown width P90 (m, {int(CROWN_P90_WINDOW_M)} m window)",
            window_note=(
                f"window = {int(CROWN_P90_WINDOW_M)} m, "
                "min 3 crowns per cell required"
            ),
        )
    else:
        logger.warning("P90 rasters not found — crown segmentation may need to be re-run.")

    logger.info("=== Done: %s ===", s)


if __name__ == "__main__":
    main()
