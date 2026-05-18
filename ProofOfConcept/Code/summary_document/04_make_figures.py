"""
04_make_figures.py — Generate demonstration figures for the CSDV summary document.

Produces two publication-quality figures at 300 dpi:

Figure 1: fig01_data_sources.png
    Three-panel comparison of NAIP RGB, NAIP-CHM, and NEON ALS CHM for the
    same SCBI spatial extent. Shows the visual character of the three primary
    data sources used in the proof-of-concept.

Figure 2: fig02_derived_metrics.png
    Five-panel figure showing NAIP RGB and four derived metrics: gap fraction,
    GLCM texture entropy, crown segmentation polygons (lidR), and crown width
    CV per 50m window. Demonstrates that the metrics described in the proposal
    are computable from real data.

All inputs must be present before running (see 01–03 scripts). Paths are
resolved relative to this script's location.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject, Resampling, calculate_default_transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
POC = PROJECT_ROOT / "ProofOfConcept"

NAIP_DIR = POC / "Data" / "NAIP" / "Imagery"
NAIPCHM_DIR = POC / "Data" / "NAIP" / "National_CHM"
NEON_CHM = POC / "Data" / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif"
INTERMEDIATE = POC / "Results" / "summary_document" / "intermediate"
FIGURES_OUT = POC / "Results" / "summary_document" / "figures"

OUT_FIG1 = FIGURES_OUT / "fig01_data_sources.png"
OUT_FIG2 = FIGURES_OUT / "fig02_derived_metrics.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FONT_SIZE = 9
LABEL_FONT_SIZE = 11
DPI = 300
PANEL_LABEL_KWARGS = dict(fontsize=LABEL_FONT_SIZE, fontweight="bold",
                          va="top", ha="left", color="white",
                          bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.6, lw=0))


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "sans-serif",
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "image.interpolation": "none",
    })


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def find_latest(directory: Path, prefix: str) -> Path:
    """Return the most recently modified .tif in directory matching prefix."""
    tifs = sorted(directory.glob(f"{prefix}*.tif"), key=lambda p: p.stat().st_mtime)
    if not tifs:
        raise FileNotFoundError(f"No .tif matching '{prefix}*' in {directory}")
    return tifs[-1]


def read_rgb(path: Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read bands 1,2,3 as uint8-clipped RGB (2–98 pct stretch)."""
    with rasterio.open(path) as src:
        profile = src.profile
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
    bands = []
    for band in (r, g, b):
        p2, p98 = np.percentile(band[band > 0], [2, 98])
        stretched = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
        bands.append(stretched)
    rgb = np.stack(bands, axis=-1)  # (H, W, 3)
    return rgb, profile


def read_chm(path: Path, scale_factor: float = 1.0) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read a single-band CHM; apply scale_factor (e.g. 1/100 for NAIP-CHM UInt16)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile
    if nodata is not None:
        data[data == nodata] = np.nan
    data *= scale_factor
    data[data < 0] = np.nan
    return data, profile


def read_metric(path: Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read a single-band metric raster as float32."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


def reproject_match(
    src_path: Path,
    ref_path: Path,
    scale_factor: float = 1.0,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Reproject src raster to match the CRS, extent, and resolution of ref."""
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(src_path) as src:
        dst_array = np.empty((1, ref_height, ref_width), dtype=np.float32)
        reproject(
            source=src.read(1).astype(np.float32),
            destination=dst_array[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    data = dst_array[0] * scale_factor
    data[data < 0] = np.nan
    profile = {"crs": ref_crs, "transform": ref_transform, "width": ref_width, "height": ref_height}
    return data, profile


def get_extent(profile: dict) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) extent in projected coordinates."""
    t = profile["transform"]
    h, w = profile["height"], profile["width"]
    left = t.c
    top = t.f
    right = left + w * t.a
    bottom = top + h * t.e  # t.e is negative
    return left, right, bottom, top


# ---------------------------------------------------------------------------
# Figure components
# ---------------------------------------------------------------------------

def add_scalebar(ax: plt.Axes, extent: tuple, length_m: float = 200, label: str = "200 m") -> None:
    """Add a simple manual scale bar in the lower-right corner."""
    left, right, bottom, top = extent
    x_range = right - left
    y_range = top - bottom
    # Position: lower-right, 5% margin
    x0 = right - 0.08 * x_range - length_m
    x1 = x0 + length_m
    y0 = bottom + 0.04 * y_range
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.005 * y_range, y0 + 0.005 * y_range], color="white", lw=2)
    ax.plot([x1, x1], [y0 - 0.005 * y_range, y0 + 0.005 * y_range], color="white", lw=2)
    ax.text((x0 + x1) / 2, y0 + 0.012 * y_range, label,
            ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")


def add_north_arrow(ax: plt.Axes, extent: tuple) -> None:
    """Add a simple north arrow in the upper-right corner."""
    left, right, bottom, top = extent
    x_range = right - left
    y_range = top - bottom
    x = right - 0.05 * x_range
    y_base = top - 0.12 * y_range
    y_tip = y_base + 0.08 * y_range
    ax.annotate(
        "N",
        xy=(x, y_tip), xytext=(x, y_base),
        fontsize=8, color="white", fontweight="bold",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )


def styled_colorbar(ax: plt.Axes, im, label: str) -> None:
    """Attach a thin colorbar on the right side of the axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FONT_SIZE - 1)
    cb.ax.tick_params(labelsize=FONT_SIZE - 2)


def label_panel(ax: plt.Axes, letter: str, extent: tuple) -> None:
    """Add bold panel label (a, b, c...) in upper-left corner."""
    left, right, bottom, top = extent
    x_range = right - left
    y_range = top - bottom
    ax.text(left + 0.02 * x_range, top - 0.02 * y_range,
            letter, **PANEL_LABEL_KWARGS, transform=ax.transData)


# ---------------------------------------------------------------------------
# Figure 1: Data Sources
# ---------------------------------------------------------------------------

def make_figure1(
    naip_path: Path,
    naip_chm_path: Path,
    neon_chm_path: Path,
    out_path: Path = OUT_FIG1,
) -> None:
    """Three-panel data source comparison."""
    logger.info("Building Figure 1: Data Sources")

    # Load data; reproject NAIP and NAIP-CHM to match NEON CHM extent/CRS
    naip_rgb, naip_prof = read_rgb(naip_path)
    with rasterio.open(naip_path) as src:
        naip_full_prof = dict(src.profile)
        naip_full_prof["height"] = src.height
        naip_full_prof["width"] = src.width

    neon_chm, neon_prof = read_chm(neon_chm_path)
    with rasterio.open(neon_chm_path) as ref:
        ref_profile = dict(ref.profile)
        ref_profile["height"] = ref.height
        ref_profile["width"] = ref.width

    # Reproject NAIP RGB to NEON CHM extent for spatial alignment
    with rasterio.open(naip_path) as src:
        dst_rgb = np.empty((3, ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for i in range(3):
            reproject(
                source=src.read(i + 1).astype(np.float32),
                destination=dst_rgb[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    # Stretch each band
    rgb_aligned = np.zeros_like(dst_rgb)
    for i in range(3):
        band = dst_rgb[i]
        valid = band[~np.isnan(band) & (band > 0)]
        if valid.size:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb_aligned[i] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb_aligned = np.moveaxis(rgb_aligned, 0, -1)  # (H, W, 3)

    # Reproject NAIP-CHM; values are UInt16 / 100 = meters
    naip_chm, _ = reproject_match(naip_chm_path, neon_chm_path, scale_factor=1 / 100)

    extent = get_extent(ref_profile)

    # CHM display range
    vmin_chm, vmax_chm = 0, 35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.93, top=0.90, bottom=0.02)

    titles = [
        "NAIP true-color  |  0.6–1.0 m  |  2022",
        "NAIP-CHM (Morford et al. 2025)  |  0.6 m",
        "NEON ALS CHM  |  1 m  |  2023",
    ]

    # Panel A: NAIP RGB
    ax = axes[0]
    ax.imshow(rgb_aligned, extent=extent, origin="upper", aspect="equal")
    ax.set_title(titles[0], pad=4)
    add_scalebar(ax, extent, 200)
    add_north_arrow(ax, extent)
    label_panel(ax, "a", extent)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: NAIP-CHM
    ax = axes[1]
    im = ax.imshow(naip_chm, extent=extent, origin="upper", cmap="viridis",
                   vmin=vmin_chm, vmax=vmax_chm, aspect="equal")
    ax.set_title(titles[1], pad=4)
    add_scalebar(ax, extent, 200)
    label_panel(ax, "b", extent)
    ax.set_xticks([])
    ax.set_yticks([])
    styled_colorbar(ax, im, "Height (m)")

    # Panel C: NEON CHM
    ax = axes[2]
    im = ax.imshow(neon_chm, extent=extent, origin="upper", cmap="viridis",
                   vmin=vmin_chm, vmax=vmax_chm, aspect="equal")
    ax.set_title(titles[2], pad=4)
    add_scalebar(ax, extent, 200)
    label_panel(ax, "c", extent)
    ax.set_xticks([])
    ax.set_yticks([])
    styled_colorbar(ax, im, "Height (m)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2: Derived Metrics
# ---------------------------------------------------------------------------

def make_figure2(
    naip_path: Path,
    gap_frac_path: Path,
    entropy_path: Path,
    crowns_path: Path,
    crown_cv_path: Path,
    neon_chm_path: Path,
    out_path: Path = OUT_FIG2,
) -> None:
    """Five-panel derived metrics figure."""
    logger.info("Building Figure 2: Derived Metrics")

    # Reference extent from NEON CHM
    with rasterio.open(neon_chm_path) as ref:
        ref_profile = dict(ref.profile)
        ref_profile["height"] = ref.height
        ref_profile["width"] = ref.width

    extent = get_extent(ref_profile)

    # Load NAIP RGB (reproject to NEON CHM extent)
    with rasterio.open(naip_path) as src:
        dst_rgb = np.empty((3, ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for i in range(3):
            reproject(
                source=src.read(i + 1).astype(np.float32),
                destination=dst_rgb[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    rgb_aligned = np.zeros_like(dst_rgb)
    for i in range(3):
        band = dst_rgb[i]
        valid = band[~np.isnan(band) & (band > 0)]
        if valid.size:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb_aligned[i] = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb_aligned = np.moveaxis(rgb_aligned, 0, -1)

    # Load metrics
    neon_chm, _ = read_chm(neon_chm_path)
    gap_frac, gap_prof = read_metric(gap_frac_path)
    entropy, ent_prof = read_metric(entropy_path)
    crown_cv, cv_prof = read_metric(crown_cv_path)

    gap_extent = get_extent(gap_prof)
    ent_extent = get_extent(ent_prof)
    cv_extent = get_extent(cv_prof)

    # Load crown polygons
    crowns_present = crowns_path.exists()
    if crowns_present:
        crowns = gpd.read_file(crowns_path)
        if str(crowns.crs) != str(ref_profile["crs"]):
            crowns = crowns.to_crs(ref_profile["crs"])

    # -- Layout: 2 rows × 3 cols, panels a–e, panel f is blank/note
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    fig.subplots_adjust(wspace=0.12, hspace=0.12, left=0.01, right=0.94, top=0.95, bottom=0.02)

    # Panel a: NAIP RGB
    ax = axes[0, 0]
    ax.imshow(rgb_aligned, extent=extent, origin="upper", aspect="equal")
    ax.set_title("a.  NAIP true-color  |  2022", pad=4)
    add_scalebar(ax, extent, 200)
    add_north_arrow(ax, extent)
    label_panel(ax, "a", extent)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel b: Gap fraction
    ax = axes[0, 1]
    im = ax.imshow(gap_frac, extent=gap_extent, origin="upper", cmap="RdYlGn_r",
                   vmin=0, vmax=1, aspect="equal")
    ax.set_title("b.  Gap Fraction  |  25 m windows  |  h < 2 m", pad=4)
    label_panel(ax, "b", gap_extent)
    add_scalebar(ax, gap_extent, 200)
    ax.set_xticks([]); ax.set_yticks([])
    styled_colorbar(ax, im, "Gap Fraction")

    # Panel c: GLCM entropy
    ax = axes[0, 2]
    valid_ent = entropy[~np.isnan(entropy)]
    vmin_e = float(np.percentile(valid_ent, 2)) if valid_ent.size else 0
    vmax_e = float(np.percentile(valid_ent, 98)) if valid_ent.size else 1
    im = ax.imshow(entropy, extent=ent_extent, origin="upper", cmap="plasma",
                   vmin=vmin_e, vmax=vmax_e, aspect="equal")
    ax.set_title("c.  GLCM Texture Entropy  |  NIR band  |  7×7 px", pad=4)
    label_panel(ax, "c", ent_extent)
    add_scalebar(ax, ent_extent, 200)
    ax.set_xticks([]); ax.set_yticks([])
    styled_colorbar(ax, im, "Entropy (bits)")

    # Panel d: Crown segmentation on NEON CHM background
    ax = axes[1, 0]
    ax.imshow(neon_chm, extent=extent, origin="upper", cmap="Greys_r",
              vmin=0, vmax=35, aspect="equal")
    if crowns_present and len(crowns) > 0:
        crowns.boundary.plot(ax=ax, color="gold", linewidth=0.4, alpha=0.85)
        n_crowns = len(crowns)
    else:
        n_crowns = 0
        ax.text(0.5, 0.5, "Crown polygons not available\n(run 02_crown_segmentation.R)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="red")
    ax.set_title(f"d.  Crown Segmentation (lidR)  |  n={n_crowns:,}", pad=4)
    label_panel(ax, "d", extent)
    add_scalebar(ax, extent, 200)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel e: Crown width CV
    ax = axes[1, 1]
    valid_cv = crown_cv[~np.isnan(crown_cv)]
    if valid_cv.size > 0:
        vmin_cv = float(np.percentile(valid_cv, 2))
        vmax_cv = float(np.percentile(valid_cv, 98))
        im = ax.imshow(crown_cv, extent=cv_extent, origin="upper", cmap="RdYlGn",
                       vmin=vmin_cv, vmax=vmax_cv, aspect="equal")
        styled_colorbar(ax, im, "Crown Width CV")
    else:
        ax.text(0.5, 0.5, "Crown CV not available\n(run 02_crown_segmentation.R)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="red")
    ax.set_title("e.  Crown Width CV  |  50 m windows", pad=4)
    label_panel(ax, "e", cv_extent if valid_cv.size else extent)
    add_scalebar(ax, cv_extent if valid_cv.size else extent, 200)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel f: blank / explanatory note
    ax = axes[1, 2]
    ax.axis("off")
    note = (
        "SCBI — Smithsonian Conservation Biology Institute, VA\n"
        "Mixed oak-hickory hardwood forest\n\n"
        "Data sources:\n"
        "  NAIP: USDA, 1.0 m (VA 2022 cycle)\n"
        "  NAIP-CHM: Morford et al. (2025), 0.6 m\n"
        "  NEON ALS CHM: NEON AOP 2023, 1 m\n\n"
        "Metrics:\n"
        "  Gap fraction: h < 2 m threshold, 25 m windows\n"
        "  GLCM entropy: NIR band, 7×7 px window\n"
        "  Crown seg.: lidR lmf + Dalponte (2016)\n"
        "  Crown width CV: σ/μ of est. diameters, 50 m"
    )
    ax.text(0.05, 0.95, note, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc", lw=0.8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    setup_style()

    # Resolve input paths
    try:
        naip_path = find_latest(NAIP_DIR, "NAIP_SCBI")
        logger.info("NAIP: %s", naip_path.name)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    try:
        naip_chm_path = find_latest(NAIPCHM_DIR, "NAIPCHM_SCBI")
        logger.info("NAIP-CHM: %s", naip_chm_path.name)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    for path, label in [
        (NEON_CHM, "NEON CHM"),
        (INTERMEDIATE / "gap_fraction_25m_SCBI.tif", "Gap fraction"),
        (INTERMEDIATE / "glcm_entropy_NAIP_SCBI.tif", "GLCM entropy"),
    ]:
        if not path.exists():
            logger.error("%s not found: %s. Run prerequisite scripts.", label, path)
            raise SystemExit(1)

    logger.info("=== Making Figure 1: Data Sources ===")
    make_figure1(naip_path, naip_chm_path, NEON_CHM)

    logger.info("=== Making Figure 2: Derived Metrics ===")
    make_figure2(
        naip_path=naip_path,
        gap_frac_path=INTERMEDIATE / "gap_fraction_25m_SCBI.tif",
        entropy_path=INTERMEDIATE / "glcm_entropy_NAIP_SCBI.tif",
        crowns_path=INTERMEDIATE / "crown_polygons_SCBI.gpkg",
        crown_cv_path=INTERMEDIATE / "crown_cv_50m_SCBI.tif",
        neon_chm_path=NEON_CHM,
    )

    logger.info("=== All figures complete ===")
    logger.info("  Figure 1: %s", OUT_FIG1)
    logger.info("  Figure 2: %s", OUT_FIG2)


if __name__ == "__main__":
    main()
