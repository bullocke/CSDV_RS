"""
08_compare_chm_sources.py — Compare NEON ALS CHM and NAIP CHM as inputs for
structural metrics: crown segmentation, gap fraction, and crown width CV.

Three figures per site, each showing three panels:
  Left:   NAIP true-color RGB
  Middle: Metric derived from NEON ALS CHM (1 m lidar)
  Right:  Metric derived from NAIP CHM (0.6 m deep learning, Morford et al. 2025)

The figures use a deep zoom extent — a square window (default 400 m) centered
on the bottom-left quadrant of the full NEON CHM subset. This is more zoomed
than the existing _zoom figures (which cover the full SW quadrant) and shows
individual crowns clearly. The NEON and NAIP panels share identical color
scaling for direct visual comparison.

Intermediate outputs are written to:
  Results/summary_document/intermediate/deepzoom/{SITE}/

Figures are saved to:
  Results/summary_document/figures/
  fig_compare_crownseg_{SITE}.png
  fig_compare_gapfrac_{SITE}.png
  fig_compare_crownCV_{SITE}.png

Usage
-----
  python 08_compare_chm_sources.py --site SCBI
  python 08_compare_chm_sources.py --site HARV

To add a new site: add a SiteConfig entry to SITES and run the matching
download script first (01_download_data.py for SCBI, 01b_download_data_harv.py
for HARV).
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.features import rasterize as rio_rasterize
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine
from shapely.geometry import box, mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
POC = PROJECT_ROOT / "ProofOfConcept"
DATA = POC / "Data"

R_SCRIPT = HERE.parent / "02_crown_segmentation.R"
MICROMAMBA = Path("/home/bullocke/micromamba/micromamba")
ENV_PATH = Path("/home/bullocke/vscode_projects/csdv/.micromamba/envs/CSDV")

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------


@dataclass
class SiteConfig:
    site_code: str
    neon_chm_path: Path
    naip_rgb_dir: Path
    naip_rgb_prefix: str
    naip_chm_dir: Path
    naip_chm_prefix: str
    deepzoom_size_m: float = 400.0
    mediumzoom_size_m: float = 800.0
    label: str = ""


SITES: dict[str, SiteConfig] = {
    "SCBI": SiteConfig(
        site_code="SCBI",
        neon_chm_path=DATA / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif",
        naip_rgb_dir=DATA / "NAIP" / "Imagery",
        naip_rgb_prefix="NAIP_SCBI",
        naip_chm_dir=DATA / "NAIP" / "National_CHM",
        naip_chm_prefix="NAIPCHM_SCBI",
        deepzoom_size_m=400.0,
        label="SCBI — Smithsonian Conservation Biology Institute, VA",
    ),
    "HARV": SiteConfig(
        site_code="HARV",
        neon_chm_path=DATA / "NEON" / "CHM" / "NEON_CHM_HARV_Subset_2023.tif",
        naip_rgb_dir=DATA / "NAIP" / "Imagery",
        naip_rgb_prefix="NAIP_HARV",
        naip_chm_dir=DATA / "NAIP" / "National_CHM",
        naip_chm_prefix="NAIPCHM_HARV",
        deepzoom_size_m=400.0,
        label="HARV — Harvard Forest, MA",
    ),
}

# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------


def find_latest_nozoom(directory: Path, prefix: str) -> Path:
    """Return most recent .tif matching prefix, excluding _zoom and _deepzoom."""
    tifs = sorted(
        [
            p
            for p in directory.glob(f"{prefix}*.tif")
            if "_zoom" not in p.stem and "_deepzoom" not in p.stem
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not tifs:
        raise FileNotFoundError(
            f"No .tif matching '{prefix}*' (non-zoom) in {directory}"
        )
    return tifs[-1]


def compute_deepzoom_bbox(
    neon_chm_path: Path,
    size_m: float,
) -> tuple[float, float, float, float]:
    """
    Return (west, south, east, north) for a square window centered on the
    bottom-left quadrant of the NEON CHM subset.

    The bottom-left quadrant spans rows [h//2 : h], cols [0 : w//2]. Its
    geographic center is used as the window center, so the deep zoom sits
    inside the area already shown in the existing _zoom figures.
    """
    with rasterio.open(neon_chm_path) as src:
        h, w = src.height, src.width
        center_row = h // 2 + (h - h // 2) // 2
        center_col = (w // 2) // 2
        cx, cy = src.xy(center_row, center_col)

    half = size_m / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)
    logger.info(
        "Deep zoom bbox: W=%.0f S=%.0f E=%.0f N=%.0f  (%.0f m × %.0f m)",
        *bbox,
        size_m,
        size_m,
    )
    return bbox


def clip_raster_to_bbox(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Clip any raster (single- or multi-band) to a geographic bbox and save."""
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rasterio_mask(src, [geom], crop=True, all_touched=True)
        profile = src.profile.copy()
        profile.update(
            height=out_data.shape[1],
            width=out_data.shape[2],
            transform=out_transform,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data)
    logger.info("Clipped -> %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)


def clip_and_convert_naip_chm(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    scale: float = 0.01,
) -> None:
    """
    Clip the NAIP CHM (UInt16, stored as height_m * 100) to bbox and write
    float32 in meters. The output is directly usable by compute_gap_fraction()
    and 02_crown_segmentation.R without further scaling.
    """
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rasterio_mask(src, [geom], crop=True, all_touched=True)
        src_nodata = src.nodata
        profile = src.profile.copy()

    arr = out_data[0].astype(np.float32) * scale
    if src_nodata is not None:
        arr[out_data[0] == src_nodata] = -9999.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        height=arr.shape[0],
        width=arr.shape[1],
        transform=out_transform,
        dtype="float32",
        nodata=-9999.0,
        count=1,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr[np.newaxis])
    logger.info(
        "NAIP CHM clip (meters) -> %s (%.1f MB)",
        out_path.name,
        out_path.stat().st_size / 1e6,
    )


def run_r_segmentation(
    chm_path: Path,
    out_crowns: Path,
    out_cv: Path,
) -> None:
    """Run 02_crown_segmentation.R on the given CHM clip via micromamba."""
    cmd = [
        str(MICROMAMBA),
        "run",
        "-p",
        str(ENV_PATH),
        "Rscript",
        str(R_SCRIPT),
        str(chm_path),
        str(out_crowns),
        str(out_cv),
    ]
    logger.info("Running R segmentation on: %s", chm_path.name)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"R segmentation failed for {chm_path.name}")


def resolve_neon_chm(cfg: SiteConfig) -> Path:
    """
    Locate the NEON CHM file for a site, accounting for the wxee timestamp
    suffix that the download script may append (e.g. .time.19700101T000000).
    Returns the configured path if it exists; otherwise returns the most recent
    .tif whose stem starts with the configured stem.
    """
    if cfg.neon_chm_path.exists():
        return cfg.neon_chm_path
    matches = sorted(
        cfg.neon_chm_path.parent.glob(f"{cfg.neon_chm_path.stem}*.tif"),
        key=lambda p: p.stat().st_mtime,
    )
    if matches:
        logger.info(
            "NEON CHM resolved via glob: %s", matches[-1].name
        )
        return matches[-1]
    raise FileNotFoundError(
        f"NEON CHM not found: {cfg.neon_chm_path}\n"
        "Run the download script first (01_download_data.py or 01b_download_data_harv.py)."
    )


def _import_compute_metrics() -> object:
    """Import compute_gap_fraction and save_raster from 03_compute_metrics.py."""
    path = HERE.parent / "03_compute_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_metrics_08", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["compute_metrics_08"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Crown mask and CV helpers
# ---------------------------------------------------------------------------


def rasterize_crowns(crowns_path: Path, ref_chm_path: Path) -> np.ndarray:
    """
    Rasterize crown polygons to a binary uint8 mask matching the ref CHM grid.

    1 = pixel inside any crown polygon, 0 = outside.
    Both sources are reprojected to the ref CHM CRS before rasterization so
    the output mask is always in the same pixel space as ref_chm_path.
    Returns a zeros array if the GeoPackage is empty.
    """
    with rasterio.open(ref_chm_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        ref_crs = src.crs

    crowns = gpd.read_file(crowns_path).to_crs(ref_crs)
    if len(crowns) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    shapes = (
        (mapping(geom), 1)
        for geom in crowns.geometry
        if geom is not None and not geom.is_empty
    )
    return rio_rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def compute_crown_cv_python(
    crowns_path: Path,
    ref_chm_path: Path,
    window_m: float,
) -> tuple[np.ndarray | None, Affine | None, str | None]:
    """
    Compute crown width CV per spatial window in Python.

    Replicates the R tapply step from 02_crown_segmentation.R at any window
    size, avoiding repeated R calls for multi-window figures. Requires the
    crown GeoPackage to contain a 'crown_diam_m' column (produced by R).

    Returns (cv_grid, affine_transform, crs_str), or (None, None, None) if
    the GeoPackage is empty or has no crowns with valid diameters.
    """
    crowns = gpd.read_file(crowns_path)
    if len(crowns) == 0 or "crown_diam_m" not in crowns.columns:
        return None, None, None

    with rasterio.open(ref_chm_path) as src:
        bounds = src.bounds
        ref_crs = src.crs
        pixel_size = abs(src.transform.a)

    crowns = crowns.to_crs(ref_crs)
    centroids = crowns.geometry.centroid
    diams = crowns["crown_diam_m"].values.astype(np.float64)

    xmin, ymax = bounds.left, bounds.top
    # Use floor division to discard partial edge cells, consistent with
    # gap_fraction() which uses (shape // window_px).
    window_px = max(1, int(round(window_m / pixel_size)))
    n_cols = int((bounds.right - xmin) / pixel_size) // window_px
    n_rows = int((ymax - bounds.bottom) / pixel_size) // window_px

    col_idx = ((centroids.x.values - xmin) / window_m).astype(int)
    row_idx = ((ymax - centroids.y.values) / window_m).astype(int)

    valid = (col_idx >= 0) & (col_idx < n_cols) & (row_idx >= 0) & (row_idx < n_rows)
    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    diams = diams[valid]

    cell_ids = row_idx * n_cols + col_idx
    cv_flat = np.full(n_rows * n_cols, np.nan, dtype=np.float32)

    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        if mask.sum() < 3:
            continue
        d = diams[mask]
        mu = d.mean()
        if mu > 0:
            cv_flat[cell_id] = float(d.std() / mu)

    cv_grid = cv_flat.reshape(n_rows, n_cols)
    transform = Affine(window_m, 0.0, xmin, 0.0, -window_m, ymax)
    return cv_grid, transform, str(ref_crs)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

FONT_SIZE = 9
DPI = 300
_PANEL_LABEL_KW = dict(
    fontsize=11,
    fontweight="bold",
    va="top",
    ha="left",
    color="white",
    bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.6, lw=0),
)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "sans-serif",
            "axes.titlesize": FONT_SIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "image.interpolation": "none",
        }
    )


def _profile_extent(profile: dict) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) from a rasterio profile dict."""
    t = profile["transform"]
    h, w = profile["height"], profile["width"]
    left = t.c
    top = t.f
    right = left + w * t.a
    bottom = top + h * t.e  # t.e is negative
    return left, right, bottom, top


def _read_rgb(path: Path) -> tuple[np.ndarray, dict]:
    """Read bands 1,2,3 as float [0,1] with 2–98 pct stretch."""
    with rasterio.open(path) as src:
        profile: dict = {"transform": src.transform, "height": src.height, "width": src.width}
        bands = [src.read(i).astype(np.float32) for i in (1, 2, 3)]
    rgb = []
    for b in bands:
        valid = b[b > 0]
        if valid.size:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb.append(np.clip((b - p2) / (p98 - p2 + 1e-6), 0, 1))
        else:
            rgb.append(np.zeros_like(b))
    return np.stack(rgb, axis=-1), profile


def _read_chm_meters(path: Path) -> tuple[np.ndarray, dict]:
    """Read a CHM that is already in meters; mask nodata as nan."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile: dict = {"transform": src.transform, "height": src.height, "width": src.width}
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


def _read_metric(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single-band metric raster as float32; mask nodata as nan."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile: dict = {"transform": src.transform, "height": src.height, "width": src.width}
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


def _add_scalebar(
    ax: plt.Axes,
    disp: tuple[float, float, float, float],
    length_m: float = 100.0,
) -> None:
    left, right, bottom, top = disp
    xr = right - left
    yr = top - bottom
    x0 = right - 0.08 * xr - length_m
    x1 = x0 + length_m
    y0 = bottom + 0.05 * yr
    ax.plot([x0, x1], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.01 * yr, y0 + 0.01 * yr], color="white", lw=2)
    ax.plot([x1, x1], [y0 - 0.01 * yr, y0 + 0.01 * yr], color="white", lw=2)
    ax.text(
        (x0 + x1) / 2,
        y0 + 0.02 * yr,
        f"{int(length_m)} m",
        ha="center",
        va="bottom",
        fontsize=7,
        color="white",
        fontweight="bold",
    )


def _label_panel(
    ax: plt.Axes,
    letter: str,
    disp: tuple[float, float, float, float],
) -> None:
    left, right, bottom, top = disp
    xr = right - left
    yr = top - bottom
    ax.text(
        left + 0.02 * xr,
        top - 0.02 * yr,
        letter,
        **_PANEL_LABEL_KW,
        transform=ax.transData,
    )


def _styled_colorbar(ax: plt.Axes, im: object, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FONT_SIZE - 1)
    cb.ax.tick_params(labelsize=FONT_SIZE - 2)


def _set_ax(ax: plt.Axes, disp: tuple[float, float, float, float]) -> None:
    left, right, bottom, top = disp
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Figure makers
# ---------------------------------------------------------------------------


def make_crownseg_figure(
    naip_rgb_clip: Path,
    neon_chm_clip: Path,
    naip_chm_clip: Path,
    neon_crowns: Path,
    naip_crowns: Path,
    disp: tuple[float, float, float, float],
    out_path: Path,
    site_label: str = "",
    outline_color: str = "gold",
    outline_lw: float = 0.5,
    add_chm_colorbar: bool = False,
) -> None:
    """
    Three-panel crown segmentation comparison.

    Left: NAIP RGB. Middle: NEON CHM (gray) with crown outlines and count.
    Right: NAIP CHM (gray) with crown outlines and count.
    Both CHM panels share the same 0–35 m grayscale range.

    Parameters
    ----------
    outline_color, outline_lw
        Crown polygon outline color and linewidth. Defaults preserve the
        original styling; pass higher contrast / thicker values for the
        improved _v2 figures.
    add_chm_colorbar
        If True, attach a colorbar to the rightmost CHM panel labeled
        "CHM height (m)".
    """
    logger.info("Building crown segmentation comparison figure ...")

    rgb, rgb_prof = _read_rgb(naip_rgb_clip)
    neon_chm, neon_prof = _read_chm_meters(neon_chm_clip)
    naip_chm, naip_prof = _read_chm_meters(naip_chm_clip)

    rgb_ext = _profile_extent(rgb_prof)
    neon_ext = _profile_extent(neon_prof)
    naip_ext = _profile_extent(naip_prof)

    neon_gdf = gpd.read_file(neon_crowns) if neon_crowns.exists() else None
    naip_gdf = gpd.read_file(naip_crowns) if naip_crowns.exists() else None
    n_neon = len(neon_gdf) if neon_gdf is not None else 0
    n_naip = len(naip_gdf) if naip_gdf is not None else 0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.97, top=0.88, bottom=0.02)
    if site_label:
        fig.suptitle(
            f"Crown segmentation — {site_label}  |  same lidR parameters (lmf + Dalponte 2016)",
            fontsize=FONT_SIZE + 1,
            fontweight="bold",
        )

    vmin_chm, vmax_chm = 0, 35

    ax = axes[0]
    ax.imshow(rgb, extent=rgb_ext, origin="upper", aspect="equal")
    ax.set_title("NAIP true-color  |  2022", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "a", disp)
    _set_ax(ax, disp)

    ax = axes[1]
    im_neon = ax.imshow(
        neon_chm, extent=neon_ext, origin="upper",
        cmap="Greys_r", vmin=vmin_chm, vmax=vmax_chm, aspect="equal",
    )
    if neon_gdf is not None and len(neon_gdf) > 0:
        neon_gdf.to_crs("EPSG:5070").boundary.plot(
            ax=ax, color=outline_color, linewidth=outline_lw, alpha=0.95
        )
    ax.set_title(f"NEON ALS CHM  |  n = {n_neon:,} crowns", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "b", disp)
    _set_ax(ax, disp)

    ax = axes[2]
    im_naip = ax.imshow(
        naip_chm, extent=naip_ext, origin="upper",
        cmap="Greys_r", vmin=vmin_chm, vmax=vmax_chm, aspect="equal",
    )
    if naip_gdf is not None and len(naip_gdf) > 0:
        naip_gdf.to_crs("EPSG:5070").boundary.plot(
            ax=ax, color=outline_color, linewidth=outline_lw, alpha=0.95
        )
    ax.set_title(f"NAIP CHM (Morford et al. 2025)  |  n = {n_naip:,} crowns", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "c", disp)
    _set_ax(ax, disp)

    if add_chm_colorbar:
        _styled_colorbar(axes[2], im_naip, "CHM height (m)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def make_raster_comparison_figure(
    naip_rgb_clip: Path,
    neon_metric_path: Path,
    naip_metric_path: Path,
    disp: tuple[float, float, float, float],
    out_path: Path,
    cmap: str,
    cbar_label: str,
    title_neon: str,
    title_naip: str,
    site_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Three-panel raster metric comparison (gap fraction or crown width CV).

    Left: NAIP RGB. Middle: NEON-derived metric. Right: NAIP CHM-derived metric.
    Both metric panels share identical vmin/vmax for direct comparison; derived
    from the 2nd/98th percentile of the combined data if not specified.
    """
    logger.info("Building raster comparison figure: %s ...", out_path.name)

    rgb, rgb_prof = _read_rgb(naip_rgb_clip)
    neon_metric, neon_prof = _read_metric(neon_metric_path)
    naip_metric, naip_prof = _read_metric(naip_metric_path)

    rgb_ext = _profile_extent(rgb_prof)
    neon_ext = _profile_extent(neon_prof)
    naip_ext = _profile_extent(naip_prof)

    combined = np.concatenate(
        [neon_metric[~np.isnan(neon_metric)], naip_metric[~np.isnan(naip_metric)]]
    )
    if vmin is None:
        vmin = float(np.percentile(combined, 2)) if combined.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(combined, 98)) if combined.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.97, top=0.88, bottom=0.02)
    if site_label:
        fig.suptitle(
            f"{cbar_label} comparison — {site_label}",
            fontsize=FONT_SIZE + 1,
            fontweight="bold",
        )

    ax = axes[0]
    ax.imshow(rgb, extent=rgb_ext, origin="upper", aspect="equal")
    ax.set_title("NAIP true-color  |  2022", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "a", disp)
    _set_ax(ax, disp)

    ax = axes[1]
    im = ax.imshow(
        neon_metric, extent=neon_ext, origin="upper",
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
    )
    ax.set_title(title_neon, pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "b", disp)
    _set_ax(ax, disp)
    _styled_colorbar(ax, im, cbar_label)

    ax = axes[2]
    im = ax.imshow(
        naip_metric, extent=naip_ext, origin="upper",
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
    )
    ax.set_title(title_naip, pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "c", disp)
    _set_ax(ax, disp)
    _styled_colorbar(ax, im, cbar_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# Categorical colormap for IoU agreement map (TN, TP, FP, FN)
_AGREE_CMAP = ListedColormap(["#404040", "#4caf50", "#2196f3", "#f44336"])
_AGREE_LEGEND = [
    Patch(color="#404040", label="True Negative (neither detects)"),
    Patch(color="#4caf50", label="True Positive (both detect)"),
    Patch(color="#2196f3", label="False Positive (NAIP only)"),
    Patch(color="#f44336", label="False Negative (NEON only)"),
]


def make_iou_figure(
    naip_rgb_clip: Path,
    neon_chm_clip: Path,
    naip_chm_clip: Path,
    neon_crowns: Path,
    naip_crowns: Path,
    disp: tuple[float, float, float, float],
    out_path: Path,
    site_label: str = "",
    outline_color_neon: str = "#ffc107",
    outline_color_naip: str = "#00e5ff",
    outline_lw: float = 0.4,
    add_chm_colorbar: bool = False,
) -> None:
    """
    Four-panel crown segmentation IoU figure.

    Computes pixel-level IoU, precision, and recall treating NEON ALS as
    ground truth. Displays individual crown outlines overlaid on the matching
    CHM background so individual crowns are distinguishable:

      (a) NAIP true-color RGB
      (b) NEON ALS CHM (greyscale 0–35 m) + NEON crown polygon outlines (gold)
      (c) NAIP CHM (greyscale 0–35 m) + NAIP crown polygon outlines (cyan)
      (d) NAIP RGB + per-pixel agreement overlay (TP/FP/FN/TN)

    The agreement overlay (d) uses semi-transparency over the RGB so the
    canopy structure remains visible through the colour coding.
    """
    logger.info("Building IoU figure ...")

    rgb, rgb_prof = _read_rgb(naip_rgb_clip)
    rgb_ext = _profile_extent(rgb_prof)

    neon_chm, neon_prof = _read_chm_meters(neon_chm_clip)
    naip_chm, naip_prof = _read_chm_meters(naip_chm_clip)
    neon_ext = _profile_extent(neon_prof)
    naip_ext = _profile_extent(naip_prof)

    # Binary masks for IoU computation only (not displayed directly)
    neon_mask = rasterize_crowns(neon_crowns, neon_chm_clip)
    naip_mask = rasterize_crowns(naip_crowns, neon_chm_clip)  # same reference grid

    tp = int(np.sum((neon_mask == 1) & (naip_mask == 1)))
    fp = int(np.sum((neon_mask == 0) & (naip_mask == 1)))
    fn = int(np.sum((neon_mask == 1) & (naip_mask == 0)))
    denom_iou = tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Agreement map for overlay: 0=TN (skip), 1=TP, 2=FP, 3=FN
    agree_rgba = np.zeros((*neon_mask.shape, 4), dtype=np.float32)
    agree_rgba[(neon_mask == 1) & (naip_mask == 1)] = [0.30, 0.69, 0.31, 0.55]  # green TP
    agree_rgba[(neon_mask == 0) & (naip_mask == 1)] = [0.13, 0.59, 0.95, 0.55]  # blue FP
    agree_rgba[(neon_mask == 1) & (naip_mask == 0)] = [0.96, 0.26, 0.21, 0.55]  # red FN

    # Crown geodataframes
    neon_gdf = gpd.read_file(neon_crowns) if neon_crowns.exists() else None
    naip_gdf = gpd.read_file(naip_crowns) if naip_crowns.exists() else None
    n_neon = len(neon_gdf) if neon_gdf is not None else 0
    n_naip = len(naip_gdf) if naip_gdf is not None else 0

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.97, top=0.84, bottom=0.02)
    title = (
        f"Crown segmentation IoU — {site_label}  |  "
        f"IoU = {iou:.3f}  |  Precision = {precision:.3f}  |  Recall = {recall:.3f}"
    )
    fig.suptitle(title, fontsize=FONT_SIZE + 0.5, fontweight="bold")

    # Panel a: NAIP RGB
    ax = axes[0]
    ax.imshow(rgb, extent=rgb_ext, origin="upper", aspect="equal")
    ax.set_title("NAIP true-color  |  2022", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "a", disp)
    _set_ax(ax, disp)

    # Panel b: NEON CHM + NEON crown outlines
    ax = axes[1]
    im_neon = ax.imshow(neon_chm, extent=neon_ext, origin="upper",
                        cmap="Greys_r", vmin=0, vmax=35, aspect="equal")
    if neon_gdf is not None and len(neon_gdf) > 0:
        neon_gdf.to_crs("EPSG:5070").boundary.plot(
            ax=ax, color=outline_color_neon, linewidth=outline_lw, alpha=0.95
        )
    ax.set_title(f"NEON ALS CHM  |  {n_neon:,} crowns", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "b", disp)
    _set_ax(ax, disp)

    # Panel c: NAIP CHM + NAIP crown outlines
    ax = axes[2]
    im_naip = ax.imshow(naip_chm, extent=naip_ext, origin="upper",
                        cmap="Greys_r", vmin=0, vmax=35, aspect="equal")
    if naip_gdf is not None and len(naip_gdf) > 0:
        naip_gdf.to_crs("EPSG:5070").boundary.plot(
            ax=ax, color=outline_color_naip, linewidth=outline_lw, alpha=0.95
        )
    ax.set_title(f"NAIP CHM (Morford 2025)  |  {n_naip:,} crowns", pad=4)
    _add_scalebar(ax, disp)
    _label_panel(ax, "c", disp)
    _set_ax(ax, disp)

    if add_chm_colorbar:
        _styled_colorbar(axes[2], im_naip, "CHM height (m)")

    # Panel d: NAIP RGB + semi-transparent agreement overlay
    ax = axes[3]
    ax.imshow(rgb, extent=rgb_ext, origin="upper", aspect="equal")
    ax.imshow(agree_rgba, extent=neon_ext, origin="upper", aspect="equal")
    ax.set_title("Agreement  |  NEON = truth", pad=4)
    ax.legend(
        handles=_AGREE_LEGEND,
        loc="lower left",
        fontsize=6,
        framealpha=0.85,
        handlelength=1.0,
        borderpad=0.4,
    )
    _label_panel(ax, "d", disp)
    _set_ax(ax, disp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s  (IoU=%.3f)", out_path.name, iou)


def make_multiwindow_figure(
    naip_rgb_clip: Path,
    neon_metrics: dict[float, Path],
    naip_metrics: dict[float, Path],
    window_sizes: list[float],
    disp: tuple[float, float, float, float],
    out_path: Path,
    cmap: str,
    cbar_label: str,
    metric_name: str,
    site_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    3×3 multi-window comparison figure (9 panels, figsize 9×9 inches).

    Each row shows one window size. Columns: NAIP RGB | NEON metric | NAIP metric.
    All metric panels share identical vmin/vmax. A single colorbar appears to
    the right of the last column. Scale bar on the first panel only.
    """
    logger.info("Building multi-window figure: %s ...", out_path.name)

    rgb, rgb_prof = _read_rgb(naip_rgb_clip)
    rgb_ext = _profile_extent(rgb_prof)

    # Load all metric arrays and collect combined value range
    neon_arrays: dict[float, tuple[np.ndarray, dict]] = {}
    naip_arrays: dict[float, tuple[np.ndarray, dict]] = {}
    all_vals: list[np.ndarray] = []

    for w in window_sizes:
        if w in neon_metrics and neon_metrics[w].exists():
            arr, prof = _read_metric(neon_metrics[w])
            neon_arrays[w] = (arr, prof)
            valid = arr[~np.isnan(arr)]
            if valid.size:
                all_vals.append(valid)
        if w in naip_metrics and naip_metrics[w].exists():
            arr, prof = _read_metric(naip_metrics[w])
            naip_arrays[w] = (arr, prof)
            valid = arr[~np.isnan(arr)]
            if valid.size:
                all_vals.append(valid)

    if vmin is None or vmax is None:
        combined = np.concatenate(all_vals) if all_vals else np.array([0.0, 1.0])
        if vmin is None:
            vmin = float(np.percentile(combined, 2))
        if vmax is None:
            vmax = float(np.percentile(combined, 98))

    n_rows = len(window_sizes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 9))
    fig.subplots_adjust(
        wspace=0.04, hspace=0.12,
        left=0.08, right=0.91, top=0.92, bottom=0.02,
    )
    if site_label:
        fig.suptitle(
            f"{metric_name} by window size — {site_label}  |  NEON ALS vs. NAIP CHM",
            fontsize=FONT_SIZE + 1,
            fontweight="bold",
        )

    last_im = None
    for ri, w in enumerate(window_sizes):
        row_label = f"{int(w)} m"

        # Column 0: NAIP RGB
        ax = axes[ri, 0]
        ax.imshow(rgb, extent=rgb_ext, origin="upper", aspect="equal")
        ax.set_ylabel(row_label, fontsize=FONT_SIZE, fontweight="bold", labelpad=4)
        if ri == 0:
            ax.set_title("NAIP true-color", pad=4, fontsize=FONT_SIZE)
            _add_scalebar(ax, disp)
            _label_panel(ax, "a", disp)
        _set_ax(ax, disp)

        # Column 1: NEON metric
        ax = axes[ri, 1]
        if w in neon_arrays:
            arr, prof = neon_arrays[w]
            ext = _profile_extent(prof)
            im = ax.imshow(arr, extent=ext, origin="upper",
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            last_im = im
        else:
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=7)
        if ri == 0:
            ax.set_title("NEON ALS CHM", pad=4, fontsize=FONT_SIZE)
            _label_panel(ax, "b", disp)
        _set_ax(ax, disp)

        # Column 2: NAIP metric
        ax = axes[ri, 2]
        if w in naip_arrays:
            arr, prof = naip_arrays[w]
            ext = _profile_extent(prof)
            im = ax.imshow(arr, extent=ext, origin="upper",
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            last_im = im
        else:
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=7)
        if ri == 0:
            ax.set_title("NAIP CHM", pad=4, fontsize=FONT_SIZE)
            _label_panel(ax, "c", disp)
        _set_ax(ax, disp)

    # Single shared colorbar to the right of the last column
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.75])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label(cbar_label, fontsize=FONT_SIZE)
        cb.ax.tick_params(labelsize=FONT_SIZE - 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_mediumzoom_pipeline(cfg: SiteConfig) -> None:
    """
    Medium zoom pipeline (800 m × 800 m) producing three additional figures:
      fig_iou_crownseg_{SITE}.png       — pixel-level IoU vs NEON truth
      fig_multiwindow_gapfrac_{SITE}.png — gap fraction at 25/50/100 m windows
      fig_multiwindow_crownCV_{SITE}.png — crown width CV at 25/50/100 m windows
    """
    site = cfg.site_code
    intermediate = (
        POC / "Results" / "summary_document" / "intermediate" / "mediumzoom" / site
    )
    figures_out = POC / "Results" / "summary_document" / "figures"
    intermediate.mkdir(parents=True, exist_ok=True)

    naip_rgb_src = find_latest_nozoom(cfg.naip_rgb_dir, cfg.naip_rgb_prefix)
    naip_chm_src = find_latest_nozoom(cfg.naip_chm_dir, cfg.naip_chm_prefix)
    neon_chm_src = resolve_neon_chm(cfg)

    bbox = compute_deepzoom_bbox(neon_chm_src, cfg.mediumzoom_size_m)
    west, south, east, north = bbox
    disp = (west, east, south, north)

    # --- Step 1: clip rasters ---
    logger.info("=== Medium zoom Step 1: Clipping rasters ===")
    neon_clip = intermediate / f"neon_chm_{site}_mediumzoom.tif"
    naip_rgb_clip = intermediate / f"naip_rgb_{site}_mediumzoom.tif"
    naip_chm_clip = intermediate / f"naip_chm_{site}_mediumzoom_meters.tif"

    clip_raster_to_bbox(neon_chm_src, bbox, neon_clip)
    clip_raster_to_bbox(naip_rgb_src, bbox, naip_rgb_clip)
    clip_and_convert_naip_chm(naip_chm_src, bbox, naip_chm_clip, scale=0.01)

    # --- Step 2: crown segmentation on both CHMs ---
    logger.info("=== Medium zoom Step 2: Crown segmentation (NEON CHM) ===")
    neon_crowns = intermediate / f"crowns_neon_{site}_mediumzoom.gpkg"
    neon_crown_cv_r = intermediate / f"crown_cv_neon_{site}_mediumzoom.tif"
    run_r_segmentation(neon_clip, neon_crowns, neon_crown_cv_r)

    logger.info("=== Medium zoom Step 3: Crown segmentation (NAIP CHM) ===")
    naip_crowns = intermediate / f"crowns_naip_{site}_mediumzoom.gpkg"
    naip_crown_cv_r = intermediate / f"crown_cv_naip_{site}_mediumzoom.tif"
    run_r_segmentation(naip_chm_clip, naip_crowns, naip_crown_cv_r)

    # --- Step 3: IoU figures (native 800m + 1/4-area 400m deepzoom) ---
    logger.info("=== Medium zoom Step 4: IoU figure (800 m) ===")
    make_iou_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_chm_clip=neon_clip,
        naip_chm_clip=naip_chm_clip,
        neon_crowns=neon_crowns,
        naip_crowns=naip_crowns,
        disp=disp,
        out_path=figures_out / f"fig_iou_crownseg_{site}.png",
        site_label=cfg.label,
    )

    logger.info("=== Medium zoom Step 4b: IoU figure (400 m deepzoom) ===")
    dz_bbox = compute_deepzoom_bbox(neon_chm_src, cfg.deepzoom_size_m)
    dz_intermediate = (
        POC / "Results" / "summary_document" / "intermediate" / "deepzoom" / site
    )
    dz_neon_clip = dz_intermediate / f"neon_chm_{site}_deepzoom.tif"
    dz_naip_rgb = dz_intermediate / f"naip_rgb_{site}_deepzoom.tif"
    dz_naip_chm = dz_intermediate / f"naip_chm_{site}_deepzoom_meters.tif"
    dz_neon_crowns = dz_intermediate / f"crowns_neon_{site}_deepzoom.gpkg"
    dz_naip_crowns = dz_intermediate / f"crowns_naip_{site}_deepzoom.gpkg"

    # Clip inputs to deepzoom bbox if not already done
    for src_p, dst_p, label in [
        (neon_clip, dz_neon_clip, "NEON CHM deepzoom"),
        (naip_rgb_clip, dz_naip_rgb, "NAIP RGB deepzoom"),
        (naip_chm_clip, dz_naip_chm, "NAIP CHM deepzoom"),
    ]:
        if not dst_p.exists():
            clip_raster_to_bbox(src_p, dz_bbox, dst_p)
            logger.info("Clipped %s -> %s", label, dst_p.name)

    # Crown segmentation on deepzoom clips
    dz_neon_cv = dz_intermediate / f"crown_cv_neon_{site}_deepzoom.tif"
    dz_naip_cv = dz_intermediate / f"crown_cv_naip_{site}_deepzoom.tif"
    if not dz_neon_crowns.exists():
        run_r_segmentation(dz_neon_clip, dz_neon_crowns, dz_neon_cv)
    if not dz_naip_crowns.exists():
        run_r_segmentation(dz_naip_chm, dz_naip_crowns, dz_naip_cv)

    dz_disp = (dz_bbox[0], dz_bbox[2], dz_bbox[1], dz_bbox[3])
    make_iou_figure(
        naip_rgb_clip=dz_naip_rgb,
        neon_chm_clip=dz_neon_clip,
        naip_chm_clip=dz_naip_chm,
        neon_crowns=dz_neon_crowns,
        naip_crowns=dz_naip_crowns,
        disp=dz_disp,
        out_path=figures_out / f"fig_iou_crownseg_{site}_zoom.png",
        site_label=cfg.label + "  (400 m detail)",
    )

    # --- Step 4: gap fraction at 3 window sizes ---
    logger.info("=== Medium zoom Step 5: Gap fraction (multi-window) ===")
    m = _import_compute_metrics()
    window_sizes = [25.0, 50.0, 100.0]
    gap_neon: dict[float, Path] = {}
    gap_naip: dict[float, Path] = {}

    for w in window_sizes:
        p = intermediate / f"gap_frac_neon_{site}_mediumzoom_{int(w)}m.tif"
        gf, gt, crs = m.compute_gap_fraction(neon_clip, window_m=w, height_threshold_m=2.0)
        m.save_raster(gf, p, gt, crs)
        gap_neon[w] = p

        p = intermediate / f"gap_frac_naip_{site}_mediumzoom_{int(w)}m.tif"
        gf, gt, crs = m.compute_gap_fraction(naip_chm_clip, window_m=w, height_threshold_m=2.0)
        m.save_raster(gf, p, gt, crs)
        gap_naip[w] = p

    make_multiwindow_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_metrics=gap_neon,
        naip_metrics=gap_naip,
        window_sizes=window_sizes,
        disp=disp,
        out_path=figures_out / f"fig_multiwindow_gapfrac_{site}.png",
        cmap="RdYlGn_r",
        cbar_label="Gap Fraction",
        metric_name="Gap Fraction",
        site_label=cfg.label,
        vmin=0.0,
        vmax=1.0,
    )

    # --- Step 5: crown width CV at 3 window sizes (Python) ---
    logger.info("=== Medium zoom Step 6: Crown width CV (multi-window) ===")
    cv_neon: dict[float, Path] = {}
    cv_naip: dict[float, Path] = {}

    for w in window_sizes:
        cv, transform, crs = compute_crown_cv_python(neon_crowns, neon_clip, w)
        if cv is not None:
            p = intermediate / f"crown_cv_neon_{site}_mediumzoom_{int(w)}m.tif"
            m.save_raster(cv, p, transform, crs)
            cv_neon[w] = p
        else:
            logger.warning("No crown CV data for NEON at %d m window", int(w))

        cv, transform, crs = compute_crown_cv_python(naip_crowns, naip_chm_clip, w)
        if cv is not None:
            p = intermediate / f"crown_cv_naip_{site}_mediumzoom_{int(w)}m.tif"
            m.save_raster(cv, p, transform, crs)
            cv_naip[w] = p
        else:
            logger.warning("No crown CV data for NAIP at %d m window", int(w))

    make_multiwindow_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_metrics=cv_neon,
        naip_metrics=cv_naip,
        window_sizes=window_sizes,
        disp=disp,
        out_path=figures_out / f"fig_multiwindow_crownCV_{site}.png",
        cmap="RdYlGn",
        cbar_label="Crown Width CV",
        metric_name="Crown Width CV",
        site_label=cfg.label,
    )

    logger.info("=== Medium zoom pipeline complete for %s ===", site)


def run_site_pipeline(cfg: SiteConfig) -> None:
    """Run the full CHM comparison pipeline for one site."""
    site = cfg.site_code
    intermediate = (
        POC / "Results" / "summary_document" / "intermediate" / "deepzoom" / site
    )
    figures_out = POC / "Results" / "summary_document" / "figures"
    intermediate.mkdir(parents=True, exist_ok=True)

    naip_rgb_src = find_latest_nozoom(cfg.naip_rgb_dir, cfg.naip_rgb_prefix)
    naip_chm_src = find_latest_nozoom(cfg.naip_chm_dir, cfg.naip_chm_prefix)
    neon_chm_src = resolve_neon_chm(cfg)
    logger.info("NAIP RGB : %s", naip_rgb_src.name)
    logger.info("NAIP CHM : %s", naip_chm_src.name)
    logger.info("NEON CHM : %s", neon_chm_src.name)

    # Compute deep zoom bbox
    bbox = compute_deepzoom_bbox(neon_chm_src, cfg.deepzoom_size_m)
    west, south, east, north = bbox
    # (left, right, bottom, top) format used by matplotlib extent and axis limits
    disp = (west, east, south, north)

    # Step 1: clip source rasters
    logger.info("=== Step 1: Clipping rasters ===")
    neon_clip = intermediate / f"neon_chm_{site}_deepzoom.tif"
    naip_rgb_clip = intermediate / f"naip_rgb_{site}_deepzoom.tif"
    naip_chm_clip = intermediate / f"naip_chm_{site}_deepzoom_meters.tif"

    clip_raster_to_bbox(neon_chm_src, bbox, neon_clip)
    clip_raster_to_bbox(naip_rgb_src, bbox, naip_rgb_clip)
    clip_and_convert_naip_chm(naip_chm_src, bbox, naip_chm_clip, scale=0.01)

    # Step 2: crown segmentation — NEON CHM
    logger.info("=== Step 2: Crown segmentation (NEON CHM) ===")
    neon_crowns = intermediate / f"crowns_neon_{site}_deepzoom.gpkg"
    neon_crown_cv = intermediate / f"crown_cv_neon_{site}_deepzoom.tif"
    run_r_segmentation(neon_clip, neon_crowns, neon_crown_cv)

    # Step 3: crown segmentation — NAIP CHM (same R script, CHM already in meters)
    logger.info("=== Step 3: Crown segmentation (NAIP CHM) ===")
    naip_crowns = intermediate / f"crowns_naip_{site}_deepzoom.gpkg"
    naip_crown_cv = intermediate / f"crown_cv_naip_{site}_deepzoom.tif"
    run_r_segmentation(naip_chm_clip, naip_crowns, naip_crown_cv)

    # Step 4: gap fraction from both CHMs
    logger.info("=== Step 4: Gap fraction ===")
    m = _import_compute_metrics()
    gap_neon = intermediate / f"gap_frac_neon_{site}_deepzoom.tif"
    gap_naip = intermediate / f"gap_frac_naip_{site}_deepzoom.tif"

    gf, gt, crs = m.compute_gap_fraction(neon_clip, window_m=25.0, height_threshold_m=2.0)
    m.save_raster(gf, gap_neon, gt, crs)

    gf, gt, crs = m.compute_gap_fraction(naip_chm_clip, window_m=25.0, height_threshold_m=2.0)
    m.save_raster(gf, gap_naip, gt, crs)

    # Step 5: figures
    logger.info("=== Step 5: Generating figures ===")

    make_crownseg_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_chm_clip=neon_clip,
        naip_chm_clip=naip_chm_clip,
        neon_crowns=neon_crowns,
        naip_crowns=naip_crowns,
        disp=disp,
        out_path=figures_out / f"fig_compare_crownseg_{site}.png",
        site_label=cfg.label,
    )

    make_raster_comparison_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_metric_path=gap_neon,
        naip_metric_path=gap_naip,
        disp=disp,
        out_path=figures_out / f"fig_compare_gapfrac_{site}.png",
        cmap="RdYlGn_r",
        cbar_label="Gap Fraction",
        title_neon="NEON ALS CHM  |  gap fraction  |  25 m windows",
        title_naip="NAIP CHM  |  gap fraction  |  25 m windows",
        site_label=cfg.label,
        vmin=0.0,
        vmax=1.0,
    )

    make_raster_comparison_figure(
        naip_rgb_clip=naip_rgb_clip,
        neon_metric_path=neon_crown_cv,
        naip_metric_path=naip_crown_cv,
        disp=disp,
        out_path=figures_out / f"fig_compare_crownCV_{site}.png",
        cmap="RdYlGn",
        cbar_label="Crown Width CV",
        title_neon="NEON ALS CHM  |  crown width CV  |  50 m windows",
        title_naip="NAIP CHM  |  crown width CV  |  50 m windows",
        site_label=cfg.label,
    )

    logger.info("=== Deep zoom pipeline complete for %s ===", site)

    # Medium zoom: IoU, multi-window gap fraction, multi-window crown CV
    run_mediumzoom_pipeline(cfg)

    logger.info("=== All figures complete for %s ===", site)
    logger.info("  Figures: %s", figures_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--site",
    type=click.Choice(list(SITES.keys()), case_sensitive=False),
    default="SCBI",
    show_default=True,
    help="Site to process. Data must be downloaded first.",
)
def main(site: str) -> None:
    """Compare NEON ALS CHM and NAIP CHM as structural metric inputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _setup_style()
    run_site_pipeline(SITES[site.upper()])


if __name__ == "__main__":
    main()
