"""
07_make_fig03_naip_consistency.py — Multi-temporal NAIP consistency figure.

Produces Figure 3 for the CSDV summary document (Section 5.2): a grid of RGB
panels, one per available NAIP acquisition year, all stretched identically
using global percentiles from a reference image (most recent year by default).
The identical stretch preserves inter-date radiometric differences so that
atmospheric, phenological, and calibration variation is visible to the reader.

The spatial extent is the center 1/8th of the reference raster's area, chosen
to show representative forest interior free of edge effects.

Reusable: all key functions accept explicit parameters so this script can be
called for any site, any set of NAIP files, any zoom level.

Usage
-----
  # Default (SCBI, center 1/8th, all NAIP_SCBI_*.tif files):
  python 07_make_fig03_naip_consistency.py

  # Different zoom or scale bar:
  python 07_make_fig03_naip_consistency.py \\
      --area-fraction 0.25 --scale-bar-m 100 --ncols 3

  # Different output path:
  python 07_make_fig03_naip_consistency.py \\
      --out-path /path/to/custom_figure.png
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve()
POC = HERE.parents[2]   # ProofOfConcept/

DEFAULT_DATA_DIR = POC / "Data" / "NAIP" / "Imagery"
DEFAULT_OUT_PATH = POC / "Results" / "summary_document" / "figures" / "fig03_naip_consistency.png"


# ---------------------------------------------------------------------------
# importlib helper
# ---------------------------------------------------------------------------

def _import_script(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def compute_center_bbox(
    ref_path: Path,
    area_fraction: float = 0.125,
) -> tuple[float, float, float, float]:
    """
    Return (west, south, east, north) for the center sub-region of a raster.

    The returned region has `area_fraction` of the input raster's total area,
    centered on the raster's centroid.  For area_fraction=0.125 (1/8th), each
    linear dimension is sqrt(0.125) ≈ 35.4% of the original.

    Parameters
    ----------
    ref_path : Path
        Reference raster whose extent defines the crop geometry.
    area_fraction : float
        Fraction of the raster's area to retain (0 < area_fraction <= 1).

    Returns
    -------
    (west, south, east, north) in the raster's CRS.
    """
    frac = math.sqrt(area_fraction)  # linear fraction per axis
    with rasterio.open(ref_path) as src:
        h, w = src.height, src.width
        t = src.transform

    row_start = int(h * (1.0 - frac) / 2.0)
    row_end = h - row_start
    col_start = int(w * (1.0 - frac) / 2.0)
    col_end = w - col_start

    # Geographic coordinates (positive x east, t.e is negative for north-up)
    left = t.c + col_start * t.a
    right = t.c + col_end * t.a
    top = t.f + row_start * t.e      # t.e negative → top < t.f
    bottom = t.f + row_end * t.e

    west, east = min(left, right), max(left, right)
    south, north = min(top, bottom), max(top, bottom)

    logger.info(
        "Center bbox (fraction=%.3f): W=%.0f S=%.0f E=%.0f N=%.0f  (~%.0fx%.0f m)",
        area_fraction, west, south, east, north, east - west, north - south,
    )
    return west, south, east, north


# ---------------------------------------------------------------------------
# Stretch helpers
# ---------------------------------------------------------------------------

def compute_global_stretch(
    ref_path: Path,
    bands: tuple[int, ...] = (1, 2, 3),
    p_low: float = 2.0,
    p_high: float = 98.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> dict[int, tuple[float, float]]:
    """
    Compute per-band percentile stretch from a single reference image.

    If bbox is provided the stretch is computed only over the cropped area.
    Returns dict {band_index: (lo, hi)}.
    """
    stretch: dict[int, tuple[float, float]] = {}
    with rasterio.open(ref_path) as src:
        for b in bands:
            if bbox is not None:
                west, south, east, north = bbox
                geom = box(west, south, east, north)
                data, _ = rasterio_mask(src, [geom], crop=True, all_touched=True, indexes=[b])
                arr = data[0].astype(np.float32)
            else:
                arr = src.read(b).astype(np.float32)

            valid = arr[arr > 0]
            if valid.size == 0:
                stretch[b] = (0.0, 255.0)
            else:
                lo, hi = np.percentile(valid, [p_low, p_high])
                stretch[b] = (float(lo), float(hi))
            logger.debug("  Band %d stretch: [%.1f, %.1f]", b, *stretch[b])

    return stretch


def compute_pooled_stretch(
    paths: list[Path],
    bands: tuple[int, ...] = (1, 2, 3),
    p_low: float = 2.0,
    p_high: float = 98.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    sample_fraction: float = 0.05,
) -> dict[int, tuple[float, float]]:
    """
    Compute per-band percentile stretch pooled across all input images.

    Samples `sample_fraction` of valid pixels from each file to keep memory
    manageable, then computes a single (lo, hi) per band from the combined
    sample.  This produces a stretch that is representative of the full time
    series rather than any single year, avoiding the problem where one year's
    unusually narrow channel range dominates the display.

    Parameters
    ----------
    paths : list of Path
        All NAIP GeoTIFFs to include in the pooled sample.
    bands : tuple of int
        1-based band indices.
    p_low, p_high : float
        Percentile cutpoints (default 2–98).
    bbox : (west, south, east, north) or None
        Spatial crop applied before sampling.
    sample_fraction : float
        Fraction of valid pixels to keep per file (random subsample).

    Returns
    -------
    dict {band_index: (lo, hi)}
    """
    rng = np.random.default_rng(seed=42)
    pooled: dict[int, list[np.ndarray]] = {b: [] for b in bands}

    for path in paths:
        with rasterio.open(path) as src:
            for b in bands:
                if bbox is not None:
                    west, south, east, north = bbox
                    geom = box(west, south, east, north)
                    data, _ = rasterio_mask(src, [geom], crop=True, all_touched=True, indexes=[b])
                    arr = data[0].astype(np.float32)
                else:
                    arr = src.read(b).astype(np.float32)

                valid = arr[arr > 0]
                if valid.size == 0:
                    continue
                n_sample = max(1, int(valid.size * sample_fraction))
                idx = rng.choice(valid.size, size=n_sample, replace=False)
                pooled[b].append(valid[idx])

    stretch: dict[int, tuple[float, float]] = {}
    for b in bands:
        if not pooled[b]:
            stretch[b] = (0.0, 255.0)
        else:
            combined = np.concatenate(pooled[b])
            lo, hi = np.percentile(combined, [p_low, p_high])
            stretch[b] = (float(lo), float(hi))
        logger.info("  Band %d pooled stretch: [%.1f, %.1f]", b, *stretch[b])

    return stretch


def read_rgb_stretched(
    path: Path,
    stretch_params: dict[int, tuple[float, float]],
    bands: tuple[int, ...] = (1, 2, 3),
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Read RGB bands from a raster, clip to bbox, apply global stretch.

    Returns
    -------
    rgb : np.ndarray, shape (H, W, 3), float32 in [0, 1]
    profile : dict  — rasterio profile of the clipped area
    """
    with rasterio.open(path) as src:
        if bbox is not None:
            west, south, east, north = bbox
            geom = box(west, south, east, north)
            out_data, out_transform = rasterio_mask(
                src, [geom], crop=True, all_touched=True
            )
            profile = dict(src.profile)
            profile.update(
                height=out_data.shape[1],
                width=out_data.shape[2],
                transform=out_transform,
            )
            band_arrays = [out_data[i].astype(np.float32) for i in range(len(bands))]
        else:
            band_arrays = [src.read(b).astype(np.float32) for b in bands]
            profile = dict(src.profile)

    stretched = []
    for i, b in enumerate(bands):
        lo, hi = stretch_params[b]
        arr = np.clip((band_arrays[i] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        stretched.append(arr)

    return np.stack(stretched, axis=-1), profile


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_naip_consistency_figure(
    naip_files: list[Path],
    labels: list[str],
    stretch_params: dict[int, tuple[float, float]],
    bbox: tuple[float, float, float, float],
    out_path: Path,
    ncols: int = 4,
    scale_bar_m: float = 50.0,
    dpi: int = 300,
) -> None:
    """
    Generate a multi-panel NAIP time series figure.

    All panels share an identical RGB stretch, making inter-date radiometric
    differences directly visible.  Scale bar and north arrow appear on the
    first panel only.

    Parameters
    ----------
    naip_files : list of Path
        Per-year NAIP GeoTIFFs, chronologically sorted.
    labels : list of str
        Panel title for each file (typically the acquisition year).
    stretch_params : dict {band: (lo, hi)}
        Global stretch from compute_global_stretch().
    bbox : (west, south, east, north)
        Geographic crop extent (from compute_center_bbox).
    out_path : Path
        Output PNG path.
    ncols : int
        Panels per row.
    scale_bar_m : float
        Scale bar length in metres.
    dpi : int
        Output resolution.
    """
    n = len(naip_files)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    west, south, east, north = bbox
    extent = (west, east, south, north)  # matplotlib imshow extent order
    x_range = east - west
    y_range = north - south
    aspect = y_range / x_range  # height/width ratio

    panel_w = 3.2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_w, nrows * panel_w * aspect),
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.01)

    # Flatten axes for easy indexing; handle 1-row or 1-col edge cases
    ax_flat = np.array(axes).flatten() if hasattr(axes, "__len__") else np.array([axes])

    panel_labels = [chr(ord("a") + i) for i in range(n)]

    for i, (f, label, letter) in enumerate(zip(naip_files, labels, panel_labels)):
        ax = ax_flat[i]
        try:
            rgb, _ = read_rgb_stretched(f, stretch_params, bbox=bbox)
        except Exception as exc:
            logger.warning("Could not read %s: %s", f.name, exc)
            ax.axis("off")
            ax.text(0.5, 0.5, f"Error\n{label}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="red")
            continue

        ax.imshow(rgb, extent=extent, origin="upper", aspect="equal")
        ax.set_title(label, pad=3, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel letter label (upper left)
        ax.text(
            west + 0.02 * x_range, north - 0.02 * y_range,
            letter,
            fontsize=10, fontweight="bold", va="top", ha="left", color="white",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.6, lw=0),
            transform=ax.transData,
        )

        # Scale bar and north arrow on first panel only
        if i == 0:
            _add_scalebar(ax, extent, scale_bar_m)
            _add_north_arrow(ax, extent)

    # Turn off unused axes
    for j in range(n, len(ax_flat)):
        ax_flat[j].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Annotation helpers (standalone so this script has no hard dependency on
# 04_make_figures.py, while still matching the figure style)
# ---------------------------------------------------------------------------

def _add_scalebar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    length_m: float,
) -> None:
    """White scale bar in the lower-right corner."""
    left, right, bottom, top = extent
    x_range = right - left
    y_range = top - bottom
    label = f"{int(length_m)} m"
    x0 = right - 0.06 * x_range - length_m
    x1 = x0 + length_m
    y0 = bottom + 0.05 * y_range
    tick_h = 0.008 * y_range
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.0, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", lw=2)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="white", lw=2)
    ax.text((x0 + x1) / 2, y0 + 0.015 * y_range, label,
            ha="center", va="bottom", fontsize=6.5, color="white", fontweight="bold")


def _add_north_arrow(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    """Simple N arrow in the upper-right corner."""
    left, right, bottom, top = extent
    x_range = right - left
    y_range = top - bottom
    x = right - 0.05 * x_range
    y_base = top - 0.14 * y_range
    y_tip = y_base + 0.08 * y_range
    ax.annotate(
        "N",
        xy=(x, y_tip), xytext=(x, y_base),
        fontsize=7.5, color="white", fontweight="bold",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--data-dir", default=None, type=click.Path(exists=True),
    help="Directory containing NAIP_*.tif files.",
)
@click.option(
    "--site", default="SCBI", show_default=True,
    help="Filename prefix to match (e.g. 'SCBI' matches NAIP_SCBI_*.tif).",
)
@click.option(
    "--out-path", default=None, type=click.Path(),
    help=f"Output PNG path (default: {DEFAULT_OUT_PATH}).",
)
@click.option(
    "--stretch-mode",
    type=click.Choice(["single", "pooled"], case_sensitive=False),
    default="pooled", show_default=True,
    help=(
        "'single': stretch from one reference year (--stretch-ref). "
        "'pooled': stretch from all dates combined (fairer cross-date comparison)."
    ),
)
@click.option(
    "--stretch-ref", default=None,
    help="Year string to use as stretch reference (only used with --stretch-mode single; default: most recent).",
)
@click.option(
    "--area-fraction", default=0.125, show_default=True, type=float,
    help="Fraction of reference raster area to show (center crop).",
)
@click.option("--ncols", default=4, show_default=True, type=int)
@click.option("--scale-bar-m", default=50.0, show_default=True, type=float)
@click.option("--dpi", default=300, show_default=True, type=int)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    data_dir: Optional[str],
    site: str,
    out_path: Optional[str],
    stretch_mode: str,
    stretch_ref: Optional[str],
    area_fraction: float,
    ncols: int,
    scale_bar_m: float,
    dpi: int,
    verbose: bool,
) -> None:
    """Generate multi-temporal NAIP consistency figure (Figure 3)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    plt.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "image.interpolation": "none",
    })

    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    fig_out = Path(out_path) if out_path else DEFAULT_OUT_PATH

    # Discover NAIP files matching prefix, sorted by year
    pattern = f"NAIP_{site}_[0-9][0-9][0-9][0-9].tif"
    naip_files = sorted(data_path.glob(pattern))
    if not naip_files:
        logger.error(
            "No files matching '%s' found in %s.\n"
            "Run 06_download_naip_multitemporal.py first.",
            pattern, data_path,
        )
        raise SystemExit(1)

    labels = [f.stem.split("_")[-1] for f in naip_files]  # extract year from stem
    logger.info("Found %d NAIP files: %s", len(naip_files), labels)

    # Reference image for bbox computation (most recent by default)
    if stretch_ref is not None:
        ref_matches = [f for f in naip_files if stretch_ref in f.stem]
        if not ref_matches:
            logger.error("No file matching stretch-ref year '%s'", stretch_ref)
            raise SystemExit(1)
        ref_file = ref_matches[0]
    else:
        ref_file = naip_files[-1]
    logger.info("Spatial reference (for bbox): %s", ref_file.name)

    # Compute center bbox from reference raster
    bbox = compute_center_bbox(ref_file, area_fraction=area_fraction)

    # Compute stretch
    if stretch_mode == "pooled":
        logger.info("Computing pooled stretch from all %d files …", len(naip_files))
        stretch = compute_pooled_stretch(naip_files, bands=(1, 2, 3), p_low=2, p_high=98, bbox=bbox)
    else:
        logger.info("Computing single-reference stretch from %s …", ref_file.name)
        stretch = compute_global_stretch(ref_file, bands=(1, 2, 3), p_low=2, p_high=98, bbox=bbox)
    logger.info(
        "Stretch  R=[%.0f–%.0f]  G=[%.0f–%.0f]  B=[%.0f–%.0f]",
        *stretch[1], *stretch[2], *stretch[3],
    )

    # Generate figure
    logger.info("=== Generating Figure 3 ===")
    make_naip_consistency_figure(
        naip_files=naip_files,
        labels=labels,
        stretch_params=stretch,
        bbox=bbox,
        out_path=fig_out,
        ncols=ncols,
        scale_bar_m=scale_bar_m,
        dpi=dpi,
    )

    logger.info("=== Done ===")
    logger.info("Output: %s", fig_out)


if __name__ == "__main__":
    main()
