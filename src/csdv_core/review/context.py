"""csdv_core.review.context: what the review map and year strip show around a polygon.

- :func:`nearby_objects` reads every detected object within a radius of the
  focus polygon from the run's ``disturbances.gpkg`` (bbox filter, so only a
  few rows are read), with the refined outline where there is one.
- :func:`onset_colours` gives each onset year the colour of the exported onset
  raster (:func:`csdv_core.grid.export.onset_colormap`).
- :func:`year_strip` renders a small strip of every year from the 3 m NAIP and
  CHM quicklooks around the polygon, with its neighbours, and the mean canopy
  height inside the polygon and in a ring around it.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from csdv_core.grid.export import CHM_STEP_M, ONSET_YEAR_BASE
from csdv_core.review.sample import OBJECTS_LAYER, REFINED_LAYER, ExportedRun

logger = logging.getLogger(__name__)

#: Colour of the polygon under review, on the map and in the strip.
FOCUS_COLOUR = "#00e5ff"
#: Radius around the focus polygon from which neighbours are read, in metres.
CONTEXT_RADIUS_M = 1500.0
#: Background of strip panels outside the quicklook (beyond the unit).
OUTSIDE_COLOUR = "#d9d9d9"
#: Width of the ring around the polygon used as the height reference.
RING_M = 30.0

__all__ = [
    "FOCUS_COLOUR",
    "chm_series",
    "nearby_objects",
    "onset_colours",
    "year_strip",
]


def onset_colours(years: list[int]) -> dict[int, str]:
    """Hex colour of each onset year, as in the exported onset raster."""
    from csdv_core.grid.export import onset_colormap

    table = onset_colormap([y - ONSET_YEAR_BASE for y in years[1:]])
    return {
        y: "#{:02x}{:02x}{:02x}".format(*table[y - ONSET_YEAR_BASE][:3])
        for y in years[1:]
    }


def nearby_objects(
    run: ExportedRun,
    geometry_utm: BaseGeometry,
    radius_m: float = CONTEXT_RADIUS_M,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Detected objects near a polygon, in the run's UTM CRS.

    Returns:
        The outlines (refined where available, else the 19.2 m object, with
        ``geom_src``) and the 19.2 m objects themselves.
    """
    bbox = geometry_utm.buffer(radius_m).bounds
    cells = gpd.read_file(run.gpkg, layer=OBJECTS_LAYER, bbox=bbox)
    outlines = cells.copy()
    outlines["geom_src"] = "object"
    try:
        refined = gpd.read_file(run.gpkg, layer=REFINED_LAYER, bbox=bbox)
    except Exception:  # older runs have no refined layer
        refined = gpd.GeoDataFrame()
    if len(refined):
        refined = refined[refined.geometry.notna() & ~refined.geometry.is_empty]
        geom = refined.set_index("object_id").geometry
        hit = outlines["object_id"].isin(geom.index)
        outlines.loc[hit, "geometry"] = geom.loc[outlines.loc[hit, "object_id"]].values
        outlines.loc[hit, "geom_src"] = "refined"
    return outlines, cells


def _read_window(
    path: Path, bounds: tuple[float, ...]
) -> tuple[np.ma.MaskedArray, tuple]:
    """A window of a raster, masked outside the raster and where it has no data."""
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        data = src.read(window=win, boundless=True, masked=True)
        transform = src.window_transform(win)
    return data, transform


def chm_series(
    quicklook_dir: Path, years: list[int], geometry_utm: BaseGeometry
) -> pd.DataFrame:
    """Mean CHM height inside the polygon and in a ring around it, per year."""
    ring = geometry_utm.buffer(RING_M).difference(geometry_utm)
    bounds = geometry_utm.buffer(RING_M + 6).bounds
    rows = []
    for year in years:
        path = quicklook_dir / f"chm_{year}_3m.tif"
        if not path.exists():
            continue
        data, transform = _read_window(path, bounds)
        band = data[0]
        valid = ~np.ma.getmaskarray(band)
        heights = band.filled(0).astype("float32") * CHM_STEP_M
        values = {}
        for name, geom in (("inside", geometry_utm), ("ring", ring)):
            inside = ~geometry_mask(
                [geom], out_shape=band.shape, transform=transform, all_touched=False
            )
            sel = inside & valid
            values[name] = float(heights[sel].mean()) if sel.any() else np.nan
        rows.append({"year": year, **values})
    return pd.DataFrame(rows)


def year_strip(
    run: ExportedRun,
    quicklook_dir: Path,
    focus_utm: BaseGeometry,
    neighbours: gpd.GeoDataFrame,
    year_before: int,
    onset_year: int,
    *,
    pad_m: float = 60.0,
    min_side_m: float = 300.0,
) -> bytes:
    """PNG strip of every year: NAIP 3 m, CHM 3 m and the height series.

    Args:
        run: The run.
        quicklook_dir: ``quicklooks/<unit>/`` of the export.
        focus_utm: The polygon under review, in the run's CRS.
        neighbours: Outlines near it (from :func:`nearby_objects`).
        year_before: First image of the detected interval (blue frame).
        onset_year: Second image of the detected interval (red frame).
        pad_m: Margin around the polygon.
        min_side_m: Smallest side of a panel.
    """
    from csdv_core.disturbance import _figure_helpers as helpers
    from csdv_core.disturbance.chips import (
        AFTER_FRAME_COLOR,
        BEFORE_FRAME_COLOR,
        ChipSettings,
        chip_bounds,
    )
    from csdv_core.viz.maps import stretch_rgb
    from csdv_core.viz.relief import canopy_cmap

    years = run.years
    bounds = chip_bounds(focus_utm, ChipSettings(pad_m=pad_m, min_chip_m=min_side_m))
    frame = box(*bounds)
    near = neighbours[neighbours.intersects(frame)]
    colours = onset_colours(years)
    series = chm_series(quicklook_dir, years, focus_utm)

    n = len(years)
    fig = plt.figure(figsize=(1.5 * n, 4.3), dpi=100)
    grid = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.75], hspace=0.08, wspace=0.04)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    for col, year in enumerate(years):
        for row, kind in enumerate(("naip", "chm")):
            ax = fig.add_subplot(grid[row, col])
            name = f"naip_{year}_3m_rgb" if kind == "naip" else f"chm_{year}_3m"
            path = quicklook_dir / f"{name}.tif"
            if path.exists():
                data, _ = _read_window(path, bounds)
                ax.set_facecolor(OUTSIDE_COLOUR)
                if kind == "naip":
                    outside = np.ma.getmaskarray(data[:3]).all(axis=0)
                    rgb = stretch_rgb(np.moveaxis(data[:3].filled(0), 0, -1))
                    rgba = np.dstack([rgb, (~outside).astype(rgb.dtype)])
                    ax.imshow(rgba, extent=extent)
                else:
                    h = data[0].astype("float32") * CHM_STEP_M
                    ax.imshow(h, extent=extent, cmap=canopy_cmap(), vmin=0, vmax=30)
            for onset, part in near.groupby("onset_year"):
                helpers.outline(
                    ax,
                    part.geometry,
                    colours.get(int(onset), "#ffffff"),
                    linewidth=0.7,
                    halo=None,
                )
            helpers.outline(ax, [focus_utm], FOCUS_COLOUR, linewidth=1.3, halo=None)
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_xticks([])
            ax.set_yticks([])
            colour = {year_before: BEFORE_FRAME_COLOR, onset_year: AFTER_FRAME_COLOR}
            edge = colour.get(year)
            for spine in ax.spines.values():
                spine.set_edgecolor(edge or "#999999")
                spine.set_linewidth(2.5 if edge else 0.5)
            if row == 0:
                ax.set_title(str(year), fontsize=9, pad=2)
    ax = fig.add_subplot(grid[2, :])
    if len(series):
        ax.plot(
            series["year"],
            series["inside"],
            "-o",
            color="#0b7a8a",
            ms=3,
            label="inside polygon",
        )
        ax.plot(
            series["year"],
            series["ring"],
            "--",
            color="#888888",
            lw=1,
            label=f"{RING_M:.0f} m ring",
        )
    ax.axvspan(year_before, onset_year, color="#d62728", alpha=0.12, lw=0)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(years)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("CHM (m)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    ax.grid(alpha=0.3, lw=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()
