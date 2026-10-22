"""csdv_core.disturbance.figures: PNG figures of a disturbance run (Agg backend).

Each figure has a ``plot_*`` function that takes explicit arrays and
``out_path`` and returns the written path. Single-panel maps and charts also
have a ``draw_*(ax, ...)`` function so they compose into other figures. The
drawing code lives in three private modules, ``_figure_maps`` (scene maps),
``_figure_charts`` (null fits, area bars, refined examples) and
``_figure_helpers`` (layout, scale bar, outlines, shared constants), and is
re-exported here. Style and saving come from :mod:`csdv_core.viz.style`.

Inputs: cell rasters ``(rows, cols)`` or year stacks ``(T, rows, cols)``,
fits, summaries, GeoDataFrames. Outputs: PNG files under ``figures/``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from rasterio.merge import merge

from csdv_core.disturbance import _figure_helpers as helpers
from csdv_core.disturbance._figure_charts import (
    DEFAULT_N_EXAMPLES,
    draw_area_by_interval,
    draw_null_fit,
    plot_area_by_interval,
    plot_null_fits,
    plot_refined_examples,
)
from csdv_core.disturbance._figure_helpers import DEFAULT_DPI, NaipBackdrop
from csdv_core.disturbance._figure_maps import (
    Z_COLOR_LIMIT,
    draw_forest,
    draw_onset,
    plot_disturbance_onset,
    plot_forest_by_year,
    plot_forest_initial,
    plot_onset_by_interval,
    plot_probability_maps,
    plot_z_maps,
)
from csdv_core.disturbance.decide import DateFit
from csdv_core.disturbance.inputs import CellGrid, SiteInputs
from csdv_core.disturbance.report import IntervalSummary
from csdv_core.viz.maps import stretch_rgb
from csdv_core.viz.style import setup_style

logger = logging.getLogger(__name__)

#: Longest side, in pixels, of a NAIP backdrop read for a map.
BACKDROP_MAX_PX = 2000
#: Percentile of cell CHM height drawn as the lightest grey on the CHM backdrop.
CHM_BACKDROP_PERCENTILE = 99.0
#: Figure name to file name inside ``figures/``.
FIGURE_FILES: dict[str, str] = {
    "forest_initial": "forest_initial.png",
    "forest_by_year": "forest_by_year.png",
    "disturbance_onset": "disturbance_onset.png",
    "onset_by_interval": "onset_by_interval.png",
    "z_maps": "z_maps.png",
    "probability_maps": "probability_maps.png",
    "null_fits": "null_fits.png",
    "area_by_interval": "area_by_interval.png",
    "refined_examples": "refined_examples.png",
}


@dataclass
class FigureInputs:
    """Everything :func:`make_figures` draws from.

    Attributes:
        grid: Common grid.
        years: NAIP years, length ``T``.
        forest: bool ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        z: float32 ``(T, rows, cols)`` quantized z.
        probability: float32 ``(T, rows, cols)`` or None (rule D1).
        onset_year: int16 ``(rows, cols)``, first detection after the MMU
            (0 none, -1 unmonitored).
        fits: One :class:`DateFit` per date.
        intervals: One :class:`IntervalSummary` per interval.
        objects: Objects GeoDataFrame.
        refined: Refined outlines GeoDataFrame, or None.
        p_boundary: Decision boundary (marked on probability maps and
            null-fit panels).
        bin_w: Histogram bin width for null-fit panels.
        site_inputs: Site inputs for CHM panels of refined examples, or None
            (the refined examples figure is then skipped).
        backdrop: NAIP backdrop, or None.
        chm_height: float32 ``(T, rows, cols)`` mean CHM height per cell, or
            None. Without a NAIP backdrop, the onset maps draw the first
            year's height in grey underneath (see :func:`chm_height_backdrop`).
        detected: bool ``(T, rows, cols)`` detections per interval, or None.
            The per-interval map then shows each interval's detections,
            otherwise the cells whose first detection falls in it.
    """

    grid: CellGrid
    years: tuple[int, ...]
    forest: np.ndarray
    monitored: np.ndarray
    z: np.ndarray
    probability: np.ndarray | None
    onset_year: np.ndarray
    fits: list[DateFit]
    intervals: list[IntervalSummary]
    objects: gpd.GeoDataFrame
    refined: gpd.GeoDataFrame | None
    p_boundary: float
    bin_w: float
    site_inputs: SiteInputs | None = None
    backdrop: NaipBackdrop | None = None
    chm_height: np.ndarray | None = None
    detected: np.ndarray | None = None


def read_naip_backdrop(
    naip_paths: Sequence[Path],
    grid: CellGrid,
    *,
    bands: tuple[int, int, int] = (1, 2, 3),
    max_px: int = BACKDROP_MAX_PX,
) -> NaipBackdrop:
    """Read and stretch a NAIP RGB backdrop over the grid extent.

    The files are mosaicked with :func:`rasterio.merge.merge` at a resolution
    that keeps the longest side at or below ``max_px``, so a large site is
    decimated on read.

    Args:
        naip_paths: NAIP GeoTIFFs of one year (grid CRS).
        grid: Common grid.
        bands: 1-based red, green and blue band numbers.
        max_px: Longest side of the backdrop in pixels.

    Returns:
        The stretched backdrop and its extent.

    Raises:
        FileNotFoundError: If ``naip_paths`` is empty.
    """
    if not naip_paths:
        raise FileNotFoundError("No NAIP files given for the map backdrop")
    pixel_rows, pixel_cols = grid.pixel_shape
    longest_side_px = max(pixel_rows, pixel_cols)
    decimation = max(1.0, longest_side_px / max_px)
    resolution_m = grid.pixel_m * decimation
    band_list = list(bands)
    path_list = list(naip_paths)
    mosaic, transform = merge(
        path_list, bounds=grid.bounds, res=resolution_m, indexes=band_list
    )
    rgb_stack = np.moveaxis(mosaic, 0, -1)
    stretched = stretch_rgb(rgb_stack)
    rgb_scaled = np.round(stretched * 255.0)
    rgb_uint8 = rgb_scaled.astype(np.uint8)
    extent = helpers.transform_extent(transform, rgb_uint8.shape)
    logger.debug("NAIP backdrop %s at %.2f m", rgb_uint8.shape, resolution_m)
    return NaipBackdrop(rgb=rgb_uint8, extent=extent)


def chm_height_backdrop(chm_height: np.ndarray, grid: CellGrid) -> NaipBackdrop:
    """Turn one year of cell CHM height into a grey map backdrop.

    Heights are scaled from 0 m to the :data:`CHM_BACKDROP_PERCENTILE`
    percentile of the finite cells, so taller canopy is lighter. Cells without
    a height are drawn white. The result is a grey RGB image so it goes through
    the same drawing path as a NAIP backdrop.

    Args:
        chm_height: float32 ``(rows, cols)`` mean CHM height (m), NaN allowed.
        grid: Common grid (sets the extent).

    Returns:
        The grey backdrop on the cell grid.
    """
    finite_heights = chm_height[np.isfinite(chm_height)]
    top_height = 1.0
    if finite_heights.size:
        percentile_height = np.percentile(finite_heights, CHM_BACKDROP_PERCENTILE)
        top_height = max(float(percentile_height), 1.0)
    scaled = chm_height / top_height
    clipped = np.clip(scaled, 0.0, 1.0)
    filled = np.where(np.isfinite(clipped), clipped, 1.0)
    gray_values = np.round(filled * 255.0)
    gray_uint8 = gray_values.astype(np.uint8)
    rgb_uint8 = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
    extent = grid.extent()
    return NaipBackdrop(rgb=rgb_uint8, extent=extent)


def onset_backdrop(inputs: FigureInputs) -> NaipBackdrop | None:
    """Backdrop of the onset maps: NAIP when given, else grey CHM height.

    Args:
        inputs: Figure inputs.

    Returns:
        ``inputs.backdrop`` when set, otherwise the grey CHM height of the
        first year (the baseline of the first interval) when
        ``inputs.chm_height`` is set, otherwise None.
    """
    if inputs.backdrop is not None:
        return inputs.backdrop
    if inputs.chm_height is None:
        return None
    first_year_height = inputs.chm_height[0]
    return chm_height_backdrop(first_year_height, inputs.grid)


def make_figures(
    inputs: FigureInputs, figures_dir: Path, *, dpi: int = DEFAULT_DPI
) -> dict[str, Path]:
    """Write every figure of :data:`FIGURE_FILES` that the inputs allow.

    ``probability_maps`` is skipped when ``inputs.probability`` is None and
    ``refined_examples`` when ``inputs.refined`` or ``inputs.site_inputs`` is
    None or no object was refined. A figure that fails logs the error and is
    left out instead of stopping the run.

    Args:
        inputs: Arrays, fits, summaries and vectors of the run.
        figures_dir: Output folder (created when missing).
        dpi: Output resolution.

    Returns:
        Figure name to written path, for the figures that were written.
    """
    setup_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    years = inputs.years
    grid = inputs.grid
    onset_map_backdrop = onset_backdrop(inputs)
    jobs: dict[str, Callable[[Path], Path]] = {
        "forest_initial": lambda path: plot_forest_initial(
            inputs.forest[0], grid, years[0], path, backdrop=inputs.backdrop, dpi=dpi
        ),
        "forest_by_year": lambda path: plot_forest_by_year(
            inputs.forest, grid, years, path, dpi=dpi
        ),
        "disturbance_onset": lambda path: plot_disturbance_onset(
            inputs.onset_year, years, grid, path, backdrop=onset_map_backdrop, dpi=dpi
        ),
        "onset_by_interval": lambda path: plot_onset_by_interval(
            inputs.onset_year,
            years,
            grid,
            path,
            detected=inputs.detected,
            backdrop=onset_map_backdrop,
            dpi=dpi,
        ),
        "z_maps": lambda path: plot_z_maps(
            inputs.z, years, grid, path, monitored=inputs.monitored, dpi=dpi
        ),
        "null_fits": lambda path: plot_null_fits(
            inputs.z,
            inputs.monitored,
            inputs.fits,
            path,
            bin_w=inputs.bin_w,
            p_boundary=inputs.p_boundary,
            dpi=dpi,
        ),
        "area_by_interval": lambda path: plot_area_by_interval(
            inputs.intervals, path, dpi=dpi
        ),
    }
    if inputs.probability is not None:
        jobs["probability_maps"] = lambda path: plot_probability_maps(
            inputs.probability, years, grid, path, p_boundary=inputs.p_boundary, dpi=dpi
        )
    else:
        logger.info("Skipping probability_maps: the rule has no probability")
    refined_ready = (
        inputs.refined is not None
        and inputs.site_inputs is not None
        and not inputs.refined.empty
    )
    if refined_ready:
        jobs["refined_examples"] = lambda path: plot_refined_examples(
            inputs.objects, inputs.refined, inputs.site_inputs, path, dpi=dpi
        )
    else:
        logger.info("Skipping refined_examples: no refined outlines or site inputs")
    written: dict[str, Path] = {}
    for name, file_name in FIGURE_FILES.items():
        job = jobs.get(name)
        if job is None:
            continue
        out_path = figures_dir / file_name
        try:
            written[name] = job(out_path)
        except Exception:
            logger.exception("Figure %s failed and is left out", name)
            plt.close("all")
    return written


__all__ = [
    "BACKDROP_MAX_PX",
    "CHM_BACKDROP_PERCENTILE",
    "DEFAULT_DPI",
    "DEFAULT_N_EXAMPLES",
    "FIGURE_FILES",
    "Z_COLOR_LIMIT",
    "FigureInputs",
    "NaipBackdrop",
    "chm_height_backdrop",
    "draw_area_by_interval",
    "draw_forest",
    "draw_null_fit",
    "draw_onset",
    "make_figures",
    "onset_backdrop",
    "plot_area_by_interval",
    "plot_disturbance_onset",
    "plot_forest_by_year",
    "plot_forest_initial",
    "plot_null_fits",
    "plot_onset_by_interval",
    "plot_probability_maps",
    "plot_refined_examples",
    "plot_z_maps",
    "read_naip_backdrop",
]
