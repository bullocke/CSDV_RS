"""csdv_core.disturbance._pipeline_figures: figures of a new or saved run.

Private helpers of :mod:`csdv_core.disturbance.pipeline`:
:func:`write_run_figures` assembles the figure inputs of a run and writes the
PNGs, and :func:`rebuild_figures` redraws the figures and ``report.md`` of a
saved run folder.

Inputs: run arrays or a saved run folder. Outputs: PNGs under ``figures/``
and, for a rebuild, ``report.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance._pipeline_inputs import discover_with_saved_grid
from csdv_core.disturbance.decide import DateFit, skipped_fit
from csdv_core.disturbance.figures import (
    FigureInputs,
    NaipBackdrop,
    make_figures,
    read_naip_backdrop,
)
from csdv_core.disturbance.inputs import CellGrid, SiteInputs
from csdv_core.disturbance.layers import CellLayers, monitored_mask
from csdv_core.disturbance.report import (
    IntervalSummary,
    read_summary_json,
    refresh_summary,
    write_report_markdown,
)
from csdv_core.disturbance.store import (
    OBJECTS_LAYER,
    REFINED_OUTLINES_LAYER,
    RunPaths,
    load_run,
    read_detected_by_interval,
    read_year_stack,
)

logger = logging.getLogger(__name__)


def _naip_backdrop(
    site_inputs: SiteInputs | None, years: tuple[int, ...]
) -> NaipBackdrop | None:
    """Read the NAIP RGB backdrop of the last year, or None when unavailable.

    Args:
        site_inputs: Inputs discovered with NAIP files, or None.
        years: NAIP years of the run.

    Returns:
        The backdrop, or None (with a warning) when no NAIP file is known.
    """
    if site_inputs is None:
        logger.warning("No site inputs, the maps are drawn without a NAIP backdrop")
        return None
    last_year = years[-1]
    year_input = site_inputs.year_input(last_year)
    if not year_input.naip_paths:
        logger.warning("No NAIP files for %s, no NAIP backdrop", last_year)
        return None
    return read_naip_backdrop(year_input.naip_paths, site_inputs.grid)


def write_run_figures(
    *,
    params: DisturbanceConfig,
    run_paths: RunPaths,
    grid: CellGrid,
    layers: CellLayers,
    z: np.ndarray,
    probability: np.ndarray | None,
    onset_year: np.ndarray,
    detected: np.ndarray,
    fits: list[DateFit],
    intervals: list[IntervalSummary],
    objects: gpd.GeoDataFrame,
    refined: gpd.GeoDataFrame | None,
    site_inputs: SiteInputs | None,
    naip_backdrop: bool,
) -> dict[str, Path]:
    """Assemble :class:`FigureInputs` and write the figures.

    Args:
        params: Run parameters (``decide`` and ``figures`` sections).
        run_paths: Paths of the run.
        grid: Common grid.
        layers: Cell layers.
        z: Quantized z.
        probability: Decision probability, or None.
        onset_year: ``onset_year`` raster after the MMU.
        detected: bool ``(T, rows, cols)`` detections per interval.
        fits: Null fit per date.
        intervals: Summary per interval.
        objects: Objects GeoDataFrame.
        refined: Refined outlines, or None.
        site_inputs: Site inputs, or None.
        naip_backdrop: Draw the NAIP RGB backdrop.

    Returns:
        Figure name to written path.
    """
    years = layers.years
    monitored = monitored_mask(layers.forest)
    backdrop = None
    if naip_backdrop:
        backdrop = _naip_backdrop(site_inputs, years)
    figure_inputs = FigureInputs(
        grid=grid,
        years=years,
        forest=layers.forest,
        monitored=monitored,
        z=z,
        probability=probability,
        onset_year=onset_year,
        fits=fits,
        intervals=intervals,
        objects=objects,
        refined=refined,
        detected=detected,
        p_boundary=params.decide.p_boundary,
        bin_w=params.decide.bin_w,
        site_inputs=site_inputs,
        backdrop=backdrop,
        chm_height=layers.chm_height,
    )
    return make_figures(figure_inputs, run_paths.figures_dir, dpi=params.figures.dpi)


def _read_single_band(path: Path) -> np.ndarray:
    """Read band 1 of a single-band raster such as ``onset_year.tif``.

    Args:
        path: GeoTIFF.

    Returns:
        The band as stored.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as dataset:
        return dataset.read(1)


def _read_refined(gpkg_path: Path) -> gpd.GeoDataFrame | None:
    """Read the refined outlines layer of a run's GeoPackage, if present.

    Args:
        gpkg_path: ``detection/disturbances.gpkg``.

    Returns:
        The refined outlines, or None when the layer does not exist.
    """
    layer_table = gpd.list_layers(gpkg_path)
    layer_names = set(layer_table["name"])
    if REFINED_OUTLINES_LAYER not in layer_names:
        return None
    return gpd.read_file(gpkg_path, layer=REFINED_OUTLINES_LAYER)


def _rediscover_inputs(
    site: str,
    params: DisturbanceConfig,
    grid: CellGrid,
    *,
    need_refined: bool,
    need_naip: bool,
) -> SiteInputs | None:
    """Rediscover a saved run's source files for figures, when needed.

    Args:
        site: Site name.
        params: Saved run parameters.
        grid: Saved grid.
        need_refined: The refined examples figure needs the CHMs.
        need_naip: The NAIP backdrop is requested.

    Returns:
        Site inputs on the saved grid, or None when they are not needed or the
        files are missing (logged as a warning).
    """
    if not need_refined and not need_naip:
        return None
    try:
        return discover_with_saved_grid(site, params, grid, need_naip=need_naip)
    except (FileNotFoundError, ValueError) as error:
        logger.warning(
            "Source files unavailable (%s), refined examples and the NAIP "
            "backdrop are skipped",
            error,
        )
        return None


def rebuild_figures(
    run_dir: Path, *, naip_backdrop: bool | None = None
) -> dict[str, Path]:
    """Rebuild ``figures/`` and ``report.md`` of an existing run in place.

    Reloads the run (:func:`csdv_core.disturbance.store.load_run`), the
    detection rasters, the GeoPackage and ``summary.json``. Site inputs are
    rediscovered for the refined examples and the NAIP backdrop when the data
    are available, otherwise those parts are skipped with a warning. Existing
    PNGs are overwritten (``figures/`` is derived output, unlike the run
    folder itself). ``summary.json`` is not changed.

    Args:
        run_dir: Run folder.
        naip_backdrop: Draw NAIP RGB under the maps. None uses the saved
            run's ``figures.naip_backdrop``.

    Returns:
        Figure name to written path.

    Raises:
        FileNotFoundError: If the run folder or its files are missing.
    """
    saved = load_run(run_dir)
    run_paths = saved.run_paths
    params = saved.params
    use_backdrop = params.figures.naip_backdrop
    if naip_backdrop is not None:
        use_backdrop = naip_backdrop
    summary = read_summary_json(run_paths.summary_json)
    onset_year = _read_single_band(run_paths.onset_year)
    detected = read_detected_by_interval(run_paths, saved.years)
    probability = None
    if run_paths.probability.is_file():
        probability_stack = read_year_stack(run_paths.probability)
        probability = probability_stack.values
    objects = gpd.read_file(run_paths.disturbances_gpkg, layer=OBJECTS_LAYER)
    refined = _read_refined(run_paths.disturbances_gpkg)
    need_refined = refined is not None and not refined.empty
    site_inputs = _rediscover_inputs(
        saved.site,
        params,
        saved.grid,
        need_refined=need_refined,
        need_naip=use_backdrop,
    )
    interval_fits = [interval.fit for interval in summary.intervals]
    first_fit = skipped_fit(saved.years[0], n_cells=0)
    all_fits = [first_fit, *interval_fits]
    figure_paths = write_run_figures(
        params=params,
        run_paths=run_paths,
        grid=saved.grid,
        layers=saved.layers,
        z=saved.z,
        probability=probability,
        onset_year=onset_year,
        detected=detected,
        fits=all_fits,
        intervals=summary.intervals,
        objects=objects,
        refined=refined,
        site_inputs=site_inputs,
        naip_backdrop=use_backdrop,
    )
    refreshed = refresh_summary(summary, run_paths)
    write_report_markdown(refreshed, run_paths.report_md, figure_paths=figure_paths)
    return figure_paths


__all__ = ["rebuild_figures", "write_run_figures"]
