"""Multi-date stage and metric cube readers.

Phase 5 reads per-year stage rasters (Phase 4 outputs) and optionally the
metric stacks (Phase 2-3 outputs), stacking them into ``(T, H, W)`` cubes
for trajectory classification. Pure I/O. Grid (transform + CRS + shape)
must agree across years; mismatches raise ``ValueError``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from csdv_core.io.grids import GridSpec
from csdv_core.io.metrics_io import read_metric_stack
from csdv_core.io.paths import ProjectPaths
from csdv_core.io.raster import read_band

logger = logging.getLogger(__name__)


def read_stage_cube(
    paths: ProjectPaths,
    site: str,
    years: Sequence[int],
    window_m: int | float,
) -> tuple[np.ndarray, GridSpec]:
    """Stack per-year ``stage.tif`` rasters into a ``(T, H, W)`` cube.

    Args:
        paths: Resolved project paths.
        site: Site code.
        years: Ordered list of NAIP years (length T >= 2).
        window_m: Analysis window size in meters; selects the year/window
            subdirectory under ``stages/<site>/<year>/<window_m>m/``.

    Returns:
        ``(cube, grid)`` where ``cube`` is uint8 of shape ``(T, H, W)`` with
        0 = unclassified, and ``grid`` is the shared :class:`GridSpec`.

    Raises:
        ValueError: If fewer than two years are provided or any year's
            grid disagrees with the first year's.
        FileNotFoundError: If any year's ``stage.tif`` is missing.
    """
    years = list(years)
    if len(years) < 2:
        raise ValueError(f"need at least 2 years, got {years!r}")

    arrays: list[np.ndarray] = []
    grid: GridSpec | None = None
    ref_shape: tuple[int, int] | None = None
    for year in years:
        path = paths.stages_dir(site, year, window_m) / "stage.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing stage raster: {path}")
        rr = read_band(path, nodata_to_nan=False)
        arr = rr.data.astype("uint8", copy=False)
        crs_str = str(rr.crs) if rr.crs is not None else ""
        if grid is None:
            grid = GridSpec(
                transform=rr.transform,
                crs=crs_str,
                pixel_size_m=float(abs(rr.transform.a)),
            )
            ref_shape = arr.shape
        else:
            if rr.transform != grid.transform or crs_str != grid.crs:
                raise ValueError(
                    f"Stage raster {path} disagrees with cube grid: "
                    f"transform/crs mismatch."
                )
            if arr.shape != ref_shape:
                raise ValueError(
                    f"Stage raster {path} shape {arr.shape} != {ref_shape}"
                )
        arrays.append(arr)

    assert grid is not None
    cube = np.stack(arrays, axis=0).astype("uint8", copy=False)
    logger.info(
        "read_stage_cube: site=%s years=%s window=%sm shape=%s",
        site,
        years,
        window_m,
        cube.shape,
    )
    return cube, grid


def read_metric_cube(
    paths: ProjectPaths,
    site: str,
    years: Sequence[int],
    window_m: int | float,
    metrics: Sequence[str],
) -> tuple[dict[str, np.ndarray], GridSpec]:
    """Stack per-year metric rasters into a ``{name: (T, H, W)}`` mapping.

    Args:
        paths: Resolved project paths.
        site: Site code.
        years: Ordered list of NAIP years (length T >= 2).
        window_m: Analysis window size in meters.
        metrics: Metric names to load. Each must exist for every year.

    Returns:
        ``(cubes, grid)`` where ``cubes[name]`` is float32 ``(T, H, W)`` with
        NaN where source nodata, and ``grid`` is the shared :class:`GridSpec`.

    Raises:
        ValueError: If years count < 2 or per-year grids disagree.
        FileNotFoundError: If any required metric is missing for any year.
    """
    years = list(years)
    metric_names = list(metrics)
    if len(years) < 2:
        raise ValueError(f"need at least 2 years, got {years!r}")
    if not metric_names:
        return {}, _grid_from_first_metric_year(paths, site, years[0], window_m)

    grid: GridSpec | None = None
    per_year: list[dict[str, np.ndarray]] = []
    for year in years:
        mdir = paths.metrics_dir(site, year, window_m)
        if not mdir.exists():
            raise FileNotFoundError(f"Metrics directory missing: {mdir}")
        arrays, year_grid = read_metric_stack(mdir, metric_names)
        if grid is None:
            grid = year_grid
        elif (
            year_grid.transform != grid.transform
            or year_grid.crs != grid.crs
            or next(iter(arrays.values())).shape
            != next(iter(per_year[0].values())).shape
        ):
            raise ValueError(
                f"Metric grid for year {year} ({mdir}) disagrees with first year."
            )
        per_year.append(arrays)

    cubes: dict[str, np.ndarray] = {}
    for name in metric_names:
        cubes[name] = np.stack([per_year[t][name] for t in range(len(years))], axis=0)

    assert grid is not None
    logger.info(
        "read_metric_cube: site=%s years=%s metrics=%s window=%sm",
        site,
        years,
        metric_names,
        window_m,
    )
    return cubes, grid


def _grid_from_first_metric_year(
    paths: ProjectPaths,
    site: str,
    year: int,
    window_m: int | float,
) -> GridSpec:
    """Resolve a GridSpec when no metrics are requested but a grid is needed."""
    mdir = paths.metrics_dir(site, year, window_m)
    if not mdir.exists():
        raise FileNotFoundError(f"Metrics directory missing: {mdir}")
    _, grid = read_metric_stack(mdir)
    return grid
