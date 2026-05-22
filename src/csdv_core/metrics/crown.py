"""csdv_core.metrics.crown - Per-window statistics over crown diameters.

Pure-function port of ``poc_lib/crowns.py::crown_stats_per_window``. The
caller supplies a ``GeoDataFrame`` of crown polygons (or points) with a
``crown_diam_m`` column; no file I/O happens here.

Supported statistics: ``cv, p90, mean, median, count, std``.
Windows with fewer than ``min_crowns`` crowns return NaN (except ``count``,
which always returns the raw count including zeros and 1-2).
"""

from __future__ import annotations

import logging
from typing import Literal

import geopandas as gpd
import numpy as np

from csdv_core.io.grids import GridSpec
from csdv_core.metrics._result import MetricResult, make_result
from csdv_core.metrics._window import (
    assign_points_to_cells,
    window_pixels,
    window_transform,
)
from csdv_core.metrics.registry import register

logger = logging.getLogger(__name__)

CrownStat = Literal["cv", "p90", "mean", "median", "count", "std"]
_VALID_STATS: tuple[CrownStat, ...] = ("cv", "p90", "mean", "median", "count", "std")


def crown_stats(
    crowns: gpd.GeoDataFrame,
    grid: GridSpec,
    bounds: tuple[float, float, float, float],
    *,
    window_m: float = 50.0,
    stat: CrownStat = "cv",
    diam_column: str = "crown_diam_m",
    min_crowns: int = 3,
) -> MetricResult:
    """Compute a per-window statistic over crown diameters.

    Args:
        crowns: GeoDataFrame with a ``diam_column`` column. Geometries may be
            polygons or points; centroids are used for cell assignment.
        grid: :class:`GridSpec` for the reference grid.
        bounds: ``(xmin, ymin, xmax, ymax)`` of the reference grid in ``grid.crs``.
        window_m: Window side length in meters. Default 50.
        stat: One of ``cv, p90, mean, median, count, std``.
        diam_column: Column holding crown diameter in meters.
        min_crowns: Minimum crowns per window required for non-count stats.

    Returns:
        :class:`MetricResult` array sized by floor-divided bounds / window.

    Raises:
        ValueError: If ``stat`` is not supported or ``diam_column`` is missing.
    """
    if stat not in _VALID_STATS:
        raise ValueError(f"stat must be one of {_VALID_STATS}, got {stat!r}")
    if diam_column not in crowns.columns:
        raise ValueError(f"crowns is missing required column {diam_column!r}")

    xmin, ymin, xmax, ymax = bounds
    window_px = window_pixels(window_m, grid.pixel_size_m)
    # Match legacy floor-division behavior.
    n_cols = int((xmax - xmin) / grid.pixel_size_m) // window_px
    n_rows = int((ymax - ymin) / grid.pixel_size_m) // window_px
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    if n_rows == 0 or n_cols == 0:
        logger.warning("crown_stats: empty output grid (%dx%d)", n_rows, n_cols)
        return make_result(
            f"crown_{stat}",
            out,
            transform=window_transform(grid.transform, window_m),
            crs=grid.crs,
            window_m=window_m,
            params={"stat": stat, "min_crowns": min_crowns},
            units="m" if stat in ("p90", "mean", "median", "std") else "",
        )

    if len(crowns) == 0:
        logger.warning("crown_stats: empty GeoDataFrame")
        return make_result(
            f"crown_{stat}",
            out,
            transform=window_transform(grid.transform, window_m),
            crs=grid.crs,
            window_m=window_m,
            params={"stat": stat, "min_crowns": min_crowns},
            units="m" if stat in ("p90", "mean", "median", "std") else "",
        )

    centroids = crowns.geometry.centroid
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()
    diams = crowns[diam_column].to_numpy(dtype=np.float64)
    row_idx, col_idx = assign_points_to_cells(
        xs,
        ys,
        origin_x=xmin,
        origin_y=ymax,
        window_m=window_m,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    keep = (row_idx >= 0) & (col_idx >= 0)
    row_idx = row_idx[keep]
    col_idx = col_idx[keep]
    diams = diams[keep]

    if stat == "count":
        # Count includes empty cells as 0 (rather than NaN).
        out[:] = 0.0
        if row_idx.size:
            np.add.at(out, (row_idx, col_idx), 1.0)
        units = ""
    else:
        cell_ids = row_idx * n_cols + col_idx
        for cid in np.unique(cell_ids):
            d = diams[cell_ids == cid]
            if d.size < min_crowns:
                continue
            r, c = divmod(int(cid), n_cols)
            if stat == "cv":
                mu = float(d.mean())
                out[r, c] = float(d.std() / mu) if mu > 0 else np.nan
            elif stat == "p90":
                out[r, c] = float(np.percentile(d, 90))
            elif stat == "mean":
                out[r, c] = float(d.mean())
            elif stat == "median":
                out[r, c] = float(np.median(d))
            elif stat == "std":
                out[r, c] = float(d.std())
        units = "m" if stat in ("p90", "mean", "median", "std") else ""

    logger.info(
        "crown_%s: %dx%d windows (%.0fm), non-NaN cells = %d",
        stat,
        n_rows,
        n_cols,
        window_m,
        int(np.sum(~np.isnan(out))),
    )
    return make_result(
        f"crown_{stat}",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"stat": stat, "min_crowns": min_crowns, "diam_column": diam_column},
        units=units,
    )


def _make_wrapper(stat: CrownStat):
    def wrapper(
        crowns: gpd.GeoDataFrame,
        grid: GridSpec,
        bounds: tuple[float, float, float, float],
        *,
        window_m: float = 50.0,
        diam_column: str = "crown_diam_m",
        min_crowns: int = 3,
    ) -> MetricResult:
        return crown_stats(
            crowns,
            grid,
            bounds,
            window_m=window_m,
            stat=stat,
            diam_column=diam_column,
            min_crowns=min_crowns,
        )

    wrapper.__name__ = f"crown_{stat}"
    wrapper.__doc__ = f"Convenience wrapper for :func:`crown_stats` with stat={stat!r}."
    return wrapper


crown_cv = register("crown_cv")(_make_wrapper("cv"))
crown_p90 = register("crown_p90")(_make_wrapper("p90"))
crown_mean = register("crown_mean")(_make_wrapper("mean"))
crown_count = register("crown_count")(_make_wrapper("count"))


__all__ = ["crown_stats", "crown_cv", "crown_p90", "crown_mean", "crown_count"]
