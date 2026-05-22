"""On-disk contract for metric rasters.

Each metric is written as a single-band float32 GeoTIFF named
``<metric>.tif`` under :meth:`csdv_core.io.paths.ProjectPaths.metrics_dir`.
A sidecar ``manifest.yaml`` lists the metrics present, their parameters,
and the source window size.

This module is the input contract for stage classification
(:mod:`csdv_core.stages.classify`) and any later trajectory work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from csdv_core.io.grids import GridSpec
from csdv_core.io.raster import write_raster
from csdv_core.metrics._result import MetricResult

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.yaml"


def write_metric(result: MetricResult, out_dir: Path | str) -> Path:
    """Write a single :class:`MetricResult` as ``<name>.tif`` in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.name}.tif"
    write_raster(
        path,
        result.array,
        transform=result.transform,
        crs=str(result.crs),
        nodata=float("nan"),
        dtype="float32",
    )
    logger.info("wrote metric %s to %s", result.name, path)
    return path


def write_manifest(
    out_dir: Path | str,
    results: list[MetricResult],
) -> Path:
    """Write a manifest listing the metrics in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for r in results:
        entries.append(
            {
                "metric": r.name,
                "window_m": float(r.window_m),
                "units": r.units,
                "params": dict(r.params),
            }
        )
    payload = {"metrics": entries}
    path = out_dir / MANIFEST_NAME
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def write_metric_stack(
    results: list[MetricResult],
    out_dir: Path | str,
) -> Path:
    """Write all metrics in ``results`` plus a manifest into ``out_dir``."""
    out_dir = Path(out_dir)
    for r in results:
        write_metric(r, out_dir)
    return write_manifest(out_dir, results)


def read_metric_stack(
    metrics_dir: Path | str,
    metrics: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], GridSpec]:
    """Read all (or a subset of) metric rasters from ``metrics_dir``.

    Args:
        metrics_dir: Directory containing ``<metric>.tif`` files.
        metrics: Optional list of metric names to read. ``None`` loads
            every ``*.tif`` in the directory.

    Returns:
        Tuple ``(arrays, grid)`` where ``arrays`` maps metric name to a
        2-D float32 array (NaN where nodata) and ``grid`` is the shared
        :class:`GridSpec`. Raises ``ValueError`` if rasters disagree on
        transform, CRS, or shape.
    """
    import rasterio

    metrics_dir = Path(metrics_dir)
    if metrics is None:
        files = sorted(p for p in metrics_dir.glob("*.tif"))
    else:
        files = [metrics_dir / f"{m}.tif" for m in metrics]

    if not files:
        raise FileNotFoundError(f"No metric rasters found in {metrics_dir}")

    arrays: dict[str, np.ndarray] = {}
    grid: GridSpec | None = None
    ref_shape: tuple[int, int] | None = None

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing metric raster: {path}")
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype("float32")
            transform = src.transform
            crs = str(src.crs) if src.crs is not None else ""
            pixel_size_m = float(abs(transform.a))
        name = path.stem
        if grid is None:
            grid = GridSpec(transform=transform, crs=crs, pixel_size_m=pixel_size_m)
            ref_shape = arr.shape
        else:
            if transform != grid.transform or crs != grid.crs or arr.shape != ref_shape:
                raise ValueError(
                    f"Metric raster {path} disagrees with stack grid: "
                    f"got transform={transform}, crs={crs}, shape={arr.shape}; "
                    f"expected transform={grid.transform}, crs={grid.crs}, "
                    f"shape={ref_shape}"
                )
        arrays[name] = arr

    assert grid is not None
    return arrays, grid


def read_manifest(metrics_dir: Path | str) -> dict[str, Any]:
    """Read the metrics manifest, returning ``{}`` if absent."""
    path = Path(metrics_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}
