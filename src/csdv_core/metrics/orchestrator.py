"""Pass-1 orchestrator for ``csdv compute-metrics``.

Drives the metric registry for a single (site, year, window_m) target. Loads
just the inputs each requested metric needs, calls the registered function,
and returns a list of :class:`~csdv_core.metrics._result.MetricResult` ready
for :func:`~csdv_core.io.metrics_io.write_metric_stack`.

Pass 1 supports four metrics, chosen to cover the seeded trajectory rules:

    gap_fraction, crown_fraction      (CHM)
    crown_cv                          (crowns vector)
    glcm_texture                      (NAIP NIR band)

Other registered metrics raise :class:`NotImplementedError` so the caller
gets a clear "not yet wired" message rather than a silent skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

from csdv_core.io.grids import GridSpec, gridspec_from_raster
from csdv_core.io.paths import project_paths
from csdv_core.metrics._result import MetricResult
from csdv_core.metrics.registry import get_metric

logger = logging.getLogger(__name__)

PASS1_METRICS: tuple[str, ...] = (
    "gap_fraction",
    "crown_fraction",
    "crown_cv",
    "glcm_texture",
)

_CHM_METRICS = {"gap_fraction", "crown_fraction"}
_CROWN_METRICS = {"crown_cv"}
_IMAGE_METRICS = {"glcm_texture"}


def _find_one_tif(directory: Path, label: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory missing: {directory}")
    files = sorted(directory.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No *.tif found in {label} directory: {directory}")
    if len(files) > 1:
        logger.warning(
            "%s directory has %d tifs; using %s", label, len(files), files[0].name
        )
    return files[0]


def _load_chm(path: Path, *, scale: float) -> tuple[np.ndarray, GridSpec]:
    """Load a CHM raster as float32 meters with NaN nodata."""
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs) if src.crs is not None else ""
        pixel_size_m = float(abs(transform.a))
    arr = raw.astype("float32") * float(scale)
    if nodata is not None:
        arr = np.where(raw == nodata, np.nan, arr).astype("float32")
    grid = GridSpec(transform=transform, crs=crs, pixel_size_m=pixel_size_m)
    return arr, grid


def _load_naip_band(path: Path, *, band: int) -> tuple[np.ndarray, GridSpec]:
    """Load a single NAIP band as float32 with NaN nodata."""
    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError(
                f"NAIP raster {path} has {src.count} bands; --naip-band={band} out of range"
            )
        raw = src.read(band)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs) if src.crs is not None else ""
        pixel_size_m = float(abs(transform.a))
    arr = raw.astype("float32")
    if nodata is not None:
        arr = np.where(raw == nodata, np.nan, arr).astype("float32")
    grid = GridSpec(transform=transform, crs=crs, pixel_size_m=pixel_size_m)
    return arr, grid


def _raster_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def _resolve_inputs(
    site: str,
    year: int,
    naip_path: Path | None,
    chm_path: Path | None,
    crowns_path: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Fill in missing input paths from :func:`project_paths`."""
    paths = project_paths()
    if naip_path is None:
        try:
            naip_path = _find_one_tif(paths.naip_dir(site, year), "NAIP")
        except FileNotFoundError:
            naip_path = None
    if chm_path is None:
        try:
            chm_path = _find_one_tif(paths.naip_chm_dir(site, year), "NAIP-CHM")
        except FileNotFoundError:
            chm_path = None
    if crowns_path is None:
        candidate = paths.crowns_dir(site, year) / "crowns.gpkg"
        crowns_path = candidate if candidate.exists() else None
    return naip_path, chm_path, crowns_path


def _check_unsupported(metric_names: list[str]) -> None:
    """Raise if any requested metric is not in Pass 1."""
    unsupported = [m for m in metric_names if m not in PASS1_METRICS]
    if unsupported:
        raise NotImplementedError(
            "Metric(s) not yet wired in the Pass-1 orchestrator: "
            f"{sorted(unsupported)}. Supported: {list(PASS1_METRICS)}"
        )


def compute_for_window(
    site: str,
    year: int,
    window_m: float,
    *,
    metric_names: list[str] | None = None,
    naip_path: Path | None = None,
    chm_path: Path | None = None,
    crowns_path: Path | None = None,
    chm_scale: float = 1.0,
    naip_band: int = 4,
) -> list[MetricResult]:
    """Compute Pass-1 metrics for one (site, year, window_m).

    Args:
        site: Site code (e.g. ``"SCBI"``).
        year: NAIP acquisition year.
        window_m: Analysis window side length in meters.
        metric_names: Subset of :data:`PASS1_METRICS` to compute. ``None``
            means all four.
        naip_path: Optional explicit NAIP RGBN raster. If omitted, resolved
            from ``project_paths().naip_dir(site, year)``.
        chm_path: Optional explicit CHM raster. If omitted, resolved from
            ``project_paths().naip_chm_dir(site, year)``.
        crowns_path: Optional explicit crowns vector. If omitted, resolved
            to ``project_paths().crowns_dir(site, year) / "crowns.gpkg"``.
        chm_scale: Multiplicative scale for the CHM raster. Use ``0.01`` for
            uint16 NAIP-CHM in centimeters; ``1.0`` for float32 meters.
        naip_band: 1-based band index used for GLCM texture (default 4 = NIR).

    Returns:
        List of :class:`MetricResult` in the order requested.
    """
    requested = list(metric_names) if metric_names else list(PASS1_METRICS)
    _check_unsupported(requested)

    naip_path, chm_path, crowns_path = _resolve_inputs(
        site, year, naip_path, chm_path, crowns_path
    )

    need_chm = bool(set(requested) & _CHM_METRICS)
    need_crowns = bool(set(requested) & _CROWN_METRICS)
    need_image = bool(set(requested) & _IMAGE_METRICS)

    if need_chm and chm_path is None:
        raise FileNotFoundError(
            f"CHM required for {sorted(set(requested) & _CHM_METRICS)} "
            f"but no raster found for site={site} year={year}"
        )
    if need_image and naip_path is None:
        raise FileNotFoundError(
            f"NAIP required for {sorted(set(requested) & _IMAGE_METRICS)} "
            f"but no raster found for site={site} year={year}"
        )
    if need_crowns and (crowns_path is None or not crowns_path.exists()):
        raise FileNotFoundError(
            f"Crowns required for {sorted(set(requested) & _CROWN_METRICS)} "
            f"but no vector found for site={site} year={year}"
        )

    chm_arr: np.ndarray | None = None
    chm_grid: GridSpec | None = None
    naip_arr: np.ndarray | None = None
    naip_grid: GridSpec | None = None
    bounds: tuple[float, float, float, float] | None = None

    if need_chm:
        assert chm_path is not None
        chm_arr, chm_grid = _load_chm(chm_path, scale=chm_scale)
        logger.info("loaded CHM %s (shape=%s)", chm_path, chm_arr.shape)
    if need_image:
        assert naip_path is not None
        naip_arr, naip_grid = _load_naip_band(naip_path, band=naip_band)
        logger.info(
            "loaded NAIP band %d from %s (shape=%s)",
            naip_band,
            naip_path,
            naip_arr.shape,
        )
    if need_crowns:
        # Bounds for crown stats must match the metric grid origin. Prefer the
        # CHM grid if available; otherwise the NAIP grid.
        if chm_path is not None:
            bounds = _raster_bounds(chm_path)
            crown_grid = chm_grid or gridspec_from_raster(chm_path)
        elif naip_path is not None:
            bounds = _raster_bounds(naip_path)
            crown_grid = naip_grid or gridspec_from_raster(naip_path)
        else:
            raise FileNotFoundError(
                "crown_cv needs a reference raster (CHM or NAIP) for bounds"
            )
    else:
        crown_grid = None

    results: list[MetricResult] = []
    for name in requested:
        spec = get_metric(name)
        params = dict(spec.defaults)
        params.setdefault("window_m", window_m)
        params["window_m"] = float(window_m)

        if name in _CHM_METRICS:
            assert chm_arr is not None and chm_grid is not None
            r = spec.fn(
                chm_arr,
                chm_grid,
                window_m=float(window_m),
                height_threshold_m=float(
                    params.get(
                        "height_threshold_m", params.get("chm_gap_threshold_m", 2.0)
                    )
                ),
            )
        elif name == "glcm_texture":
            assert naip_arr is not None and naip_grid is not None
            r = spec.fn(
                naip_arr,
                naip_grid,
                window_m=float(window_m),
                levels=int(params.get("levels", 16)),
                prop=params.get("prop", "entropy"),
            )
        elif name in _CROWN_METRICS:
            import geopandas as gpd

            assert crowns_path is not None
            assert bounds is not None and crown_grid is not None
            gdf = gpd.read_file(crowns_path)
            r = spec.fn(
                gdf,
                crown_grid,
                bounds,
                window_m=float(window_m),
                min_crowns=int(params.get("min_crowns_per_window", 3)),
            )
        else:  # pragma: no cover - guarded by _check_unsupported
            raise NotImplementedError(name)

        logger.info("computed %s: shape=%s window_m=%s", name, r.array.shape, window_m)
        results.append(r)

    return results


__all__ = ["compute_for_window", "PASS1_METRICS"]
