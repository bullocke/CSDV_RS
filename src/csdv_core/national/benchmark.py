"""csdv_core.national.benchmark — time crown engines block by block on real tiles.

The national cost estimate needs seconds, memory and output bytes per unit of
canopy for each engine. This module reads a NAIP-CHM tile one block at a time
(an interior block plus a halo, the same layout as
:func:`csdv_core.zonal.crowns.segment_scene_crowns`), runs the Python watershed
or the lidR ``dalponte2016`` engine on that block, and records per-stage wall
time, crowns kept in the interior, and canopy pixels. Windowed reads keep
memory to one block, so the same code runs on a laptop with a few blocks per
tile and on CHPC over every block of a tile.

lidR runs in a fresh ``Rscript`` per block, so its wall time includes R start
up and package loading. The R script times its own steps, and their sum is the
compute time that would remain if a production run looped blocks inside one R
session. Both numbers are recorded.

Inputs: a CHM COG and segmentation parameters. Outputs: one
:class:`BlockResult` per block and engine, and output-size measurements.
"""

from __future__ import annotations

import logging
import platform
import resource
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np

from csdv_core.segmentation.params import DEFAULT_PARAMS, SegmentationParams

logger = logging.getLogger(__name__)

ENGINES = ("watershed", "lidr")

__all__ = [
    "ENGINES",
    "BlockResult",
    "OutputSizes",
    "Window4",
    "benchmark_tile",
    "cpu_model",
    "interior_crowns",
    "iter_blocks",
    "measure_output_sizes",
    "tile_canopy_stats",
]

#: ``(row0, col0, row1, col1)`` of a block interior, end exclusive.
Window4 = tuple[int, int, int, int]


@dataclass
class BlockResult:
    """Timing and yield of one engine on one block.

    Attributes:
        tile_id: Tile identifier.
        engine: ``watershed`` or ``lidr``.
        row0: First interior row.
        col0: First interior column.
        interior_px: Interior pixels.
        halo_px: Pixels read including the halo.
        canopy_px: Interior pixels at or above the canopy floor (raw CHM).
        n_crowns: Crowns whose centroid falls in the interior.
        compute_s: Summed stage time, excluding interpreter start up.
        wall_s: Wall time of the whole call, including the block read and,
            for lidR, R start up and the temporary GeoTIFF.
        peak_rss_mb: Peak resident memory of the process that segmented.
            For the watershed engine this is the Python process peak so far.
        stages: Seconds per stage.
        error: Empty on success.
    """

    tile_id: str
    engine: str
    row0: int
    col0: int
    interior_px: int
    halo_px: int
    canopy_px: int
    n_crowns: int = 0
    compute_s: float = float("nan")
    wall_s: float = float("nan")
    peak_rss_mb: float = float("nan")
    stages: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def as_row(self) -> dict[str, object]:
        row = {k: v for k, v in self.__dict__.items() if k != "stages"}
        row.update({f"t_{k}": v for k, v in self.stages.items()})
        return row


@dataclass(frozen=True)
class OutputSizes:
    """Bytes per crown and write time for each candidate storage format."""

    n_crowns: int
    bytes_per_crown: dict[str, float]
    write_s: dict[str, float]


def cpu_model() -> str:
    """CPU model string, so timings from different machines can be told apart."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _peak_rss_mb() -> float:
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def iter_blocks(height: int, width: int, block_px: int) -> list[Window4]:
    """Interior blocks covering a raster, in row-major order."""
    return [
        (r0, c0, min(height, r0 + block_px), min(width, c0 + block_px))
        for r0 in range(0, height, block_px)
        for c0 in range(0, width, block_px)
    ]


def interior_crowns(
    crowns: gpd.GeoDataFrame, interior_bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Crowns whose centroid lies in a block interior (half-open box).

    Same rule as :func:`csdv_core.zonal.crowns.segment_scene_crowns`, so block
    counts add up to what a full scene run keeps.
    """
    if crowns.empty:
        return crowns
    c = crowns.geometry.centroid
    x0, y0, x1, y1 = interior_bounds
    keep = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1)
    return crowns.loc[keep]


def tile_canopy_stats(
    path: Path | str, *, scale_factor: float = 0.01, min_height_m: float = 2.0
) -> dict[str, float]:
    """Full-resolution canopy fraction of a tile, read in strips.

    Used to check that the overview census tracks real canopy cover.
    """
    import rasterio
    from rasterio.windows import Window

    n_valid = n_canopy = n_ge5 = 0
    with rasterio.open(path) as src:
        step = 1024
        for r0 in range(0, src.height, step):
            win = Window(0, r0, src.width, min(step, src.height - r0))
            data = src.read(1, window=win, masked=True)
            h = data.astype("float32").filled(np.nan) * np.float32(scale_factor)
            ok = np.isfinite(h)
            n_valid += int(ok.sum())
            n_canopy += int((h >= min_height_m).sum())
            n_ge5 += int((h >= 5.0).sum())
        n_px = src.width * src.height
    return {
        "full_px": float(n_px),
        "valid_frac_full": n_valid / n_px if n_px else float("nan"),
        "frac_ge2_full": n_canopy / n_valid if n_valid else float("nan"),
        "frac_ge5_full": n_ge5 / n_valid if n_valid else float("nan"),
        "canopy_px_full": float(n_canopy),
    }


def _read_block(
    src: object, interior: Window4, halo_px: int, scale_factor: float
) -> tuple[np.ndarray, object, tuple[float, float, float, float]]:
    """Read an interior block plus halo as float32 metres with NaN nodata."""
    import rasterio.windows
    from rasterio.windows import Window

    r0, c0, r1, c1 = interior
    er0, ec0 = max(0, r0 - halo_px), max(0, c0 - halo_px)
    er1 = min(src.height, r1 + halo_px)  # type: ignore[attr-defined]
    ec1 = min(src.width, c1 + halo_px)  # type: ignore[attr-defined]
    win = Window(ec0, er0, ec1 - ec0, er1 - er0)
    data = src.read(1, window=win, masked=True)  # type: ignore[attr-defined]
    chm = data.astype("float32").filled(np.nan) * np.float32(scale_factor)
    transform = rasterio.windows.transform(win, src.transform)  # type: ignore[attr-defined]
    bounds = rasterio.windows.bounds(
        Window(c0, r0, c1 - c0, r1 - r0), src.transform  # type: ignore[attr-defined]
    )
    return chm, transform, bounds


def _run_watershed(
    chm: np.ndarray, transform: object, crs: object, params: SegmentationParams
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    from csdv_core.segmentation.chm_watershed import segment_crowns

    timings: dict[str, float] = {}
    crowns = segment_crowns(chm, transform, crs, params=params, timings=timings)
    return crowns, timings


def _run_lidr(
    chm: np.ndarray,
    transform: object,
    crs: object,
    params: SegmentationParams,
    work_dir: Path,
    th_seed: float,
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    import rasterio

    from csdv_core.segmentation.lidr_bridge import run_lidr_segmentation

    tif = work_dir / "block.tif"
    gpkg = work_dir / "block_crowns.gpkg"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=chm.shape[0],
        width=chm.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(chm, 1)
    timings: dict[str, float] = {}
    try:
        run_lidr_segmentation(
            tif, gpkg, params=params, scale_factor=1.0, th_seed=th_seed,
            timings=timings,
        )  # fmt: skip
    except RuntimeError as exc:
        # The R script stops when a block has no tree tops. That is a valid
        # empty result for the benchmark, not a failure.
        if "No tree tops" in str(exc) or not np.any(chm >= params.min_height_m):
            return gpd.GeoDataFrame(geometry=[], crs=crs), timings
        raise
    crowns = gpd.read_file(gpkg)
    return crowns, timings


def benchmark_tile(
    path: Path | str,
    tile_id: str,
    *,
    engines: tuple[str, ...] = ENGINES,
    params: SegmentationParams = DEFAULT_PARAMS,
    block_px: int = 2048,
    halo_px: int | None = None,
    blocks: list[Window4] | None = None,
    scale_factor: float = 0.01,
    th_seed: float = 0.45,
    work_dir: Path | str | None = None,
    keep_crowns: bool = False,
) -> tuple[list[BlockResult], dict[str, gpd.GeoDataFrame]]:
    """Run each engine on chosen blocks of one tile.

    Args:
        path: Tile COG.
        tile_id: Identifier carried into the results.
        engines: Engines to run, from :data:`ENGINES`.
        params: Segmentation parameters, shared by both engines.
        block_px: Interior block side in pixels.
        halo_px: Halo in pixels. Defaults to
            :func:`csdv_core.zonal.crowns.default_overlap_px`.
        blocks: Interior windows to run. Defaults to every block of the tile.
        scale_factor: Multiplier from stored values to metres.
        th_seed: lidR ``dalponte2016`` seed threshold.
        work_dir: Scratch directory for lidR's temporary files.
        keep_crowns: Return the interior crowns of each engine, concatenated.

    Returns:
        ``(results, crowns)``. ``crowns`` maps engine to its interior crowns
        when ``keep_crowns`` is set, and is empty otherwise.
    """
    import pandas as pd
    import rasterio

    from csdv_core.zonal.crowns import default_overlap_px

    unknown = set(engines) - set(ENGINES)
    if unknown:
        raise ValueError(f"Unknown engines: {sorted(unknown)}")
    results: list[BlockResult] = []
    kept: dict[str, list[gpd.GeoDataFrame]] = {e: [] for e in engines}
    tmp_ctx = tempfile.TemporaryDirectory(dir=work_dir)
    tmp = Path(tmp_ctx.name)
    try:
        with rasterio.open(path) as src:
            px = float(abs(src.transform.a))
            halo = default_overlap_px(params, px) if halo_px is None else halo_px
            windows = blocks or iter_blocks(src.height, src.width, block_px)
            for interior in windows:
                t_read = time.perf_counter()
                chm, transform, bounds = _read_block(src, interior, halo, scale_factor)
                read_s = time.perf_counter() - t_read
                r0, c0, r1, c1 = interior
                # Offset of the interior inside the haloed array. The halo is
                # cut short at the raster edge.
                dr, dc = r0 - max(0, r0 - halo), c0 - max(0, c0 - halo)
                inner = chm[dr : dr + (r1 - r0), dc : dc + (c1 - c0)]
                canopy_px = int(np.sum(inner >= params.min_height_m))
                for engine in engines:
                    res = BlockResult(
                        tile_id=tile_id,
                        engine=engine,
                        row0=r0,
                        col0=c0,
                        interior_px=int((r1 - r0) * (c1 - c0)),
                        halo_px=int(chm.size),
                        canopy_px=canopy_px,
                    )
                    t0 = time.perf_counter()
                    try:
                        if canopy_px == 0:
                            crowns, stages = gpd.GeoDataFrame(geometry=[]), {}
                        elif engine == "watershed":
                            crowns, stages = _run_watershed(
                                chm, transform, src.crs, params
                            )
                        else:
                            crowns, stages = _run_lidr(
                                chm, transform, src.crs, params, tmp, th_seed
                            )
                        peak = stages.pop("peak_rss_mb", _peak_rss_mb())
                        crowns = interior_crowns(crowns, bounds)
                        res.n_crowns = len(crowns)
                        res.stages = {"read": read_s, **stages}
                        res.compute_s = float(sum(stages.values()))
                        res.wall_s = read_s + time.perf_counter() - t0
                        res.peak_rss_mb = float(peak)
                        if keep_crowns and not crowns.empty:
                            kept[engine].append(crowns)
                    except Exception as exc:  # noqa: BLE001  record and continue
                        logger.exception(
                            "%s %s block %s failed", tile_id, engine, interior
                        )
                        res.error = str(exc)[:500]
                    results.append(res)
                    logger.info(
                        "%s %-9s block (%5d,%5d): %6d crowns, %5.1f%% canopy, "
                        "%.2f s compute, %.2f s wall",
                        tile_id,
                        engine,
                        r0,
                        c0,
                        res.n_crowns,
                        100.0 * canopy_px / max(res.interior_px, 1),
                        res.compute_s,
                        res.wall_s,
                    )
            crs = src.crs
    finally:
        tmp_ctx.cleanup()
    crowns_out: dict[str, gpd.GeoDataFrame] = {}
    if keep_crowns:
        for engine, parts in kept.items():
            crowns_out[engine] = (
                gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
                if parts
                else gpd.GeoDataFrame(geometry=[], crs=crs)
            )
    return results, crowns_out


def measure_output_sizes(
    crowns: gpd.GeoDataFrame,
    work_dir: Path | str,
    *,
    simplify_m: float = 0.3,
) -> OutputSizes:
    """Write crowns in each candidate format and report bytes per crown.

    Formats: GeoPackage, GeoParquet with WKB geometry, GeoParquet with native
    GeoArrow coordinates, GeoParquet after removing pixel stair steps with a
    ``simplify_m`` Douglas-Peucker tolerance, and the attribute table alone.
    All Parquet variants use zstd.
    """
    from csdv_core.io.vector import write_vector

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    n = len(crowns)
    if n == 0:
        return OutputSizes(0, {}, {})
    simplified = crowns.copy()
    simplified["geometry"] = crowns.geometry.simplify(simplify_m)
    variants = {
        "gpkg": (crowns, "crowns.gpkg", {}),
        "parquet_wkb": (crowns, "crowns_wkb.parquet", {}),
        "parquet_geoarrow": (
            crowns, "crowns_geoarrow.parquet", {"geometry_encoding": "geoarrow"}
        ),
        f"parquet_wkb_simplify{simplify_m:g}": (
            simplified, "crowns_simplified.parquet", {}
        ),
    }  # fmt: skip
    sizes: dict[str, float] = {}
    times: dict[str, float] = {}
    for name, (frame, fname, kwargs) in variants.items():
        out = work_dir / fname
        out.unlink(missing_ok=True)
        t0 = time.perf_counter()
        write_vector(frame, out, **kwargs)
        times[name] = time.perf_counter() - t0
        sizes[name] = out.stat().st_size / n
        out.unlink()
    attrs = work_dir / "crowns_attributes.parquet"
    t0 = time.perf_counter()
    crowns.drop(columns="geometry").to_parquet(attrs, compression="zstd")
    times["attributes_only"] = time.perf_counter() - t0
    sizes["attributes_only"] = attrs.stat().st_size / n
    attrs.unlink()
    return OutputSizes(n, sizes, times)
