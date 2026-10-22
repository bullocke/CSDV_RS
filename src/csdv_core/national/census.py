"""csdv_core.national.census — a canopy census of every tile from its overviews.

Segmentation cost and crown count both scale with the amount of canopy in a
tile, and CONUS tiles range from bare desert to closed forest. Reading every
tile at 0.6 m to find out would mean downloading the whole product. Each COG
carries internal overviews, so one low-resolution level (by default 1/32, about
19 m) costs a header and a handful of blocks over HTTP. The census records
canopy fractions and heights at that level for every tile.

Overview pixels are block means, so the fractions here are not canopy cover at
0.6 m. A 19 m cell holding one 10 m tree reads lower than 10 m. The cost model
is fitted on these same overview metrics, so the bias is absorbed there, and
the benchmark tiles check that the relationship to full-resolution canopy is
monotonic.

Inputs: tile ids and COG URLs (or paths). Outputs: one :class:`TileCensus` per
tile, written as resumable parquet parts.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Overview decimation read by default. 1/32 of 0.6 m is about 19 m.
DEFAULT_DECIMATION = 32

#: Height thresholds (m) at which canopy fractions are reported.
HEIGHT_THRESHOLDS_M = (2.0, 5.0, 15.0)

#: GDAL options for reading COGs over HTTP without listing directories.
VSICURL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
}

__all__ = [
    "DEFAULT_DECIMATION",
    "TileCensus",
    "canopy_metrics",
    "census_tile",
    "load_census",
    "run_census",
]


@dataclass(frozen=True)
class TileCensus:
    """Canopy summary of one tile, read from one overview level.

    Attributes:
        tile_id: Index ``system:index`` of the tile.
        decimation: Overview factor actually read (1 means full resolution).
        pixel_m: Pixel size of the level read, in metres.
        full_width: Tile width at full resolution, in pixels.
        full_height: Tile height at full resolution, in pixels.
        valid_frac: Fraction of overview pixels that hold data.
        frac_ge2: Fraction of valid pixels at or above 2 m.
        frac_ge5: Same at 5 m.
        frac_ge15: Same at 15 m.
        mean_h: Mean height over valid pixels, in metres.
        mean_h_ge2: Mean height over pixels at or above 2 m.
        p95_h: 95th percentile height over valid pixels.
        seconds: Wall time of the open and read.
        error: Empty on success, otherwise the failure message.
    """

    tile_id: str
    decimation: int
    pixel_m: float
    full_width: int
    full_height: int
    valid_frac: float
    frac_ge2: float
    frac_ge5: float
    frac_ge15: float
    mean_h: float
    mean_h_ge2: float
    p95_h: float
    seconds: float
    error: str = ""

    @property
    def full_px(self) -> int:
        return self.full_width * self.full_height


def canopy_metrics(heights_m: np.ndarray) -> dict[str, float]:
    """Canopy fractions and heights of one array.

    Args:
        heights_m: Heights in metres, NaN for nodata.

    Returns:
        ``valid_frac``, ``frac_ge2``, ``frac_ge5``, ``frac_ge15``, ``mean_h``,
        ``mean_h_ge2`` and ``p95_h``. Fractions are over valid pixels. Every
        value is NaN when no pixel is valid, except ``valid_frac``, which is 0.
    """
    arr = np.asarray(heights_m, dtype="float64")
    valid = np.isfinite(arr)
    out: dict[str, float] = {"valid_frac": float(valid.mean()) if arr.size else 0.0}
    if not valid.any():
        for t in HEIGHT_THRESHOLDS_M:
            out[f"frac_ge{t:g}"] = float("nan")
        out.update(mean_h=float("nan"), mean_h_ge2=float("nan"), p95_h=float("nan"))
        return out
    v = arr[valid]
    for t in HEIGHT_THRESHOLDS_M:
        out[f"frac_ge{t:g}"] = float((v >= t).mean())
    canopy = v[v >= 2.0]
    out["mean_h"] = float(v.mean())
    out["mean_h_ge2"] = float(canopy.mean()) if canopy.size else float("nan")
    out["p95_h"] = float(np.percentile(v, 95))
    return out


def _pick_decimation(available: list[int], wanted: int) -> int:
    """Overview factor nearest to ``wanted``, or 1 when there are none."""
    if wanted <= 1 or not available:
        return 1
    return min(available, key=lambda f: (abs(math.log(f / wanted)), f))


def _vsicurl(url_or_path: str) -> str:
    if url_or_path.startswith(("http://", "https://")):
        return "/vsicurl/" + url_or_path
    return url_or_path


def census_tile(
    tile_id: str,
    url_or_path: str,
    *,
    decimation: int = DEFAULT_DECIMATION,
    scale_factor: float = 0.01,
) -> TileCensus:
    """Read one overview level of one tile and summarise its canopy.

    Args:
        tile_id: Identifier carried into the result.
        url_or_path: COG URL or local path.
        decimation: Wanted overview factor. The nearest available level is
            read, and the factor actually used is recorded.
        scale_factor: Multiplier from stored values to metres.

    Returns:
        A :class:`TileCensus`. Failures are recorded in ``error`` rather than
        raised, so one bad tile cannot stop a national run.
    """
    import rasterio

    start = time.perf_counter()
    try:
        with rasterio.Env(**VSICURL_OPTIONS):
            with rasterio.open(_vsicurl(url_or_path)) as src:
                factor = _pick_decimation(list(src.overviews(1)), decimation)
                shape = (
                    math.ceil(src.height / factor),
                    math.ceil(src.width / factor),
                )
                data = src.read(1, out_shape=shape, masked=True)
                full_w, full_h = src.width, src.height
                pixel_m = float(abs(src.transform.a)) * src.width / shape[1]
        heights = data.astype("float64").filled(np.nan) * float(scale_factor)
        metrics = canopy_metrics(heights)
        return TileCensus(
            tile_id=tile_id,
            decimation=int(factor),
            pixel_m=pixel_m,
            full_width=int(full_w),
            full_height=int(full_h),
            seconds=time.perf_counter() - start,
            **metrics,
        )
    except Exception as exc:  # noqa: BLE001  record and continue
        logger.warning("census %s failed: %s", tile_id, exc)
        nan = float("nan")
        return TileCensus(
            tile_id, 0, nan, 0, 0, nan, nan, nan, nan, nan, nan, nan,
            time.perf_counter() - start, error=str(exc)[:500],
        )  # fmt: skip


def _done_ids(out_dir: Path) -> set[str]:
    if not out_dir.exists():
        return set()
    done: set[str] = set()
    for part in sorted(out_dir.glob("part_*.parquet")):
        frame = pd.read_parquet(part, columns=["tile_id", "error"])
        done.update(frame.loc[frame["error"].fillna("") == "", "tile_id"])
    return done


def run_census(
    tiles: pd.DataFrame,
    out_dir: Path | str,
    *,
    workers: int = 16,
    chunk_size: int = 500,
    decimation: int = DEFAULT_DECIMATION,
    limit: int | None = None,
) -> Path:
    """Census many tiles, writing one parquet part per chunk.

    The run resumes. Tiles already in a part without an error are skipped, so
    a rerun after an interruption or a batch of failures fetches only what is
    missing.

    Args:
        tiles: Frame with ``tile_id`` and ``chm_url``, and optionally
            ``scale_factor`` (multiplier to metres, default 0.01).
        out_dir: Directory for ``part_*.parquet``.
        workers: Concurrent HTTP readers.
        chunk_size: Tiles per part.
        decimation: Overview factor to read.
        limit: Census at most this many new tiles (for timing a first run).

    Returns:
        ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_dir)
    todo = tiles.loc[~tiles["tile_id"].isin(done), ["tile_id", "chm_url"]].copy()
    todo["scale_factor"] = (
        tiles.loc[todo.index, "scale_factor"] if "scale_factor" in tiles else 0.01
    )
    if limit is not None:
        todo = todo.head(int(limit))
    logger.info(
        "Census: %d tiles to read, %d already done, %d workers",
        len(todo),
        len(done),
        workers,
    )
    next_part = len(list(out_dir.glob("part_*.parquet")))
    start = time.perf_counter()
    n_read = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for c0 in range(0, len(todo), chunk_size):
            chunk = todo.iloc[c0 : c0 + chunk_size]
            jobs: Iterable[TileCensus] = pool.map(
                lambda row: census_tile(
                    row[0],
                    row[1],
                    decimation=decimation,
                    scale_factor=float(row[2]),
                ),
                chunk.itertuples(index=False, name=None),
            )
            frame = pd.DataFrame([asdict(r) for r in jobs])
            part = out_dir / f"part_{next_part:05d}.parquet"
            frame.to_parquet(part, index=False)
            next_part += 1
            n_read += len(frame)
            elapsed = time.perf_counter() - start
            logger.info(
                "Census %d/%d tiles, %d failed in this part, %.2f tiles/s, wrote %s",
                n_read,
                len(todo),
                int((frame["error"] != "").sum()),
                n_read / elapsed if elapsed > 0 else 0.0,
                part.name,
            )
    return out_dir


def load_census(out_dir: Path | str) -> pd.DataFrame:
    """Concatenate census parts, keeping the latest successful row per tile."""
    parts = sorted(Path(out_dir).glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=[f for f in TileCensus.__dataclass_fields__])
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame["ok"] = frame["error"].fillna("") == ""
    frame = frame.sort_values("ok", kind="stable")
    frame = frame.drop_duplicates("tile_id", keep="last").drop(columns="ok")
    frame["full_px"] = frame["full_width"].astype("int64") * frame["full_height"]
    return frame.reset_index(drop=True)
