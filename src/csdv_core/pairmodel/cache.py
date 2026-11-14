"""csdv_core.pairmodel.cache: decoded chips for fast loading.

Inputs: a chip folder of a label run. Outputs: a cache folder with one
uncompressed ``<chip>.npy`` per chip and ``manifest.json``.

The chip rasters are compressed and pixel-interleaved, so reading a crop of
two dates decodes every band of the whole chip. That costs about 50 ms per
sample and is most of an epoch. The cache holds each chip decoded once, as a
single uint8 array that every loader process opens as a memory map. A crop
is then a slice, and the operating system shares the pages among the
processes.

Layout of a chip with ``n`` dates, along the first axis: ``4 n`` NAIP bands
(date-major), ``n - 1`` change bands, ``n - 1`` agent bands and ``n`` forest
bands. See :func:`layout`.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)

VERSION = 1
PARTS = ("naip", "change", "agent", "aux")
MANIFEST = "manifest.json"

__all__ = ["CacheReport", "build_cache", "cache_path", "layout", "read_cached"]


@dataclass(frozen=True)
class CacheReport:
    """What a cache build did.

    Attributes:
        written: Chips decoded in this build.
        kept: Chips whose cached array was still valid.
        gigabytes: Size of the cache on disk.
        seconds: Duration of the build.
    """

    written: int
    kept: int
    gigabytes: float
    seconds: float


def cache_path(cache_dir: Path, chip_id: str) -> Path:
    return Path(cache_dir) / f"{chip_id}.npy"


def layout(n_dates: int) -> dict[str, int]:
    """First band of each part of a cached chip, and ``bands`` in total."""
    naip = 4 * n_dates
    return {
        "naip": 0,
        "change": naip,
        "agent": naip + n_dates - 1,
        "forest": naip + 2 * (n_dates - 1),
        "bands": naip + 2 * (n_dates - 1) + n_dates,
    }


def _signature(chips_dir: Path, chip_id: str) -> list[list[int]]:
    """Size and modification time of the chip's source files."""
    out = []
    for part in PARTS:
        stat = (Path(chips_dir) / f"{chip_id}_{part}.tif").stat()
        out.append([stat.st_size, stat.st_mtime_ns])
    return out


def _decode(chips_dir: Path, chip_id: str, n_dates: int) -> np.ndarray:
    """Every band a loader needs of one chip, as one uint8 array."""
    stem = Path(chips_dir) / chip_id
    expected = {"naip": 4 * n_dates, "change": n_dates - 1, "agent": n_dates - 1}
    parts = []
    for part, count in expected.items():
        with rasterio.open(f"{stem}_{part}.tif") as src:
            if src.count != count:
                raise ValueError(
                    f"{chip_id}_{part}.tif has {src.count} bands, expected {count}"
                )
            parts.append(src.read())
    with rasterio.open(f"{stem}_aux.tif") as src:
        if src.count != 2 * n_dates - 1:
            raise ValueError(
                f"{chip_id}_aux.tif has {src.count} bands, expected {2 * n_dates - 1}"
            )
        parts.append(src.read(list(range(n_dates, 2 * n_dates))))
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.uint8)


def _write_chip(task: tuple[str, str, str, int]) -> tuple[str, int]:
    """Decode one chip into the cache. Returns its ID and file size."""
    chips_dir, cache_dir, chip_id, n_dates = task
    target = cache_path(Path(cache_dir), chip_id)
    partial = target.with_suffix(f".{os.getpid()}.tmp")
    with partial.open("wb") as handle:
        np.save(handle, _decode(Path(chips_dir), chip_id, n_dates))
    partial.replace(target)
    return chip_id, target.stat().st_size


def _read_manifest(cache_dir: Path) -> dict[str, dict]:
    path = Path(cache_dir) / MANIFEST
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except ValueError:
        return {}
    if manifest.get("version") != VERSION:
        return {}
    return manifest.get("chips", {})


def build_cache(
    chips_dir: Path,
    cache_dir: Path,
    chips: pd.DataFrame,
    *,
    workers: int = 1,
) -> CacheReport:
    """Decode chips into ``cache_dir``, leaving valid entries alone.

    A chip is decoded again when its array is missing or has another size
    than recorded, or when the size or modification time of one of its
    source files differs from the manifest.

    Args:
        chips_dir: Chip folder of the label run.
        cache_dir: Cache folder, created when missing. About 12 MB per chip
            of 512 px and seven dates.
        chips: Table with ``chip_id`` and ``n_dates``, one row per chip or
            more (duplicates are dropped).
        workers: Processes that decode.

    Returns:
        Counts, size and duration.
    """
    start = time.perf_counter()
    chips_dir, cache_dir = Path(chips_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = chips.drop_duplicates("chip_id")
    known = _read_manifest(cache_dir)
    entries: dict[str, dict] = {}
    tasks = []
    for chip_id, n_dates in zip(wanted["chip_id"], wanted["n_dates"], strict=True):
        signature = _signature(chips_dir, chip_id)
        entry = known.get(chip_id)
        target = cache_path(cache_dir, chip_id)
        if (
            entry is not None
            and entry.get("signature") == signature
            and entry.get("n_dates") == int(n_dates)
            and target.is_file()
            and target.stat().st_size == entry.get("bytes")
        ):
            entries[chip_id] = entry
            continue
        entries[chip_id] = {"signature": signature, "n_dates": int(n_dates)}
        tasks.append((str(chips_dir), str(cache_dir), chip_id, int(n_dates)))
    kept = len(wanted) - len(tasks)
    if tasks:
        if workers > 1:
            # Spawned, not forked: the caller may already have loaded torch.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(workers, mp_context=context) as pool:
                done = list(pool.map(_write_chip, tasks, chunksize=8))
        else:
            done = [_write_chip(task) for task in tasks]
        for chip_id, size in done:
            entries[chip_id]["bytes"] = size
    # Entries of chips outside this build stay valid for other runs.
    merged = {**known, **entries}
    manifest = cache_dir / MANIFEST
    partial = manifest.with_suffix(".tmp")
    partial.write_text(json.dumps({"version": VERSION, "chips": merged}))
    partial.replace(manifest)
    size = sum(entries[c]["bytes"] for c in entries) / 1e9
    report = CacheReport(len(tasks), kept, size, time.perf_counter() - start)
    logger.info(
        "Chip cache %s: %d decoded, %d kept, %.1f GB, %.0f s",
        cache_dir,
        report.written,
        report.kept,
        report.gigabytes,
        report.seconds,
    )
    return report


def read_cached(
    cache_dir: Path,
    chip_id: str,
    k: int,
    n_dates: int,
    window_a: tuple[int, int, int] | None = None,
    window_b: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interval ``k`` of a cached chip.

    Args:
        window_a: ``(row, col, size)`` of a square crop of date A and of the
            labels. Default: the whole chip.
        window_b: The same for date B. Default: ``window_a``.

    Returns:
        ``(a, b, change, forest, agent)`` as copies: two ``(4, rows, cols)``
        images, the change band, the forest masks of both dates
        ``(2, rows, cols)`` and the agent band.

    Raises:
        ValueError: If the cached chip does not hold ``n_dates`` dates.
    """
    data = np.load(cache_path(cache_dir, chip_id), mmap_mode="r")
    at = layout(n_dates)
    if data.shape[0] != at["bands"]:
        raise ValueError(
            f"Cached chip {chip_id} has {data.shape[0]} bands, expected "
            f"{at['bands']} for {n_dates} dates"
        )

    def crop(bands: slice | int, spec: tuple[int, int, int] | None) -> np.ndarray:
        if spec is None:
            return np.array(data[bands])
        row, col, size = spec
        return np.array(data[bands, row : row + size, col : col + size])

    b_spec = window_b or window_a
    a = crop(slice(4 * k, 4 * k + 4), window_a)
    b = crop(slice(4 * (k + 1), 4 * (k + 1) + 4), b_spec)
    change = crop(at["change"] + k, window_a)
    agent = crop(at["agent"] + k, window_a)
    forest = crop(slice(at["forest"] + k, at["forest"] + k + 2), window_a)
    return a, b, change, forest, agent
