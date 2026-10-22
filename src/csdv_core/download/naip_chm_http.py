"""csdv_core.download.naip_chm_http — National NAIP-CHM tiles over HTTP.

The CONUS NAIP-CHM (Morford et al., 2026) is published as one Cloud Optimized
GeoTIFF per NAIP quarter quad on the NTSG Rangeland server, with a CSV index of
every tile. This module fetches that index and downloads whole tiles. It is the
free route to the full-resolution product. The same files sit in a requester
pays Google Cloud bucket, and :mod:`csdv_core.download.chm_model` exports
clips from the Earth Engine copy.

Every download records its size and wall time, so the same code that fetches
tiles also measures server throughput for the national feasibility study.

Inputs: tile URLs from the index. Outputs: COGs on disk and one
:class:`TileDownload` record per tile.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import click

logger = logging.getLogger(__name__)

NTSG_BASE_URL = "https://rangeland.ntsg.umt.edu/data/naip-chm"
INDEX_URL = f"{NTSG_BASE_URL}/index.csv"

__all__ = [
    "INDEX_URL",
    "NTSG_BASE_URL",
    "TileDownload",
    "download_tile",
    "download_tiles",
    "fetch_index",
    "https_url",
]


@dataclass(frozen=True)
class TileDownload:
    """Outcome of one tile download.

    Attributes:
        url: Source URL.
        path: Destination file.
        n_bytes: Bytes on disk after the download, or 0 on failure.
        expected_bytes: ``Content-Length`` reported by the server, or -1.
        seconds: Wall time of the transfer. Zero when the file was skipped.
        skipped: True when a complete file was already on disk.
        error: Empty on success, otherwise the last error message.
    """

    url: str
    path: str
    n_bytes: int
    expected_bytes: int
    seconds: float
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def https_url(url: str) -> str:
    """Rewrite an index ``http://`` URL to ``https://``.

    The index lists ``http`` URLs, and the server answers them with a 301 to
    the ``https`` address. Asking for ``https`` directly saves a round trip
    per tile, which adds up over two hundred thousand tiles.
    """
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def fetch_index(
    dest_dir: Path | str, *, url: str = INDEX_URL, force: bool = False
) -> Path:
    """Download the NTSG tile index (about 270 MB of CSV).

    Args:
        dest_dir: Directory for ``index.csv``.
        url: Index URL.
        force: Download again even when the file exists.

    Returns:
        Path to the index.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(url).name
    if target.exists() and not force:
        logger.info("Index already present at %s", target)
        return target
    record = download_tile(url, dest_dir, force=True)
    if not record.ok:
        raise RuntimeError(f"Index download failed: {record.error}")
    return target


def _content_length(response: object) -> int:
    value = getattr(response, "headers", {}).get("Content-Length")
    try:
        return int(value) if value is not None else -1
    except ValueError:
        return -1


def download_tile(
    url: str,
    out_dir: Path | str,
    *,
    force: bool = False,
    retries: int = 3,
    timeout_s: float = 120.0,
    backoff_s: float = 5.0,
) -> TileDownload:
    """Download one file, writing to ``.part`` and renaming on success.

    A file counts as complete only when its size matches ``Content-Length``,
    so an interrupted transfer is never mistaken for a tile.

    Args:
        url: Source URL. ``http`` is rewritten to ``https``.
        out_dir: Destination directory. The file keeps its URL basename.
        force: Download even when a file of the right name exists.
        retries: Attempts before giving up.
        timeout_s: Socket timeout per attempt.
        backoff_s: Wait before a retry, multiplied by the attempt number.

    Returns:
        A :class:`TileDownload`. Failures are recorded, not raised, so one bad
        tile cannot stop a batch.
    """
    url = https_url(url)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(url).name
    if target.exists() and not force:
        size = target.stat().st_size
        return TileDownload(url, str(target), size, size, 0.0, skipped=True)

    tmp = target.with_suffix(target.suffix + ".part")
    error = ""
    expected = -1
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout_s) as response,
                open(tmp, "wb") as fh,
            ):
                expected = _content_length(response)
                shutil.copyfileobj(response, fh, length=1 << 20)
            seconds = time.perf_counter() - start
            n_bytes = tmp.stat().st_size
            if expected >= 0 and n_bytes != expected:
                raise OSError(f"short read: {n_bytes} of {expected} bytes")
            tmp.replace(target)
            return TileDownload(url, str(target), n_bytes, expected, seconds)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            error = f"attempt {attempt}: {exc}"
            logger.warning("%s failed (%s)", url, error)
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    return TileDownload(url, str(target), 0, expected, 0.0, error=error)


def download_tiles(
    urls: Iterable[str],
    out_dir: Path | str,
    *,
    workers: int = 8,
    force: bool = False,
) -> list[TileDownload]:
    """Download many tiles with a thread pool.

    Args:
        urls: Tile URLs.
        out_dir: Destination directory.
        workers: Concurrent transfers. Keep this modest on a shared server.
        force: Re-download files that already exist.

    Returns:
        One :class:`TileDownload` per URL, in input order.
    """
    url_list = list(urls)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        records = list(
            pool.map(lambda u: download_tile(u, out_dir, force=force), url_list)
        )
    elapsed = time.perf_counter() - start
    moved = sum(r.n_bytes for r in records if not r.skipped and r.ok)
    n_failed = sum(not r.ok for r in records)
    logger.info(
        "Downloaded %d tiles (%d skipped, %d failed), %.2f GB in %.0f s, "
        "%.1f MB/s aggregate at %d workers",
        len(records),
        sum(r.skipped for r in records),
        n_failed,
        moved / 1e9,
        elapsed,
        moved / 1e6 / elapsed if elapsed > 0 else 0.0,
        workers,
    )
    return records


@click.command("naip-chm-http")
@click.option(
    "--urls-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Text file with one tile URL per line, or a CSV with a chm_url column.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination directory for the COGs.",
)
@click.option("--workers", type=int, default=8, show_default=True)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional CSV with one row per tile (bytes, seconds, error).",
)
@click.option("--force", is_flag=True, default=False, help="Re-download.")
def cli(
    urls_file: Path, out_dir: Path, workers: int, report: Path | None, force: bool
) -> None:
    """Download national NAIP-CHM tiles from the NTSG server."""
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    if urls_file.suffix.lower() == ".csv":
        urls = pd.read_csv(urls_file)["chm_url"].astype(str).tolist()
    else:
        urls = [u.strip() for u in urls_file.read_text().splitlines() if u.strip()]
    records = download_tiles(urls, out_dir, workers=workers, force=force)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([asdict(r) for r in records])
        frame["workers"] = workers
        frame.to_csv(report, index=False)
        logger.info("Wrote %s", report)
    n_failed = sum(not r.ok for r in records)
    if n_failed:
        raise SystemExit(f"{n_failed} of {len(records)} downloads failed")
