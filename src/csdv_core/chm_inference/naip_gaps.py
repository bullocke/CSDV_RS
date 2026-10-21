"""csdv_core.chm_inference.naip_gaps — carry NAIP gaps into a predicted CHM.

The NAIP-CHM model has no notion of missing imagery. Where a NAIP mosaic holds
no data, which both NAIP downloaders write as zero in every band, the model
still predicts a height (usually near 0 m), and a later disturbance run reads
the gap as cleared forest. :func:`mask_naip_gaps` sets those CHM pixels to the
CHM's nodata value. It also masks a band of ``edge_m`` around each gap, where
the model saw the black fill as context.

Gaps are found by the "zero in all four bands" test rather than the NAIP
file's mask, so mosaics written before the downloaders recorded a mask are
handled too. The band around a gap is measured on coarse cells of about
``edge_m / 10``, and only cells that are at least half gap seed it, so a stray
black pixel inside real imagery is masked on its own without opening a hole.

Inputs: a NAIP GeoTIFF and the CHM predicted from it. Outputs: the CHM
rewritten with the gaps as nodata, and a :class:`GapMaskResult`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage

logger = logging.getLogger(__name__)

#: Suffix of the file written before it replaces the target. It does not end
#: in ``.tif``, so a ``*.tif`` glob never picks up a half-written CHM.
PARTIAL_SUFFIX = ".partial"
#: Suffix of the unmasked CHM kept beside a CHM masked in place.
ORIGINAL_SUFFIX = ".unmasked"
#: Coarse cells per ``edge_m`` when measuring the band around gaps.
EDGE_CELLS = 10
#: Least gap share of a coarse cell for it to seed the band around a gap.
SEED_SHARE = 0.5

__all__ = [
    "EDGE_CELLS",
    "ORIGINAL_SUFFIX",
    "PARTIAL_SUFFIX",
    "SEED_SHARE",
    "GapMaskResult",
    "chm_nodata_value",
    "coarse_block_px",
    "edge_band",
    "mask_naip_gaps",
    "naip_gap",
]


@dataclass(frozen=True)
class GapMaskResult:
    """Outcome of :func:`mask_naip_gaps`.

    Attributes:
        chm_path: The CHM with NAIP gaps as nodata (unchanged when the NAIP
            has no gap).
        original_path: The unmasked CHM kept beside it, or None.
        n_pixels: Pixels in the CHM.
        gap_pixels: Pixels where the NAIP has no data.
        edge_pixels: Pixels with NAIP data inside the band around a gap.
        newly_masked: Pixels set to nodata that held a value before.
        nodata: Nodata value of the CHM.
        edge_m: Width of the band masked around gaps (m).
        rewritten: False when the NAIP has no gap and the CHM was left alone.
    """

    chm_path: Path
    original_path: Path | None
    n_pixels: int
    gap_pixels: int
    edge_pixels: int
    newly_masked: int
    nodata: float
    edge_m: float
    rewritten: bool

    @property
    def gap_share(self) -> float:
        """Share of CHM pixels without NAIP imagery."""
        return self.gap_pixels / self.n_pixels

    @property
    def edge_share(self) -> float:
        """Share of CHM pixels masked only because they lie near a gap."""
        return self.edge_pixels / self.n_pixels

    @property
    def newly_masked_share(self) -> float:
        """Share of CHM pixels turned from a value into nodata."""
        return self.newly_masked / self.n_pixels


def naip_gap(block: np.ndarray) -> np.ndarray:
    """Pixels of a ``(bands, rows, cols)`` NAIP block that are zero in every band.

    Args:
        block: NAIP pixel values.

    Returns:
        bool ``(rows, cols)``, True where the NAIP has no imagery.
    """
    return ~block.any(axis=0)


def chm_nodata_value(dtype: str, nodata: float | None) -> float:
    """Nodata value to write into a CHM.

    Args:
        dtype: CHM data type.
        nodata: The CHM's declared nodata value, or None.

    Returns:
        ``nodata`` when declared, else the type's maximum for integers or NaN
        for floats.
    """
    if nodata is not None:
        return float(nodata)
    kind = np.dtype(dtype)
    if np.issubdtype(kind, np.integer):
        return float(np.iinfo(kind).max)
    return float("nan")


def coarse_block_px(edge_m: float, pixel_m: float) -> int:
    """Side, in pixels, of the coarse cells that measure the band around gaps.

    Args:
        edge_m: Width of the band (m), positive.
        pixel_m: CHM pixel size (m).

    Returns:
        About ``edge_m / EDGE_CELLS`` in pixels, at least 1.
    """
    cells = edge_m / EDGE_CELLS / pixel_m
    return max(1, int(round(cells)))


def edge_band(
    gap_share: np.ndarray,
    cell_m: float,
    edge_m: float,
    *,
    seed_share: float = SEED_SHARE,
) -> np.ndarray:
    """Coarse cells within ``edge_m`` of a gap.

    Args:
        gap_share: float ``(rows, cols)`` share of each coarse cell without
            NAIP imagery.
        cell_m: Coarse cell side (m).
        edge_m: Width of the band (m).
        seed_share: Least gap share for a cell to count as gap.

    Returns:
        bool ``(rows, cols)``, True for gap cells and cells whose centre lies
        within ``edge_m`` of a gap cell centre.
    """
    seed = gap_share >= seed_share
    if not seed.any():
        return np.zeros(gap_share.shape, dtype=bool)
    distance = ndimage.distance_transform_edt(~seed, sampling=cell_m)
    return distance <= edge_m


def _pixels_per_cell(height: int, width: int, block_px: int) -> np.ndarray:
    """Pixels inside the raster for each coarse cell (edge cells hold fewer)."""
    row_starts = np.arange(0, height, block_px)
    col_starts = np.arange(0, width, block_px)
    rows_in = np.minimum(block_px, height - row_starts)
    cols_in = np.minimum(block_px, width - col_starts)
    return np.outer(rows_in, cols_in)


def _strip_rows(src: Any, requested: int) -> int:
    """Strip height rounded down to whole CHM blocks, at least one block."""
    block_rows = src.block_shapes[0][0]
    whole_blocks = requested // block_rows
    return max(1, whole_blocks) * block_rows


@contextmanager
def _naip_on_chm_grid(naip_src: Any, chm_src: Any) -> Iterator[Any]:
    """Yield the NAIP as a reader on the CHM grid.

    The CHM normally shares the NAIP grid, and then the NAIP is read directly.
    Otherwise it is read through a nearest-neighbour warp onto the CHM grid,
    which also turns area outside the NAIP into zero, that is, gap.
    """
    same_grid = (
        naip_src.crs == chm_src.crs
        and naip_src.shape == chm_src.shape
        and naip_src.transform.almost_equals(chm_src.transform)
    )
    if same_grid:
        yield naip_src
        return
    logger.info("NAIP and CHM grids differ, warping the NAIP onto the CHM grid")
    with WarpedVRT(
        naip_src,
        crs=chm_src.crs,
        transform=chm_src.transform,
        width=chm_src.width,
        height=chm_src.height,
        resampling=Resampling.nearest,
    ) as vrt:
        yield vrt


def _scan_gaps(
    reader: Any, strip_rows: int, block_px: int
) -> tuple[list[np.ndarray], int, np.ndarray | None]:
    """Read the NAIP once: gap bits per strip, gap total, gap count per cell.

    Args:
        reader: NAIP reader on the CHM grid.
        strip_rows: Rows per strip.
        block_px: Coarse cell side in pixels, 0 to skip the coarse counts.

    Returns:
        ``(packed, gap_total, counts)``: one ``np.packbits`` array per strip,
        the number of gap pixels, and the gap pixels per coarse cell (None
        when ``block_px`` is 0).
    """
    height, width = reader.height, reader.width
    packed: list[np.ndarray] = []
    gap_total = 0
    counts = None
    if block_px:
        n_cell_rows = -(-height // block_px)
        n_cell_cols = -(-width // block_px)
        counts = np.zeros((n_cell_rows, n_cell_cols), dtype=np.int64)
        pad_cols = n_cell_cols * block_px - width
    for row0 in range(0, height, strip_rows):
        n_rows = min(strip_rows, height - row0)
        block = reader.read(window=Window(0, row0, width, n_rows))
        gap = naip_gap(block)
        packed.append(np.packbits(gap, axis=1))
        gap_total += int(gap.sum())
        if counts is None:
            continue
        padded = np.pad(gap, ((0, 0), (0, pad_cols)))
        col_sums = padded.reshape(n_rows, n_cell_cols, block_px).sum(axis=2)
        cell_rows = (row0 + np.arange(n_rows)) // block_px
        starts = np.flatnonzero(np.r_[True, np.diff(cell_rows) > 0])
        counts[cell_rows[starts]] += np.add.reduceat(col_sums, starts, axis=0)
    return packed, gap_total, counts


def _copy_metadata(src: Any, dst: Any) -> None:
    """Copy dataset and band tags, descriptions, scales, offsets and units."""
    dst.update_tags(**src.tags())
    for band in range(1, src.count + 1):
        dst.update_tags(band, **src.tags(band))
    dst.descriptions = src.descriptions
    dst.scales = src.scales
    dst.offsets = src.offsets
    dst.units = src.units


def _write_masked(
    chm_src: Any,
    out_path: Path,
    packed: list[np.ndarray],
    edge: np.ndarray | None,
    block_px: int,
    strip_rows: int,
    nodata: float,
) -> tuple[int, int]:
    """Write the CHM with gaps and the band around them as nodata.

    Returns:
        ``(edge_pixels, newly_masked)``.
    """
    height, width = chm_src.height, chm_src.width
    profile = chm_src.profile.copy()
    profile.update(driver="GTiff", nodata=nodata, BIGTIFF="IF_SAFER")
    nodata_is_nan = np.isnan(nodata)
    col_cells = np.arange(width) // block_px if block_px else None
    edge_pixels = 0
    newly_masked = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        _copy_metadata(chm_src, dst)
        for strip_index, row0 in enumerate(range(0, height, strip_rows)):
            n_rows = min(strip_rows, height - row0)
            window = Window(0, row0, width, n_rows)
            chm = chm_src.read(window=window)
            gap_bits = np.unpackbits(packed[strip_index], axis=1, count=width)
            mask = gap_bits.astype(bool)
            if edge is not None:
                row_cells = (row0 + np.arange(n_rows)) // block_px
                near_gap = edge[np.ix_(row_cells, col_cells)]
                edge_pixels += int((near_gap & ~mask).sum())
                mask |= near_gap
            if nodata_is_nan:
                was_nodata = np.isnan(chm[0])
            else:
                was_nodata = chm[0] == nodata
            newly_masked += int((mask & ~was_nodata).sum())
            chm[:, mask] = nodata
            dst.write(chm, window=window)
    return edge_pixels, newly_masked


def mask_naip_gaps(
    naip_path: Path | str,
    chm_path: Path | str,
    *,
    edge_m: float = 0.0,
    out_path: Path | str | None = None,
    keep_original: bool = False,
    strip_rows: int = 1024,
    seed_share: float = SEED_SHARE,
) -> GapMaskResult:
    """Set CHM pixels without NAIP imagery, and a band around them, to nodata.

    The NAIP is read once, in strips, to find the gaps. When it has none the
    CHM is left untouched. Otherwise the CHM is written to
    ``<target>.partial`` and moved over the target when complete, so a failed
    run never leaves a half-written CHM behind.

    Args:
        naip_path: NAIP GeoTIFF the CHM was predicted from.
        chm_path: Predicted CHM.
        edge_m: Width (m) of the band masked around each gap. 0 masks the gaps
            only. The band is resolved to coarse cells of about
            ``edge_m / EDGE_CELLS``.
        out_path: Where to write the masked CHM. Default: over ``chm_path``.
        keep_original: When masking in place, move the unmasked CHM to
            ``<chm_path>.unmasked`` instead of replacing it.
        strip_rows: Rows read at a time (rounded to whole CHM blocks).
        seed_share: Least gap share of a coarse cell for it to seed the band.

    Returns:
        What was masked.

    Raises:
        ValueError: If ``edge_m`` is negative, the CHM is not in a projected
            CRS while ``edge_m`` is positive, or ``keep_original`` is set with
            a separate ``out_path``.
        FileExistsError: If the kept original already exists, which means the
            CHM was masked before.
    """
    naip_path = Path(naip_path)
    chm_path = Path(chm_path)
    target = chm_path if out_path is None else Path(out_path)
    if edge_m < 0:
        raise ValueError(f"edge_m must not be negative, got {edge_m}")
    if keep_original and target != chm_path:
        raise ValueError("keep_original only applies when masking in place")
    original = None
    if keep_original:
        original = chm_path.with_name(chm_path.name + ORIGINAL_SUFFIX)
        if original.exists():
            raise FileExistsError(
                f"{original} exists, so {chm_path.name} looks masked already"
            )
    partial = target.with_name(target.name + PARTIAL_SUFFIX)

    with rasterio.open(naip_path) as naip_src, rasterio.open(chm_path) as chm_src:
        if edge_m > 0 and not chm_src.crs.is_projected:
            raise ValueError(f"{chm_path.name} is not in a projected CRS")
        pixel_m = abs(chm_src.transform.a)
        block_px = coarse_block_px(edge_m, pixel_m) if edge_m > 0 else 0
        strip_rows = _strip_rows(chm_src, strip_rows)
        n_pixels = chm_src.height * chm_src.width
        nodata = chm_nodata_value(chm_src.dtypes[0], chm_src.nodata)
        with _naip_on_chm_grid(naip_src, chm_src) as reader:
            packed, gap_pixels, counts = _scan_gaps(reader, strip_rows, block_px)
        if gap_pixels == 0:
            logger.info(
                "%s has no gap, %s left unchanged", naip_path.name, chm_path.name
            )
            return GapMaskResult(
                chm_path=chm_path,
                original_path=None,
                n_pixels=n_pixels,
                gap_pixels=0,
                edge_pixels=0,
                newly_masked=0,
                nodata=nodata,
                edge_m=edge_m,
                rewritten=False,
            )
        edge = None
        if counts is not None:
            cell_pixels = _pixels_per_cell(chm_src.height, chm_src.width, block_px)
            gap_share = counts / cell_pixels
            cell_m = block_px * pixel_m
            edge = edge_band(gap_share, cell_m, edge_m, seed_share=seed_share)
        try:
            edge_pixels, newly_masked = _write_masked(
                chm_src, partial, packed, edge, block_px, strip_rows, nodata
            )
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    if original is not None:
        os.replace(chm_path, original)
    os.replace(partial, target)
    result = GapMaskResult(
        chm_path=target,
        original_path=original,
        n_pixels=n_pixels,
        gap_pixels=gap_pixels,
        edge_pixels=edge_pixels,
        newly_masked=newly_masked,
        nodata=nodata,
        edge_m=edge_m,
        rewritten=True,
    )
    logger.info(
        "Masked %s: %.1f%% without NAIP imagery, %.1f%% within %g m of a gap, "
        "%.1f%% newly nodata",
        target.name,
        100 * result.gap_share,
        100 * result.edge_share,
        edge_m,
        100 * result.newly_masked_share,
    )
    return result
