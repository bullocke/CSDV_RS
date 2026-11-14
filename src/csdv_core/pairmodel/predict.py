"""csdv_core.pairmodel.predict: map loss probability over a whole grid unit.

Inputs: a checkpoint, the disturbance run folder of the unit (its grid,
years and forest layer) and the unit's NAIP mosaics under
``<data_root>/naip/<unit>/<year>/``.
Outputs: ``prob_<unit>.tif``, a Cloud Optimized GeoTIFF of uint8 percent on
the model's cells with one band per pair of consecutive dates (255 where a
date has no imagery), and ``prob_forest_<unit>.tif``, the same with cells
that were not forest at the earlier date set to 0.

The cells lie on the global 0.6 m lattice of the UTM zone, at multiples of
the cell size, which is where the label chips pool to cells. The unit's own
grid starts anywhere on the 0.6 m lattice, so the map covers the unit's
extent snapped outward, and the unit's 19.2 m cell layers are sampled at the
centre of each map cell.

The unit is read in blocks with a halo. Each date of a block is encoded
once and used for both of its intervals.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # noqa: F401  (GDAL has to load before torch, see __init__)
from affine import Affine

from csdv_core.label.lattice import GSD, Bounds
from csdv_core.pairmodel.data import NO_IMAGERY_SHARE, normalise
from csdv_core.pairmodel.metrics import to_percent

logger = logging.getLogger(__name__)

NODATA = 255

__all__ = [
    "Block",
    "PredGrid",
    "PredictResult",
    "blocks_of",
    "cells_at_centres",
    "predict_stack",
    "predict_unit",
    "unit_pred_grid",
]


@dataclass(frozen=True)
class PredGrid:
    """The cells of a map, on the global lattice.

    Attributes:
        epsg: EPSG code of the UTM zone.
        col0: Lattice pixel column of the left edge (``x / 0.6``), a
            multiple of ``cell_px``.
        row0: Lattice pixel row of the top edge (``y / 0.6``), a multiple of
            ``cell_px``. Rows count north, so the map runs down from it.
        cols: Number of cell columns.
        rows: Number of cell rows.
        cell_px: Pixels per cell side.
    """

    epsg: int
    col0: int
    row0: int
    cols: int
    rows: int
    cell_px: int

    @property
    def cell_m(self) -> float:
        return self.cell_px * GSD

    @property
    def transform(self) -> Affine:
        """Affine transform of the cells (north up)."""
        return Affine(
            self.cell_m,
            0.0,
            self.col0 * 6 / 10.0,
            0.0,
            -self.cell_m,
            self.row0 * 6 / 10.0,
        )

    @property
    def bounds(self) -> Bounds:
        left, top = self.col0 * 6 / 10.0, self.row0 * 6 / 10.0
        return (
            left,
            (self.row0 - self.rows * self.cell_px) * 6 / 10.0,
            (self.col0 + self.cols * self.cell_px) * 6 / 10.0,
            top,
        )

    def cell_window(self, bounds: Bounds) -> tuple[slice, slice]:
        """Row and column slices of the cells inside ``bounds`` (clipped)."""
        col_a = round(bounds[0] / GSD) - self.col0
        col_b = round(bounds[2] / GSD) - self.col0
        row_a = self.row0 - round(bounds[3] / GSD)
        row_b = self.row0 - round(bounds[1] / GSD)
        rows = slice(
            max(math.floor(row_a / self.cell_px), 0),
            min(math.ceil(row_b / self.cell_px), self.rows),
        )
        cols = slice(
            max(math.floor(col_a / self.cell_px), 0),
            min(math.ceil(col_b / self.cell_px), self.cols),
        )
        return rows, cols


@dataclass(frozen=True)
class Block:
    """A block of cells and the pixels read for it.

    Attributes:
        row: First cell row of the block's interior.
        col: First cell column.
        rows: Cell rows of the interior.
        cols: Cell columns of the interior.
        halo: Cells read around the interior on every side.
        bounds: Extent read, halo included, on the 0.6 m lattice.
    """

    row: int
    col: int
    rows: int
    cols: int
    halo: int
    bounds: Bounds


@dataclass
class PredictResult:
    """What a mapping run produced."""

    unit: str
    prob_path: Path
    forest_path: Path | None
    intervals: list[str]
    blocks: int
    seconds: float
    grid: PredGrid
    notes: list[str] = field(default_factory=list)


def unit_pred_grid(bounds: Bounds, epsg: int, cell_px: int) -> PredGrid:
    """The cells that cover ``bounds``, snapped outward to the global lattice."""
    left = math.floor(round(bounds[0] / GSD) / cell_px) * cell_px
    right = math.ceil(round(bounds[2] / GSD) / cell_px) * cell_px
    bottom = math.floor(round(bounds[1] / GSD) / cell_px) * cell_px
    top = math.ceil(round(bounds[3] / GSD) / cell_px) * cell_px
    return PredGrid(
        epsg, left, top, (right - left) // cell_px, (top - bottom) // cell_px, cell_px
    )


def blocks_of(grid: PredGrid, block_cells: int, halo_cells: int) -> list[Block]:
    """Blocks that tile the grid, north to south, west to east."""
    out = []
    for row in range(0, grid.rows, block_cells):
        for col in range(0, grid.cols, block_cells):
            rows = min(block_cells, grid.rows - row)
            cols = min(block_cells, grid.cols - col)
            left = grid.col0 + (col - halo_cells) * grid.cell_px
            top = grid.row0 - (row - halo_cells) * grid.cell_px
            width = (cols + 2 * halo_cells) * grid.cell_px
            height = (rows + 2 * halo_cells) * grid.cell_px
            bounds = (
                left * 6 / 10.0,
                (top - height) * 6 / 10.0,
                (left + width) * 6 / 10.0,
                top * 6 / 10.0,
            )
            out.append(Block(row, col, rows, cols, halo_cells, bounds))
    return out


def central_blocks(blocks: Sequence[Block], grid: PredGrid, n: int) -> list[Block]:
    """The ``n`` blocks nearest to the centre of the grid (for a quick run)."""

    def distance(block: Block) -> float:
        return math.hypot(
            block.row + block.rows / 2 - grid.rows / 2,
            block.col + block.cols / 2 - grid.cols / 2,
        )

    return sorted(blocks, key=distance)[:n]


def cells_at_centres(layer: np.ndarray, cell_transform: Affine, grid: PredGrid) -> Any:
    """A unit cell layer sampled at the centre of every map cell.

    Args:
        layer: ``(bands, rows, cols)`` on the unit's cell grid.
        cell_transform: Affine transform of that cell grid.
        grid: The map's cells.

    Returns:
        ``(values, inside)``: ``(bands, grid.rows, grid.cols)`` with the
        dtype of ``layer`` (zeros outside the unit's grid), and a bool
        ``(grid.rows, grid.cols)`` array of the cells inside it.
    """
    left, _, _, top = grid.bounds
    xs = left + (np.arange(grid.cols) + 0.5) * grid.cell_m
    ys = top - (np.arange(grid.rows) + 0.5) * grid.cell_m
    cols = np.floor((xs - cell_transform.c) / cell_transform.a).astype(int)
    rows = np.floor((cell_transform.f - ys) / -cell_transform.e).astype(int)
    in_c = (cols >= 0) & (cols < layer.shape[2])
    in_r = (rows >= 0) & (rows < layer.shape[1])
    inside = in_r[:, None] & in_c[None, :]
    values = np.zeros((layer.shape[0], grid.rows, grid.cols), dtype=layer.dtype)
    r = np.clip(rows, 0, layer.shape[1] - 1)
    c = np.clip(cols, 0, layer.shape[2] - 1)
    values[:] = layer[:, r[:, None], c[None, :]]
    values[:, ~inside] = 0
    return values, inside


def read_block(
    mosaics: Sequence[Sequence[str]], epsg: int, bounds: Bounds
) -> np.ndarray:
    """Every date of a block, ``(dates, 4, rows, cols)`` uint8 on the lattice."""
    from csdv_core.label.naip import warp_sources

    return np.stack([warp_sources(list(paths), epsg, bounds)[0] for paths in mosaics])


def predict_stack(
    model: Any,
    stack: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    *,
    amp: bool = False,
) -> np.ndarray:
    """Loss percent for every consecutive pair of a stack of dates.

    Args:
        model: A :class:`~csdv_core.pairmodel.model.SiamesePairNet` in
            evaluation mode.
        stack: ``(dates, 4, rows, cols)`` uint8, rows and columns multiples
            of the model's cell size.
        mean: Per-band mean of the checkpoint.
        std: Per-band standard deviation of the checkpoint.

    Returns:
        ``(dates - 1, rows / cell_px, cols / cell_px)`` uint8 percent, 255
        where either date lacks imagery over half of the cell or more.
    """
    import torch
    import torch.nn.functional as F

    cell_px = model.cell_px
    dates, _, height, width = stack.shape
    rows, cols = height // cell_px, width // cell_px
    # The encoder halves the image five times, so it needs multiples of 32.
    pad_h, pad_w = (-height) % 32, (-width) % 32
    features, has_imagery = [], []
    with torch.no_grad():
        for image in stack:
            valid = image.any(axis=0)
            share = valid[: rows * cell_px, : cols * cell_px].reshape(
                rows, cell_px, cols, cell_px
            )
            has_imagery.append(share.mean(axis=(1, 3)) > 1.0 - NO_IMAGERY_SHARE)
            x = torch.from_numpy(normalise(image, mean, std)).unsqueeze(0).to(device)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            with torch.autocast(device.type, enabled=amp):
                features.append(model.encode(x))
        out = np.full((dates - 1, rows, cols), NODATA, dtype="uint8")
        for k in range(dates - 1):
            with torch.autocast(device.type, enabled=amp):
                logits = model.decode(features[k], features[k + 1])
            prob = torch.sigmoid(logits.float())[0, 0, :rows, :cols].cpu().numpy()
            usable = has_imagery[k] & has_imagery[k + 1]
            out[k][usable] = to_percent(prob)[usable]
    return out


class _BlockReader:
    """Blocks of a unit as a dataset, so that loader workers do the reading."""

    def __init__(
        self, blocks: Sequence[Block], mosaics: Sequence[Sequence[str]], epsg: int
    ):
        self.blocks = list(blocks)
        self.mosaics = [list(paths) for paths in mosaics]
        self.epsg = epsg

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, item: int) -> tuple[int, Any]:
        import torch

        stack = read_block(self.mosaics, self.epsg, self.blocks[item].bounds)
        return item, torch.from_numpy(stack)


def predict_unit(
    checkpoint: Path,
    run_dir: Path,
    data_root: Path,
    out_dir: Path,
    *,
    block_cells: int = 128,
    halo_cells: int = 16,
    workers: int = 4,
    limit_blocks: int | None = None,
    device: str = "auto",
) -> PredictResult:
    """Map every consecutive pair of dates of a unit.

    Args:
        checkpoint: ``best.pt`` of a training run.
        run_dir: Disturbance run folder of the unit (the run itself or its
            exported copy), for the grid, the years and the forest layer.
        data_root: Root that holds ``naip/<unit>/<year>/``.
        out_dir: Folder for the rasters.
        block_cells: Cells per side of a block's interior.
        halo_cells: Cells read around a block and thrown away.
        workers: Processes that read blocks.
        limit_blocks: Map only this many blocks near the centre (a quick
            run). The other cells stay 255.

    Raises:
        FileNotFoundError: If a year has no mosaic.
    """
    from torch.utils.data import DataLoader

    from csdv_core.grid.export import write_cog
    from csdv_core.label.inputs import fill_forest_dips, load_unit_inputs
    from csdv_core.label.naip import local_mosaics
    from csdv_core.pairmodel.train import _device, load_checkpoint

    clock = time.perf_counter()
    unit = load_unit_inputs(Path(run_dir))
    target = _device(device)
    model, ckpt = load_checkpoint(Path(checkpoint), target)
    mean, std = np.asarray(ckpt["mean"], "float32"), np.asarray(ckpt["std"], "float32")
    mosaics = []
    for year in unit.years:
        paths = local_mosaics(Path(data_root), unit.unit, year)
        if not paths:
            raise FileNotFoundError(
                f"No NAIP mosaic for {unit.unit} {year} under {data_root}/naip"
            )
        mosaics.append([str(p) for p in paths])
    grid = unit_pred_grid(unit.grid.bounds, unit.epsg, model.cell_px)
    blocks = blocks_of(grid, block_cells, halo_cells)
    notes = []
    if limit_blocks is not None and limit_blocks < len(blocks):
        blocks = central_blocks(blocks, grid, limit_blocks)
        notes.append(f"quick run: {len(blocks)} blocks near the centre")
    dates = [unit.dates[y] for y in unit.years]
    intervals = [f"{a}_{b}" for a, b in zip(dates[:-1], dates[1:], strict=True)]
    logger.info(
        "%s: %d by %d cells of %.1f m, %d dates, %d blocks",
        unit.unit,
        grid.rows,
        grid.cols,
        grid.cell_m,
        len(dates),
        len(blocks),
    )
    prob = np.full((len(intervals), grid.rows, grid.cols), NODATA, dtype="uint8")
    loader = DataLoader(
        _BlockReader(blocks, mosaics, unit.epsg),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
    )
    amp = target.type == "cuda"
    for done, (item, stack) in enumerate(loader, start=1):
        block = blocks[int(item)]
        cells = predict_stack(model, stack.numpy(), mean, std, target, amp=amp)
        inner = cells[
            :,
            block.halo : block.halo + block.rows,
            block.halo : block.halo + block.cols,
        ]
        prob[
            :, block.row : block.row + block.rows, block.col : block.col + block.cols
        ] = inner
        if done % 10 == 0 or done == len(blocks):
            logger.info(
                "%d of %d blocks, %.0f s",
                done,
                len(blocks),
                time.perf_counter() - clock,
            )
    out_dir = Path(out_dir)
    crs = f"EPSG:{unit.epsg}"
    prob_path = write_cog(
        out_dir / f"prob_{unit.unit}.tif",
        prob,
        crs=crs,
        transform=grid.transform,
        nodata=NODATA,
        resampling="AVERAGE",
        descriptions=intervals,
    )
    forest_path = None
    if unit.has_layers:
        forest, inside = cells_at_centres(
            fill_forest_dips(unit.layer("forest")), unit.grid.cell_transform, grid
        )
        masked = prob.copy()
        for k in range(len(intervals)):
            band = masked[k]
            band[(band != NODATA) & inside & ~forest[k]] = 0
        forest_path = write_cog(
            out_dir / f"prob_forest_{unit.unit}.tif",
            masked,
            crs=crs,
            transform=grid.transform,
            nodata=NODATA,
            resampling="AVERAGE",
            descriptions=intervals,
        )
    else:
        notes.append("no forest layer on this machine, so no masked map")
    seconds = time.perf_counter() - clock
    logger.info("%s mapped in %.0f s: %s", unit.unit, seconds, prob_path)
    return PredictResult(
        unit.unit, prob_path, forest_path, intervals, len(blocks), seconds, grid, notes
    )
