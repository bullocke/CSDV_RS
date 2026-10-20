"""csdv_core.disturbance.layers: per-year cell means, forest and monitoring masks.

Each year's 0.6 m CHM is read in full-width strips on the common grid and
aggregated to cells of ``block_px x block_px`` pixels: mean height, canopy
cover (share of valid pixels above ``forest.height_m``) and the share of
valid pixels. Cells with too few valid pixels are NaN. NDVI is aggregated
the same way from NAIP red and NIR when the ``d_ndvi`` input is used. The
change layer is the difference between consecutive years, and rules D7 and
D7s also use the two-step changes of :func:`bracket_changes`.

Unlike ``01_layers.py``, CHM layers are not masked where NAIP has no data,
because the core pipeline reads NAIP only for NDVI.

Inputs: :class:`~csdv_core.disturbance.inputs.SiteInputs`, parameters.
Outputs: :class:`CellLayers` year stacks ``(T, cell_rows, cell_cols)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csdv_core.config._models import DisturbanceConfig, DisturbanceInput
from csdv_core.disturbance.inputs import CellGrid, SiteInputs
from csdv_core.disturbance.windows import read_window
from csdv_core.masks.forest import block_view

logger = logging.getLogger(__name__)

#: Default strip height in cells for :func:`build_cell_layers`.
DEFAULT_STRIP_CELLS = 16
#: State layer that each change input differences.
INPUT_STATE_LAYER: dict[str, str] = {
    "d_chm_h": "chm_height",
    "d_chm_cov": "chm_cover",
    "d_ndvi": "ndvi",
}
#: Year-stack fields of :class:`CellLayers`, one ``layers/<name>.tif`` each.
LAYER_NAMES: tuple[str, ...] = (
    "chm_height",
    "chm_cover",
    "valid_share",
    "forest",
    "ndvi",
)


@dataclass(frozen=True)
class BlockAggregate:
    """Cell statistics of one CHM strip or scene.

    All arrays are float32 ``(rows // block_px, cols // block_px)``.

    Attributes:
        height: Mean CHM height (m) over valid pixels. NaN where
            ``valid_share < min_valid_share``.
        cover: Share of valid pixels with height strictly above ``height_m``,
            in [0, 1]. NaN where ``valid_share < min_valid_share``.
        valid_share: Share of the cell's pixels that are finite, in [0, 1].
            Never NaN.
    """

    height: np.ndarray
    cover: np.ndarray
    valid_share: np.ndarray


@dataclass(frozen=True)
class CellLayers:
    """Per-year cell layers of a site.

    Every stack is ``(T, cell_rows, cell_cols)`` with index ``t`` matching
    ``years[t]``. Float stacks are float32 with NaN for missing cells.

    Attributes:
        years: NAIP years, ascending, length ``T``.
        chm_height: Mean CHM height (m).
        chm_cover: Canopy cover share (pixels above ``forest.height_m``).
        valid_share: Share of valid CHM pixels per cell (never NaN).
        forest: bool, ``chm_cover >= forest.min_cover`` (NaN cover is False).
        ndvi: Mean NDVI, or None when NDVI was not computed.
    """

    years: tuple[int, ...]
    chm_height: np.ndarray
    chm_cover: np.ndarray
    valid_share: np.ndarray
    forest: np.ndarray
    ndvi: np.ndarray | None = None

    @property
    def cell_shape(self) -> tuple[int, int]:
        """``(cell_rows, cell_cols)``."""
        return (self.chm_height.shape[1], self.chm_height.shape[2])


def aggregate_block(
    chm: np.ndarray,
    block_px: int,
    *,
    height_m: float,
    min_valid_share: float,
) -> BlockAggregate:
    """Aggregate a CHM array to cells (mean height, cover, valid share).

    Matches ``01_layers._aggregate``: cover counts pixels with
    ``height > height_m`` among valid pixels, and cells with a valid share
    below ``min_valid_share`` get NaN height and cover.

    Args:
        chm: float32 ``(rows, cols)`` heights in metres, NaN = nodata. Both
            dimensions must be multiples of ``block_px``.
        block_px: Pixels per cell side.
        height_m: Canopy height threshold (m).
        min_valid_share: Least valid share for a defined cell.

    Returns:
        The cell statistics.

    Raises:
        ValueError: If a dimension is not a multiple of ``block_px``.
    """
    _check_block_shape(chm.shape, block_px)
    blocks = block_view(chm, block_px)
    is_valid = np.isfinite(blocks)
    valid_count = is_valid.sum(-1)
    filled = np.where(is_valid, blocks, 0.0)
    height_sum = filled.sum(-1)
    is_canopy = (filled > height_m) & is_valid
    canopy_count = is_canopy.sum(-1)
    pixels_per_cell = block_px * block_px
    with np.errstate(invalid="ignore", divide="ignore"):
        height = height_sum / valid_count
        cover = canopy_count / valid_count
    valid_share = valid_count / pixels_per_cell
    too_sparse = valid_share < min_valid_share
    height[too_sparse] = np.nan
    cover[too_sparse] = np.nan
    return BlockAggregate(
        height=height.astype("float32"),
        cover=cover.astype("float32"),
        valid_share=valid_share.astype("float32"),
    )


def aggregate_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    block_px: int,
    *,
    min_valid_share: float,
) -> np.ndarray:
    """Aggregate NAIP red and NIR to cell-mean NDVI.

    A pixel is valid where both bands are finite and ``red + nir > 0`` (NAIP
    fill is 0). Pixel NDVI is ``(nir - red) / max(nir + red, 1)``, as in
    ``01_layers.py``. The cell value is the mean over valid pixels.

    Args:
        red: float32 ``(rows, cols)`` raw digital numbers, NaN allowed.
        nir: float32 ``(rows, cols)`` on the same grid.
        block_px: Pixels per cell side.
        min_valid_share: Least valid share for a defined cell.

    Returns:
        float32 ``(rows // block_px, cols // block_px)``, NaN where the valid
        share is below ``min_valid_share``.

    Raises:
        ValueError: If shapes differ or are not multiples of ``block_px``.
    """
    if red.shape != nir.shape:
        raise ValueError(f"Red {red.shape} and NIR {nir.shape} shapes differ")
    _check_block_shape(red.shape, block_px)
    band_sum = nir + red
    is_valid = np.isfinite(red) & np.isfinite(nir) & (band_sum > 0)
    denominator = np.maximum(band_sum, 1.0)
    with np.errstate(invalid="ignore"):
        pixel_ndvi = (nir - red) / denominator
    filled = np.where(is_valid, pixel_ndvi, 0.0)
    valid_blocks = block_view(is_valid, block_px)
    valid_count = valid_blocks.sum(-1)
    ndvi_blocks = block_view(filled, block_px)
    ndvi_sum = ndvi_blocks.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_ndvi = ndvi_sum / valid_count
    pixels_per_cell = block_px * block_px
    valid_share = valid_count / pixels_per_cell
    too_sparse = valid_share < min_valid_share
    cell_ndvi[too_sparse] = np.nan
    return cell_ndvi.astype("float32")


def _check_block_shape(shape: tuple[int, ...], block_px: int) -> None:
    """Check that a 2-D shape is a whole number of blocks.

    Args:
        shape: Array shape ``(rows, cols)``.
        block_px: Pixels per cell side.

    Raises:
        ValueError: If a dimension is not a multiple of ``block_px``.
    """
    rows, cols = shape
    if rows % block_px or cols % block_px:
        raise ValueError(f"Shape {shape} is not a multiple of block_px {block_px}")


def forest_mask(chm_cover: np.ndarray, *, min_cover: float) -> np.ndarray:
    """Cell forest mask, ``chm_cover >= min_cover`` with NaN as non-forest.

    Args:
        chm_cover: float32 cover share, any shape (usually ``(T, rows, cols)``).
        min_cover: Least canopy share for forest.

    Returns:
        bool array of the same shape.
    """
    with np.errstate(invalid="ignore"):
        is_forest = chm_cover >= min_cover
    return is_forest


def build_cell_layers(
    site_inputs: SiteInputs,
    params: DisturbanceConfig,
    *,
    include_ndvi: bool = False,
    strip_cells: int = DEFAULT_STRIP_CELLS,
) -> CellLayers:
    """Read every year in strips and build the cell layers.

    For each year, strips of ``strip_cells * block_px`` pixel rows are read
    with :func:`csdv_core.disturbance.windows.read_window` (CHM scaled to
    metres) and aggregated with :func:`aggregate_block`. The result does not
    depend on ``strip_cells``. Uses ``params.forest.height_m``,
    ``params.forest.min_cover`` and ``params.grid.min_valid_share``. NAIP bands
    are ``params.inputs.naip_red_band`` and ``naip_nir_band``.

    Args:
        site_inputs: Discovered inputs and grid.
        params: Run parameters.
        include_ndvi: Also read NAIP and fill ``ndvi``.
        strip_cells: Strip height in cells (memory control only).

    Returns:
        The layers for all years of ``site_inputs``.

    Raises:
        ValueError: If ``include_ndvi`` is True and a year has no NAIP files,
            or ``strip_cells < 1``.
    """
    if strip_cells < 1:
        raise ValueError(f"strip_cells must be at least 1, got {strip_cells}")
    if include_ndvi:
        for year_input in site_inputs.years:
            if not year_input.naip_paths:
                raise ValueError(
                    f"NDVI needs NAIP files, but year {year_input.year} has none"
                )
    grid = site_inputs.grid
    year_count = len(site_inputs.years)
    stack_shape = (year_count, grid.cell_rows, grid.cell_cols)
    chm_height = np.full(stack_shape, np.nan, dtype="float32")
    chm_cover = np.full(stack_shape, np.nan, dtype="float32")
    valid_share = np.zeros(stack_shape, dtype="float32")
    ndvi = None
    if include_ndvi:
        ndvi = np.full(stack_shape, np.nan, dtype="float32")
    for year_index, year_input in enumerate(site_inputs.years):
        for first_cell_row in range(0, grid.cell_rows, strip_cells):
            strip_cell_rows = min(strip_cells, grid.cell_rows - first_cell_row)
            row_off = first_cell_row * grid.block_px
            n_rows = strip_cell_rows * grid.block_px
            chm = read_window(
                year_input.chm_paths,
                grid,
                row_off,
                n_rows,
                scale=year_input.chm_scale,
                nodata=year_input.chm_nodata,
            )
            aggregate = aggregate_block(
                chm,
                grid.block_px,
                height_m=params.forest.height_m,
                min_valid_share=params.grid.min_valid_share,
            )
            strip_slice = slice(first_cell_row, first_cell_row + strip_cell_rows)
            chm_height[year_index, strip_slice] = aggregate.height
            chm_cover[year_index, strip_slice] = aggregate.cover
            valid_share[year_index, strip_slice] = aggregate.valid_share
            if ndvi is not None:
                strip_ndvi = _strip_ndvi(
                    year_input.naip_paths, grid, row_off, n_rows, params
                )
                ndvi[year_index, strip_slice] = strip_ndvi
        _log_year(year_input.year, chm_cover[year_index], params.forest.min_cover)
    forest = forest_mask(chm_cover, min_cover=params.forest.min_cover)
    return CellLayers(
        years=site_inputs.year_list,
        chm_height=chm_height,
        chm_cover=chm_cover,
        valid_share=valid_share,
        forest=forest,
        ndvi=ndvi,
    )


def _strip_ndvi(
    naip_paths: tuple[Path, ...],
    grid: CellGrid,
    row_off: int,
    n_rows: int,
    params: DisturbanceConfig,
) -> np.ndarray:
    """Read NAIP red and NIR for one strip and aggregate them to cell NDVI.

    NAIP is read with each file's own nodata metadata (none for typical
    NAIP), so validity comes from ``red + nir > 0`` as in ``01_layers.py``.

    Args:
        naip_paths: NAIP GeoTIFFs of one year.
        grid: Common grid.
        row_off: First pixel row of the strip.
        n_rows: Strip height in pixels.
        params: Run parameters (band numbers, ``min_valid_share``).

    Returns:
        float32 cell NDVI of the strip.
    """
    red_band = params.inputs.naip_red_band
    nir_band = params.inputs.naip_nir_band
    red = read_window(naip_paths, grid, row_off, n_rows, band=red_band)
    nir = read_window(naip_paths, grid, row_off, n_rows, band=nir_band)
    return aggregate_ndvi(
        red, nir, grid.block_px, min_valid_share=params.grid.min_valid_share
    )


def _log_year(year: int, cover: np.ndarray, min_cover: float) -> None:
    """Log the forest share of a year's valid cells.

    Args:
        year: NAIP year.
        cover: float32 cell cover of the year, NaN = undefined.
        min_cover: Least canopy share for forest.
    """
    is_defined = np.isfinite(cover)
    defined_count = int(is_defined.sum())
    is_forest = forest_mask(cover, min_cover=min_cover)
    forest_count = int(is_forest.sum())
    forest_share = forest_count / max(defined_count, 1)
    logger.info(
        "Year %d: %d of %d defined cells are forest (%.3f)",
        year,
        forest_count,
        defined_count,
        forest_share,
    )


def monitored_mask(forest: np.ndarray) -> np.ndarray:
    """Cells watched for loss at each date: forest at the previous date.

    Args:
        forest: bool ``(T, rows, cols)``.

    Returns:
        bool ``(T, rows, cols)`` with ``out[0]`` all False and
        ``out[t] = forest[t - 1]`` for ``t >= 1``.
    """
    monitored = np.zeros_like(forest)
    monitored[1:] = forest[:-1]
    return monitored


def population_mask(forest: np.ndarray) -> np.ndarray:
    """Cells that form the standardization population at each date.

    Same as :func:`monitored_mask` except ``out[0] = forest[0]`` (the
    research ``02_standardize._population``). Index 0 only matters for state
    inputs, since change layers are NaN at index 0.

    Args:
        forest: bool ``(T, rows, cols)``.

    Returns:
        bool ``(T, rows, cols)``.
    """
    population = monitored_mask(forest)
    population[0] = forest[0]
    return population


def change_layer(layers: CellLayers, input_name: DisturbanceInput) -> np.ndarray:
    """Difference of a state layer between consecutive years.

    Args:
        layers: Cell layers.
        input_name: ``d_chm_h``, ``d_chm_cov`` or ``d_ndvi`` (see
            :data:`INPUT_STATE_LAYER`).

    Returns:
        float32 ``(T, rows, cols)``: ``out[0]`` all NaN and
        ``out[t] = state[t] - state[t - 1]`` (NaN where either is NaN).

    Raises:
        ValueError: If ``input_name`` is ``d_ndvi`` and ``layers.ndvi`` is None,
            or the name is unknown.
    """
    state = _state_layer(layers, input_name)
    change = np.full(state.shape, np.nan, dtype="float32")
    change[1:] = state[1:] - state[:-1]
    return change


def _state_layer(layers: CellLayers, input_name: DisturbanceInput) -> np.ndarray:
    """State layer that a change input differences.

    Args:
        layers: Cell layers.
        input_name: ``d_chm_h``, ``d_chm_cov`` or ``d_ndvi``.

    Returns:
        float32 ``(T, rows, cols)`` state stack.

    Raises:
        ValueError: If the name is unknown or the layer is missing.
    """
    if input_name not in INPUT_STATE_LAYER:
        raise ValueError(
            f"Unknown input {input_name!r}, expected one of {list(INPUT_STATE_LAYER)}"
        )
    layer_name = INPUT_STATE_LAYER[input_name]
    state = getattr(layers, layer_name)
    if state is None:
        raise ValueError(f"Input {input_name} needs the {layer_name} layer")
    return state


def bracket_changes(
    layers: CellLayers, input_name: DisturbanceInput
) -> tuple[np.ndarray, np.ndarray]:
    """Two-step change layers that bracket each interval (rules D7 and D7s).

    For the interval ending at date ``t``, the back change compares the
    later image with the image before the interval's earlier image, and the
    forward change compares the image after the interval with its earlier
    image. A spike in the earlier image or a dip in the later image shows up
    in the one-step change but not in these two.

    Args:
        layers: Cell layers.
        input_name: ``d_chm_h``, ``d_chm_cov`` or ``d_ndvi``.

    Returns:
        ``(back, forward)``, float32 ``(T, rows, cols)`` each.
        ``back[t] = state[t] - state[t - 2]`` for ``t >= 2`` and
        ``forward[t] = state[t + 1] - state[t - 1]`` for ``1 <= t <= T - 2``.
        Every other date, and cells where either state is NaN, are NaN.

    Raises:
        ValueError: If the name is unknown or the layer is missing.
    """
    state = _state_layer(layers, input_name)
    back = np.full(state.shape, np.nan, dtype="float32")
    forward = np.full(state.shape, np.nan, dtype="float32")
    back[2:] = state[2:] - state[:-2]
    forward[1:-1] = state[2:] - state[:-2]
    return back, forward


__all__ = [
    "DEFAULT_STRIP_CELLS",
    "INPUT_STATE_LAYER",
    "LAYER_NAMES",
    "BlockAggregate",
    "CellLayers",
    "aggregate_block",
    "aggregate_ndvi",
    "bracket_changes",
    "build_cell_layers",
    "change_layer",
    "forest_mask",
    "monitored_mask",
    "population_mask",
]
