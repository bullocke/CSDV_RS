"""csdv_core.disturbance.inputs: find site inputs and build the common grid.

Each NAIP-CHM year lives in ``$CSDV_DATA_ROOT/naip_chm/<site>/<year>/`` as one
or more GeoTIFFs (``CHM_<site>_<date>_0p6m.tif`` locally,
``NAIP_<site>_<date>_0p6m_chm.tif`` from CHPC inference). The reference year
fixes the CRS, pixel size and pixel alignment. The common extent is the
intersection of all years, snapped to reference pixels and trimmed to a whole
number of cells. :mod:`csdv_core.disturbance.windows` reads pixels on that
grid.

Inputs: site name, data roots, parameters. Outputs: :class:`SiteInputs` and
:class:`CellGrid`.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine
from rasterio.crs import CRS

from csdv_core.config._models import DisturbanceConfig
from csdv_core.io.paths import ProjectPaths, project_paths
from csdv_core.io.raster import resolve_scale_factor

logger = logging.getLogger(__name__)

#: Acquisition date embedded in input file names, as ``_YYYYMMDD_``.
DATE_PATTERN = re.compile(r"_(\d{8})_")
#: Glob for raster files inside a year folder (CHM and NAIP alike).
RASTER_GLOB = "*.tif"
#: Square metres per hectare.
M2_PER_HA = 10_000.0
#: Relative tolerance when comparing pixel sizes across years.
PIXEL_SIZE_RTOL = 1e-6
#: Tolerance in pixels when snapping extents to pixel edges (absorbs float error).
SNAP_TOLERANCE_PX = 1e-6


@dataclass(frozen=True)
class YearInput:
    """The input files of one NAIP year.

    Attributes:
        year: NAIP year (the folder name).
        chm_paths: CHM GeoTIFFs of the year, sorted by name. Several files are
            mosaicked on read.
        acquisition_dates: Date parsed from each CHM file name, aligned with
            ``chm_paths`` (None where the name holds no ``_YYYYMMDD_``).
        chm_dtype: Stored data type of the CHM files (e.g. ``"uint16"``).
        chm_scale: Multiplier from stored values to metres (resolved from the
            config or :func:`csdv_core.io.raster.resolve_scale_factor`).
        chm_nodata: Stored nodata value used on read (config override, else
            file metadata, else None with a logged warning).
        crs: CRS of the CHM files (identical across a year's files).
        pixel_m: CHM pixel size in metres.
        naip_paths: NAIP GeoTIFFs of the year, only filled when NDVI or an RGB
            backdrop is requested (empty tuple otherwise).
    """

    year: int
    chm_paths: tuple[Path, ...]
    acquisition_dates: tuple[datetime.date | None, ...]
    chm_dtype: str
    chm_scale: float
    chm_nodata: float | None
    crs: CRS
    pixel_m: float
    naip_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CellGrid:
    """The common 0.6 m pixel grid and the cell grid laid over it.

    Pixel row ``r`` and column ``c`` belong to cell ``(r // block_px,
    c // block_px)``. The pixel grid covers exactly ``cell_rows * block_px``
    by ``cell_cols * block_px`` pixels.

    Attributes:
        crs: Coordinate reference system shared by all years.
        pixel_transform: Affine transform of the pixel grid over the common
            extent (north up, origin at the top-left corner).
        pixel_m: Pixel size in metres (e.g. 0.6).
        block_px: Pixels per cell side, ``round(grain_m / pixel_m)``.
        cell_rows: Number of cell rows.
        cell_cols: Number of cell columns.
    """

    crs: CRS
    pixel_transform: Affine
    pixel_m: float
    block_px: int
    cell_rows: int
    cell_cols: int

    @property
    def cell_m(self) -> float:
        """Cell side in metres, ``pixel_m * block_px`` (19.2 for 32 x 0.6 m)."""
        return self.pixel_m * self.block_px

    @property
    def cell_ha(self) -> float:
        """Cell area in hectares."""
        return self.cell_m**2 / M2_PER_HA

    @property
    def cell_transform(self) -> Affine:
        """Affine transform of the cell grid (same origin, ``block_px`` larger)."""
        return self.pixel_transform @ Affine.scale(self.block_px)

    @property
    def cell_shape(self) -> tuple[int, int]:
        """Cell grid shape ``(cell_rows, cell_cols)``."""
        return (self.cell_rows, self.cell_cols)

    @property
    def pixel_shape(self) -> tuple[int, int]:
        """Pixel grid shape ``(cell_rows * block_px, cell_cols * block_px)``."""
        return (self.cell_rows * self.block_px, self.cell_cols * self.block_px)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Extent as ``(left, bottom, right, top)`` in CRS units (rasterio order)."""
        pixel_rows, pixel_cols = self.pixel_shape
        left = self.pixel_transform.c
        top = self.pixel_transform.f
        right = left + pixel_cols * self.pixel_transform.a
        bottom = top + pixel_rows * self.pixel_transform.e
        return (left, bottom, right, top)

    def extent(self) -> tuple[float, float, float, float]:
        """Extent as ``(left, right, bottom, top)`` for ``matplotlib.imshow``."""
        left, bottom, right, top = self.bounds
        return (left, right, bottom, top)


@dataclass(frozen=True)
class SiteInputs:
    """Everything the pipeline needs to read a site.

    Attributes:
        site: Site name (folder name under ``naip_chm/``).
        chm_root: ``<data_root>/naip_chm/<site>``.
        naip_root: ``<data_root>/naip/<site>`` (may not exist when NAIP is
            not needed).
        years: One :class:`YearInput` per year, in ascending year order.
        grid: The common grid.
        reference_year: Year that defined CRS, pixel size and alignment.
    """

    site: str
    chm_root: Path
    naip_root: Path
    years: tuple[YearInput, ...]
    grid: CellGrid
    reference_year: int

    @property
    def year_list(self) -> tuple[int, ...]:
        """The years in order, ``tuple(y.year for y in years)``."""
        return tuple(year_input.year for year_input in self.years)

    def year_input(self, year: int) -> YearInput:
        """Return the :class:`YearInput` of ``year``.

        Args:
            year: NAIP year.

        Returns:
            The year's inputs.

        Raises:
            KeyError: If ``year`` is not part of the run.
        """
        for candidate in self.years:
            if candidate.year == year:
                return candidate
        raise KeyError(f"Year {year} is not part of this run: {self.year_list}")


def parse_acquisition_date(path: Path) -> datetime.date | None:
    """Parse the ``_YYYYMMDD_`` acquisition date from a file name.

    Works for both ``CHM_<site>_20190612_0p6m.tif`` and
    ``NAIP_<site>_20190612_0p6m_chm.tif``. The first match in ``path.name``
    is used.

    Args:
        path: Input file path.

    Returns:
        The date, or None when the name holds no valid ``_YYYYMMDD_`` token.
    """
    match = DATE_PATTERN.search(path.name)
    if match is None:
        return None
    token = match.group(1)
    try:
        parsed = datetime.datetime.strptime(token, "%Y%m%d")
    except ValueError:
        return None
    return parsed.date()


def discover_years(
    chm_root: Path,
    *,
    years: Sequence[int] | None = None,
    naip_root: Path | None = None,
    chm_scale: float | None = None,
    chm_nodata: float | None = None,
) -> list[YearInput]:
    """List the year folders under ``chm_root`` that hold at least one ``.tif``.

    Year folders are subfolders whose name is an integer. Each year's CHM files
    are opened once for metadata (dtype, nodata, CRS, pixel size). All files of
    a year must share dtype and CRS. A warning is logged when no nodata value
    is known for a year.

    Args:
        chm_root: ``<data_root>/naip_chm/<site>``.
        years: Years to keep (None = every year found). Order does not matter,
            the result is ascending.
        naip_root: ``<data_root>/naip/<site>``. When given, each year's NAIP
            files (``naip_root/<year>/*.tif``) are listed into ``naip_paths``.
        chm_scale: Explicit stored-to-metres multiplier (None = 0.01 for
            integer dtypes, 1.0 for floats).
        chm_nodata: Explicit nodata value (None = file metadata).

    Returns:
        Year inputs in ascending year order.

    Raises:
        FileNotFoundError: If ``chm_root`` does not exist, a requested year has
            no CHM files, or ``naip_root`` is given and a year has no NAIP file.
        ValueError: If fewer than two years remain, or a year mixes dtypes or
            CRSs across its files.
    """
    if not chm_root.is_dir():
        raise FileNotFoundError(f"CHM site folder not found: {chm_root}")
    found_paths: dict[int, list[Path]] = {}
    for folder in sorted(chm_root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        tifs = _list_rasters(folder)
        if tifs:
            found_paths[int(folder.name)] = tifs
    if years is None:
        selected_years = sorted(found_paths)
    else:
        requested_years = set(years)
        missing_set = requested_years - found_paths.keys()
        missing_years = sorted(missing_set)
        if missing_years:
            raise FileNotFoundError(
                f"No CHM GeoTIFF for year(s) {missing_years} under {chm_root} "
                f"(found {sorted(found_paths)})"
            )
        selected_years = sorted(requested_years)
    if len(selected_years) < 2:
        raise ValueError(
            f"Need at least two CHM years under {chm_root}, got {selected_years}"
        )
    year_inputs = []
    for year in selected_years:
        naip_paths: list[Path] = []
        if naip_root is not None:
            naip_folder = naip_root / str(year)
            naip_paths = _list_rasters(naip_folder)
            if not naip_paths:
                raise FileNotFoundError(f"No NAIP GeoTIFF for {year} in {naip_folder}")
        year_input = _read_year_metadata(
            year,
            found_paths[year],
            naip_paths,
            chm_scale=chm_scale,
            chm_nodata=chm_nodata,
        )
        year_inputs.append(year_input)
    return year_inputs


def _list_rasters(folder: Path) -> list[Path]:
    """List the GeoTIFFs of a folder, sorted by name.

    Args:
        folder: Year folder (may not exist).

    Returns:
        Sorted ``*.tif`` paths, empty when the folder is missing.
    """
    if not folder.is_dir():
        return []
    matches = folder.glob(RASTER_GLOB)
    return sorted(path for path in matches if path.is_file())


def _read_year_metadata(
    year: int,
    chm_paths: Sequence[Path],
    naip_paths: Sequence[Path],
    *,
    chm_scale: float | None,
    chm_nodata: float | None,
) -> YearInput:
    """Open a year's CHM files once and collect their metadata.

    Args:
        year: NAIP year.
        chm_paths: The year's CHM files, sorted by name.
        naip_paths: The year's NAIP files (may be empty).
        chm_scale: Explicit stored-to-metres multiplier, or None.
        chm_nodata: Explicit nodata value, or None.

    Returns:
        The year input.

    Raises:
        ValueError: If the files mix dtypes or CRSs.
    """
    dtypes = []
    crs_values = []
    nodata_values = []
    pixel_sizes = []
    for path in chm_paths:
        with rasterio.open(path) as src:
            dtypes.append(src.dtypes[0])
            crs_values.append(src.crs)
            nodata_values.append(src.nodata)
            pixel_sizes.append(src.res[0])
    dtype_set = set(dtypes)
    unique_dtypes = sorted(dtype_set)
    if len(unique_dtypes) > 1:
        raise ValueError(f"Year {year} mixes CHM dtypes {unique_dtypes}")
    first_crs = crs_values[0]
    for path, crs in zip(chm_paths, crs_values, strict=True):
        if crs != first_crs:
            raise ValueError(
                f"Year {year} mixes CRSs: {chm_paths[0].name} has {first_crs}, "
                f"{path.name} has {crs}"
            )
    dtype = unique_dtypes[0]
    scale = resolve_scale_factor(chm_scale, dtype)
    unique_nodata = set(nodata_values)
    nodata: float | None = None
    if chm_nodata is not None:
        nodata = float(chm_nodata)
    elif len(unique_nodata) > 1:
        logger.warning(
            "Year %d: CHM files disagree on nodata %s, each file uses its own",
            year,
            nodata_values,
        )
    elif nodata_values[0] is None:
        logger.warning(
            "Year %d: CHM files have no nodata value and none was given "
            "(set inputs.chm_nodata), so only uncovered pixels count as missing",
            year,
        )
    else:
        nodata = float(nodata_values[0])
    dates = tuple(parse_acquisition_date(path) for path in chm_paths)
    logger.info(
        "Year %d: %d CHM file(s), dtype %s, scale %g, nodata %s, NAIP files %d",
        year,
        len(chm_paths),
        dtype,
        scale,
        nodata,
        len(naip_paths),
    )
    return YearInput(
        year=year,
        chm_paths=tuple(chm_paths),
        acquisition_dates=dates,
        chm_dtype=dtype,
        chm_scale=scale,
        chm_nodata=nodata,
        crs=first_crs,
        pixel_m=float(pixel_sizes[0]),
        naip_paths=tuple(naip_paths),
    )


def build_common_grid(
    year_inputs: Sequence[YearInput],
    grain_m: float,
    *,
    reference_year: int | None = None,
) -> CellGrid:
    """Build the intersection grid of all years on the reference year's pixels.

    The reference year (first year when None) defines CRS, pixel size and
    pixel alignment. The extent is the intersection of every year's bounds
    (all files of a year unioned), shrunk inward to reference pixel edges and
    then trimmed at the right and bottom to a whole number of cells.
    ``block_px = round(grain_m / pixel_m)``.

    Args:
        year_inputs: Years from :func:`discover_years`.
        grain_m: Requested cell size in metres.
        reference_year: Year that defines the pixel grid.

    Returns:
        The common grid.

    Raises:
        ValueError: If CRSs or pixel sizes differ across years (the message
            names the years), ``reference_year`` is not among the years,
            ``grain_m`` is smaller than one pixel, or the intersection holds
            no whole cell.
    """
    year_numbers = [year_input.year for year_input in year_inputs]
    if reference_year is None:
        reference = year_inputs[0]
    elif reference_year in year_numbers:
        reference_index = year_numbers.index(reference_year)
        reference = year_inputs[reference_index]
    else:
        raise ValueError(
            f"reference_year {reference_year} is not among the years {year_numbers}"
        )
    _check_same_crs_and_pixel(year_inputs, reference)

    with rasterio.open(reference.chm_paths[0]) as reference_src:
        reference_transform = reference_src.transform
    if reference_transform.b != 0 or reference_transform.d != 0:
        raise ValueError(f"Year {reference.year}: rotated rasters are not supported")
    if reference_transform.e >= 0:
        raise ValueError(f"Year {reference.year}: raster is not north up")
    pixel_m = reference_transform.a
    if grain_m < pixel_m * (1.0 - PIXEL_SIZE_RTOL):
        raise ValueError(f"grain_m {grain_m} is smaller than one pixel ({pixel_m} m)")
    block_px = round(grain_m / pixel_m)
    cell_m = block_px * pixel_m
    if not math.isclose(cell_m, grain_m, rel_tol=PIXEL_SIZE_RTOL):
        logger.info(
            "grain_m %.3f m is not a whole number of %.3f m pixels, cells are "
            "%d px = %.3f m",
            grain_m,
            pixel_m,
            block_px,
            cell_m,
        )

    left, bottom, right, top = _intersect_bounds(year_inputs)
    origin_x = reference_transform.c
    origin_y = reference_transform.f
    first_col = math.ceil((left - origin_x) / pixel_m - SNAP_TOLERANCE_PX)
    stop_col = math.floor((right - origin_x) / pixel_m + SNAP_TOLERANCE_PX)
    first_row = math.ceil((origin_y - top) / pixel_m - SNAP_TOLERANCE_PX)
    stop_row = math.floor((origin_y - bottom) / pixel_m + SNAP_TOLERANCE_PX)
    cell_rows = (stop_row - first_row) // block_px
    cell_cols = (stop_col - first_col) // block_px
    if cell_rows < 1 or cell_cols < 1:
        raise ValueError(
            f"The intersection of years {year_numbers} holds no whole "
            f"{cell_m:.2f} m cell (bounds {left}, {bottom}, {right}, {top})"
        )
    grid_left = origin_x + first_col * reference_transform.a
    grid_top = origin_y + first_row * reference_transform.e
    pixel_transform = Affine(
        reference_transform.a, 0.0, grid_left, 0.0, reference_transform.e, grid_top
    )
    grid = CellGrid(
        crs=reference.crs,
        pixel_transform=pixel_transform,
        pixel_m=float(pixel_m),
        block_px=block_px,
        cell_rows=cell_rows,
        cell_cols=cell_cols,
    )
    logger.info(
        "Common grid: reference year %d, %d x %d cells of %d px (%.2f m), bounds %s",
        reference.year,
        cell_rows,
        cell_cols,
        block_px,
        grid.cell_m,
        grid.bounds,
    )
    return grid


def _check_same_crs_and_pixel(
    year_inputs: Sequence[YearInput], reference: YearInput
) -> None:
    """Check that every year shares the reference year's CRS and pixel size.

    Args:
        year_inputs: All years of the run.
        reference: The reference year.

    Raises:
        ValueError: Naming the first year that differs.
    """
    for year_input in year_inputs:
        if year_input.crs != reference.crs:
            raise ValueError(
                f"CRS of year {year_input.year} ({year_input.crs}) differs from "
                f"reference year {reference.year} ({reference.crs}). Reproject "
                "the CHMs to one CRS first."
            )
        same_pixel = math.isclose(
            year_input.pixel_m, reference.pixel_m, rel_tol=PIXEL_SIZE_RTOL
        )
        if not same_pixel:
            raise ValueError(
                f"Pixel size of year {year_input.year} ({year_input.pixel_m} m) "
                f"differs from reference year {reference.year} "
                f"({reference.pixel_m} m)"
            )


def _intersect_bounds(
    year_inputs: Sequence[YearInput],
) -> tuple[float, float, float, float]:
    """Intersect the years' bounds, each year being the union of its files.

    Args:
        year_inputs: All years of the run.

    Returns:
        ``(left, bottom, right, top)`` of the intersection (may be empty, with
        ``left >= right`` or ``bottom >= top``).
    """
    lefts, bottoms, rights, tops = [], [], [], []
    for year_input in year_inputs:
        file_bounds = []
        for path in year_input.chm_paths:
            with rasterio.open(path) as src:
                file_bounds.append(src.bounds)
        lefts.append(min(bounds.left for bounds in file_bounds))
        bottoms.append(min(bounds.bottom for bounds in file_bounds))
        rights.append(max(bounds.right for bounds in file_bounds))
        tops.append(max(bounds.top for bounds in file_bounds))
    return (max(lefts), max(bottoms), min(rights), min(tops))


def discover_site_inputs(
    site: str,
    params: DisturbanceConfig,
    *,
    paths: ProjectPaths | None = None,
    need_naip: bool = False,
) -> SiteInputs:
    """Discover a site's years and build its grid from the run parameters.

    Uses ``params.inputs.years``, ``chm_scale``, ``chm_nodata``,
    ``params.grid.grain_m`` and ``reference_year``.

    Args:
        site: Site name.
        params: Run parameters.
        paths: Project roots (None = :func:`csdv_core.io.paths.project_paths`).
        need_naip: List NAIP files too (``d_ndvi`` input or RGB backdrop).

    Returns:
        The site inputs with their common grid.

    Raises:
        FileNotFoundError: See :func:`discover_years`.
        ValueError: See :func:`discover_years` and :func:`build_common_grid`.
    """
    if paths is None:
        paths = project_paths()
    chm_root = paths.naip_chm_site_dir(site)
    naip_root = paths.naip_site_dir(site)
    input_params = params.inputs
    year_inputs = discover_years(
        chm_root,
        years=input_params.years,
        naip_root=naip_root if need_naip else None,
        chm_scale=input_params.chm_scale,
        chm_nodata=input_params.chm_nodata,
    )
    grid = build_common_grid(
        year_inputs,
        params.grid.grain_m,
        reference_year=params.grid.reference_year,
    )
    reference_year = params.grid.reference_year
    if reference_year is None:
        reference_year = year_inputs[0].year
    return SiteInputs(
        site=site,
        chm_root=chm_root,
        naip_root=naip_root,
        years=tuple(year_inputs),
        grid=grid,
        reference_year=reference_year,
    )


def grid_to_dict(grid: CellGrid) -> dict[str, Any]:
    """Serialize a grid to JSON-ready values.

    Args:
        grid: Common grid.

    Returns:
        Dict with ``crs`` (WKT string), ``pixel_transform`` (the six GDAL
        geotransform coefficients from ``Affine.to_gdal()``), ``pixel_m``,
        ``block_px``, ``cell_rows``, ``cell_cols``.
    """
    coefficients = list(grid.pixel_transform.to_gdal())
    return {
        "crs": grid.crs.to_wkt(),
        "pixel_transform": coefficients,
        "pixel_m": grid.pixel_m,
        "block_px": grid.block_px,
        "cell_rows": grid.cell_rows,
        "cell_cols": grid.cell_cols,
    }


def grid_from_dict(data: dict[str, Any]) -> CellGrid:
    """Inverse of :func:`grid_to_dict`.

    Args:
        data: Dict from :func:`grid_to_dict`.

    Returns:
        The grid.

    Raises:
        KeyError: If a field is missing.
    """
    crs = CRS.from_wkt(data["crs"])
    pixel_transform = Affine.from_gdal(*data["pixel_transform"])
    return CellGrid(
        crs=crs,
        pixel_transform=pixel_transform,
        pixel_m=float(data["pixel_m"]),
        block_px=int(data["block_px"]),
        cell_rows=int(data["cell_rows"]),
        cell_cols=int(data["cell_cols"]),
    )


__all__ = [
    "DATE_PATTERN",
    "M2_PER_HA",
    "CellGrid",
    "SiteInputs",
    "YearInput",
    "build_common_grid",
    "discover_site_inputs",
    "discover_years",
    "grid_from_dict",
    "grid_to_dict",
    "parse_acquisition_date",
]
