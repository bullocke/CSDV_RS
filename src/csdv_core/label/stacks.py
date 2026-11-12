"""csdv_core.label.stacks: the multi-date NAIP stack of one window.

Inputs: a :class:`~csdv_core.label.lattice.Window` and its unit.
Outputs: one four-band GeoTIFF per year under a stack folder, with
``meta.json`` (dates, sources, native resolution). The stack covers the
window plus a collar and is read back as a :class:`WindowStack`.

A stack is written once and then reused, so the vision-model images, the
chips and the review sheets all come from the same pixels.

Beside the imagery the folder holds ``cell_ndvi.tif``: the NDVI of every
19.2 m cell of the unit that the stack touches, one band per year. It is a
few kilobytes, so it can be copied to a machine that does not hold the
imagery, where the label rules use it to tell ground that was never green
(roofs, rock) from forest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.disturbance.inputs import CellGrid
from csdv_core.label.inputs import UnitInputs
from csdv_core.label.lattice import (
    GSD,
    Bounds,
    Window,
    cell_index,
    cells_to_pixels,
    grid_transform,
    pixel_slices,
)
from csdv_core.label.naip import NaipRead, read_naip

logger = logging.getLogger(__name__)

#: Smallest share that a year must cover to be kept, measured over the part
#: of the window (with its collar) that lies inside the unit's cell grid.
MIN_COVERAGE = 0.98
BAND_NAMES = ("R", "G", "B", "NIR")
#: File of the cell NDVI inside a stack folder.
CELL_NDVI_FILE = "cell_ndvi.tif"
#: NDVI is stored as int16 after multiplying by this.
NDVI_SCALE = 10_000
NDVI_NODATA = -32768
#: Smallest share of a cell's pixels that must hold imagery for its NDVI.
NDVI_MIN_SHARE = 0.9

__all__ = [
    "BAND_NAMES",
    "CELL_NDVI_FILE",
    "MIN_COVERAGE",
    "CellNdvi",
    "WindowStack",
    "build_stack",
    "inside_unit",
    "load_cell_ndvi",
    "load_stack",
    "unit_cell_ndvi",
    "write_cell_ndvi",
    "write_four_band",
]


@dataclass
class WindowStack:
    """Every year of a window.

    Attributes:
        bounds: Extent of the arrays on the 0.6 m lattice.
        epsg: EPSG code of the lattice.
        years: NAIP years, ascending.
        dates: Acquisition date of each year (``YYYY-MM-DD``).
        native_m: Native resolution of each year, metres (None if unknown).
        data: ``(years, 4, rows, cols)`` uint8.
    """

    bounds: Bounds
    epsg: int
    years: list[int]
    dates: list[str]
    native_m: list[float | None]
    data: np.ndarray

    def crop(self, bounds: Bounds) -> np.ndarray:
        """``(years, 4, rows, cols)`` view of ``bounds`` inside the stack."""
        rows, cols = pixel_slices(bounds, self.bounds)
        return self.data[:, :, rows, cols]


def write_four_band(
    path: Path, data: np.ndarray, epsg: int, bounds: Bounds, tags: dict[str, str]
) -> None:
    """Write a ``(4, rows, cols)`` uint8 block with named bands.

    The fourth band is near-infrared. GDAL marks the fourth band of an RGB
    file as alpha unless told otherwise, which makes viewers treat it as
    transparency, so the file is written as plain grey bands.
    """
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=4,
        dtype="uint8",
        crs=f"EPSG:{epsg}",
        transform=grid_transform(bounds),
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        photometric="MINISBLACK",
    ) as dst:
        dst.write(data)
        dst.descriptions = BAND_NAMES
        dst.update_tags(**tags)


def inside_unit(unit: UnitInputs, bounds: Bounds) -> np.ndarray:
    """Pixels of ``bounds`` that lie inside the unit's cell grid.

    A window on the edge of a unit reaches past the unit's mosaic. Ground
    outside the cell grid has no cell layer and is never labelled, so
    imagery is not required there.
    """
    cells = np.ones(unit.grid.cell_shape, dtype=bool)
    return cells_to_pixels(cells, unit.grid, bounds)


def _check_same_window(
    meta: dict[str, Any], unit: UnitInputs, bounds: Bounds, folder: Path
) -> None:
    same_unit = meta.get("unit") == unit.unit
    same_ground = np.allclose(meta.get("bounds", []) or [np.nan] * 4, bounds, atol=1e-6)
    if not (same_unit and same_ground):
        raise ValueError(
            f"{folder} holds the stack of another window (unit "
            f"{meta.get('unit')}, bounds {meta.get('bounds')}). The sample was "
            "probably drawn again after rendering. Use a new label folder."
        )


def build_stack(
    unit: UnitInputs,
    window: Window,
    folder: Path,
    *,
    collar_px: int = 64,
    source: str | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Fetch every year of a window once and return its metadata.

    A year is kept when its imagery covers the part of the stack that lies
    inside the unit's cell grid. The mosaics of a unit end at its edge, so a
    window there has pixels without imagery (0 in every band) beyond it. The
    remote sources could fill that ground from the neighbouring quads, but
    it is left empty for every source, so that a window looks the same
    whichever machine rendered it.

    Args:
        unit: The window's unit.
        window: The window.
        folder: Stack folder (one per window). An existing ``meta.json``
            without failed years means the stack is done. Failed years are
            tried again.
        collar_px: Pixels kept around the window.
        source: ``local`` to read the unit's mosaics. Default: each year's
            own source as recorded for the unit (``pc`` or ``rest``).
        data_root: Root of the local mosaics, for ``source="local"``.

    Returns:
        ``meta.json`` as a dict. ``inside_unit`` is the share of the stack
        inside the unit's cell grid. ``failed`` lists years that could not
        be read or cover less than :data:`MIN_COVERAGE` of that part.

    Raises:
        ValueError: If the folder holds the stack of another window. Window
            IDs follow the order of the sample, so a sample drawn again into
            a folder that was already rendered would otherwise reuse the
            imagery of other ground.
    """
    meta_path = folder / "meta.json"
    bounds = window.stack_bounds(collar_px)
    done: dict[str, Any] = {}
    if meta_path.exists():
        earlier = json.loads(meta_path.read_text())
        _check_same_window(earlier, unit, bounds, folder)
        if not earlier.get("failed"):
            return earlier
        done = {
            year: info
            for year, info in earlier["years"].items()
            if (folder / f"{year}.tif").exists()
        }
    inside = inside_unit(unit, bounds)
    keep = None if inside.all() else inside
    meta: dict[str, Any] = {
        "window_id": window.window_id,
        "unit": unit.unit,
        "epsg": window.epsg,
        "bounds": list(bounds),
        "inside_unit": round(float(inside.mean()), 4),
        "clipped_to_unit": True,
        "years": done,
        "failed": {},
    }
    for year in unit.years:
        if str(year) in done:
            continue
        year_source = source or unit.sources.get(year, "pc")
        try:
            read: NaipRead = read_naip(
                year_source,
                unit=unit.unit,
                year=year,
                epsg=window.epsg,
                bounds=bounds,
                data_root=data_root,
                keep=keep,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s %d (%s): %s", window.window_id, year, year_source, exc)
            meta["failed"][str(year)] = repr(exc)[:200]
            continue
        covered = read.data.any(axis=0)
        coverage = float(covered[inside].mean()) if inside.any() else 0.0
        if coverage < MIN_COVERAGE:
            meta["failed"][str(year)] = (
                f"covers {coverage:.2f} of the part inside the unit "
                f"({read.coverage:.2f} of the whole stack)"
            )
            continue
        write_four_band(
            folder / f"{year}.tif",
            read.data,
            window.epsg,
            bounds,
            {"date": read.date, "source": read.source},
        )
        meta["years"][str(year)] = {
            "date": read.date,
            "source": read.source,
            "native_m": read.native_m,
            "item_ids": list(read.item_ids),
            "coverage": round(read.coverage, 4),
            "date_rule": read.date_rule,
            "quad": read.quad,
            "date_share": read.date_share,
            "other_dates": list(read.other_dates),
            "mixed_dates": read.mixed_dates,
        }
    folder.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def load_stack(folder: Path) -> WindowStack:
    """Read a stack written by :func:`build_stack`.

    Raises:
        FileNotFoundError: If the folder has no ``meta.json``.
    """
    import rasterio

    meta = json.loads((folder / "meta.json").read_text())
    years = sorted(int(y) for y in meta["years"])
    arrays = []
    for year in years:
        with rasterio.open(folder / f"{year}.tif") as src:
            arrays.append(src.read())
    info = [meta["years"][str(y)] for y in years]
    return WindowStack(
        bounds=tuple(meta["bounds"]),
        epsg=int(meta["epsg"]),
        years=years,
        dates=[i["date"] for i in info],
        native_m=[i.get("native_m") for i in info],
        data=np.stack(arrays) if arrays else np.zeros((0, 4, 0, 0), "uint8"),
    )


def stack_gsd_note(stack: WindowStack) -> str:
    """Short text of the native resolutions, for sheet headers."""
    parts = [
        f"{y}: {m:g} m" if m else f"{y}: ?"
        for y, m in zip(stack.years, stack.native_m, strict=True)
    ]
    return ", ".join(parts) + f" (stored at {GSD} m)"


@dataclass
class CellNdvi:
    """NDVI of the unit's cells under a window, one layer per year.

    Attributes:
        years: NAIP years, ascending.
        row0: Row of the unit's cell grid at the top of ``values``.
        col0: Column of the unit's cell grid at the left of ``values``.
        values: ``(years, rows, cols)`` float32, NaN where a cell has too
            little imagery.
    """

    years: list[int]
    row0: int
    col0: int
    values: np.ndarray

    def on_unit(self, cell_shape: tuple[int, int]) -> np.ndarray:
        """``values`` placed on the unit's whole cell grid (NaN elsewhere)."""
        out = np.full((len(self.years), *cell_shape), np.nan, dtype="float32")
        rows, cols = self.values.shape[1:]
        out[:, self.row0 : self.row0 + rows, self.col0 : self.col0 + cols] = self.values
        return out


def unit_cell_ndvi(
    grid: CellGrid, stack: WindowStack, *, min_share: float = NDVI_MIN_SHARE
) -> CellNdvi:
    """NDVI of every cell of the unit that the stack touches, per year.

    The NDVI of a cell is taken from the sums of its near-infrared and red
    digital numbers, ``(NIR - R) / (NIR + R)``. That weights bright pixels
    more than a mean of pixel values would, so sunlit crowns count for more
    than their shadows. The digital numbers are not calibrated between
    flights, which is why the label rules use the largest value over all
    years and not the value of one year.

    Args:
        grid: Cell grid of the unit.
        stack: Imagery of the window.
        min_share: Smallest share of a cell's pixels that must hold imagery.

    Returns:
        The cells from the first to the last row and column that the stack
        reaches inside the grid. Empty when the stack lies outside it.
    """
    rows, cols = cell_index(grid, stack.bounds)
    ok_r = (rows >= 0) & (rows < grid.cell_rows)
    ok_c = (cols >= 0) & (cols < grid.cell_cols)
    n_years = len(stack.years)
    if not ok_r.any() or not ok_c.any():
        return CellNdvi(list(stack.years), 0, 0, np.zeros((n_years, 0, 0), "float32"))
    row0, col0 = int(rows[ok_r].min()), int(cols[ok_c].min())
    n_rows = int(rows[ok_r].max()) - row0 + 1
    n_cols = int(cols[ok_c].max()) - col0 + 1
    inside = ok_r[:, None] & ok_c[None, :]
    index = ((rows - row0)[:, None] * n_cols + (cols - col0)[None, :])[inside]
    size = n_rows * n_cols
    cell_px = round(grid.cell_transform.a / GSD) ** 2
    values = np.full((n_years, size), np.nan, dtype="float32")
    for t in range(n_years):
        block = stack.data[t]
        seen = block.any(axis=0)[inside]
        red = block[0][inside].astype("float64") * seen
        nir = block[3][inside].astype("float64") * seen
        count = np.bincount(index, weights=seen, minlength=size)
        sum_red = np.bincount(index, weights=red, minlength=size)
        sum_nir = np.bincount(index, weights=nir, minlength=size)
        total = sum_nir + sum_red
        enough = (count >= min_share * cell_px) & (total > 0)
        values[t, enough] = (sum_nir[enough] - sum_red[enough]) / total[enough]
    return CellNdvi(
        list(stack.years), row0, col0, values.reshape(n_years, n_rows, n_cols)
    )


def write_cell_ndvi(unit: UnitInputs, stack: WindowStack, folder: Path) -> Path:
    """Write the cell NDVI of a stack to ``cell_ndvi.tif`` in its folder.

    One int16 band per year (NDVI times :data:`NDVI_SCALE`), georeferenced on
    the unit's cell grid.
    """
    import rasterio
    from affine import Affine

    ndvi = unit_cell_ndvi(unit.grid, stack)
    cell = unit.grid.cell_transform
    transform = Affine(
        cell.a,
        0.0,
        cell.c + ndvi.col0 * cell.a,
        0.0,
        cell.e,
        cell.f + ndvi.row0 * cell.e,
    )
    scaled = np.where(
        np.isnan(ndvi.values), NDVI_NODATA, np.round(ndvi.values * NDVI_SCALE)
    ).astype("int16")
    path = folder / CELL_NDVI_FILE
    folder.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=max(scaled.shape[1], 1),
        width=max(scaled.shape[2], 1),
        count=len(ndvi.years),
        dtype="int16",
        crs=f"EPSG:{stack.epsg}",
        transform=transform,
        nodata=NDVI_NODATA,
        compress="DEFLATE",
    ) as dst:
        if scaled.size:
            dst.write(scaled)
        dst.descriptions = tuple(str(y) for y in ndvi.years)
        dst.update_tags(row0=ndvi.row0, col0=ndvi.col0, scale=NDVI_SCALE)
    return path


def load_cell_ndvi(folder: Path) -> CellNdvi | None:
    """Read ``cell_ndvi.tif`` of a stack folder, or None when there is none."""
    import rasterio

    path = folder / CELL_NDVI_FILE
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        raw = src.read()
        years = [int(d) for d in src.descriptions]
        tags = src.tags()
    values = np.where(raw == NDVI_NODATA, np.nan, raw / NDVI_SCALE).astype("float32")
    return CellNdvi(years, int(tags["row0"]), int(tags["col0"]), values)
