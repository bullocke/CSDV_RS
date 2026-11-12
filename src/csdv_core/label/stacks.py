"""csdv_core.label.stacks: the multi-date NAIP stack of one window.

Inputs: a :class:`~csdv_core.label.lattice.Window` and its unit.
Outputs: one four-band GeoTIFF per year under a stack folder, with
``meta.json`` (dates, sources, native resolution). The stack covers the
window plus a collar and is read back as a :class:`WindowStack`.

A stack is written once and then reused, so the vision-model images, the
chips and the review sheets all come from the same pixels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.label.inputs import UnitInputs
from csdv_core.label.lattice import GSD, Bounds, Window, grid_transform, pixel_slices
from csdv_core.label.naip import NaipRead, read_naip

logger = logging.getLogger(__name__)

#: Smallest share of a window that a year must cover to be kept.
MIN_COVERAGE = 0.98
BAND_NAMES = ("R", "G", "B", "NIR")

__all__ = [
    "BAND_NAMES",
    "MIN_COVERAGE",
    "WindowStack",
    "build_stack",
    "load_stack",
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

    Args:
        unit: The window's unit.
        window: The window.
        folder: Stack folder (one per window). An existing ``meta.json``
            means the stack is done.
        collar_px: Pixels kept around the window.
        source: ``local`` to read the unit's mosaics. Default: each year's
            own source as recorded for the unit (``pc`` or ``rest``).
        data_root: Root of the local mosaics, for ``source="local"``.

    Returns:
        ``meta.json`` as a dict. ``failed`` lists years that could not be
        read or cover less than :data:`MIN_COVERAGE` of the window.
    """
    meta_path = folder / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    bounds = window.stack_bounds(collar_px)
    meta: dict[str, Any] = {
        "window_id": window.window_id,
        "unit": unit.unit,
        "epsg": window.epsg,
        "bounds": list(bounds),
        "years": {},
        "failed": {},
    }
    for year in unit.years:
        year_source = source or unit.sources.get(year, "pc")
        try:
            read: NaipRead = read_naip(
                year_source,
                unit=unit.unit,
                year=year,
                epsg=window.epsg,
                bounds=bounds,
                data_root=data_root,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s %d (%s): %s", window.window_id, year, year_source, exc)
            meta["failed"][str(year)] = repr(exc)[:200]
            continue
        if read.coverage < MIN_COVERAGE:
            meta["failed"][str(year)] = f"coverage {read.coverage:.2f}"
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
