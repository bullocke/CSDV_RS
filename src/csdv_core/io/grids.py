"""csdv_core.io.grids — small grid-metadata value object.

A ``GridSpec`` captures the affine transform, CRS, and pixel size of a raster
in a single hashable value, decoupling metric functions from rasterio handles
and ``Path`` arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GridSpec:
    """Grid metadata for a raster."""

    transform: Any
    crs: str
    pixel_size_m: float


def gridspec_from_raster(path: Path | str) -> GridSpec:
    """Build a :class:`GridSpec` from an on-disk raster."""
    import rasterio

    with rasterio.open(path) as src:
        return GridSpec(
            transform=src.transform,
            crs=str(src.crs) if src.crs is not None else "",
            pixel_size_m=float(abs(src.transform.a)),
        )
