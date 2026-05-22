"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture()
def synthetic_chm(tmp_path: Path) -> Path:
    """Write a 50x50 float32 CHM raster (EPSG:5070) and return its path."""
    rng = np.random.default_rng(seed=0)
    data = rng.uniform(0.0, 25.0, size=(50, 50)).astype("float32")
    # Punch some nodata holes.
    data[0:5, 0:5] = -9999.0
    transform = from_origin(west=1_500_000.0, north=2_000_000.0, xsize=0.6, ysize=0.6)
    path = tmp_path / "synthetic_chm.tif"
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": "EPSG:5070",
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path
