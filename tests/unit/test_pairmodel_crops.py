"""Tests of csdv_core.pairmodel.crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.pairmodel.crops import read_boxes, snap_bounds, write_crops


def _mosaic(root: Path, unit: str, year: int) -> None:
    folder = root / "naip" / unit / str(year)
    folder.mkdir(parents=True)
    data = np.full((4, 200, 200), year % 100, dtype=np.uint8)
    with rasterio.open(
        folder / f"{unit}_{year}.tif",
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=4,
        dtype="uint8",
        crs="EPSG:26910",
        transform=from_origin(1000.0, 2000.0, 0.6, 0.6),
    ) as dst:
        dst.write(data)


def test_snap_bounds_widens_to_whole_pixels():
    assert snap_bounds((1000.1, 1999.0, 1010.0, 2005.3)) == pytest.approx(
        (1000.2 - 0.6, 1998.6, 1010.4, 2005.8)
    )


def test_crops_are_written_per_box_and_year(tmp_path):
    _mosaic(tmp_path, "q0000001", 2014)
    _mosaic(tmp_path, "q0000001", 2016)
    boxes = tmp_path / "boxes.csv"
    pd.DataFrame(
        {
            "unit": ["q0000001"],
            "box": ["F001"],
            "epsg": [26910],
            "minx": [1010.0],
            "miny": [1900.0],
            "maxx": [1070.0],
            "maxy": [1960.0],
            "years": ["2014 2016 2018"],
        }
    ).to_csv(boxes, index=False)
    report = write_crops(read_boxes(boxes), tmp_path, tmp_path / "out")
    assert [p.name for p in report.written] == ["F001_2014.tif", "F001_2016.tif"]
    assert report.missing == ["q0000001 2018"]
    with rasterio.open(tmp_path / "out" / "q0000001" / "F001_2016.tif") as src:
        assert src.count == 4 and src.res == pytest.approx((0.6, 0.6))
        assert src.read(1).mean() == pytest.approx(16)


def test_read_boxes_refuses_huge_boxes(tmp_path):
    path = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "unit": ["u"],
            "box": ["b"],
            "epsg": [26910],
            "minx": [0.0],
            "miny": [0.0],
            "maxx": [5000.0],
            "maxy": [10.0],
            "years": ["2014"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_boxes(path)
