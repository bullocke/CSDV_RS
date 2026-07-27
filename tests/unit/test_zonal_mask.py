"""Tests for stand windowing and the in-stand pixel mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box

from csdv_core.zonal.mask import read_stand_array, stand_window

# 30 x 30 m raster on 1 m pixels, origin at (0, 30).
TRANSFORM = from_origin(west=0.0, north=30.0, xsize=1.0, ysize=1.0)
SHAPE = (30, 30)


def test_square_stand_pixel_count_is_exact():
    """A 10 x 10 m square on 1 m pixels contains exactly 100 pixel centres."""
    sw = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE, stand_id="s")
    assert sw.n_inside == 100
    assert sw.area_m2 == pytest.approx(100.0)
    assert sw.pixel_size_m == pytest.approx(1.0)


def test_bbox_fill_is_one_for_a_grid_aligned_square():
    sw = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE)
    assert sw.bbox_fill_fraction == pytest.approx(1.0)


def test_bbox_fill_reports_a_thin_diagonal_stand():
    """A diagonal sliver fills only part of its bounding box."""
    sliver = Polygon([(2, 2), (28, 26), (28, 28), (2, 4)])
    sw = stand_window(sliver, TRANSFORM, SHAPE)
    assert 0.0 < sw.bbox_fill_fraction < 0.25
    assert sw.n_inside == int(sw.mask.sum())


def test_multipolygon_parts_are_both_included():
    parts = MultiPolygon([box(2.0, 2.0, 7.0, 7.0), box(20.0, 20.0, 25.0, 25.0)])
    sw = stand_window(parts, TRANSFORM, SHAPE)
    assert sw.n_inside == 50  # two 5 x 5 m squares
    # The bounding box spans both parts, so most of it is outside the stand.
    assert sw.bbox_fill_fraction < 0.15


def test_pixel_centre_rule_excludes_a_pixel_whose_centre_is_outside():
    """Growing the stand by less than half a pixel adds no pixels."""
    tight = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE)
    grown = stand_window(box(4.6, 4.6, 15.4, 15.4), TRANSFORM, SHAPE)
    assert grown.n_inside == tight.n_inside


def test_all_touched_includes_partly_covered_pixels():
    """Edges falling past a pixel centre drop that pixel under the centre rule."""
    edges = box(5.6, 5.6, 14.4, 14.4)
    centres = stand_window(edges, TRANSFORM, SHAPE)
    touched = stand_window(edges, TRANSFORM, SHAPE, all_touched=True)
    assert centres.n_inside == 64  # centres 6.5 .. 13.5, eight per side
    assert touched.n_inside == 100  # pixels 5 .. 14, ten per side


def test_padding_widens_the_block_but_not_the_mask():
    plain = stand_window(box(10.0, 10.0, 15.0, 15.0), TRANSFORM, SHAPE)
    padded = stand_window(box(10.0, 10.0, 15.0, 15.0), TRANSFORM, SHAPE, pad_px=4)
    assert padded.mask.shape[0] == plain.mask.shape[0] + 8
    assert padded.n_inside == plain.n_inside
    assert padded.bbox_fill_fraction < plain.bbox_fill_fraction


def test_padding_is_clipped_at_the_raster_edge():
    padded = stand_window(box(0.0, 0.0, 5.0, 5.0), TRANSFORM, SHAPE, pad_px=50)
    assert padded.mask.shape == SHAPE


def test_window_transform_places_the_block_correctly():
    sw = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE)
    # Origin of the block is the top-left of the stand bounding box.
    assert sw.transform.c == pytest.approx(5.0)
    assert sw.transform.f == pytest.approx(15.0)


def test_geometry_outside_the_raster_raises():
    with pytest.raises(ValueError, match="does not overlap"):
        stand_window(box(100.0, 100.0, 110.0, 110.0), TRANSFORM, SHAPE, stand_id="off")


def test_stand_smaller_than_a_pixel_raises():
    with pytest.raises(ValueError, match="no pixel centres"):
        stand_window(box(5.01, 5.01, 5.09, 5.09), TRANSFORM, SHAPE, stand_id="tiny")


@pytest.fixture()
def chm_raster(tmp_path: Path) -> Path:
    """A 30 x 30 raster of heights in centimetres with a nodata corner."""
    data = np.full(SHAPE, 1500, dtype="uint16")  # 15.00 m
    data[0:3, 0:3] = 65535  # nodata
    path = tmp_path / "chm_cm.tif"
    profile = {
        "driver": "GTiff",
        "height": SHAPE[0],
        "width": SHAPE[1],
        "count": 1,
        "dtype": "uint16",
        "transform": TRANSFORM,
        "crs": "EPSG:26916",
        "nodata": 65535,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_read_stand_array_applies_nodata_and_scale(chm_raster: Path):
    arr, sw = read_stand_array(
        chm_raster, box(5.0, 5.0, 15.0, 15.0), scale=0.01, stand_id="s"
    )
    assert arr.shape == sw.mask.shape
    assert arr[sw.mask] == pytest.approx(15.0)


def test_read_stand_array_masks_nodata_as_nan(chm_raster: Path):
    arr, sw = read_stand_array(chm_raster, box(0.0, 25.0, 5.0, 30.0), scale=0.01)
    # The top-left 3 x 3 block is nodata, so some in-stand pixels are NaN.
    assert np.isnan(arr[sw.mask]).sum() == 9
