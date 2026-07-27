"""Tests for the map panel helpers, especially MultiPolygon outlines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon, box  # noqa: E402

from csdv_core.viz.maps import (  # noqa: E402
    chm_panel,
    draw_stand_outline,
    padded_bounds,
    read_rgb_window,
    rgb_panel,
    stretch_rgb,
)

TRANSFORM = from_origin(west=0.0, north=60.0, xsize=0.6, ysize=0.6)
SHAPE = (100, 100)


@pytest.fixture()
def naip(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(30, 210, size=(4, *SHAPE)).astype("uint8")
    path = tmp_path / "naip.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=4,
        dtype="uint8",
        transform=TRANSFORM,
        crs="EPSG:26916",
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture()
def chm(tmp_path: Path) -> Path:
    data = np.full(SHAPE, 1500, dtype="uint16")
    data[0:10, 0:10] = 65535
    path = tmp_path / "chm.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=1,
        dtype="uint16",
        transform=TRANSFORM,
        crs="EPSG:26916",
        nodata=65535,
    ) as dst:
        dst.write(data, 1)
    return path


def test_outline_handles_a_multipolygon():
    """The legacy helper read polygon.exterior and raised here. Every geometry
    in the calibration delivery is a MultiPolygon."""
    geometry = MultiPolygon([box(5.0, 5.0, 15.0, 15.0), box(25.0, 25.0, 35.0, 35.0)])
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(SHAPE))
    draw_stand_outline(ax, geometry, TRANSFORM)
    assert len(ax.lines) == 2  # one ring per part
    plt.close(fig)


def test_outline_draws_interior_rings():
    ring = Polygon(
        [(5, 5), (35, 5), (35, 35), (5, 35)],
        holes=[[(15, 15), (25, 15), (25, 25), (15, 25)]],
    )
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(SHAPE))
    draw_stand_outline(ax, ring, TRANSFORM)
    assert len(ax.lines) == 2  # exterior plus one hole
    plt.close(fig)


def test_outline_converts_map_coordinates_to_pixels():
    geometry = box(6.0, 24.0, 12.0, 30.0)
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(SHAPE))
    draw_stand_outline(ax, geometry, TRANSFORM)
    xs = ax.lines[0].get_xdata()
    ys = ax.lines[0].get_ydata()
    # x 6 m is column 10 at 0.6 m; y 30 m is row 50 from a north edge of 60 m.
    assert min(xs) == pytest.approx(10.0)
    assert min(ys) == pytest.approx(50.0)
    plt.close(fig)


def test_stretch_keeps_band_balance():
    """A joint stretch must not shift the colour of a uniform grey block."""
    block = np.dstack(
        [np.full((8, 8), 120.0), np.full((8, 8), 120.0), np.full((8, 8), 120.0)]
    )
    out = stretch_rgb(block)
    assert np.allclose(out[..., 0], out[..., 1])
    assert np.allclose(out[..., 1], out[..., 2])


def test_stretch_passes_through_data_already_in_zero_to_one():
    block = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert stretch_rgb(block) == pytest.approx(0.5)


def test_stretch_bounds_the_output():
    rng = np.random.default_rng(1)
    block = rng.integers(0, 255, size=(16, 16, 3)).astype(np.float32)
    out = stretch_rgb(block)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_read_rgb_window_clips_to_the_raster(naip: Path):
    block, _ = read_rgb_window(naip, (-100.0, -100.0, 1000.0, 1000.0))
    assert block.shape == (*SHAPE, 3)


def test_rgb_panel_draws_without_error(naip: Path):
    fig, ax = plt.subplots()
    rgb_panel(
        ax,
        naip,
        (5.0, 5.0, 40.0, 40.0),
        geometry=box(10.0, 10.0, 30.0, 30.0),
        title="2018",
    )
    assert ax.get_title() == "2018"
    assert ax.get_xticks().size == 0
    plt.close(fig)


def test_chm_panel_scales_centimetres_to_metres(chm: Path):
    fig, ax = plt.subplots()
    image, _ = chm_panel(ax, chm, (20.0, 20.0, 40.0, 40.0), scale=0.01)
    assert np.nanmax(image.get_array()) == pytest.approx(15.0)
    plt.close(fig)


def test_chm_panel_shows_nodata_as_nan(chm: Path):
    fig, ax = plt.subplots()
    image, _ = chm_panel(ax, chm, (0.0, 54.0, 6.0, 60.0), scale=0.01)
    assert np.isnan(np.asarray(image.get_array())).any()
    plt.close(fig)


def test_padded_bounds_widens_by_the_longer_side():
    bounds = padded_bounds(box(0.0, 0.0, 100.0, 50.0), pad_fraction=0.25)
    assert bounds == pytest.approx((-25.0, -25.0, 125.0, 75.0))


def test_padded_bounds_respects_a_minimum():
    bounds = padded_bounds(box(0.0, 0.0, 10.0, 10.0), pad_fraction=0.25, min_pad_m=20.0)
    assert bounds == pytest.approx((-20.0, -20.0, 30.0, 30.0))
