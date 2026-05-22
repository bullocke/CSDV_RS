"""Unit tests for csdv_core.preprocess.windows.multiscale_grid."""

from __future__ import annotations

from rasterio.transform import Affine

from csdv_core.io.grids import GridSpec
from csdv_core.preprocess.windows import multiscale_grid


def _grid(pixel_size_m: float = 1.0) -> GridSpec:
    return GridSpec(
        transform=Affine(pixel_size_m, 0.0, 0.0, 0.0, -pixel_size_m, 0.0),
        crs="EPSG:5070",
        pixel_size_m=pixel_size_m,
    )


def test_multiscale_grid_emits_one_row_per_scale() -> None:
    """One yield per requested window size, in input order."""
    grid = _grid(pixel_size_m=1.0)
    sizes = [25.0, 50.0, 100.0]
    rows, cols = 200, 200
    out = list(multiscale_grid(grid, rows=rows, cols=cols, window_sizes_m=sizes))
    assert [w for w, _, _ in out] == sizes


def test_multiscale_grid_tile_counts() -> None:
    """Tile shape is the floor-divided count of windows at each scale."""
    grid = _grid(pixel_size_m=1.0)
    out = list(
        multiscale_grid(grid, rows=200, cols=200, window_sizes_m=[25.0, 50.0, 100.0])
    )
    shapes = {w: shape for w, shape, _ in out}
    assert shapes[25.0] == (200 // 25, 200 // 25)
    assert shapes[50.0] == (200 // 50, 200 // 50)
    assert shapes[100.0] == (200 // 100, 200 // 100)


def test_multiscale_grid_window_transform_pixel_size() -> None:
    """Each yielded transform has its pixel size set to the window size."""
    grid = _grid(pixel_size_m=0.6)
    out = list(multiscale_grid(grid, rows=500, cols=500, window_sizes_m=[25.0, 100.0]))
    for window_m, _, wt in out:
        assert abs(wt.a) == window_m
        assert abs(wt.e) == window_m
