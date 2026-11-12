"""Chips and windows on the global 0.6 m lattice."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.label.lattice import (
    GSD,
    Chip,
    cells_to_pixels,
    pixel_slices,
    window_for,
)


def _on_lattice(value: float) -> bool:
    return abs(value / GSD - round(value / GSD)) < 1e-6


def test_chip_edges_are_lattice_multiples() -> None:
    chip = Chip(26919, 977, 14650)
    x0, y0, x1, y1 = chip.bounds
    assert x0 == pytest.approx(977 * 307.2)
    assert x1 - x0 == pytest.approx(307.2)
    assert all(_on_lattice(v) for v in chip.bounds)
    assert chip.transform.c == x0 and chip.transform.f == y1


def test_window_covers_bounds_with_margin() -> None:
    bounds = (300100.0, 4499000.0, 300500.0, 4499200.0)
    window = window_for(bounds, window_id="w", unit="u", epsg=26919, margin_m=19.2)
    x0, y0, x1, y1 = window.bounds
    assert x0 <= bounds[0] - 19.2 and x1 >= bounds[2] + 19.2
    assert y0 <= bounds[1] - 19.2 and y1 >= bounds[3] + 19.2
    assert len(window.chips()) == window.nx * window.ny
    # North-west chip first.
    first = window.chips()[0]
    assert first.bounds[0] == x0 and first.bounds[3] == y1


def test_window_is_capped_around_the_anchor() -> None:
    bounds = (300000.0, 4490000.0, 305000.0, 4495000.0)
    anchor = (304000.0, 4491000.0)
    window = window_for(
        bounds, window_id="w", unit="u", epsg=26919, max_chips=3, anchor=anchor
    )
    assert (window.nx, window.ny) == (3, 3)
    x0, y0, x1, y1 = window.bounds
    assert x0 <= anchor[0] <= x1 and y0 <= anchor[1] <= y1


def test_stack_bounds_are_square_and_hold_the_window() -> None:
    window = window_for(
        (300100.0, 4499000.0, 300900.0, 4499100.0),
        window_id="w",
        unit="u",
        epsg=26919,
    )
    assert window.nx > window.ny
    sx0, sy0, sx1, sy1 = window.stack_bounds(64)
    assert sx1 - sx0 == pytest.approx(sy1 - sy0)
    assert sx1 - sx0 == pytest.approx(window.nx * 307.2 + 2 * 64 * GSD)
    x0, y0, x1, y1 = window.bounds
    assert sx0 <= x0 - 38.4 + 1e-6 and sx1 >= x1 + 38.4 - 1e-6
    assert sy0 <= y0 - 38.4 + 1e-6 and sy1 >= y1 + 38.4 - 1e-6
    assert all(_on_lattice(v) for v in (sx0, sy0, sx1, sy1))


def test_pixel_slices() -> None:
    outer = (0.0, 0.0, 60.0, 60.0)
    rows, cols = pixel_slices((6.0, 12.0, 18.0, 30.0), outer)
    assert (cols.start, cols.stop) == (10, 30)
    assert (rows.start, rows.stop) == (50, 80)
    with pytest.raises(ValueError):
        pixel_slices((-6.0, 0.0, 6.0, 6.0), outer)


def test_cells_map_to_pixels_by_position(label_grid) -> None:
    grid = label_grid
    origin = (grid.pixel_transform.c, grid.pixel_transform.f)
    cells = np.arange(40 * 40, dtype="float32").reshape(40, 40)
    # A chip-sized block whose corner is not on a cell corner.
    x0 = origin[0] + 10.2
    y1 = origin[1] - 4.8
    bounds = (x0, y1 - 60.0, x0 + 60.0, y1)
    out = cells_to_pixels(cells, grid, bounds)
    assert out.shape == (100, 100)
    # Pixel (0, 0) has its centre 10.5 m east and 5.1 m south of the origin.
    assert out[0, 0] == cells[0, 0]
    # 19.2 m east of the origin is 9.0 m into the block: pixel column 15.
    assert out[0, 14] == cells[0, 0] and out[0, 15] == cells[0, 1]
    # 19.2 m south of the origin is 14.4 m into the block: pixel row 24.
    assert out[23, 0] == cells[0, 0] and out[24, 0] == cells[1, 0]


def test_cells_outside_the_grid_are_filled(label_grid) -> None:
    grid = label_grid
    origin = (grid.pixel_transform.c, grid.pixel_transform.f)
    forest = np.ones((2, 40, 40), dtype=bool)
    bounds = (origin[0] - 6.0, origin[1] - 6.0, origin[0] + 6.0, origin[1] + 6.0)
    out = cells_to_pixels(forest, grid, bounds)
    assert out.shape == (2, 20, 20) and out.dtype == bool
    assert out[:, 10:, 10:].all() and not out[:, :10, :].any()
    values = cells_to_pixels(forest[0].astype("float32"), grid, bounds)
    assert np.isnan(values[0, 0]) and values[19, 19] == 1.0
