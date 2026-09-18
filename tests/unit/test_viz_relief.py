"""Tests for csdv_core.viz.relief."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import rgb_to_hsv

from csdv_core.viz.relief import canopy_cmap, hillshade, shade_rgb


def test_flat_surface_shades_to_sine_of_altitude() -> None:
    shade = hillshade(np.zeros((20, 20)), 1.0, altitude_deg=30.0)
    assert np.allclose(shade, 0.5, atol=1e-6)


def test_slope_facing_the_sun_is_brighter() -> None:
    # Height rises to the east, so the slope faces west.
    z = np.tile(np.arange(30, dtype=float), (30, 1))
    west_sun = hillshade(z, 1.0, azimuth_deg=270.0)
    east_sun = hillshade(z, 1.0, azimuth_deg=90.0)
    assert west_sun.mean() > east_sun.mean()


def test_north_facing_slope_with_row_zero_at_north() -> None:
    # Height rises toward the south (larger row), so the slope faces north.
    z = np.tile(np.arange(30, dtype=float)[:, None], (1, 30))
    assert (
        hillshade(z, 1.0, azimuth_deg=0.0).mean()
        > hillshade(z, 1.0, azimuth_deg=180.0).mean()
    )


def test_canopy_cmap_value_is_monotonic() -> None:
    colours = canopy_cmap()(np.linspace(0, 1, 64))[:, :3]
    luminance = colours @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) > 0)
    assert rgb_to_hsv(colours[0])[2] < 0.15


def test_shade_rgb_leaves_flat_ground_unchanged() -> None:
    rgb = np.full((5, 5, 3), 0.4, dtype=np.float32)
    out = shade_rgb(rgb, np.full((5, 5), 0.7, dtype=np.float32))
    assert np.allclose(out, 0.4, atol=1e-6)
