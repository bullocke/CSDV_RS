"""csdv_core.viz.relief — shaded relief for canopy height and terrain.

A canopy height model drawn with a colour ramp alone reads as a smooth stain.
Multiplying the colour by a hillshade of the same surface brings out individual
crowns, which is what makes a height map look like a forest.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CANOPY_VMAX_M = 20.0

_CANOPY_STOPS = (
    (0.00, "#14171c"),
    (0.06, "#1d2a2a"),
    (0.18, "#1f4a3a"),
    (0.38, "#2f7d4f"),
    (0.60, "#7fb557"),
    (0.80, "#d6de6f"),
    (1.00, "#fbf6d9"),
)

__all__ = [
    "CANOPY_VMAX_M",
    "canopy_cmap",
    "hillshade",
    "multidirectional_hillshade",
    "shade_rgb",
    "shaded_canopy_rgb",
]


def canopy_cmap() -> LinearSegmentedColormap:
    """Return the canopy height ramp: charcoal ground rising to pale cream.

    Lightness rises monotonically with height, so the ramp survives greyscale
    printing and tall riparian canopy stands out against a dark floodplain.
    """
    return LinearSegmentedColormap.from_list("csdv_canopy", list(_CANOPY_STOPS))


def hillshade(
    z: np.ndarray,
    pixel_size_m: float,
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Return Lambertian illumination of surface ``z`` in the range 0 to 1.

    Args:
        z: Surface heights with row 0 at the north edge. NaN is treated as 0.
        pixel_size_m: Ground size of one cell.
        azimuth_deg: Compass bearing of the sun, clockwise from north.
        altitude_deg: Sun elevation above the horizon.
        z_factor: Vertical exaggeration applied before shading.
    """
    surface = np.nan_to_num(np.asarray(z, dtype=np.float32)) * np.float32(z_factor)
    d_row, d_col = np.gradient(surface, pixel_size_m)
    # Rows run south, so the northward slope is the negative row gradient.
    dz_dx, dz_dy = d_col, -d_row
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    sun = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
    norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
    shade = (-dz_dx * sun[0] - dz_dy * sun[1] + sun[2]) / norm
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def multidirectional_hillshade(
    z: np.ndarray,
    pixel_size_m: float,
    *,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Return a hillshade lit mostly from the northwest with three fill lights.

    A single light leaves slopes facing away from it black. The fill lights
    keep detail there while the northwest key light sets the relief direction.
    """
    weights = {315.0: 0.55, 270.0: 0.15, 360.0: 0.15, 225.0: 0.15}
    out = np.zeros(np.shape(z), dtype=np.float32)
    for azimuth, weight in weights.items():
        out += weight * hillshade(
            z,
            pixel_size_m,
            azimuth_deg=azimuth,
            altitude_deg=altitude_deg,
            z_factor=z_factor,
        )
    return out


def shade_rgb(
    rgb: np.ndarray, shade: np.ndarray, *, strength: float = 0.7
) -> np.ndarray:
    """Multiply ``(H, W, 3)`` colour by a hillshade, normalised so flat ground is unchanged.

    ``strength`` of 0 returns the colour as is, 1 applies the full shade.
    """
    flat = float(np.nanmedian(shade)) or 1.0
    factor = 1.0 + strength * (shade / flat - 1.0)
    return np.clip(rgb[..., :3] * factor[..., None], 0.0, 1.0).astype(np.float32)


def shaded_canopy_rgb(
    chm: np.ndarray,
    pixel_m: float,
    *,
    dem: np.ndarray | None = None,
    vmax: float = CANOPY_VMAX_M,
    canopy_z_factor: float = 2.0,
    terrain_z_factor: float = 3.0,
) -> np.ndarray:
    """Colour a canopy height array and shade it with canopy and terrain relief."""
    rgb = canopy_cmap()(np.clip(np.nan_to_num(chm) / vmax, 0.0, 1.0))[..., :3]
    crowns = multidirectional_hillshade(chm, pixel_m, z_factor=canopy_z_factor)
    rgb = shade_rgb(rgb, crowns, strength=0.75)
    if dem is not None:
        terrain = hillshade(dem, pixel_m, altitude_deg=40.0, z_factor=terrain_z_factor)
        terrain = terrain / (float(np.nanmedian(terrain)) or 1.0)
        # Terrain relief lifts open ground only. Under canopy it would fight
        # the crown shading.
        openness = np.clip(1.0 - np.nan_to_num(chm) / 3.0, 0.0, 1.0)[..., None]
        lifted = np.clip(
            rgb * terrain[..., None] + 0.10 * (terrain[..., None] - 0.8), 0, 1
        )
        rgb = rgb * (1.0 - openness) + lifted * openness
    return rgb.astype(np.float32)
