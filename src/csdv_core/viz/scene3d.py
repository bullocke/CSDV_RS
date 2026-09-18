"""csdv_core.viz.scene3d — oblique 3D renders of a height surface with PyVista.

The renders run off screen on the CPU, so they work on a machine without a
usable GPU. Colour is always applied as a texture rather than per vertex. That
lets the drape keep the full image resolution while the mesh stays coarse
enough to fit in memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CameraPreset", "build_skirt", "build_surface", "render_view"]


@dataclass(frozen=True)
class CameraPreset:
    """An orbit camera described the way a map reader would.

    Attributes:
        look_azimuth_deg: Compass bearing the camera looks toward.
        elevation_deg: Camera angle above the horizon (90 is nadir).
        distance_factor: Camera distance as a multiple of the scene diagonal.
        fov_deg: Vertical field of view. Small values flatten perspective.
        target_frac: Look-at point as (x, y) fractions of the scene extent,
            measured from the west and south edges.
    """

    look_azimuth_deg: float = 0.0
    elevation_deg: float = 30.0
    distance_factor: float = 1.0
    fov_deg: float = 28.0
    target_frac: tuple[float, float] = (0.5, 0.5)


def build_surface(z: np.ndarray, transform: Any) -> Any:
    """Return a ``pv.StructuredGrid`` of ``z`` with texture coordinates.

    Coordinates are metres from the southwest corner of the grid rather than
    map coordinates, because VTK renders in float32 and UTM northings lose
    sub-metre precision there.
    """
    import pyvista as pv

    height, width = z.shape
    res_x, res_y = abs(transform.a), abs(transform.e)
    xs = (np.arange(width, dtype=np.float32) + 0.5) * res_x
    ys = (height - np.arange(height, dtype=np.float32) - 0.5) * res_y
    xx, yy = np.meshgrid(xs, ys)
    grid = pv.StructuredGrid(xx, yy, np.nan_to_num(z).astype(np.float32))
    u = np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))
    v = np.tile(np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None], (1, width))
    # StructuredGrid stores points in Fortran order of the input arrays.
    grid.active_texture_coordinates = np.column_stack(
        [u.ravel(order="F"), v.ravel(order="F")]
    )
    return grid


def build_skirt(z: np.ndarray, transform: Any, base_z: float) -> Any:
    """Return vertical walls from the edge of the surface down to ``base_z``."""
    import pyvista as pv

    height, width = z.shape
    res_x, res_y = abs(transform.a), abs(transform.e)
    rows = np.r_[
        np.zeros(width - 1),
        np.arange(height - 1),
        np.full(width - 1, height - 1),
        np.arange(height - 1, 0, -1),
    ].astype(int)
    cols = np.r_[
        np.arange(width - 1),
        np.full(height - 1, width - 1),
        np.arange(width - 1, 0, -1),
        np.zeros(height - 1),
    ].astype(int)
    x = (cols + 0.5) * res_x
    y = (height - rows - 0.5) * res_y
    top = np.column_stack([x, y, np.nan_to_num(z[rows, cols])])
    bottom = np.column_stack([x, y, np.full(len(x), base_z)])
    n = len(x)
    i = np.arange(n)
    j = (i + 1) % n
    faces = np.column_stack([np.full(n, 4), i, j, j + n, i + n]).ravel()
    return pv.PolyData(np.vstack([top, bottom]).astype(np.float32), faces)


def _sun_position(
    center: np.ndarray, radius: float, azimuth_deg: float, altitude_deg: float
) -> tuple[float, float, float]:
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    offset = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    return tuple(center + 10.0 * radius * offset)


def render_view(
    surface: Any,
    texture_rgb: np.ndarray,
    out_path: Path | str,
    *,
    camera: CameraPreset,
    window_size: tuple[int, int] = (3840, 2160),
    skirt: Any | None = None,
    sun_azimuth_deg: float = 300.0,
    sun_altitude_deg: float = 35.0,
    sun_intensity: float = 1.0,
    ambient: float = 0.35,
    ssao_radius_m: float | None = 12.0,
    shadows: bool = False,
    background: tuple[str, str] = ("#0d1015", "#232a33"),
) -> np.ndarray:
    """Render ``surface`` draped with ``texture_rgb`` and return the RGB image.

    Args:
        surface: Mesh from :func:`build_surface`.
        texture_rgb: ``(H, W, 3)`` drape, float 0 to 1 or uint8, north up.
        out_path: PNG to write.
        camera: View description.
        window_size: Output size in pixels.
        skirt: Optional walls from :func:`build_skirt`.
        sun_azimuth_deg: Compass bearing of the key light.
        sun_altitude_deg: Elevation of the key light.
        sun_intensity: Key light intensity.
        ambient: Ambient term of the surface material.
        ssao_radius_m: Ambient occlusion radius, or None to disable. It darkens
            the gaps between crowns, which gives the canopy its depth.
        shadows: Cast shadows from the key light. Slow on the CPU.
        background: Bottom and top colours of the background gradient.
    """
    import pyvista as pv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex = np.asarray(texture_rgb)
    if tex.dtype != np.uint8:
        tex = (np.clip(tex, 0.0, 1.0) * 255).astype(np.uint8)

    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    extent = np.array([xmax - xmin, ymax - ymin])
    diagonal = float(np.hypot(*extent))
    tx, ty = camera.target_frac
    target = np.array([xmin + tx * extent[0], ymin + ty * extent[1], zmin])
    az, el = np.deg2rad(camera.look_azimuth_deg), np.deg2rad(camera.elevation_deg)
    distance = camera.distance_factor * diagonal
    position = target + distance * np.array(
        [-np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)]
    )

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size), lighting=None)
    plotter.set_background(background[0], top=background[1])
    plotter.add_mesh(
        surface,
        texture=pv.Texture(np.ascontiguousarray(tex)),
        smooth_shading=True,
        ambient=ambient,
        diffuse=0.85,
        specular=0.0,
        show_scalar_bar=False,
    )
    if skirt is not None:
        plotter.add_mesh(skirt, color="#1a1e24", ambient=0.6, diffuse=0.4, specular=0.0)

    sun = pv.Light(
        position=_sun_position(target, diagonal, sun_azimuth_deg, sun_altitude_deg),
        focal_point=tuple(target),
        intensity=sun_intensity,
        light_type="scene light",
    )
    sun.positional = False
    plotter.add_light(sun)
    fill = pv.Light(
        position=_sun_position(target, diagonal, sun_azimuth_deg + 160.0, 60.0),
        focal_point=tuple(target),
        intensity=0.25 * sun_intensity,
        light_type="scene light",
    )
    fill.positional = False
    plotter.add_light(fill)

    plotter.camera.position = tuple(position)
    plotter.camera.focal_point = tuple(target)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.view_angle = camera.fov_deg
    plotter.camera.clipping_range = (0.01 * distance, 10.0 * distance)

    if shadows:
        plotter.enable_shadows()
    if ssao_radius_m:
        plotter.enable_ssao(
            radius=ssao_radius_m, bias=0.01 * ssao_radius_m, kernel_size=128
        )
    plotter.enable_anti_aliasing("ssaa")
    image = plotter.screenshot(str(out_path), return_img=True)
    plotter.close()
    logger.info("Rendered %s (%d x %d)", out_path.name, *window_size)
    return image
