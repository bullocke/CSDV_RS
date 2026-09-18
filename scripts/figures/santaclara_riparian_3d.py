"""Oblique 3D views of the Santa Clara River riparian corridor.

The rendered surface is the 3DEP terrain with the 2024 NAIP canopy height model
stacked on it. Terrain and canopy take separate vertical exaggeration, because
at true scale a 15 m cottonwood disappears against a kilometre of valley. Three
views step in from the whole corridor to a single stand, and each can be draped
with the canopy height ramp or with the NAIP true-colour image.

Run ``scripts/data/download_santaclara_context.py`` first to fetch the DEM and
the NAIP image.

Example::

    python scripts/figures/santaclara_riparian_3d.py --view all --drape both
    python scripts/figures/santaclara_riparian_3d.py --view close --draft
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from santaclara_riparian_map import CHM, DEM, OUT_DIR, REPO, read_to_grid  # noqa: E402

from csdv_core.preprocess.chm import smooth_chm  # noqa: E402
from csdv_core.viz.maps import (  # noqa: E402
    read_band_window,
    read_rgb_window,
    stretch_rgb,
)
from csdv_core.viz.relief import (  # noqa: E402
    CANOPY_VMAX_M,
    canopy_cmap,
    shaded_canopy_rgb,
)
from csdv_core.viz.scene3d import (  # noqa: E402
    CameraPreset,
    build_skirt,
    build_surface,
    render_view,
)

logger = logging.getLogger(__name__)

NAIP = REPO / "data/naip/SantaClara/2024/m_3411948_ne_11_060_20240728_north.tif"
INK, MUTED = "#e8ecef", "#9aa6af"


@dataclass(frozen=True)
class View:
    """One rendered view: its ground window, mesh density, relief and camera."""

    label: str
    bounds: tuple[float, float, float, float]
    mesh_px: int
    terrain_exaggeration: float
    canopy_exaggeration: float
    ssao_radius_m: float
    camera: CameraPreset
    smooth: bool = False


VIEWS = {
    "wide": View(
        label="The corridor, 4 km across",
        bounds=(312300.0, 3802800.0, 316308.0, 3805400.0),
        mesh_px=2600,
        terrain_exaggeration=1.5,
        canopy_exaggeration=2.5,
        ssao_radius_m=25.0,
        camera=CameraPreset(
            look_azimuth_deg=20.0,
            elevation_deg=32.0,
            distance_factor=1.27,
            fov_deg=26.0,
            target_frac=(0.5, 0.3),
        ),
    ),
    "reach": View(
        label="The densest reach, 2.3 km across",
        bounds=(314000.0, 3803100.0, 316300.0, 3804600.0),
        mesh_px=2400,
        terrain_exaggeration=1.2,
        canopy_exaggeration=2.0,
        ssao_radius_m=12.0,
        camera=CameraPreset(
            look_azimuth_deg=335.0,
            elevation_deg=26.0,
            distance_factor=1.2,
            fov_deg=26.0,
            target_frac=(0.5, 0.4),
        ),
    ),
    "close": View(
        label="A single stand, 700 m across",
        bounds=(313800.0, 3803350.0, 314500.0, 3803850.0),
        mesh_px=1200,
        terrain_exaggeration=1.0,
        canopy_exaggeration=1.5,
        ssao_radius_m=5.0,
        camera=CameraPreset(
            look_azimuth_deg=10.0,
            elevation_deg=20.0,
            distance_factor=1.2,
            fov_deg=28.0,
            target_frac=(0.5, 0.35),
        ),
        smooth=True,
    ),
}


def compose(
    image: np.ndarray, view: View, drape: str, out_path: Path, *, vmax: float, dpi: int
) -> None:
    """Add the title, caption and colour bar over a rendered image."""
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), facecolor="#0d1015")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    scale = width / 3840.0
    fig.text(
        0.025,
        0.085,
        "Santa Clara River riparian corridor",
        color=INK,
        fontsize=30 * scale,
        fontweight="bold",
    )
    surface = "canopy height" if drape == "height" else "NAIP true colour"
    fig.text(
        0.025,
        0.045,
        f"{view.label}. 2024 NAIP canopy height on 3DEP terrain, draped with {surface}. "
        f"Vertical exaggeration {view.canopy_exaggeration:g}x (canopy), "
        f"{view.terrain_exaggeration:g}x (terrain).",
        color=MUTED,
        fontsize=17 * scale,
    )
    if drape == "height":
        cax = fig.add_axes((0.765, 0.915, 0.21, 0.016))
        bar = fig.colorbar(
            ScalarMappable(Normalize(0, vmax), canopy_cmap()),
            cax=cax,
            orientation="horizontal",
            extend="max",
        )
        bar.set_label("Canopy height (m)", color=INK, fontsize=17 * scale)
        bar.ax.tick_params(colors=MUTED, labelsize=15 * scale)
        bar.outline.set_edgecolor(MUTED)
    fig.savefig(out_path, dpi=dpi, facecolor="#0d1015")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def render(view_name: str, drape: str, args: argparse.Namespace) -> None:
    view = VIEWS[view_name]
    mesh_px = view.mesh_px // 2 if args.draft else view.mesh_px
    window = (1920, 1080) if args.draft else (3840, 2160)

    chm, transform = read_band_window(args.chm, view.bounds, scale=0.01, max_px=mesh_px)
    if view.smooth:
        chm = smooth_chm(chm, kernel=3)
    with rasterio.open(args.chm) as src:
        crs = src.crs
    dem = read_to_grid(args.dem, transform, chm.shape, crs)
    dem = dem - np.nanmin(dem)
    z = view.terrain_exaggeration * dem + view.canopy_exaggeration * np.nan_to_num(chm)
    logger.info(
        "%s: mesh %s at %.2f m, relief %.0f m", view_name, z.shape, transform.a, z.max()
    )

    # The drape is read finer than the mesh. Texture detail costs little, and
    # it is what makes crowns look sharp on a mesh coarser than the imagery.
    tex_px = min(8000, 2 * mesh_px)
    if drape == "naip":
        rgb, _ = read_rgb_window(args.naip, view.bounds, max_px=tex_px)
        texture = stretch_rgb(rgb, percentiles=(0.5, 99.5), gamma=1.25)
    else:
        fine, fine_transform = read_band_window(
            args.chm, view.bounds, scale=0.01, max_px=tex_px
        )
        fine_dem = read_to_grid(args.dem, fine_transform, fine.shape, crs)
        texture = shaded_canopy_rgb(
            fine, abs(fine_transform.a), dem=fine_dem, vmax=args.vmax
        )
        # The renderer supplies the lighting, so lift the ramp's dark end. At
        # its map values open ground would render almost black.
        texture = np.clip(0.06 + 1.15 * texture, 0.0, 1.0)

    base = -0.04 * float(np.hypot(*z.shape)) * abs(transform.a)
    raw = args.out_dir / "raw" / f"{view_name}_{drape}.png"
    image = render_view(
        build_surface(z, transform),
        texture,
        raw,
        camera=view.camera,
        window_size=window,
        skirt=build_skirt(z, transform, base),
        ssao_radius_m=view.ssao_radius_m,
        shadows=args.shadows,
        # The NAIP image carries its own shadows, so it takes flatter light.
        ambient=0.55 if drape == "naip" else 0.35,
        sun_intensity=0.8 if drape == "naip" else 1.0,
    )
    index = {"wide": 2, "reach": 3, "close": 4}[view_name]
    suffix = "_draft" if args.draft else ""
    out = args.out_dir / f"fig{index}_3d_{view_name}_{drape}{suffix}.png"
    compose(image, view, drape, out, vmax=args.vmax, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", choices=[*VIEWS, "all"], default="all")
    parser.add_argument("--drape", choices=["height", "naip", "both"], default="both")
    parser.add_argument("--draft", action="store_true", help="Half size, coarse mesh.")
    parser.add_argument("--shadows", action="store_true", help="Cast shadows (slow).")
    parser.add_argument("--vmax", type=float, default=CANOPY_VMAX_M)
    parser.add_argument("--chm", type=Path, default=CHM)
    parser.add_argument("--dem", type=Path, default=DEM)
    parser.add_argument("--naip", type=Path, default=NAIP)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    views = list(VIEWS) if args.view == "all" else [args.view]
    drapes = ["height", "naip"] if args.drape == "both" else [args.drape]
    for view_name in views:
        for drape in drapes:
            if drape == "naip" and not args.naip.exists():
                logger.warning("%s not found, skipping the NAIP drape", args.naip)
                continue
            render(view_name, drape, args)


if __name__ == "__main__":
    main()
