"""
08b_regenerate_crownseg_figures.py — Regenerate crown segmentation comparison
and IoU figures with improved styling (CHM colorbars, thicker/unified crown
outlines), saving them as ``_v2.png`` alongside the originals.

Reuses ``make_crownseg_figure`` and ``make_iou_figure`` from
``08_compare_chm_sources.py`` and reads the cached deepzoom and mediumzoom
intermediate rasters / GeoPackages directly. No re-clipping, no re-running of
the R segmentation.

Usage
-----
    python 08b_regenerate_crownseg_figures.py --site SCBI
    python 08b_regenerate_crownseg_figures.py --site HARV
    python 08b_regenerate_crownseg_figures.py --site all
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import rasterio

import importlib.util
import sys

_HERE = Path(__file__).resolve()
_MODULE_PATH = _HERE.parent / "08_compare_chm_sources.py"
_spec = importlib.util.spec_from_file_location("compare_chm_sources", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["compare_chm_sources"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

SITES = _mod.SITES
make_crownseg_figure = _mod.make_crownseg_figure
make_iou_figure = _mod.make_iou_figure
_setup_style = _mod._setup_style

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures" / "crown_segmentation"
_INTERMEDIATE = _POC / "Results" / "summary_document" / "intermediate"

OUTLINE_COLOR = "#00e5ff"   # cyan, high contrast over greyscale CHM
OUTLINE_LW = 1.1


def _disp_from_raster(path: Path) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) display extent from a raster's bounds."""
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.right, b.bottom, b.top)


def regenerate_for_site(site: str) -> None:
    cfg = SITES[site]
    site_code = cfg.site_code
    cfg_label = cfg.label

    # ---- Deepzoom: fig_compare_crownseg_{SITE}_v2.png ----
    dz = _INTERMEDIATE / "deepzoom" / site_code
    dz_naip_rgb = dz / f"naip_rgb_{site_code}_deepzoom.tif"
    dz_neon_chm = dz / f"neon_chm_{site_code}_deepzoom.tif"
    dz_naip_chm = dz / f"naip_chm_{site_code}_deepzoom_meters.tif"
    dz_neon_crowns = dz / f"crowns_neon_{site_code}_deepzoom.gpkg"
    dz_naip_crowns = dz / f"crowns_naip_{site_code}_deepzoom.gpkg"

    for p in (dz_naip_rgb, dz_neon_chm, dz_naip_chm,
              dz_neon_crowns, dz_naip_crowns):
        if not p.exists():
            raise FileNotFoundError(f"Missing deepzoom intermediate: {p}")

    dz_disp = _disp_from_raster(dz_neon_chm)

    out_compare = _FIGURES_OUT / f"fig_compare_crownseg_{site_code}_v2.png"
    logger.info("Regenerating %s ...", out_compare.name)
    make_crownseg_figure(
        naip_rgb_clip=dz_naip_rgb,
        neon_chm_clip=dz_neon_chm,
        naip_chm_clip=dz_naip_chm,
        neon_crowns=dz_neon_crowns,
        naip_crowns=dz_naip_crowns,
        disp=dz_disp,
        out_path=out_compare,
        site_label=cfg_label,
        outline_color=OUTLINE_COLOR,
        outline_lw=OUTLINE_LW,
        add_chm_colorbar=True,
    )

    # ---- Mediumzoom: fig_iou_crownseg_{SITE}_v2.png ----
    mz = _INTERMEDIATE / "mediumzoom" / site_code
    mz_naip_rgb = mz / f"naip_rgb_{site_code}_mediumzoom.tif"
    mz_neon_chm = mz / f"neon_chm_{site_code}_mediumzoom.tif"
    mz_naip_chm = mz / f"naip_chm_{site_code}_mediumzoom_meters.tif"
    mz_neon_crowns = mz / f"crowns_neon_{site_code}_mediumzoom.gpkg"
    mz_naip_crowns = mz / f"crowns_naip_{site_code}_mediumzoom.gpkg"

    for p in (mz_naip_rgb, mz_neon_chm, mz_naip_chm,
              mz_neon_crowns, mz_naip_crowns):
        if not p.exists():
            raise FileNotFoundError(f"Missing mediumzoom intermediate: {p}")

    mz_disp = _disp_from_raster(mz_neon_chm)

    out_iou = _FIGURES_OUT / f"fig_iou_crownseg_{site_code}_v2.png"
    logger.info("Regenerating %s ...", out_iou.name)
    make_iou_figure(
        naip_rgb_clip=mz_naip_rgb,
        neon_chm_clip=mz_neon_chm,
        naip_chm_clip=mz_naip_chm,
        neon_crowns=mz_neon_crowns,
        naip_crowns=mz_naip_crowns,
        disp=mz_disp,
        out_path=out_iou,
        site_label=cfg_label,
        outline_color_neon=OUTLINE_COLOR,
        outline_color_naip=OUTLINE_COLOR,
        outline_lw=OUTLINE_LW,
        add_chm_colorbar=True,
    )


@click.command()
@click.option(
    "--site",
    type=click.Choice(["SCBI", "HARV", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
def main(site: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _setup_style()
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    sites = ["SCBI", "HARV"] if site.lower() == "all" else [site.upper()]
    for s in sites:
        regenerate_for_site(s)
    logger.info("Done.")


if __name__ == "__main__":
    main()
