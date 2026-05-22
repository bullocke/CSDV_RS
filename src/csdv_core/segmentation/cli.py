"""csdv_core.segmentation.cli \u2014 Click command for crown segmentation.

Dispatches between the Python watershed engine
(:mod:`csdv_core.segmentation.chm_watershed`) and the R/lidR bridge
(:mod:`csdv_core.segmentation.lidr_bridge`) via ``--engine``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("segment-crowns")
@click.option(
    "--chm",
    "chm_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input CHM GeoTIFF (metres).",
)
@click.option(
    "--out-crowns",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output crown polygons (GeoPackage for lidR; any GeoPandas-writable format for watershed).",
)
@click.option(
    "--engine",
    type=click.Choice(["watershed", "lidr"], case_sensitive=False),
    default="watershed",
    show_default=True,
    help="Segmentation engine.",
)
@click.option(
    "--out-cv-raster",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="(lidR only) Output crown-CV GeoTIFF.",
)
@click.option(
    "--scale-factor",
    type=float,
    default=1.0,
    show_default=True,
    help="(lidR only) CHM scale factor; use 0.01 for uint16 NAIP-CHM.",
)
@click.option(
    "--min-height-m",
    type=float,
    default=2.0,
    show_default=True,
    help="(watershed) Pixels below this height are ignored.",
)
@click.option(
    "--min-peak-distance-m",
    type=float,
    default=3.0,
    show_default=True,
    help="(watershed) Minimum separation between local maxima.",
)
def cli(
    chm_path: Path,
    out_crowns: Path,
    engine: str,
    out_cv_raster: Path | None,
    scale_factor: float,
    min_height_m: float,
    min_peak_distance_m: float,
) -> None:
    """Segment tree crowns from a CHM with the chosen engine."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    engine = engine.lower()
    out_crowns.parent.mkdir(parents=True, exist_ok=True)

    if engine == "lidr":
        from csdv_core.segmentation.lidr_bridge import run_lidr_segmentation

        if out_cv_raster is None:
            raise click.UsageError(
                "--out-cv-raster is required when --engine lidr is selected."
            )
        run_lidr_segmentation(
            chm_path=chm_path,
            out_crowns=out_crowns,
            out_cv_raster=out_cv_raster,
            scale_factor=scale_factor,
        )
        click.echo(str(out_crowns))
        return

    # Watershed engine.
    import rasterio

    from csdv_core.segmentation.chm_watershed import segment_crowns

    with rasterio.open(chm_path) as src:
        chm = src.read(1, masked=True).filled(float("nan")).astype("float32")
        transform = src.transform
        crs = src.crs
    gdf = segment_crowns(
        chm,
        transform=transform,
        crs=crs,
        min_height_m=min_height_m,
        min_peak_distance_m=min_peak_distance_m,
    )
    gdf.to_file(out_crowns)
    logger.info("Wrote %d crowns to %s", len(gdf), out_crowns)
    click.echo(str(out_crowns))


__all__ = ["cli"]
