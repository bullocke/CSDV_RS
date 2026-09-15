"""csdv_core.segmentation.cli — Click command for crown segmentation.

Dispatches between the Python watershed engine
(:mod:`csdv_core.segmentation.chm_watershed`) and the R/lidR reference
(:mod:`csdv_core.segmentation.lidr_bridge`) via ``--engine``. Both engines take
the same parameters, in metres, so ``--engine lidr`` is a genuine check on the
Python result rather than a different configuration.

Segmenting a module-sized CHM in one pass needs more memory than it is worth,
so the watershed engine runs in overlapping blocks by default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from csdv_core.segmentation.params import (
    DEFAULT_PARAMS,
    WINDOW_FUNCTIONS,
    SegmentationParams,
    WindowFunction,
)

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
    help="Output crown polygons. GeoPackage is the expected format.",
)
@click.option(
    "--engine",
    type=click.Choice(["watershed", "lidr"], case_sensitive=False),
    default="watershed",
    show_default=True,
    help="Segmentation engine. Use lidr only as an independent check.",
)
@click.option(
    "--window",
    type=click.Choice(sorted(WINDOW_FUNCTIONS), case_sensitive=False),
    default=None,
    help="Named search-window function. Overridden by the --ws-* options.",
)
@click.option("--ws-a", type=float, default=None, help="Window intercept (m).")
@click.option("--ws-b", type=float, default=None, help="Window linear term.")
@click.option("--ws-c", type=float, default=None, help="Window quadratic term.")
@click.option("--ws-lo", type=float, default=None, help="Window lower clip (m).")
@click.option("--ws-hi", type=float, default=None, help="Window upper clip (m).")
@click.option(
    "--smooth-radius-m",
    type=float,
    default=DEFAULT_PARAMS.smooth_radius_m,
    show_default=True,
    help="Mean-filter radius in metres. 0 disables smoothing.",
)
@click.option(
    "--min-height-m",
    type=float,
    default=DEFAULT_PARAMS.min_height_m,
    show_default=True,
    help="Canopy floor. Pixels below this are not forest.",
)
@click.option(
    "--th-cr",
    type=float,
    default=DEFAULT_PARAMS.th_cr,
    show_default=True,
    help="Crown extent bound as a fraction of tree-top height. 0 disables it.",
)
@click.option(
    "--max-crown-radius-m",
    type=float,
    default=DEFAULT_PARAMS.max_crown_radius_m,
    show_default=True,
    help="Crown radius ceiling in metres. Negative disables it.",
)
@click.option(
    "--min-crown-area-m2",
    type=float,
    default=DEFAULT_PARAMS.min_crown_area_m2,
    show_default=True,
    help="Segments smaller than this are dropped as noise.",
)
@click.option(
    "--block-px",
    type=int,
    default=2048,
    show_default=True,
    help="(watershed) Interior block side. 0 segments in a single pass.",
)
@click.option(
    "--scale-factor",
    type=float,
    default=1.0,
    show_default=True,
    help="(lidR) CHM scale factor. Use 0.01 for uint16 NAIP-CHM.",
)
@click.option(
    "--out-cv-raster",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="(lidR) Optional crown-CV GeoTIFF.",
)
def cli(
    chm_path: Path,
    out_crowns: Path,
    engine: str,
    window: str | None,
    ws_a: float | None,
    ws_b: float | None,
    ws_c: float | None,
    ws_lo: float | None,
    ws_hi: float | None,
    smooth_radius_m: float,
    min_height_m: float,
    th_cr: float,
    max_crown_radius_m: float,
    min_crown_area_m2: float,
    block_px: int,
    scale_factor: float,
    out_cv_raster: Path | None,
) -> None:
    """Segment tree crowns from a CHM with the chosen engine."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    out_crowns.parent.mkdir(parents=True, exist_ok=True)

    base = WINDOW_FUNCTIONS[window] if window else DEFAULT_PARAMS.window
    ws = WindowFunction(
        a=base.a if ws_a is None else ws_a,
        b=base.b if ws_b is None else ws_b,
        c=base.c if ws_c is None else ws_c,
        lo=base.lo if ws_lo is None else ws_lo,
        hi=base.hi if ws_hi is None else ws_hi,
        name=base.name if (ws_a, ws_b, ws_c, ws_lo, ws_hi) == (None,) * 5 else "custom",
    )
    params = SegmentationParams(
        min_height_m=min_height_m,
        smooth_radius_m=smooth_radius_m,
        window=ws,
        th_cr=th_cr,
        max_crown_radius_m=None if max_crown_radius_m < 0 else max_crown_radius_m,
        min_crown_area_m2=min_crown_area_m2,
    )
    logger.info("Segmentation %s: %s", params.key, params.describe())

    if engine.lower() == "lidr":
        from csdv_core.segmentation.lidr_bridge import run_lidr_segmentation

        run_lidr_segmentation(
            chm_path=chm_path,
            out_crowns=out_crowns,
            params=params,
            scale_factor=scale_factor,
            out_cv_raster=out_cv_raster,
        )
        _write_sidecar(out_crowns, params, chm_path, engine="lidr")
        click.echo(str(out_crowns))
        return

    import rasterio

    from csdv_core.segmentation.chm_watershed import segment_crowns
    from csdv_core.zonal.crowns import segment_scene_crowns

    with rasterio.open(chm_path) as src:
        chm = src.read(1, masked=True).filled(float("nan")).astype("float32")
        transform = src.transform
        crs = src.crs
    if block_px and min(chm.shape) > block_px:
        gdf = segment_scene_crowns(
            chm, transform, crs, block_px=block_px, params=params
        )
    else:
        gdf = segment_crowns(chm, transform, crs, params=params)
    gdf.to_file(out_crowns)
    _write_sidecar(out_crowns, params, chm_path, engine="watershed")
    logger.info("Wrote %d crowns to %s", len(gdf), out_crowns)
    click.echo(str(out_crowns))


def _write_sidecar(
    out_crowns: Path, params: SegmentationParams, chm_path: Path, *, engine: str
) -> None:
    """Record what produced a crown file, next to the crown file.

    Crown artefacts used to carry no record of their parameters, so a re-run
    after a parameter change silently reused whatever was already on disk.
    """
    sidecar = out_crowns.with_suffix(out_crowns.suffix + ".params.json")
    sidecar.write_text(
        json.dumps(
            {
                "key": params.key,
                "engine": engine,
                "chm": str(chm_path),
                "params": params.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["cli"]
