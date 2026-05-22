"""``csdv stratify`` command.

Reads a per-site DEM and (optional) per-site soils raster stack, runs
:func:`csdv_core.stratification.topo.compute_topo` and
:func:`csdv_core.stratification.assign.assign_site_types`, and writes the
results under ``results_root/stratification/<site>/<window_m>m/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import rasterio

from csdv_core.config import load_site_types
from csdv_core.io.paths import project_paths
from csdv_core.io.raster import read_band, write_raster
from csdv_core.stratification.assign import assign_site_types
from csdv_core.stratification.soils import REQUIRED_SOIL_VARS
from csdv_core.stratification.topo import compute_topo

logger = logging.getLogger(__name__)


@click.command("stratify")
@click.option("--site", required=True, help="Site code (e.g. SCBI).")
@click.option(
    "--window-m",
    type=float,
    required=True,
    help="Analysis window size in meters; selects the output subdirectory.",
)
@click.option(
    "--dem",
    "dem_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the per-site DEM. Defaults to <topo_dir>/dem.tif.",
)
@click.option(
    "--soils-dir",
    "soils_dir_override",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory of soil rasters. Defaults to <soils_dir>.",
)
@click.option(
    "--accumulation-threshold",
    type=float,
    default=1000.0,
    show_default=True,
    help="Cells of flow accumulation that define a drainage pixel for HAND.",
)
def cli(
    site: str,
    window_m: float,
    dem_path: Path | None,
    soils_dir_override: Path | None,
    accumulation_threshold: float,
) -> None:
    """Compute stratification variables and assign site types for one site."""
    paths = project_paths()
    topo_dir = paths.topo_dir(site)
    soils_dir = soils_dir_override or paths.soils_dir(site)
    out_dir = paths.stratification_dir(site, window_m)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dem_path is None:
        dem_path = topo_dir / "dem.tif"
    if not dem_path.exists():
        raise click.UsageError(f"DEM not found: {dem_path}")

    logger.info("reading DEM %s", dem_path)
    dem_read = read_band(dem_path, nodata_to_nan=True)
    pixel_size_m = float(abs(dem_read.transform.a))
    transform = dem_read.transform
    crs = str(dem_read.crs) if dem_read.crs is not None else ""

    logger.info("computing topographic derivatives")
    topo_vars = compute_topo(
        dem_read.data,
        pixel_size_m,
        accumulation_threshold=accumulation_threshold,
    )
    for name, arr in topo_vars.items():
        write_raster(
            out_dir / f"{name}.tif",
            arr,
            transform=transform,
            crs=crs,
            nodata=float("nan"),
            dtype="float32",
        )

    soil_vars: dict[str, np.ndarray] = {}
    for name in REQUIRED_SOIL_VARS:
        soil_path = soils_dir / f"{name}.tif"
        if not soil_path.exists():
            logger.warning("soil raster missing: %s (skipping)", soil_path)
            continue
        with rasterio.open(soil_path) as src:
            arr = src.read(1, masked=True).filled(0)
            if name in {"parmat_kind", "texture_class", "bedrock_kind"}:
                arr = arr.astype("uint8", copy=False)
            else:
                arr = arr.astype("float32", copy=False)
        soil_vars[name] = arr

    stratvars: dict[str, np.ndarray] = dict(topo_vars)
    stratvars.update(soil_vars)

    rules = load_site_types()
    logger.info("assigning %d site-type rules", len(rules.site_types))
    site_type, match_score = assign_site_types(stratvars, rules)

    write_raster(
        out_dir / "site_type.tif",
        site_type,
        transform=transform,
        crs=crs,
        nodata=0,
        dtype="uint8",
    )
    write_raster(
        out_dir / "site_type_match_score.tif",
        match_score,
        transform=transform,
        crs=crs,
        nodata=float("nan"),
        dtype="float32",
    )
    n_assigned = int((site_type != 0).sum())
    logger.info(
        "wrote %s; %d/%d pixels classified",
        out_dir,
        n_assigned,
        site_type.size,
    )
