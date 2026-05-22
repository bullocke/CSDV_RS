"""``csdv classify-stages`` command.

Reads metric rasters from ``paths.metrics_dir(site, year, window_m)`` plus
the site-type raster from ``paths.stratification_dir(site, window_m)``,
runs :func:`csdv_core.stages.classify.classify_stages`, and writes the
output rasters to ``paths.stages_dir(site, year, window_m)``.
"""

from __future__ import annotations

import logging

import click

from csdv_core.config import load_stages
from csdv_core.io.metrics_io import read_metric_stack
from csdv_core.io.paths import project_paths
from csdv_core.io.raster import read_band, write_raster
from csdv_core.stages.classify import classify_stages

logger = logging.getLogger(__name__)


@click.command("classify-stages")
@click.option("--site", required=True, help="Site code (e.g. SCBI).")
@click.option("--year", type=int, required=True, help="NAIP acquisition year.")
@click.option(
    "--window-m",
    type=float,
    required=True,
    help="Analysis window size in meters; selects metric+stage subdirectories.",
)
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum match-score for a stage to be assigned.",
)
def cli(site: str, year: int, window_m: float, min_score: float) -> None:
    """Classify metric rasters into stand developmental stages."""
    paths = project_paths()
    metrics_dir = paths.metrics_dir(site, year, window_m)
    strat_dir = paths.stratification_dir(site, window_m)
    out_dir = paths.stages_dir(site, year, window_m)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        raise click.UsageError(f"Metrics directory missing: {metrics_dir}")
    if not (strat_dir / "site_type.tif").exists():
        raise click.UsageError(
            f"Site-type raster missing: {strat_dir / 'site_type.tif'}; "
            "run `csdv stratify` first."
        )

    logger.info("loading metric stack from %s", metrics_dir)
    metrics, grid = read_metric_stack(metrics_dir)
    sample = next(iter(metrics.values()))
    metrics_shape = sample.shape

    logger.info("loading site-type raster")
    site_read = read_band(strat_dir / "site_type.tif", nodata_to_nan=False)
    site_type = site_read.data.astype("uint8", copy=False)
    if site_type.shape != metrics_shape:
        raise click.UsageError(
            f"site_type shape {site_type.shape} != metrics shape {metrics_shape}"
        )

    stages_cfg = load_stages()
    logger.info(
        "classifying %d metrics x %d stages on %s",
        len(metrics),
        len(stages_cfg.stages),
        metrics_shape,
    )
    stage, score, evaluated = classify_stages(
        metrics, site_type, stages_cfg, min_score=min_score
    )

    write_raster(
        out_dir / "stage.tif",
        stage,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
        dtype="uint8",
    )
    write_raster(
        out_dir / "stage_score.tif",
        score,
        transform=grid.transform,
        crs=grid.crs,
        nodata=float("nan"),
        dtype="float32",
    )
    write_raster(
        out_dir / "stage_evaluated_count.tif",
        evaluated,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
        dtype="uint8",
    )
    n_assigned = int((stage != 0).sum())
    logger.info(
        "wrote %s; %d/%d pixels classified",
        out_dir,
        n_assigned,
        stage.size,
    )
