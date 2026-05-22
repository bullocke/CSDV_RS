"""``csdv classify-trajectories`` command.

Reads per-year stage rasters (Phase 4 outputs) and any required metric
rasters (Phase 2-3 outputs) for ``--site --window-m --years a,b,c``,
plus the year-invariant site-type raster if any rule references it,
runs :func:`csdv_core.trajectories.classify.classify_trajectories`, and
writes outputs to ``paths.trajectories_dir(site, window_m)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from csdv_core.config import load_stages, load_trajectories
from csdv_core.io.paths import project_paths
from csdv_core.io.raster import read_band, write_raster
from csdv_core.io.stages_io import read_metric_cube, read_stage_cube
from csdv_core.trajectories.classify import (
    all_required_metrics,
    classify_trajectories,
)

logger = logging.getLogger(__name__)


def _parse_years(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 2:
        raise click.UsageError(
            f"--years must list at least 2 comma-separated years; got {value!r}"
        )
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.UsageError(f"--years contains a non-integer: {value!r}") from exc


def _rules_need_site_type(traj_cfg) -> bool:
    for rule in traj_cfg.trajectories.values():
        for pred in rule.signature:
            if pred.dim == "site_type":
                return True
    return False


@click.command("classify-trajectories")
@click.option("--site", required=True, help="Site code (e.g. SCBI).")
@click.option(
    "--window-m",
    type=float,
    required=True,
    help="Analysis window size in meters; selects metric+stage subdirectories.",
)
@click.option(
    "--years",
    required=True,
    help="Comma-separated NAIP years (>= 2), e.g. '2014,2018,2022'.",
)
@click.option(
    "--no-site-type",
    is_flag=True,
    default=False,
    help="Skip loading the site-type raster even if rules reference it.",
)
def cli(site: str, window_m: float, years: str, no_site_type: bool) -> None:
    """Classify multi-date stage cubes into V5 trajectory classes."""
    paths = project_paths()
    year_list = _parse_years(years)

    traj_cfg = load_trajectories()
    stages_cfg = load_stages()

    logger.info("loading stage cube for %s years=%s", site, year_list)
    stage_cube, grid = read_stage_cube(paths, site, year_list, window_m)

    metric_names = all_required_metrics(traj_cfg)
    if metric_names:
        logger.info("loading %d metric cubes: %s", len(metric_names), metric_names)
        metric_cubes, _mgrid = read_metric_cube(
            paths, site, year_list, window_m, metric_names
        )
    else:
        metric_cubes = {}

    site_type = None
    if _rules_need_site_type(traj_cfg) and not no_site_type:
        st_path = paths.stratification_dir(site, window_m) / "site_type.tif"
        if not st_path.exists():
            raise click.UsageError(
                f"site-type raster missing: {st_path}; "
                "run `csdv stratify` first or pass --no-site-type."
            )
        st_read = read_band(st_path, nodata_to_nan=False)
        site_type = st_read.data.astype("uint8", copy=False)

    out_dir = paths.trajectories_dir(site, window_m)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "classifying trajectories on stage_cube=%s metrics=%d",
        stage_cube.shape,
        len(metric_cubes),
    )
    trajectory, n_predicates, n_dates = classify_trajectories(
        stage_cube,
        metric_cubes,
        site_type,
        traj_cfg,
        stages_cfg,
    )

    write_raster(
        out_dir / "trajectory.tif",
        trajectory,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
        dtype="uint8",
    )
    write_raster(
        out_dir / "trajectory_n_predicates.tif",
        n_predicates,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
        dtype="uint8",
    )
    write_raster(
        out_dir / "trajectory_n_dates.tif",
        n_dates,
        transform=grid.transform,
        crs=grid.crs,
        nodata=0,
        dtype="uint8",
    )

    manifest = {
        "site": site,
        "window_m": float(window_m),
        "years": year_list,
        "metrics_used": metric_names,
        "site_type_used": site_type is not None,
        "trajectory_order": list(traj_cfg.trajectory_order),
        "trajectory_codes": dict(traj_cfg.trajectory_codes),
    }
    Path(out_dir / "trajectories_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )

    n_assigned = int((trajectory != 0).sum())
    logger.info(
        "wrote %s; %d/%d pixels classified",
        out_dir,
        n_assigned,
        trajectory.size,
    )
