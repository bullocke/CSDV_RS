"""csdv_core.satellite.cli — the ``csdv satellite`` command group.

Four steps, kept separate because they fail for different reasons and cost
different amounts. ``fetch`` is the only one that touches the network, and it is
also the slow and quota-consuming one, so it caches to disk and everything after
it reads that cache. ``annual`` and ``join`` are pure and can be rerun freely
while a threshold is being tuned.

Note for CHPC: ``fetch`` needs outbound internet, which compute nodes do not
have. Run it on a data transfer node and stage the parquet, the same way the
conditioning rasters are handled in ``docs/workflow_chpc.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Per-stand satellite index series and the metrics derived from them."""


def _read_stands(gdb: Path, layer: str):
    from csdv_core.io.stands import read_ais_stands

    return read_ais_stands(gdb, layer=layer)


def _default_dirs(site: str) -> tuple[Path, Path]:
    from csdv_core.io.paths import project_paths

    paths = project_paths()
    return paths.satellite_dir(site), paths.stands_dir(site)


@cli.command("fetch")
@click.option("--site", required=True, help="Site name, e.g. ElkinsvilleNE.")
@click.option(
    "--stands",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Stand polygon geodatabase, read by read_ais_stands.",
)
@click.option("--layer", default="DisturbancePoly", show_default=True)
@click.option("--start-year", type=int, default=None, help="Overrides satellite.yaml.")
@click.option("--end-year", type=int, default=None, help="Overrides satellite.yaml.")
@click.option("--sensors", default=None, help="Comma-separated, e.g. L5,L7,L8.")
@click.option("--indices", default=None, help="Comma-separated, e.g. ndvi,nbr.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=None)
@click.option("--project", default=None, help="Earth Engine project.")
@click.option("--max-workers", type=int, default=None)
@click.option(
    "--chunk", type=click.Choice(["year", "half_year", "month"]), default=None
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Resolve chunks and stands and print the plan without contacting Earth Engine.",
)
@click.option(
    "--replace", is_flag=True, help="Overwrite the cache instead of appending."
)
def fetch_cli(
    site: str,
    stands: Path,
    layer: str,
    start_year: int | None,
    end_year: int | None,
    sensors: str | None,
    indices: str | None,
    out_dir: Path | None,
    project: str | None,
    max_workers: int | None,
    chunk: str | None,
    dry_run: bool,
    replace: bool,
) -> None:
    """Reduce satellite archives over stand polygons into an observation cache."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from csdv_core.config import load_satellite
    from csdv_core.io.satellite_io import write_observations
    from csdv_core.satellite.extract import date_chunks, fetch_observations

    cfg = load_satellite()
    if project or chunk:
        cfg = cfg.model_copy(deep=True)
        if project:
            cfg.earth_engine.project = project
        if chunk:
            cfg.extraction.chunk = chunk

    frame = _read_stands(stands, layer)
    sensor_names = sensors.split(",") if sensors else list(cfg.extraction.sensors)
    index_names = indices.split(",") if indices else list(cfg.extraction.indices)
    first = int(start_year if start_year is not None else cfg.extraction.start_year)
    last = int(end_year if end_year is not None else cfg.extraction.end_year)
    destination = out_dir or _default_dirs(site)[0]

    if dry_run:
        chunks = date_chunks(first, last, granularity=cfg.extraction.chunk)
        vertices = sum(
            len(geom.exterior.coords)
            if geom.geom_type == "Polygon"
            else sum(len(part.exterior.coords) for part in geom.geoms)
            for geom in frame.geometry
        )
        click.echo(f"site         {site}")
        click.echo(f"stands       {len(frame)} ({vertices} vertices, {frame.crs})")
        click.echo(f"sensors      {','.join(sensor_names)}")
        click.echo(f"indices      {','.join(index_names)}")
        click.echo(f"years        {first}-{last}")
        click.echo(f"chunks       {len(chunks)} at {cfg.extraction.chunk} granularity")
        click.echo(f"destination  {destination}")
        click.echo("Nothing was contacted.")
        return

    observations, provenance = fetch_observations(
        frame,
        cfg=cfg,
        sensors=sensor_names,
        indices=index_names,
        start_year=first,
        end_year=last,
        max_workers=max_workers,
    )
    provenance["site"] = site
    provenance["stands_source"] = str(stands)
    parquet, manifest = write_observations(
        observations, destination, provenance, append=not replace
    )
    click.echo(str(parquet))
    click.echo(str(manifest))


@cli.command("annual")
@click.option("--site", required=True)
@click.option("--observations", type=click.Path(path_type=Path), default=None)
@click.option("--metrics", default=None, help="Comma-separated annual metric names.")
@click.option("--out", type=click.Path(path_type=Path), default=None)
def annual_cli(
    site: str, observations: Path | None, metrics: str | None, out: Path | None
) -> None:
    """Derive per-stand-per-year metrics from a cached observation table."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from csdv_core.io.satellite_io import read_observations, write_annual
    from csdv_core.satellite.annual import annual_table

    satellite_dir, stands_dir = _default_dirs(site)
    frame = read_observations(observations or satellite_dir)
    table = annual_table(frame, metrics=metrics.split(",") if metrics else None)
    click.echo(str(write_annual(table, out.parent if out else stands_dir)))


@cli.command("join")
@click.option("--site", required=True)
@click.option("--stand-metrics", type=click.Path(path_type=Path), default=None)
@click.option("--annual", "annual_path", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def join_cli(
    site: str, stand_metrics: Path | None, annual_path: Path | None, out: Path | None
) -> None:
    """Attach the satellite metrics to the stand metric table, in place by default."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    import pandas as pd

    from csdv_core.io.satellite_io import join_satellite_metrics, read_annual

    _, stands_dir = _default_dirs(site)
    metrics_path = stand_metrics or stands_dir / "stand_metrics.parquet"
    merged = join_satellite_metrics(
        pd.read_parquet(metrics_path), read_annual(annual_path or stands_dir)
    )
    destination = out or metrics_path
    merged.to_parquet(destination, index=False)
    click.echo(str(destination))


@cli.command("list")
def list_cli() -> None:
    """List the registered sensors, indices and annual metrics."""
    from csdv_core.satellite.registry import (
        get_annual,
        get_index,
        get_sensor,
        list_annual_metrics,
        list_indices,
        list_sensors,
    )

    click.echo("sensors")
    for name in list_sensors():
        spec = get_sensor(name)
        span = f"{spec.first_year}-{spec.last_year or 'present'}"
        click.echo(f"  {name:5s} {spec.platform:20s} {span:14s} {spec.collection_id}")
    click.echo("indices")
    for name in list_indices():
        spec = get_index(name)
        click.echo(f"  {name:5s} {'+'.join(spec.bands):16s} {spec.description}")
    click.echo("annual metrics")
    for name in list_annual_metrics():
        spec = get_annual(name)
        click.echo(f"  {name:26s} {spec.units:24s} from {spec.index}")
