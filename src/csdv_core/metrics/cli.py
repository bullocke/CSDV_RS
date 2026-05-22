"""``csdv compute-metrics`` Click command.

Pass-1 driver around :func:`csdv_core.metrics.orchestrator.compute_for_window`.
Writes a metric stack into ``paths.metrics_dir(site, year, window_m)`` via
:func:`csdv_core.io.metrics_io.write_metric_stack`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from csdv_core.io.metrics_io import write_metric_stack
from csdv_core.io.paths import project_paths
from csdv_core.metrics.orchestrator import PASS1_METRICS, compute_for_window

logger = logging.getLogger(__name__)


def _parse_metrics(value: str | None) -> list[str] | None:
    if not value:
        return None
    names = [s.strip() for s in value.split(",") if s.strip()]
    return names or None


@click.command("compute-metrics")
@click.option("--site", required=True, help="Site code (e.g. SCBI).")
@click.option("--year", type=int, required=True, help="NAIP acquisition year.")
@click.option(
    "--window-m",
    type=float,
    required=True,
    help="Analysis window size in meters.",
)
@click.option(
    "--metrics",
    "metrics_csv",
    default=None,
    help=(
        f"Comma-separated metric names. Default: all Pass-1 metrics "
        f"({', '.join(PASS1_METRICS)})."
    ),
)
@click.option(
    "--naip-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override NAIP RGBN raster path (default: resolve from paths).",
)
@click.option(
    "--chm-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override CHM raster path (default: resolve from paths).",
)
@click.option(
    "--crowns-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override crowns vector path (default: resolve from paths).",
)
@click.option(
    "--chm-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale factor applied to CHM values (0.01 for uint16 NAIP-CHM).",
)
@click.option(
    "--naip-band",
    type=int,
    default=4,
    show_default=True,
    help="1-based NAIP band used for GLCM texture (4 = NIR).",
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Overwrite an existing metric stack instead of failing.",
)
def cli(
    site: str,
    year: int,
    window_m: float,
    metrics_csv: str | None,
    naip_path: Path | None,
    chm_path: Path | None,
    crowns_path: Path | None,
    chm_scale: float,
    naip_band: int,
    force: bool,
) -> None:
    """Compute Pass-1 metric rasters for one (site, year, window_m)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    paths = project_paths()
    out_dir = paths.metrics_dir(site, year, window_m)
    manifest = out_dir / "manifest.yaml"
    if manifest.exists() and not force:
        raise click.UsageError(
            f"Metric stack already exists at {out_dir}; pass --force to overwrite."
        )

    metric_names = _parse_metrics(metrics_csv)
    logger.info(
        "compute-metrics site=%s year=%s window_m=%s metrics=%s",
        site,
        year,
        window_m,
        metric_names or list(PASS1_METRICS),
    )
    results = compute_for_window(
        site=site,
        year=year,
        window_m=window_m,
        metric_names=metric_names,
        naip_path=naip_path,
        chm_path=chm_path,
        crowns_path=crowns_path,
        chm_scale=chm_scale,
        naip_band=naip_band,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metric_stack(results, out_dir)
    click.echo(str(out_dir))


__all__ = ["cli"]
