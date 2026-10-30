"""csdv_core.grid.cli — ``csdv grid``: locate, submit, status, export, check."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

import click

__all__ = ["grid"]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )


def _parse_years(years: str | None) -> tuple[int, ...] | None:
    if not years:
        return None
    try:
        return tuple(int(t) for t in years.split(",") if t.strip())
    except ValueError as exc:
        raise click.BadParameter(f"not a list of years: {years!r}") from exc


@click.group()
def grid() -> None:
    """CONUS processing grid: one command per unit from NAIP to exports.

    A unit is a 7.5' USGS quad cell (ID like q4407159) or a box around a
    point (ID p_<label>). See docs/guides/grid_processing.md.
    """


@grid.command("locate")
@click.option("--lonlat", nargs=2, type=float, default=None, help="LON LAT.")
@click.option("--cell", default=None, help="Cell ID, e.g. q4407159.")
def locate(lonlat: tuple[float, float] | None, cell: str | None) -> None:
    """Print the cell that contains a point, or a cell's bounds."""
    from csdv_core.grid.cells import quad_bounds, quad_id_at

    if (lonlat is None) == (cell is None):
        raise click.UsageError("Give exactly one of --lonlat or --cell")
    if lonlat is not None:
        cell = quad_id_at(*lonlat)
    assert cell is not None
    west, south, east, north = quad_bounds(cell)
    click.echo(f"{cell}  bbox W S E N: {west:.4f} {south:.4f} {east:.4f} {north:.4f}")


@grid.command("submit", context_settings={"max_content_width": 100})
@click.option(
    "--batch",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch CSV (label,lon,lat,buffer_km,cell,notes).",
)
@click.option("--cell", "cells", multiple=True, help="Cell ID; repeatable.")
@click.option("--lonlat", nargs=2, type=float, default=None, help="LON LAT of a point.")
@click.option(
    "--buffer-km",
    type=float,
    default=None,
    help="With --lonlat: half-width of a point unit. Without: the containing cell.",
)
@click.option("--label", default="", help="Unit label (required for a point unit).")
@click.option("--notes", default="", help="Free-text notes for the unit record.")
@click.option("--run-tag", default="default", show_default=True)
@click.option("--years", default=None, help="Comma-separated NAIP years.")
@click.option("--start-year", default=2010, show_default=True, type=int)
@click.option(
    "--min-coverage",
    default=0.9,
    show_default=True,
    type=float,
    help="Skip NAIP years covering less of the unit than this share.",
)
@click.option("--replan", is_flag=True, help="Ask the NAIP catalogues again.")
@click.option(
    "--steps",
    default="download,chm,detect,chips,export",
    show_default=True,
    help="Steps to consider; finished steps are skipped anyway.",
)
@click.option("--no-chips", is_flag=True, help="Leave out the chips step.")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="detect-disturbance --config YAML.",
)
@click.option("--no-quicklooks", is_flag=True, help="Export without quicklooks.")
@click.option(
    "--gpu",
    type=click.Choice(
        ["notchpeak", "kingspeak", "lonepeak", "notchpeak-guest", "granite"]
    ),
    default="notchpeak",
    show_default=True,
    help="GPU partition for CHM inference (guest and granite are preemptable).",
)
@click.option("--dry-run", is_flag=True, help="Print the jobs; submit nothing.")
@click.argument("detect_args", nargs=-1, type=click.UNPROCESSED)
def submit(
    batch: Path | None,
    cells: tuple[str, ...],
    lonlat: tuple[float, float] | None,
    buffer_km: float | None,
    label: str,
    notes: str,
    run_tag: str,
    years: str | None,
    start_year: int,
    min_coverage: float,
    replan: bool,
    steps: str,
    no_chips: bool,
    config: Path | None,
    no_quicklooks: bool,
    gpu: str,
    dry_run: bool,
    detect_args: tuple[str, ...],
) -> None:
    """Submit the download, CHM, detection, chips and export jobs of units.

    Options after `--` go to detect-disturbance, e.g.
    `csdv grid submit --cell q4407159 --run-tag D6s_1ha -- --rule D6s --mmu-ha 1.0`.
    Run on a CHPC login node after `source scripts/chpc/env.sh`.
    """
    from csdv_core.grid.submit import SubmitError, SubmitOptions, submit_unit
    from csdv_core.grid.units import Unit, grid_paths, read_batch

    _setup_logging()
    units: list[Unit] = []
    if batch is not None:
        units.extend(read_batch(batch))
    units.extend(Unit.cell(c, label=label, notes=notes) for c in cells)
    if lonlat is not None:
        if buffer_km:
            units.append(Unit.point(*lonlat, buffer_km, label=label, notes=notes))
        else:
            units.append(Unit.cell_at(*lonlat, label=label, notes=notes))
    if not units:
        raise click.UsageError("Give --batch, --cell or --lonlat")
    step_list = [s.strip() for s in steps.split(",") if s.strip()]
    if no_chips:
        step_list = [s for s in step_list if s != "chips"]
    opts = SubmitOptions(
        run_tag=run_tag,
        steps=tuple(step_list),
        years=_parse_years(years),
        start_year=start_year,
        min_coverage=min_coverage,
        replan=replan,
        config=config.resolve() if config else None,
        detect_args=tuple(detect_args),
        quicklooks=not no_quicklooks,
        gpu=gpu,
        dry_run=dry_run,
    )
    paths = grid_paths()
    failures = []
    for unit in units:
        click.echo(f"\n== {unit.unit_id} {unit.label}".rstrip())
        try:
            sub = submit_unit(paths, unit, opts)
        except (SubmitError, ValueError) as exc:
            click.echo(f"   NOT SUBMITTED: {exc}")
            failures.append(unit.unit_id)
            continue
        w, s, e, n = sub.unit.bbox_4326
        click.echo(f"   bbox {w:.4f} {s:.4f} {e:.4f} {n:.4f}")
        planned = ", ".join(
            f"{y} ({v['source']})" for y, v in sorted(sub.unit.years.items())
        )
        click.echo(f"   years {planned}")
        for year, reason in sorted(sub.unit.skipped_years.items()):
            click.echo(f"   skipped {year}: {reason}")
        for step, why in sub.skipped.items():
            click.echo(f"   {step}: skipped ({why})")
        for job in sub.jobs:
            if dry_run:
                click.echo(f"   {job.step}: {shlex.join(job.command)}")
            else:
                click.echo(f"   {job.step}: job {job.job_id}")
        if not sub.jobs:
            click.echo("   nothing to submit")
    if dry_run:
        click.echo("\nDry run: nothing submitted, no records written.")
    if failures:
        raise click.ClickException(f"Not submitted: {', '.join(failures)}")


@grid.command("status")
@click.option("--run-tag", default=None, help="Only this run tag.")
@click.option(
    "--batch",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Only the units of this batch file.",
)
@click.option(
    "--problems", is_flag=True, help="Only rows that are not done or running."
)
@click.option("--no-sacct", is_flag=True, help="Do not ask Slurm about jobs.")
@click.option("--quiet", is_flag=True, help="Write status.csv without printing.")
def status(
    run_tag: str | None,
    batch: Path | None,
    problems: bool,
    no_sacct: bool,
    quiet: bool,
) -> None:
    """Show the progress of every unit and run tag; write status.csv."""
    import pandas as pd

    from csdv_core.grid.status import collect_status, write_status
    from csdv_core.grid.units import grid_paths, read_batch

    paths = grid_paths()
    unit_ids = [u.unit_id for u in read_batch(batch)] if batch else None
    rows = collect_status(
        paths, unit_ids=unit_ids, run_tag=run_tag, use_sacct=not no_sacct
    )
    # Only a full listing replaces status.csv.
    if unit_ids is None and run_tag is None:
        write_status(paths, rows)
    if quiet:
        return
    if not rows:
        click.echo(f"No units under {paths.grid_dir / 'units'}")
        return
    table = pd.DataFrame([vars(r) for r in rows]).drop(columns="run_dir")
    if problems:
        active = ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING")
        table = table[
            (table["state"] != "done")
            & ~table["state"].str.contains("|".join(active), regex=True)
        ]
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        click.echo(table.to_string(index=False))
    n_done = int((pd.DataFrame([vars(r) for r in rows])["state"] == "done").sum())
    click.echo(f"\n{n_done}/{len(rows)} done. Ledger: {paths.ledger}")


@grid.command("export")
@click.option("--unit", "unit_id", required=True, help="Unit ID.")
@click.option("--run-tag", default="default", show_default=True)
@click.option("--no-quicklooks", is_flag=True)
@click.option("--quicklook-res-m", default=3.0, show_default=True, type=float)
@click.option("--overwrite-quicklooks", is_flag=True)
def export(
    unit_id: str,
    run_tag: str,
    no_quicklooks: bool,
    quicklook_res_m: float,
    overwrite_quicklooks: bool,
) -> None:
    """Write a run's small download set to <export dir>/runs/<tag>/<unit>/."""
    from csdv_core.grid.export import export_run
    from csdv_core.grid.units import grid_paths

    _setup_logging()
    try:
        result = export_run(
            grid_paths(),
            unit_id,
            run_tag,
            quicklooks=not no_quicklooks,
            quicklook_res_m=quicklook_res_m,
            overwrite_quicklooks=overwrite_quicklooks,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{result.export_dir} ({result.size_mb:.1f} MB)")


@grid.command("check")
@click.option(
    "--export-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Export folder. Default: $CSDV_EXPORT_DIR, else results/chpc_exports.",
)
@click.option("--unit", "unit_id", default=None, help="Only this unit.")
@click.option("--run-tag", default=None, help="Only this run tag.")
@click.option("--no-write", is_flag=True, help="Do not write check.md/check.json.")
@click.option("--verbose", is_flag=True, help="Print the detail of passed checks.")
def check(
    export_dir: Path | None,
    unit_id: str | None,
    run_tag: str | None,
    no_write: bool,
    verbose: bool,
) -> None:
    """Check exported runs for completeness and internal agreement.

    Writes check.md and check.json into each run's export folder and exits
    with an error when any check fails.
    """
    import os

    from csdv_core.grid.check import FAIL, PASS, check_run, find_runs, write_check
    from csdv_core.io.paths import project_paths

    _setup_logging()
    if export_dir is None:
        env = os.environ.get("CSDV_EXPORT_DIR")
        export_dir = Path(env) if env else project_paths().results_root / "chpc_exports"
    runs = find_runs(export_dir, unit_id, run_tag)
    if not runs:
        raise click.ClickException(f"No exported runs under {export_dir / 'runs'}")
    failed = []
    for unit, tag in runs:
        result = check_run(export_dir, unit, tag)
        click.echo(f"\n== {unit} {tag}: {result.status.upper()}")
        for r in result.results:
            if verbose or r.status != PASS:
                click.echo(f"   {r.status:<4}  {r.name}: {r.detail}")
            else:
                click.echo(f"   {r.status:<4}  {r.name}")
        if not no_write:
            write_check(result)
        if result.status == FAIL:
            failed.append(f"{unit} {tag}")
    if failed:
        raise click.ClickException(f"Checks failed: {', '.join(failed)}")
