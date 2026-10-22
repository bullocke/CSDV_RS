"""csdv_core.disturbance.cli: Click commands for disturbance detection and figures.

``csdv detect-disturbance --site SITE`` resolves parameters (packaged YAML,
then ``--config``, then explicit options, with the locked sections of a
``--from-run`` source taken from its ``run_params.json``), picks the output folder
(``--output-dir`` or ``$CSDV_RESULTS_ROOT/disturbance/<site>``) and calls
:func:`csdv_core.disturbance.pipeline.run_detection`. An existing output
folder is reported as an error that names ``--output-dir``.
``csdv disturbance-figures --run-dir RUN`` rebuilds a run's figures and
report. Heavy modules are imported inside the command bodies so ``csdv
--help`` stays fast.

Inputs: command-line options. Outputs: a run folder, its path echoed on
stdout.
"""

from __future__ import annotations

import logging
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

#: Click option name to dotted config key for options that override parameters.
OPTION_TO_KEY: dict[str, str] = {
    "input_name": "inputs.input",
    "years": "inputs.years",
    "chm_scale": "inputs.chm_scale",
    "chm_nodata": "inputs.chm_nodata",
    "grain_m": "grid.grain_m",
    "reference_year": "grid.reference_year",
    "method": "standardize.method",
    "window_m": "standardize.window_m",
    "rule": "decide.rule",
    "p_boundary": "decide.p_boundary",
    "mmu_ha": "mmu.mmu_ha",
    "growth_mmu_ha": "mmu.growth_mmu_ha",
    "refine": "refine.enabled",
    "figures": "figures.enabled",
    "naip_backdrop": "figures.naip_backdrop",
}
#: Accepted ``--input`` values.
INPUT_CHOICES: tuple[str, ...] = ("d_chm_h", "d_chm_cov", "d_ndvi")
#: Accepted ``--method`` values.
METHOD_CHOICES: tuple[str, ...] = ("scene", "scenemean", "local", "local_twopass")
#: Accepted ``--rule`` values.
RULE_CHOICES: tuple[str, ...] = (
    "D1",
    "D2",
    "D5",
    "D5s",
    "D6",
    "D6s",
    "D7",
    "D7s",
)
#: Accepted ``--log-level`` values.
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
#: Format of log records on the console.
CONSOLE_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def parse_years(text: str | None) -> list[int] | None:
    """Parse ``--years`` (``"2013,2015,2017"``) into a list of ints.

    Args:
        text: Comma-separated years, or None.

    Returns:
        The years, or None when ``text`` is None or empty.

    Raises:
        click.BadParameter: If an entry is not an integer.
    """
    if text is None:
        return None
    stripped_text = text.strip()
    if not stripped_text:
        return None
    years: list[int] = []
    for entry in stripped_text.split(","):
        cleaned = entry.strip()
        try:
            year = int(cleaned)
        except ValueError as error:
            raise click.BadParameter(
                f"{cleaned!r} is not a year", param_hint="--years"
            ) from error
        years.append(year)
    return years


def cli_overrides_from_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Map Click option values to dotted config keys with :data:`OPTION_TO_KEY`.

    Options not in the mapping (``site``, ``output_dir``, ``config``, ...) are
    ignored. None values are kept (``resolve_params`` drops them), and
    ``years`` is parsed with :func:`parse_years`.

    Args:
        options: Click option name to value.

    Returns:
        Dotted-key overrides for
        :func:`csdv_core.disturbance.params.resolve_params`.
    """
    overrides: dict[str, Any] = {}
    for option_name, dotted_key in OPTION_TO_KEY.items():
        if option_name not in options:
            continue
        value = options[option_name]
        if option_name == "years":
            value = parse_years(value)
        overrides[dotted_key] = value
    return overrides


def default_output_dir(site: str) -> Path:
    """Default run folder of a site, ``project_paths().disturbance_dir(site)``.

    Args:
        site: Site name.

    Returns:
        ``$CSDV_RESULTS_ROOT/disturbance/<site>``.
    """
    from csdv_core.io.paths import project_paths

    paths = project_paths()
    return paths.disturbance_dir(site)


def configure_console_logging(level_name: str) -> int:
    """Send log records at ``level_name`` and above to stderr.

    Args:
        level_name: One of :data:`LOG_LEVELS`.

    Returns:
        The numeric logging level.
    """
    level = logging.getLevelName(level_name)
    logging.basicConfig(level=level, format=CONSOLE_LOG_FORMAT)
    return level


@click.command("detect-disturbance")
@click.option("--site", required=True, help="Site name under naip_chm/.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="New run folder (default $CSDV_RESULTS_ROOT/disturbance/<site>).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML with parameter overrides (any subset of disturbance.yaml).",
)
@click.option(
    "--from-run",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Reuse layers and z of a saved run, with its input, grid, forest and "
    "standardize parameters.",
)
@click.option(
    "--input",
    "input_name",
    type=click.Choice(INPUT_CHOICES),
    default=None,
    help="Detection input (default d_chm_h).",
)
@click.option("--years", default=None, help="Comma-separated years (default all).")
@click.option(
    "--reference-year", type=int, default=None, help="Year that sets the grid."
)
@click.option(
    "--chm-scale", type=float, default=None, help="Stored CHM value to metres."
)
@click.option("--chm-nodata", type=float, default=None, help="CHM nodata value.")
@click.option("--grain-m", type=float, default=None, help="Cell size (m).")
@click.option(
    "--method",
    type=click.Choice(METHOD_CHOICES),
    default=None,
    help="Standardization method (default local_twopass).",
)
@click.option("--window-m", type=float, default=None, help="Local window width (m).")
@click.option(
    "--rule",
    type=click.Choice(RULE_CHOICES),
    default=None,
    help="Decision rule (default D6).",
)
@click.option(
    "--p-boundary", type=float, default=None, help="Decision probability boundary."
)
@click.option(
    "--mmu-ha", type=float, default=None, help="Minimum mapping unit per interval (ha)."
)
@click.option(
    "--growth-mmu-ha",
    type=float,
    default=None,
    help="Minimum (ha) for objects that touch or overlap an earlier detection.",
)
@click.option("--refine/--no-refine", default=None, help="Refine outlines at 0.6 m.")
@click.option("--figures/--no-figures", default=None, help="Write the PNG figures.")
@click.option(
    "--naip-backdrop/--no-naip-backdrop",
    default=None,
    help="Draw NAIP RGB under the maps.",
)
@click.option("--log-level", type=click.Choice(LOG_LEVELS), default="INFO")
def detect_disturbance(**options: Any) -> None:
    """Map forest disturbance onset for one site from NAIP-CHM years."""
    from csdv_core.disturbance.params import resolve_params
    from csdv_core.disturbance.pipeline import run_detection
    from csdv_core.disturbance.store import read_run_params

    log_level = configure_console_logging(options["log_level"])
    site = options["site"]
    config_path = options["config_path"]
    from_run = options["from_run"]
    overrides = cli_overrides_from_options(options)
    try:
        saved_params = None
        if from_run is not None:
            saved_run_params = read_run_params(from_run)
            saved_params = saved_run_params["params"]
        params = resolve_params(config_path, overrides, saved_params=saved_params)
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(f"Invalid parameters: {error}") from error
    output_dir = options["output_dir"]
    if output_dir is None:
        output_dir = default_output_dir(site)
    command_line = shlex.join(sys.argv)
    try:
        run_detection(
            params,
            site,
            output_dir,
            from_run=from_run,
            command_line=command_line,
            config_path=config_path,
            log_level=log_level,
        )
    except FileExistsError as error:
        raise click.ClickException(
            f"Output folder {output_dir} already exists. Runs never overwrite a "
            "folder, pass a new --output-dir."
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(str(output_dir))


@click.command("disturbance-figures")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run folder written by detect-disturbance.",
)
@click.option(
    "--naip-backdrop/--no-naip-backdrop",
    default=None,
    help="Draw NAIP RGB under the maps (default: the saved run's setting).",
)
@click.option("--log-level", type=click.Choice(LOG_LEVELS), default="INFO")
def disturbance_figures(
    run_dir: Path, naip_backdrop: bool | None, log_level: str
) -> None:
    """Rebuild the figures and report.md of a disturbance run."""
    from csdv_core.disturbance.pipeline import rebuild_figures

    configure_console_logging(log_level)
    try:
        figure_paths = rebuild_figures(run_dir, naip_backdrop=naip_backdrop)
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    figure_count = len(figure_paths)
    logger.info("Rebuilt %d figures in %s", figure_count, run_dir)
    click.echo(str(run_dir))


@click.command("disturbance-chips")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run folder written by detect-disturbance.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="New folder for the chips (default <run-dir>/inspection).",
)
@click.option(
    "--n-detected", type=int, default=24, show_default=True, help="Detected objects."
)
@click.option(
    "--n-below-mmu",
    type=int,
    default=4,
    show_default=True,
    help="Near misses removed by the minimum mapping unit.",
)
@click.option(
    "--n-no-detection",
    type=int,
    default=0,
    show_default=True,
    help="Monitored forest cells never flagged.",
)
@click.option(
    "--min-per-interval",
    type=int,
    default=2,
    show_default=True,
    help="Minimum detected objects per NAIP interval.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Sample seed.")
@click.option(
    "--pad-m", type=float, default=40.0, show_default=True, help="Chip margin (m)."
)
@click.option(
    "--min-chip-m",
    type=float,
    default=150.0,
    show_default=True,
    help="Smallest chip side (m).",
)
@click.option("--chm/--no-chm", default=True, help="Draw the CHM row under the RGB.")
@click.option(
    "--sample-only",
    is_flag=True,
    help="Write samples.csv and samples.gpkg only, without reading imagery.",
)
@click.option("--log-level", type=click.Choice(LOG_LEVELS), default="INFO")
def disturbance_chips(
    run_dir: Path,
    output_dir: Path | None,
    n_detected: int,
    n_below_mmu: int,
    n_no_detection: int,
    min_per_interval: int,
    seed: int,
    pad_m: float,
    min_chip_m: float,
    chm: bool,
    sample_only: bool,
    log_level: str,
) -> None:
    """Sample a disturbance run and draw NAIP RGB and CHM chips for inspection."""
    from csdv_core.disturbance.chips import ChipSettings, make_inspection_chips
    from csdv_core.disturbance.sampling import SampleDesign

    configure_console_logging(log_level)
    design = SampleDesign(
        n_detected=n_detected,
        n_below_mmu=n_below_mmu,
        n_no_detection=n_no_detection,
        min_per_interval=min_per_interval,
        seed=seed,
    )
    settings = ChipSettings(pad_m=pad_m, min_chip_m=min_chip_m, show_chm=chm)
    try:
        result = make_inspection_chips(
            run_dir,
            output_dir,
            design=design,
            settings=settings,
            sample_only=sample_only,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    chip_count = len(result.chip_paths)
    logger.info("Wrote %d chips to %s", chip_count, result.output_dir)
    click.echo(str(result.output_dir))


__all__ = [
    "OPTION_TO_KEY",
    "cli_overrides_from_options",
    "configure_console_logging",
    "default_output_dir",
    "detect_disturbance",
    "disturbance_chips",
    "disturbance_figures",
    "parse_years",
]
