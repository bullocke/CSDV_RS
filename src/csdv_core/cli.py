"""csdv_core.cli — Click-based command-line entry point.

Subcommands are attached via :func:`click.Group.add_command` so each one
keeps its own option parser. Heavy submodules are imported lazily inside
factory functions to keep ``csdv --help`` cheap on Colab and CHPC login
nodes.
"""

from __future__ import annotations

import click


@click.group()
@click.version_option()
def main() -> None:
    """CSDV forest disturbance classification toolkit."""


@main.command()
def info() -> None:
    """Print package version and resolved project paths."""
    from csdv_core import __version__
    from csdv_core.io.paths import project_paths

    p = project_paths()
    click.echo(f"csdv_core {__version__}")
    try:
        import naipcrowns

        click.echo(f"naipcrowns {naipcrowns.__version__} ({naipcrowns.__file__})")
    except ImportError:
        click.echo("naipcrowns not installed (csdv segment-crowns needs it)")
    click.echo(f"  data_root:    {p.data_root}")
    click.echo(f"  results_root: {p.results_root}")
    click.echo(f"  cache_root:   {p.cache_root}")


@main.group()
def download() -> None:
    """Download external imagery (NAIP, NAIP-CHM)."""


def _attach_download_subcommands() -> None:
    """Lazy-attach download subcommands so ``ee`` is only imported when used."""
    from csdv_core.download.chm_model import cli as _chm_cli
    from csdv_core.download.chm_model import conditioning_cli as _cond_cli
    from csdv_core.download.naip_auto import cli as _naip_auto_cli
    from csdv_core.download.naip_gee import cli as _naip_cli
    from csdv_core.download.naip_pc import cli as _naip_pc_cli
    from csdv_core.download.naip_rest import cli as _naip_rest_cli

    download.add_command(_naip_cli, name="naip")
    download.add_command(_naip_pc_cli, name="naip-pc")
    download.add_command(_naip_rest_cli, name="naip-rest")
    download.add_command(_naip_auto_cli, name="naip-auto")
    download.add_command(_chm_cli, name="chm")
    download.add_command(_cond_cli, name="conditioning")


_NAIPCROWNS_MISSING = (
    "csdv segment-crowns runs naipcrowns, which is not installed. Clone "
    "https://github.com/bullocke/naip-crowns next to this repository and run "
    "<env>/bin/python -m pip install --no-deps --no-build-isolation -e ../naip-crowns"
)


def _attach_segmentation() -> None:
    """Attach ``naipcrowns segment`` under its CSDV name, ``segment-crowns``.

    Crown segmentation moved to the naipcrowns package (a sibling clone). When
    it is missing, the command still appears and says how to install it, so
    ``csdv --help`` and ``csdv check`` keep working.
    """
    try:
        from naipcrowns.cli import segment as _seg_cli
    except ImportError:

        @click.command(
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
        )
        def _seg_cli() -> None:
            """Segment tree crowns (needs the naipcrowns package)."""
            raise click.ClickException(_NAIPCROWNS_MISSING)

    main.add_command(_seg_cli, name="segment-crowns")


def _attach_chm_inference() -> None:
    from csdv_core.chm_inference.infer import cli as _inf_cli

    main.add_command(_inf_cli, name="chm-inference")


def _attach_stratification() -> None:
    from csdv_core.stratification.cli import cli as _strat_cli

    main.add_command(_strat_cli, name="stratify")


def _attach_metrics() -> None:
    from csdv_core.metrics.cli import cli as _metrics_cli

    main.add_command(_metrics_cli, name="compute-metrics")


def _attach_stages() -> None:
    from csdv_core.stages.cli import cli as _stages_cli

    main.add_command(_stages_cli, name="classify-stages")


def _attach_trajectories() -> None:
    from csdv_core.trajectories.cli import cli as _traj_cli

    main.add_command(_traj_cli, name="classify-trajectories")


def _attach_satellite() -> None:
    """Attach the satellite group. Only its ``fetch`` command imports ``ee``."""
    from csdv_core.satellite.cli import cli as _sat_cli

    main.add_command(_sat_cli, name="satellite")


def _attach_check() -> None:
    from csdv_core.check import cli as _check_cli

    main.add_command(_check_cli, name="check")


def _attach_disturbance() -> None:
    """Attach the ``detect-disturbance``, ``disturbance-figures`` and ``disturbance-chips`` commands.

    The pipeline modules (rasterio, geopandas, matplotlib) are imported inside
    the command bodies, so attaching costs only the Click definitions.
    """
    from csdv_core.disturbance.cli import (
        detect_disturbance,
        disturbance_chips,
        disturbance_figures,
    )

    main.add_command(detect_disturbance, name="detect-disturbance")
    main.add_command(disturbance_figures, name="disturbance-figures")
    main.add_command(disturbance_chips, name="disturbance-chips")


def _attach_grid() -> None:
    from csdv_core.grid.cli import grid as _grid_cli

    main.add_command(_grid_cli, name="grid")


# Attach now: imports are cheap (click + module-level imports only); heavy
# imports (ee, torch, rasterio in segmentation) happen inside the command
# bodies, not at import time.
_attach_download_subcommands()
_attach_segmentation()
_attach_chm_inference()
_attach_stratification()
_attach_metrics()
_attach_stages()
_attach_trajectories()
_attach_satellite()
_attach_check()
_attach_disturbance()
_attach_grid()


if __name__ == "__main__":
    main()
