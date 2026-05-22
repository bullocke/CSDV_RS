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
    click.echo(f"  data_root:    {p.data_root}")
    click.echo(f"  results_root: {p.results_root}")
    click.echo(f"  cache_root:   {p.cache_root}")


@main.group()
def download() -> None:
    """Download external imagery (NAIP, NAIP-CHM)."""


def _attach_download_subcommands() -> None:
    """Lazy-attach download subcommands so ``ee`` is only imported when used."""
    from csdv_core.download.chm_model import cli as _chm_cli
    from csdv_core.download.naip_gee import cli as _naip_cli

    download.add_command(_naip_cli, name="naip")
    download.add_command(_chm_cli, name="chm")


def _attach_segmentation() -> None:
    from csdv_core.segmentation.cli import cli as _seg_cli

    main.add_command(_seg_cli, name="segment-crowns")


def _attach_chm_inference() -> None:
    from csdv_core.chm_inference.infer import cli as _inf_cli

    main.add_command(_inf_cli, name="chm-inference")


def _attach_stratification() -> None:
    from csdv_core.stratification.cli import cli as _strat_cli

    main.add_command(_strat_cli, name="stratify")


def _attach_stages() -> None:
    from csdv_core.stages.cli import cli as _stages_cli

    main.add_command(_stages_cli, name="classify-stages")


def _attach_trajectories() -> None:
    from csdv_core.trajectories.cli import cli as _traj_cli

    main.add_command(_traj_cli, name="classify-trajectories")


# Attach now: imports are cheap (click + module-level imports only); heavy
# imports (ee, torch, rasterio in segmentation) happen inside the command
# bodies, not at import time.
_attach_download_subcommands()
_attach_segmentation()
_attach_chm_inference()
_attach_stratification()
_attach_stages()
_attach_trajectories()


if __name__ == "__main__":
    main()
