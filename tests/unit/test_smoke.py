"""Smoke test: package imports and version is set."""

from __future__ import annotations


def test_import_csdv_core() -> None:
    import csdv_core

    assert hasattr(csdv_core, "__version__")
    assert isinstance(csdv_core.__version__, str)


def test_cli_importable() -> None:
    from csdv_core.cli import main

    assert callable(main)


def test_segment_crowns_is_the_naipcrowns_command() -> None:
    """``csdv segment-crowns`` runs ``naipcrowns segment`` (sibling clone)."""
    from click.testing import CliRunner

    from csdv_core.cli import main

    result = CliRunner().invoke(main, ["segment-crowns", "--help"])
    assert result.exit_code == 0, result.output
    assert "--tallo-biome" in result.output


def test_moved_commands_are_gone() -> None:
    """benchmark-crowns and download naip-chm-http moved to naipcrowns."""
    from csdv_core.cli import download, main

    assert "benchmark-crowns" not in main.commands
    assert "naip-chm-http" not in download.commands
