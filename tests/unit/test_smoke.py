"""Smoke test: package imports and version is set."""

from __future__ import annotations


def test_import_csdv_core() -> None:
    import csdv_core

    assert hasattr(csdv_core, "__version__")
    assert isinstance(csdv_core.__version__, str)


def test_cli_importable() -> None:
    from csdv_core.cli import main

    assert callable(main)
