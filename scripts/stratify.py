"""Thin batch runner for ``csdv stratify``.

Mirrors the click command in :mod:`csdv_core.stratification.cli`.
"""

from __future__ import annotations

from csdv_core.stratification.cli import cli

if __name__ == "__main__":
    cli()
