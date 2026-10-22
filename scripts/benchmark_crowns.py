"""Thin batch runner for ``csdv benchmark-crowns``.

Mirrors the click command in :mod:`csdv_core.national.cli`.
"""

from __future__ import annotations

from csdv_core.national.cli import cli

if __name__ == "__main__":
    cli()
