"""Thin batch runner for ``csdv classify-stages``.

Mirrors the click command in :mod:`csdv_core.stages.cli`.
"""

from __future__ import annotations

from csdv_core.stages.cli import cli

if __name__ == "__main__":
    cli()
