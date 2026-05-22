"""Thin batch runner for ``csdv compute-metrics``.

Mirrors the click command in :mod:`csdv_core.metrics.cli`.
"""

from __future__ import annotations

from csdv_core.metrics.cli import cli

if __name__ == "__main__":
    cli()
