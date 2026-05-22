"""Thin batch runner for ``csdv classify-trajectories``.

Mirrors the click command in :mod:`csdv_core.trajectories.cli`.
"""

from __future__ import annotations

from csdv_core.trajectories.cli import cli

if __name__ == "__main__":
    cli()
