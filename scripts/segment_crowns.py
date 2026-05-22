"""Thin batch runner for ``csdv segment-crowns``.

Mirrors the click command in :mod:`csdv_core.segmentation.cli`.
"""

from __future__ import annotations

from csdv_core.segmentation.cli import cli

if __name__ == "__main__":
    cli()
