"""Thin batch runner for ``csdv download chm``.

Mirrors the click command in :mod:`csdv_core.download.chm_model`.
"""

from __future__ import annotations

from csdv_core.download.chm_model import cli

if __name__ == "__main__":
    cli()
