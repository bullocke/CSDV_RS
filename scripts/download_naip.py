"""Thin batch runner for ``csdv download naip``.

Mirrors the click command in :mod:`csdv_core.download.naip_gee` so the
script is runnable as either ``python scripts/download_naip.py ...`` or
``csdv download naip ...`` on CHPC / Colab with identical flags.
"""

from __future__ import annotations

from csdv_core.download.naip_gee import cli

if __name__ == "__main__":
    cli()
