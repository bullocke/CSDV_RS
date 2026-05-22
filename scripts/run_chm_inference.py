"""Thin batch runner for ``csdv chm-inference``.

Mirrors the click command in :mod:`csdv_core.chm_inference.infer`.
"""

from __future__ import annotations

from csdv_core.chm_inference.infer import cli

if __name__ == "__main__":
    cli()
