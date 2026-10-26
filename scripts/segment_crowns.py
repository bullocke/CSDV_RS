"""Thin batch runner for ``csdv segment-crowns``.

Runs :func:`naipcrowns.cli.segment`, the command CSDV attaches as
``csdv segment-crowns``.
"""

from __future__ import annotations

from naipcrowns.cli import segment

if __name__ == "__main__":
    segment()
