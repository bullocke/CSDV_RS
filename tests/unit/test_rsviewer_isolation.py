"""rsviewer must not depend on project code, so it can become its own package."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("leafmap")

MODULES = [
    "rsviewer",
    "rsviewer.cog",
    "rsviewer.config",
    "rsviewer.ee",
    "rsviewer.layers",
    "rsviewer.panels",
    "rsviewer.query",
    "rsviewer.tiles",
    "rsviewer.viewer",
]


def test_rsviewer_does_not_import_csdv_core() -> None:
    code = (
        "import importlib, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'csdv_core')\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
