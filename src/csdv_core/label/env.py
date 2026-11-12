"""csdv_core.label.env: API keys, read from the environment or ``~/.bashrc``.

Keys are never logged or written to a file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_EXPORT_RE = re.compile(r"^\s*export\s+(\w+)=(.*)$")

__all__ = ["api_key"]


def api_key(name: str) -> str:
    """Return the key called ``name``.

    A non-interactive shell does not source ``~/.bashrc``, so the export
    lines are parsed directly when the variable is not already set.

    Raises:
        KeyError: If the key is in neither place.
    """
    value = os.environ.get(name)
    if value:
        return value
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        for line in bashrc.read_text().splitlines():
            match = _EXPORT_RE.match(line)
            if match and match[1] == name:
                return match[2].strip().strip("'\"")
    raise KeyError(f"{name} is not set and not exported in ~/.bashrc")
