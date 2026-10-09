"""rsviewer.config: YAML layer catalogues.

A catalogue lists a project's static layers so they do not have to be built in
code. Strings may use ``${ENV}`` or ``${ENV:-default}`` for environment
variables and ``{name}`` for catalogue variables (``vars``) and loop variables
(``foreach``). Relative paths resolve against the YAML file's folder. Example::

    title: ElkinsvilleNE
    zoom: 14
    vars:
      cache: ${CSDV_CACHE_ROOT:-../.cache}/viewer/ElkinsvilleNE
    layers:
      - type: raster
        name: NAIP {year}
        path: "{cache}/naip/naip_{year}.tif"
        foreach: {year: [2016, 2018]}
        visible: {year: [2018]}
        group: NAIP
        style: {bands: [1, 2, 3]}
      - type: vector
        name: Reference
        path: ../data/reference.gpkg
        layer: disturbance
        style:
          by: chg_type
          categories: {stand_replacing: [Stand-replacing, "#e41a1c"]}
        popup: [ref_id, chg_type]
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rsviewer.layers import (
    AnyLayer,
    Category,
    RasterLayer,
    RasterStyle,
    TileLayerSpec,
    VectorLayer,
    VectorStyle,
)

logger = logging.getLogger(__name__)

_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


@dataclass
class Catalog:
    """A loaded catalogue: map settings plus layer specs in draw order."""

    title: str = ""
    center: tuple[float, float] | None = None
    zoom: int = 13
    basemap: str = "Esri World Imagery"
    layers: list[AnyLayer] = field(default_factory=list)
    vars: dict[str, str] = field(default_factory=dict)


def expand_env(text: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` with environment values.

    Raises:
        KeyError: A variable without a default is not set.
    """

    def sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise KeyError(f"environment variable {name} is not set")
        return default

    return _ENV.sub(sub, text)


def _fill(value: Any, ctx: dict[str, Any]) -> Any:
    """Format every string in a nested structure with ``ctx``."""
    if isinstance(value, str):
        return (
            expand_env(value).format(**ctx) if ("{" in value or "$" in value) else value
        )
    if isinstance(value, list):
        return [_fill(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, ctx) for k, v in value.items()}
    return value


def _resolve(path: str, root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _categories(raw: dict | None, key_type: type) -> dict | None:
    if not raw:
        return None
    out = {}
    for k, v in raw.items():
        label, color = (v["label"], v["color"]) if isinstance(v, dict) else v
        out[key_type(k)] = Category(str(label), str(color))
    return out


def _raster_style(raw: dict) -> RasterStyle:
    raw = dict(raw)
    if "bands" in raw:
        raw["bands"] = tuple(raw["bands"])
    raw["categories"] = _categories(raw.get("categories"), int)
    return RasterStyle(**raw)


def _vector_style(raw: dict) -> VectorStyle:
    raw = dict(raw)
    raw["categories"] = _categories(raw.get("categories"), str)
    return VectorStyle(**raw)


def _visible(raw: Any, loop: dict[str, Any]) -> bool:
    if isinstance(raw, dict):
        return all(
            loop.get(k) in (v if isinstance(v, list) else [v]) for k, v in raw.items()
        )
    return bool(raw)


def _layer(
    raw: dict, loop: dict[str, Any], ctx: dict[str, Any], root: Path
) -> AnyLayer:
    vis_raw = raw.get("visible", True)
    d = _fill({k: v for k, v in raw.items() if k not in ("foreach", "visible")}, ctx)
    kind = d.pop("type")
    d["visible"] = _visible(vis_raw, loop)
    if kind == "raster":
        d["path"] = _resolve(d["path"], root)
        d["style"] = _raster_style(d.get("style", {}))
        return RasterLayer(**d)
    if kind == "vector":
        d["path"] = _resolve(d["path"], root)
        d["style"] = _vector_style(d.get("style", {}))
        return VectorLayer(**d)
    if kind == "tiles":
        d["legend"] = _categories(d.get("legend"), str)
        return TileLayerSpec(**d)
    raise ValueError(f"unknown layer type {kind!r} (raster, vector or tiles)")


def load_catalog(path: Path, variables: dict[str, Any] | None = None) -> Catalog:
    """Load a YAML catalogue.

    Args:
        path: YAML file.
        variables: Extra ``{name}`` variables. They override ``vars`` entries
            of the same name.

    Returns:
        The catalogue with every ``foreach`` expanded.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: A layer has an unknown type.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text()) or {}
    root = path.parent.resolve()

    ctx: dict[str, Any] = {}
    for k, v in (raw.get("vars") or {}).items():
        ctx[k] = _fill(str(v), ctx)
    ctx.update(variables or {})

    layers: list[AnyLayer] = []
    for item in raw.get("layers") or []:
        loops = item.get("foreach") or {}
        keys = list(loops)
        for combo in itertools.product(*(loops[k] for k in keys)) if keys else [()]:
            loop = dict(zip(keys, combo, strict=True))
            layers.append(_layer(item, loop, {**ctx, **loop}, root))

    center = raw.get("center")
    return Catalog(
        title=str(raw.get("title", "")),
        center=tuple(center) if center else None,
        zoom=int(raw.get("zoom", 13)),
        basemap=str(raw.get("basemap", "Esri World Imagery")),
        layers=layers,
        vars={k: str(v) for k, v in ctx.items()},
    )
