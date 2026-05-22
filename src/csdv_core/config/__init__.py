"""csdv_core.config — packaged YAML configuration with typed loaders.

YAML files live alongside this module and are accessed via
:mod:`importlib.resources` so they ship with the installed package. Each
loader returns a Pydantic model and is cached for the lifetime of the
process; call :func:`reload_config` to clear the cache in tests.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from csdv_core.config._models import (
    MetricsConfig,
    SitesConfig,
    SiteTypesConfig,
    StagesConfig,
    TrajectoriesConfig,
)

__all__ = [
    "config_dir",
    "load_yaml",
    "load_sites",
    "load_site_types",
    "load_metrics",
    "load_stages",
    "load_trajectories",
    "reload_config",
    "SitesConfig",
    "SiteTypesConfig",
    "MetricsConfig",
    "StagesConfig",
    "TrajectoriesConfig",
]


def config_dir() -> Path:
    """Return the on-disk directory containing the packaged YAML files."""
    return Path(str(resources.files("csdv_core.config")))


def load_yaml(name: str) -> dict[str, Any]:
    """Load a packaged YAML file by stem (e.g. ``"sites"``)."""
    path = config_dir() / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config {name}.yaml must parse to a mapping, got {type(data).__name__}"
        )
    return data


@lru_cache(maxsize=1)
def load_sites() -> SitesConfig:
    """Load and validate ``sites.yaml``."""
    return SitesConfig.model_validate(load_yaml("sites"))


@lru_cache(maxsize=1)
def load_site_types() -> SiteTypesConfig:
    """Load and validate ``site_types.yaml``."""
    return SiteTypesConfig.model_validate(load_yaml("site_types"))


@lru_cache(maxsize=1)
def load_metrics() -> MetricsConfig:
    """Load and validate ``metrics.yaml``."""
    return MetricsConfig.model_validate(load_yaml("metrics"))


@lru_cache(maxsize=1)
def load_stages() -> StagesConfig:
    """Load and validate ``stages.yaml``."""
    return StagesConfig.model_validate(load_yaml("stages"))


@lru_cache(maxsize=1)
def load_trajectories() -> TrajectoriesConfig:
    """Load and validate ``trajectories.yaml``."""
    return TrajectoriesConfig.model_validate(load_yaml("trajectories"))


def reload_config() -> None:
    """Clear cached config loaders. Intended for tests."""
    for fn in (
        load_sites,
        load_site_types,
        load_metrics,
        load_stages,
        load_trajectories,
    ):
        fn.cache_clear()
