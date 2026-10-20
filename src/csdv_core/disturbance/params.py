"""csdv_core.disturbance.params: resolve run parameters from YAML and CLI options.

A run's parameters come from three layers, merged in this order: the packaged
``config/disturbance.yaml`` (defaults = best tested case), an optional user
YAML that may hold any subset of keys, and explicit CLI options. Later layers
win key by key (deep merge), and the merged mapping is validated once by
:class:`~csdv_core.config._models.DisturbanceConfig`, which rejects unknown
keys. A ``--from-run`` rerun takes the :data:`REUSE_LOCKED_SECTIONS` from the
saved run instead, and refuses explicit values that contradict them.

Inputs: an optional YAML path, CLI overrides and optional saved parameters.
Outputs: a validated ``DisturbanceConfig`` and its JSON-ready dict form.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from csdv_core.config import load_yaml
from csdv_core.config._models import DisturbanceConfig

logger = logging.getLogger(__name__)

#: Stem of the packaged YAML file that holds the pipeline defaults.
PACKAGED_CONFIG_NAME = "disturbance"
#: Separator for nested keys in CLI overrides (``"decide.rule"``).
KEY_SEPARATOR = "."
#: Sections that shaped the saved layers and z, so ``--from-run`` keeps them.
REUSE_LOCKED_SECTIONS: tuple[str, ...] = ("inputs", "grid", "forest", "standardize")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into a copy of ``base``, recursing into mappings.

    Lists and other non-mapping values in ``override`` replace those in
    ``base``. Neither input is modified.

    Args:
        base: Lower-precedence mapping.
        override: Higher-precedence mapping.

    Returns:
        A new dict holding the merged values.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping to dotted keys.

    ``{"decide": {"rule": "D6"}}`` becomes ``{"decide.rule": "D6"}``. Lists
    and other values are kept as leaf values.

    Args:
        mapping: Nested mapping (e.g. from :func:`params_to_dict`).
        prefix: Dotted prefix for the keys of ``mapping``.

    Returns:
        Flat dict in the insertion order of ``mapping``.
    """
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            nested = flatten_mapping(value, dotted_key)
            flat.update(nested)
        else:
            flat[dotted_key] = value
    return flat


def expand_dotted(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Turn dotted keys into nested dicts and drop None values.

    ``{"decide.rule": "D5", "grid.grain_m": None}`` becomes
    ``{"decide": {"rule": "D5"}}``. None means "option not given" on the
    command line, so it never overrides a value. Dotted and nested spellings
    of the same section are merged.

    Args:
        overrides: CLI overrides keyed by dotted paths or section names.

    Returns:
        Nested dict of the overrides that carry a value.

    Raises:
        ValueError: If a key is empty or has an empty path component.
    """
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(KEY_SEPARATOR)
        if not all(parts):
            raise ValueError(f"Invalid override key {key!r}")
        if isinstance(value, Mapping):
            value = expand_dotted(value)
            if not value:
                continue
        branch: dict[str, Any] = {parts[-1]: value}
        for part in reversed(parts[:-1]):
            branch = {part: branch}
        nested = deep_merge(nested, branch)
    return nested


def read_user_config(config_path: Path) -> dict[str, Any]:
    """Read a user YAML file of parameter overrides.

    Args:
        config_path: YAML file holding a (possibly partial) mapping with the
            sections of ``disturbance.yaml``.

    Returns:
        The parsed mapping (empty for an empty file).

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the file does not parse to a mapping.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        type_name = type(data).__name__
        raise ValueError(f"{config_path} must parse to a mapping, got {type_name}")
    return data


def check_reuse_compatible(
    saved_values: Mapping[str, Any], requested_values: Mapping[str, Any]
) -> None:
    """Refuse locked values that differ from those of a saved run.

    Every key of ``requested_values`` inside :data:`REUSE_LOCKED_SECTIONS` is
    compared with the saved value. Keys missing from ``requested_values`` are
    not checked, so a partial mapping of explicit options can be passed.

    Args:
        saved_values: Nested parameters of the saved run.
        requested_values: Nested parameters or explicit overrides of the rerun.

    Raises:
        ValueError: Listing every differing dotted key with both values.
    """
    differences: list[str] = []
    for section in REUSE_LOCKED_SECTIONS:
        saved_flat = flatten_mapping(saved_values[section], section)
        requested_section = requested_values.get(section, {})
        requested_flat = flatten_mapping(requested_section, section)
        for dotted_key, requested_value in requested_flat.items():
            saved_value = saved_flat.get(dotted_key)
            if requested_value != saved_value:
                differences.append(
                    f"{dotted_key}: saved {saved_value!r}, requested "
                    f"{requested_value!r}"
                )
    if differences:
        detail = "\n  ".join(differences)
        raise ValueError(
            "--from-run reuses the inputs, grid, forest and standardize "
            "parameters of the saved run, so they cannot change. "
            f"Differences:\n  {detail}"
        )


def resolve_params(
    config_path: Path | None,
    cli_overrides: Mapping[str, Any],
    *,
    saved_params: Mapping[str, Any] | None = None,
) -> DisturbanceConfig:
    """Merge packaged defaults, a user YAML and CLI overrides, then validate.

    Precedence (lowest to highest): packaged ``disturbance.yaml``, the YAML at
    ``config_path``, ``cli_overrides``. CLI overrides may use dotted keys
    (``"decide.rule"``) or nested dicts, and None values are ignored, so unset
    Click options can be passed through unchanged. With ``saved_params``
    (a ``--from-run`` rerun), the locked sections are copied from the saved
    run after :func:`check_reuse_compatible` has checked the explicit values.

    Args:
        config_path: Optional user YAML. None uses the packaged defaults only.
        cli_overrides: Explicit command-line values.
        saved_params: Nested parameters of a saved run, or None.

    Returns:
        The validated run parameters.

    Raises:
        FileNotFoundError: If ``config_path`` is given but missing.
        ValueError: If the user YAML is not a mapping, a key is malformed, or
            an explicit locked value differs from ``saved_params``.
        pydantic.ValidationError: If a key is unknown or a value is invalid.
    """
    explicit: dict[str, Any] = {}
    if config_path is not None:
        explicit = read_user_config(config_path)
        logger.info("Read parameter overrides from %s", config_path)
    cli_values = expand_dotted(cli_overrides)
    if cli_values:
        logger.debug("CLI overrides: %s", cli_values)
    explicit = deep_merge(explicit, cli_values)
    packaged = load_yaml(PACKAGED_CONFIG_NAME)
    merged = deep_merge(packaged, explicit)
    if saved_params is not None:
        check_reuse_compatible(saved_params, explicit)
        for section in REUSE_LOCKED_SECTIONS:
            merged[section] = copy.deepcopy(saved_params[section])
    return DisturbanceConfig.model_validate(merged)


def params_to_dict(params: DisturbanceConfig) -> dict[str, Any]:
    """Return the parameters as a JSON-ready nested dict.

    Args:
        params: Validated run parameters.

    Returns:
        Nested dict with one key per config section, which
        ``DisturbanceConfig.model_validate`` turns back into ``params``.
    """
    return params.model_dump(mode="json")


__all__ = [
    "PACKAGED_CONFIG_NAME",
    "REUSE_LOCKED_SECTIONS",
    "check_reuse_compatible",
    "deep_merge",
    "expand_dotted",
    "flatten_mapping",
    "params_to_dict",
    "read_user_config",
    "resolve_params",
]
