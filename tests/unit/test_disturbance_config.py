"""Tests for the disturbance config model, loader and parameter resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from csdv_core.config import DisturbanceConfig, load_disturbance, reload_config
from csdv_core.disturbance.params import (
    check_reuse_compatible,
    expand_dotted,
    flatten_mapping,
    params_to_dict,
    resolve_params,
)


def setup_function(_fn) -> None:
    reload_config()


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_defaults_equal_best_case() -> None:
    params = load_disturbance()
    assert isinstance(params, DisturbanceConfig)
    assert params.inputs.input == "d_chm_h"
    assert params.grid.grain_m == pytest.approx(19.2)
    assert params.standardize.method == "local_twopass"
    assert params.standardize.window_m == pytest.approx(150.0)
    assert params.decide.rule == "D6"
    assert params.decide.p_boundary == pytest.approx(0.5)
    assert params.decide.spatial_sigma_m == [10.0, 25.0]
    assert params.mmu.mmu_ha == pytest.approx(0.405)
    assert params.refine.enabled is True


def test_packaged_yaml_matches_model_defaults() -> None:
    assert load_disturbance() == DisturbanceConfig()


def test_refine_values_match_research_script() -> None:
    refine = load_disturbance().refine
    assert refine.pad_m == 60.0
    assert refine.ring_gap_m == 20.0
    assert refine.near_m == 10.0
    assert refine.forest_height_m == 5.0
    assert refine.min_ring_px == 2000
    assert refine.close_radius_m == 8.0
    assert refine.fill_holes_m2 == 2500.0
    assert refine.smooth_m == 3.0
    assert refine.simplify_m == 0.6


def test_resolve_without_overrides_is_packaged() -> None:
    assert resolve_params(None, {}) == load_disturbance()


def test_precedence_packaged_user_cli(tmp_path: Path) -> None:
    user_yaml = _write_yaml(
        tmp_path / "user.yaml",
        "decide:\n  rule: D5\n  p_boundary: 0.6\nmmu:\n  mmu_ha: 1.0\n",
    )
    params = resolve_params(user_yaml, {"decide.rule": "D2"})
    assert params.decide.rule == "D2"
    assert params.decide.p_boundary == pytest.approx(0.6)
    assert params.mmu.mmu_ha == pytest.approx(1.0)
    assert params.grid.grain_m == pytest.approx(19.2)
    assert params.decide.spatial_sigma_m == [10.0, 25.0]


def test_cli_nested_dict_overrides(tmp_path: Path) -> None:
    params = resolve_params(None, {"standardize": {"method": "local", "window_m": 500}})
    assert params.standardize.method == "local"
    assert params.standardize.window_m == pytest.approx(500.0)
    assert params.standardize.min_frac == pytest.approx(0.2)


def test_none_cli_values_ignored(tmp_path: Path) -> None:
    user_yaml = _write_yaml(tmp_path / "user.yaml", "grid:\n  grain_m: 9.6\n")
    overrides = {"grid.grain_m": None, "decide": {"rule": None}, "refine.enabled": None}
    params = resolve_params(user_yaml, overrides)
    assert params.grid.grain_m == pytest.approx(9.6)
    assert params.decide.rule == "D6"
    assert params.refine.enabled is True


def test_false_cli_value_is_kept() -> None:
    params = resolve_params(None, {"refine.enabled": False})
    assert params.refine.enabled is False


def test_unknown_key_in_user_yaml_raises(tmp_path: Path) -> None:
    user_yaml = _write_yaml(tmp_path / "user.yaml", "decide:\n  rul: D5\n")
    with pytest.raises(ValidationError):
        resolve_params(user_yaml, {})


def test_unknown_cli_key_raises() -> None:
    with pytest.raises(ValidationError):
        resolve_params(None, {"grid.cell_size": 10.0})


def test_unknown_section_raises() -> None:
    with pytest.raises(ValidationError):
        resolve_params(None, {"scoring.enabled": True})


def test_invalid_literal_raises() -> None:
    with pytest.raises(ValidationError):
        resolve_params(None, {"decide.rule": "D4"})


def test_missing_user_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_params(tmp_path / "missing.yaml", {})


def test_user_yaml_not_mapping_raises(tmp_path: Path) -> None:
    user_yaml = _write_yaml(tmp_path / "user.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        resolve_params(user_yaml, {})


def test_reference_year_must_be_in_years() -> None:
    overrides = {"inputs.years": [2013, 2015], "grid.reference_year": 2017}
    with pytest.raises(ValidationError):
        resolve_params(None, overrides)


def test_expand_dotted_merges_spellings() -> None:
    nested = expand_dotted({"decide.rule": "D5", "decide": {"pct0": 0.2}})
    assert nested == {"decide": {"rule": "D5", "pct0": 0.2}}


def test_expand_dotted_rejects_empty_component() -> None:
    with pytest.raises(ValueError):
        expand_dotted({"decide..rule": "D5"})


def test_params_to_dict_round_trip() -> None:
    params = resolve_params(None, {"decide.rule": "D5s", "inputs.years": [2013, 2015]})
    as_dict = params_to_dict(params)
    assert as_dict["decide"]["rule"] == "D5s"
    assert DisturbanceConfig.model_validate(as_dict) == params


def test_flatten_mapping_dotted_keys() -> None:
    flat = flatten_mapping({"decide": {"rule": "D6", "spatial_sigma_m": [10, 25]}})
    assert flat == {"decide.rule": "D6", "decide.spatial_sigma_m": [10, 25]}


def test_check_reuse_compatible() -> None:
    saved_params = DisturbanceConfig()
    saved = params_to_dict(saved_params)
    changed_decision_params = DisturbanceConfig.model_validate(
        {"decide": {"rule": "D5s"}, "mmu": {"mmu_ha": 1.0}}
    )
    changed_decision = params_to_dict(changed_decision_params)
    check_reuse_compatible(saved, changed_decision)
    changed_shape = {"grid": {"grain_m": 30.0}, "standardize": {"window_m": 200.0}}
    with pytest.raises(ValueError) as error_info:
        check_reuse_compatible(saved, changed_shape)
    message = str(error_info.value)
    assert "grid.grain_m" in message
    assert "standardize.window_m" in message
    assert "decide" not in message


def test_saved_params_fill_locked_sections(tmp_path: Path) -> None:
    """With saved parameters, locked sections come from the saved run."""
    saved_params = resolve_params(
        None, {"grid.grain_m": 30.0, "inputs.years": [2013, 2015]}
    )
    saved = params_to_dict(saved_params)
    rerun = resolve_params(None, {"decide.rule": "D6s"}, saved_params=saved)
    assert rerun.grid.grain_m == 30.0
    assert rerun.inputs.years == [2013, 2015]
    assert rerun.decide.rule == "D6s"
    matching = resolve_params(None, {"grid.grain_m": 30.0}, saved_params=saved)
    assert matching.grid.grain_m == 30.0
    with pytest.raises(ValueError, match="standardize.window_m"):
        resolve_params(None, {"standardize.window_m": 200.0}, saved_params=saved)
    config_path = _write_yaml(tmp_path / "user.yaml", "grid:\n  grain_m: 19.2\n")
    with pytest.raises(ValueError, match="grid.grain_m"):
        resolve_params(config_path, {}, saved_params=saved)
