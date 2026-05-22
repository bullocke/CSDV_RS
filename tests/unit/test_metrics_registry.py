"""Tests for the metric registry."""

from __future__ import annotations

import pytest

import csdv_core.metrics  # noqa: F401  ensures registrations happen
from csdv_core.metrics.registry import (
    get_delta,
    get_metric,
    list_deltas,
    list_metrics,
)


def test_expected_metrics_registered():
    names = set(list_metrics())
    expected = {
        "gap_fraction",
        "crown_fraction",
        "crown_cv",
        "crown_p90",
        "crown_mean",
        "crown_count",
        "glcm_texture",
        "shrub_fraction",
        "small_tree_fraction",
        "mid_canopy_fraction",
        "tall_canopy_fraction",
        "linearity_index",
        "edge_density",
        "row_directionality",
        "gap_persistence",
    }
    assert expected.issubset(names)


def test_get_metric_returns_callable():
    spec = get_metric("gap_fraction")
    assert callable(spec.fn)
    assert spec.name == "gap_fraction"


def test_defaults_resolved_from_yaml():
    spec = get_metric("gap_fraction")
    # YAML override.
    assert spec.defaults["window_m"] == 25.0
    assert spec.defaults["height_threshold_m"] == 2.0
    # Global defaults present.
    assert "window_sizes_m" in spec.defaults
    assert spec.defaults["chm_gap_threshold_m"] == 2.0


def test_get_metric_unknown_raises():
    with pytest.raises(KeyError):
        get_metric("not_a_metric")


def test_delta_registry_lists_expected():
    names = set(list_deltas())
    expected = {
        "delta_crown_fraction",
        "delta_gap_fraction",
        "delta_crown_p90",
        "delta_crown_count",
        "delta_edge_density",
    }
    assert expected.issubset(names)


def test_get_delta_returns_callable():
    fn = get_delta("delta_gap_fraction")
    assert callable(fn)


def test_get_delta_unknown_raises():
    with pytest.raises(KeyError):
        get_delta("delta_not_a_metric")
