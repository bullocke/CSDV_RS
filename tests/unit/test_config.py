"""Tests for csdv_core.config typed loaders."""

from __future__ import annotations

import pytest

from csdv_core.config import (
    MetricsConfig,
    SitesConfig,
    SiteTypesConfig,
    StagesConfig,
    TrajectoriesConfig,
    load_metrics,
    load_site_types,
    load_sites,
    load_stages,
    load_trajectories,
    reload_config,
)


def setup_function(_fn) -> None:
    reload_config()


def test_load_sites_returns_typed_config() -> None:
    cfg = load_sites()
    assert isinstance(cfg, SitesConfig)
    scbi = cfg.get("SCBI")
    assert scbi.category == "neon"
    assert scbi.state == "VA"
    assert scbi.neon is not None
    assert scbi.neon.site_code == "SCBI"


def test_load_sites_unknown_code_raises() -> None:
    with pytest.raises(KeyError):
        load_sites().get("NOPE")


def test_load_metrics_defaults() -> None:
    cfg = load_metrics()
    assert isinstance(cfg, MetricsConfig)
    assert cfg.defaults.chm_gap_threshold_m == 2.0
    assert 25 in cfg.defaults.window_sizes_m


def test_load_site_types() -> None:
    cfg = load_site_types()
    assert isinstance(cfg, SiteTypesConfig)
    assert "type_01" in cfg.site_types


def test_load_stages() -> None:
    cfg = load_stages()
    assert isinstance(cfg, StagesConfig)
    lse = cfg.stages["LSE"]
    rng = lse.envelopes["type_01"]["gap_fraction"]
    assert rng.max == 0.10


def test_load_trajectories() -> None:
    cfg = load_trajectories()
    assert isinstance(cfg, TrajectoriesConfig)
    assert cfg.trajectories["LC7"].group == "LC"
    # 19 V5 classes; codes are unique uint8 values starting at 1.
    assert len(cfg.trajectories) == 19
    assert cfg.trajectory_codes["DS1"] == 1
    assert cfg.trajectory_order[0] == "DS1"
    # Every rule has a code; every code has a rule.
    assert set(cfg.trajectory_codes) == set(cfg.trajectories)
    assert set(cfg.trajectory_order) == set(cfg.trajectories)
    # DS1 first predicate uses the new dim/reducer/in extensions.
    ds1 = cfg.trajectories["DS1"].signature[0]
    assert ds1.dim == "stage"
    assert ds1.reducer == "all"
    assert ds1.op == "in"
    assert ds1.value == ["ESE", "LSE"]


def test_loaders_are_cached() -> None:
    assert load_sites() is load_sites()
    reload_config()
    # After reload the object identity changes (cache cleared).
    a = load_sites()
    reload_config()
    b = load_sites()
    assert a is not b
