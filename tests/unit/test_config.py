"""Tests for csdv_core.config typed loaders."""

from __future__ import annotations

import pytest

from csdv_core.config import (
    MetricsConfig,
    SatelliteConfig,
    SitesConfig,
    SiteTypesConfig,
    StagesConfig,
    TrajectoriesConfig,
    load_metrics,
    load_satellite,
    load_site_types,
    load_sites,
    load_stages,
    load_trajectories,
    reload_config,
)
from csdv_core.config._models import ExtractionConfig, QaConfig


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


def test_load_satellite_returns_typed_config() -> None:
    cfg = load_satellite()
    assert isinstance(cfg, SatelliteConfig)
    assert cfg.extraction.start_year == 1985
    assert cfg.extraction.scale_m == 30.0
    assert cfg.earth_engine.project == "dyce-biomass"
    # Dilated cloud is the bit the Earth Engine prototype omitted, and the one
    # that matters most on a stand of a handful of Landsat pixels.
    assert "dilated_cloud" in cfg.qa.mask_bits
    assert cfg.qa.index_valid_range == (-1.0, 1.0)


def test_satellite_annual_metric_names_are_the_ones_the_rules_reference() -> None:
    # stages.yaml and trajectories.yaml both reference these spellings. A
    # rename here would silently unhook nine trajectory rules.
    cfg = load_satellite()
    assert set(cfg.annual_metrics) == {
        "ndvi_mean",
        "ndvi_seasonal_amplitude",
        "ndvi_trend",
    }
    assert cfg.annual_metrics["ndvi_mean"].params["doy_min"] == 152
    assert cfg.annual_metrics["ndvi_seasonal_amplitude"].params["min_obs"] == 6


def test_satellite_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="precedes start_year"):
        ExtractionConfig(
            sensors=["L5"], indices=["ndvi"], start_year=2000, end_year=1999
        )
    with pytest.raises(ValueError, match="must be increasing"):
        QaConfig(mask_bits=["cloud"], index_valid_range=(1.0, -1.0))
    with pytest.raises(ValueError):
        QaConfig(mask_bits=["not_a_qa_bit"])


def test_loaders_are_cached() -> None:
    assert load_sites() is load_sites()
    reload_config()
    # After reload the object identity changes (cache cleared).
    a = load_sites()
    reload_config()
    b = load_sites()
    assert a is not b
    assert load_satellite() is load_satellite()
