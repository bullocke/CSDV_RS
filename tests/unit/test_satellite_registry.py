"""Tests for csdv_core.satellite.registry."""

from __future__ import annotations

import pytest

from csdv_core.config import load_satellite
from csdv_core.satellite.registry import (
    SensorSpec,
    get_annual,
    get_index,
    get_sensor,
    list_annual_metrics,
    list_indices,
    list_sensors,
    register_sensor,
)


def test_landsat_band_mapping_shifts_at_landsat_8() -> None:
    # The single likeliest silent error in the module. An off-by-one here gives
    # a green-over-red ratio that still varies plausibly with season.
    for name in ("L4", "L5", "L7"):
        assert get_sensor(name).bands["red"] == "SR_B3"
        assert get_sensor(name).bands["nir"] == "SR_B4"
    for name in ("L8", "L9"):
        assert get_sensor(name).bands["red"] == "SR_B4"
        assert get_sensor(name).bands["nir"] == "SR_B5"
    # SWIR2 is SR_B7 on every Landsat, which is why the saturation bit rule
    # cannot simply be "native band number minus one" applied blindly.
    for name in list_sensors():
        assert get_sensor(name).bands["swir2"] == "SR_B7"


def test_saturation_bits_track_the_native_band_numbers() -> None:
    assert get_sensor("L5").saturation_bits["red"] == 2
    assert get_sensor("L8").saturation_bits["red"] == 3
    assert get_sensor("L5").saturation_bits["swir2"] == 6
    assert get_sensor("L8").saturation_bits["swir2"] == 6


def test_reflectance_scaling_constants() -> None:
    for name in list_sensors():
        spec = get_sensor(name)
        assert spec.reflectance_scale == pytest.approx(2.75e-05)
        assert spec.reflectance_offset == pytest.approx(-0.2)


def test_registered_sensors() -> None:
    assert list_sensors() == ["L4", "L5", "L7", "L8", "L9"]


def test_re_registering_a_sensor_overwrites(caplog: pytest.LogCaptureFixture) -> None:
    original = get_sensor("L5")
    try:
        replacement = SensorSpec(
            name="L5",
            platform="stub",
            collection_id="STUB",
            bands={"red": "B1", "nir": "B2"},
        )
        register_sensor(replacement)
        assert get_sensor("L5") is replacement
    finally:
        register_sensor(original)
    assert get_sensor("L5").collection_id == "LANDSAT/LT05/C02/T1_L2"


def test_index_specs_declare_the_bands_they_read() -> None:
    assert get_index("ndvi").bands == ("nir", "red")
    assert get_index("nbr").bands == ("nir", "swir2")
    assert get_index("ndmi").bands == ("nir", "swir1")
    assert set(list_indices()) >= {"ndvi", "nbr", "ndmi"}


def test_annual_defaults_merge_global_under_per_metric_params() -> None:
    mean = get_annual("ndvi_mean")
    # doy_min overrides the global default of 1.
    assert mean.defaults["doy_min"] == 152
    assert mean.defaults["doy_max"] == 258
    assert mean.defaults["min_obs"] == 3
    assert mean.index == "ndvi"

    amplitude = get_annual("ndvi_seasonal_amplitude")
    assert amplitude.defaults["min_obs"] == 6
    assert amplitude.defaults["doy_max"] == 366
    assert amplitude.defaults["max_condition"] == 30.0
    assert amplitude.units == "index (peak to trough)"


def test_unknown_names_raise_and_list_the_known_ones() -> None:
    for getter, label in (
        (get_sensor, "sensor"),
        (get_index, "index"),
        (get_annual, "annual metric"),
    ):
        with pytest.raises(KeyError) as excinfo:
            getter("NOPE")
        assert label in str(excinfo.value)
        assert "Known:" in str(excinfo.value)


def test_config_names_and_registered_names_agree() -> None:
    # A rename on either side silently unhooks the stage envelopes and the nine
    # trajectory rules that reference these spellings, so pin it both ways.
    cfg = load_satellite()
    assert set(cfg.annual_metrics) == set(list_annual_metrics())
    assert set(list_annual_metrics()) == {
        "ndvi_mean",
        "ndvi_seasonal_amplitude",
        "ndvi_trend",
    }
    assert set(cfg.extraction.sensors) <= set(list_sensors())
    assert set(cfg.extraction.indices) <= set(list_indices())
