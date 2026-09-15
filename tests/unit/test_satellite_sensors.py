"""Tests for csdv_core.satellite.sensors.

Everything here is the pure part: bit arithmetic, scaling constants and the
operating spans. The Earth Engine calls are covered by the integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.satellite.registry import get_sensor
from csdv_core.satellite.sensors import (
    QA_PIXEL_BITS,
    qa_bitmask,
    scale_reflectance,
    sensors_for_year,
)


def test_qa_pixel_bit_positions() -> None:
    assert QA_PIXEL_BITS == {
        "fill": 0,
        "dilated_cloud": 1,
        "cirrus": 2,
        "cloud": 3,
        "cloud_shadow": 4,
        "snow": 5,
        "clear": 6,
        "water": 7,
    }


def test_configured_mask_is_63_not_the_prototypes_24() -> None:
    configured = ["fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"]
    assert qa_bitmask(configured) == 63
    # What scripts/ee/test_retrieve_Landsat_NDVI.py masked. Kept as a named
    # comparison so the change stays visible rather than becoming folklore.
    assert qa_bitmask(["cloud", "cloud_shadow"]) == 24


def test_qa_bitmask_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="nonsense"):
        qa_bitmask(["nonsense"])


def test_reflectance_scaling_maps_the_documented_endpoints() -> None:
    spec = get_sensor("L5")
    out = scale_reflectance(np.array([7273.0, 43636.0]), spec)
    assert out[0] == pytest.approx(0.0, abs=1e-3)
    assert out[1] == pytest.approx(1.0, abs=1e-3)


def test_sensors_for_year_covers_the_handover_years() -> None:
    # Landsat 4 TM acquired until 1993, so the early record overlaps Landsat 5.
    assert sensors_for_year(1986) == ["L4", "L5"]
    # Between the end of Landsat 4 and the launch of Landsat 7, Landsat 5 flies
    # alone. This is the thinnest stretch of the record.
    assert sensors_for_year(1995) == ["L5"]
    # 2012 is the gap: Landsat 5 Collection 2 ends in May and Landsat 8 does
    # not start until March 2013, so the first Elkinsville NAIP year rests on
    # Landsat 7 alone. Excluding SLC-off data would empty it.
    assert sensors_for_year(2012) == ["L5", "L7"]
    assert sensors_for_year(2013) == ["L7", "L8"]
    assert sensors_for_year(2022) == ["L7", "L8", "L9"]
    assert sensors_for_year(2025) == ["L8", "L9"]


def test_sensors_for_year_can_be_restricted() -> None:
    assert sensors_for_year(2022, names=["L8", "L9"]) == ["L8", "L9"]
    assert sensors_for_year(1970) == []
