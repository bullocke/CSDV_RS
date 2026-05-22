"""Tests for :mod:`csdv_core.stratification.assign`."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config._models import Predicate, SiteTypeRule, SiteTypesConfig
from csdv_core.stratification import assign


def _cfg() -> SiteTypesConfig:
    return SiteTypesConfig(
        site_types={
            # type 7: TWI very high (wet cove)
            "type_07": SiteTypeRule(
                name="cove",
                group="upland_mesic",
                rules=[Predicate(var="twi", op=">=", value=8.0)],
            ),
            # type 5: moderate TWI and gentle slope
            "type_05": SiteTypeRule(
                name="mesic",
                group="upland_mesic",
                rules=[
                    Predicate(var="twi", op=">=", value=4.0),
                    Predicate(var="slope_deg", op="<=", value=20.0),
                ],
            ),
            # type 9: steep south-facing
            "type_09": SiteTypeRule(
                name="south",
                group="mountain",
                rules=[
                    Predicate(var="slope_deg", op=">=", value=15.0),
                    Predicate(var="northness", op="<=", value=0.0),
                ],
            ),
        }
    )


def test_assign_first_match_wins() -> None:
    twi = np.array([[10.0, 5.0], [5.0, 1.0]], dtype="float32")
    slope = np.array([[5.0, 5.0], [25.0, 5.0]], dtype="float32")
    northness = np.array([[1.0, 1.0], [-0.5, 1.0]], dtype="float32")
    site_type, score = assign.assign_site_types(
        {"twi": twi, "slope_deg": slope, "northness": northness}, _cfg()
    )
    # (0,0): twi=10 -> type_07 wins (first declared)
    assert site_type[0, 0] == 7
    # (0,1): twi=5, slope=5 -> type_05
    assert site_type[0, 1] == 5
    # (1,0): twi=5, slope=25 fails type_05; slope=25,north=-0.5 -> type_09
    assert site_type[1, 0] == 9
    # (1,1): nothing matches
    assert site_type[1, 1] == 0
    assert np.isnan(score[1, 1])
    assert score[0, 0] == pytest.approx(1.0)
    assert score[0, 1] == pytest.approx(1.0)


def test_assign_nan_inputs_never_match() -> None:
    twi = np.array([[np.nan]], dtype="float32")
    slope = np.array([[5.0]], dtype="float32")
    northness = np.array([[0.0]], dtype="float32")
    site_type, score = assign.assign_site_types(
        {"twi": twi, "slope_deg": slope, "northness": northness}, _cfg()
    )
    assert site_type[0, 0] == 0
    assert np.isnan(score[0, 0])


def test_assign_missing_variable_does_not_match() -> None:
    twi = np.array([[10.0]], dtype="float32")
    site_type, _ = assign.assign_site_types({"twi": twi}, _cfg())
    # type_07 should still match (only depends on twi).
    assert site_type[0, 0] == 7


def test_assign_empty_stratvars_raises() -> None:
    with pytest.raises(ValueError):
        assign.assign_site_types({}, _cfg())


def test_assign_placeholder_rule_never_matches() -> None:
    cfg = SiteTypesConfig(
        site_types={
            "type_01": SiteTypeRule(
                name="placeholder",
                group="x",
                rules=[Predicate(var="twi", op=">=", value=None)],
            )
        }
    )
    twi = np.array([[10.0]], dtype="float32")
    site_type, score = assign.assign_site_types({"twi": twi}, cfg)
    assert site_type[0, 0] == 0
    assert np.isnan(score[0, 0])
