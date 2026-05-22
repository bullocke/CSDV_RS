"""End-to-end synthetic test: stratify -> classify on a tiny grid.

Exercises :mod:`csdv_core.stratification.assign` and
:mod:`csdv_core.stages.classify` together against a fixture envelope.
Skipped if richdem is not available because :mod:`csdv_core.stratification.topo`
is not invoked here, but topographic variables would normally be the input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from csdv_core.config._models import (
    Predicate,
    SiteTypeRule,
    SiteTypesConfig,
    StagesConfig,
)
from csdv_core.stages.classify import classify_stages
from csdv_core.stratification.assign import assign_site_types

FIXTURE = Path(__file__).parent.parent / "fixtures" / "stages_synthetic.yaml"


def _load_stages() -> StagesConfig:
    return StagesConfig.model_validate(yaml.safe_load(FIXTURE.read_text()))


def _site_rules() -> SiteTypesConfig:
    return SiteTypesConfig(
        site_types={
            "type_01": SiteTypeRule(
                name="any",
                group="upland_mesic",
                rules=[Predicate(var="twi", op=">=", value=0.0)],
            )
        }
    )


@pytest.mark.slow
def test_strat_then_classify_pipeline() -> None:
    rng = np.random.default_rng(0)
    n = 16
    twi = rng.uniform(1.0, 10.0, size=(n, n)).astype("float32")
    gap = rng.uniform(0.0, 0.05, size=(n, n)).astype("float32")
    cv = rng.uniform(0.0, 0.2, size=(n, n)).astype("float32")

    site_type, site_score = assign_site_types({"twi": twi}, _site_rules())
    assert (site_type == 1).all()
    assert np.all(site_score == 1.0)

    stage, score, evaluated = classify_stages(
        {"gap_fraction": gap, "crown_width_cv": cv},
        site_type,
        _load_stages(),
        min_score=0.5,
    )
    # All cells should land in LSE (code 4) given the inputs above.
    assert (stage == 4).all()
    assert np.all(score >= 0.5)
    assert np.all(evaluated == 2)
