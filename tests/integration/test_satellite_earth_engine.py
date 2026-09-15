"""Integration tests that contact Earth Engine.

Gated on ``CSDV_EE_TESTS=1`` as well as on the ``slow`` marker, because they
need working credentials and a network, and neither belongs in a run of the
unit suite. Enable them with:

    CSDV_EE_TESTS=1 pytest tests/integration/test_satellite_earth_engine.py -q

Two things are checked. The first is that a real fetch comes back with the
shape and the physical plausibility the rest of the module assumes. The second
is the pixel-rule audit. Earth Engine reduces with an area-weighted mean and
``csdv_core.zonal.mask`` uses the pixel-centre rule, so the two disagree by an
amount that depends on how much of a stand sits on its own boundary. That
divergence is measured rather than assumed, and it is asserted only for the
large stand. For a stand a few pixels across the honest answer is that the two
estimators are not interchangeable, so the small stand's divergence is reported
and left unasserted.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CSDV_EE_TESTS") != "1",
        reason="Set CSDV_EE_TESTS=1 to run tests that contact Earth Engine.",
    ),
]

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
GDB = (
    REPO
    / "data/calibration/Indiana-ElkinsvilleNE_revised.gdb"
    / "Indiana-ElkinsvilleNE_revised.gdb"
)

#: One clear growing season, chosen because Landsat 5 and Landsat 7 were both
#: collecting, so a fetch over it exercises the band-mapping switch too.
TEST_YEAR = 2005

#: Growing-season window, matching ``ndvi_mean``'s configured DOY range.
SUMMER = (152, 258)

#: The divergence the stand metrics assume between the two pixel rules on a
#: stand large enough for its boundary to be a small share of its area.
LARGE_STAND_TOLERANCE = 0.01


def _stands():
    """The calibration stands, largest first."""
    pytest.importorskip("geopandas")
    if not GDB.exists():
        pytest.skip(f"Calibration geodatabase not present at {GDB}")
    from csdv_core.io.stands import read_ais_stands

    stands = read_ais_stands(GDB)
    return stands.assign(_area=stands.geometry.area).sort_values(
        "_area", ascending=False
    )


def test_one_real_year_over_three_stands() -> None:
    pytest.importorskip("ee")
    from csdv_core.satellite.extract import check_mask_propagation, fetch_observations

    stands = _stands().head(3)
    obs, provenance = fetch_observations(
        stands, start_year=TEST_YEAR, end_year=TEST_YEAR
    )

    assert not provenance["chunks_failed"], provenance["chunks_failed"]
    assert not obs.empty
    assert set(obs["stand_id"]) == set(stands["stand_id"])

    # The index mask has to reach the coverage band. This is the regression
    # guard for the bug that made every quality gate inert on the first run.
    assert check_mask_propagation(obs, "ndvi") == 0

    # Counts are counts.
    n_pixels = obs["n_pixels"].to_numpy(dtype=float)
    assert np.all(n_pixels >= 0)
    assert np.all(n_pixels == np.floor(n_pixels))

    # A clear scene over a whole stand recovers close to the stand's own area.
    clear = obs[obs["coverage_fraction"] > 0.98]
    assert len(clear) >= 3, "No clear scene in the test year"
    ratio = clear["pixel_weight_sum"] / clear["expected_pixels"]
    assert 0.95 <= float(ratio.median()) <= 1.05

    # Closed deciduous forest in July. A band mapping that read green instead
    # of red would not land here.
    summer = obs[obs["doy"].between(*SUMMER) & (obs["coverage_fraction"] > 0.9)]
    assert not summer.empty
    assert 0.70 <= float(summer["ndvi"].median()) <= 0.95


def _clearest_summer_scene(obs):
    clear = obs[
        (obs["coverage_fraction"] > 0.98) & obs["doy"].between(*SUMMER)
    ].sort_values("scene_cloud_cover")
    if clear.empty:
        pytest.skip("No clear growing-season scene over the test stand")
    return clear.iloc[0]


def _all_touched_mean(stand, time_ms: int) -> float:
    """The same scene reduced with the unweighted (all-touched) rule."""
    import ee

    from csdv_core.config import load_satellite
    from csdv_core.satellite.extract import stands_to_fc
    from csdv_core.satellite.sensors import build_collection

    cfg = load_satellite()
    geometry = stands_to_fc(stand).first().geometry()
    collection = build_collection(
        geometry,
        f"{TEST_YEAR}-01-01",
        f"{TEST_YEAR + 1}-01-01",
        cfg=cfg,
        sensors=cfg.extraction.sensors,
        indices=cfg.extraction.indices,
    )
    # Matched on the acquisition timestamp rather than on the image id, which
    # carries a collection-path prefix that is not the server's own index.
    image = ee.Image(
        collection.filter(ee.Filter.eq("system:time_start", int(time_ms))).first()
    )
    result = (
        image.select("ndvi")
        .reduceRegion(
            reducer=ee.Reducer.mean().unweighted(),
            geometry=geometry,
            scale=cfg.extraction.scale_m,
            maxPixels=int(1e9),
        )
        .getInfo()
    )
    return float(result["ndvi"])


def _audit(frame) -> float:
    """Fetch one stand, reduce the same scene both ways, return the gap.

    ``frame`` is a one-row GeoDataFrame, not a row. The CRS lives on the frame,
    and both the fetch and the geometry conversion need it.
    """
    from csdv_core.satellite.extract import fetch_observations

    obs, _ = fetch_observations(frame, start_year=TEST_YEAR, end_year=TEST_YEAR)
    row = _clearest_summer_scene(obs)
    weighted = float(row["ndvi"])
    all_touched = _all_touched_mean(frame, int(row["time_ms"]))
    divergence = abs(all_touched - weighted)
    logger.info(
        "Stand %s (%.1f ac, %d pixels): area-weighted %.4f, all-touched %.4f, "
        "divergence %.4f",
        row["stand_id"],
        float(frame.geometry.iloc[0].area) / 4046.856,
        int(row["n_pixels"]),
        weighted,
        all_touched,
        divergence,
    )
    return divergence


def test_pixel_rules_agree_on_a_large_stand() -> None:
    pytest.importorskip("ee")
    big = _stands().iloc[[0]]
    acres = float(big.geometry.iloc[0].area) / 4046.856
    divergence = _audit(big)
    assert divergence < LARGE_STAND_TOLERANCE, (
        f"Pixel rules diverge by {divergence:.4f} on a {acres:.0f} acre stand, "
        f"more than the {LARGE_STAND_TOLERANCE} the stand metrics assume."
    )


def test_pixel_rule_divergence_on_a_small_stand_is_reported_not_asserted() -> None:
    """No assertion, by design.

    On a stand a few Landsat pixels across, most of the area is boundary and
    the two rules are measuring different things. Asserting they agree would be
    asserting something false. What this test is for is putting the number in
    the log next to the large stand's, so the size at which the satellite
    metrics stop being comparable to the NAIP ones is on the record.
    """
    pytest.importorskip("ee")
    divergence = _audit(_stands().iloc[[-1]])
    logger.warning(
        "Small-stand pixel-rule divergence is %.4f NDVI. Not asserted: at this "
        "size the two rules are not estimating the same quantity.",
        divergence,
    )
