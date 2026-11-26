"""Tests for the layers the map viewer derives from a run's rasters."""

from __future__ import annotations

import numpy as np

from csdv_core.pairmodel.predict import flagged_agents
from csdv_core.view import derive

YEARS = [2014, 2016, 2018]


def _stacks() -> tuple[np.ndarray, np.ndarray]:
    """Three intervals over a 2 by 3 grid."""
    prob = np.array(
        [
            [[90, 10, 255], [0, 50, 100]],
            [[95, 85, 255], [0, 80, 0]],
            [[0, 99, 255], [0, 90, 255]],
        ],
        dtype=np.uint8,
    )
    agent = np.array(
        [
            [[1, 2, 255], [3, 4, 5]],
            [[2, 3, 255], [4, 5, 6]],
            [[3, 4, 255], [5, 6, 255]],
        ],
        dtype=np.uint8,
    )
    return prob, agent


def test_loss_mask_rounds_the_threshold_and_skips_no_imagery() -> None:
    prob = np.array([79, 80, 81, 255], dtype=np.uint8)
    assert derive.loss_mask(prob, 0.8).tolist() == [False, True, True, False]
    assert derive.loss_mask(prob, 0.796).tolist() == [False, True, True, False]


def test_date_is_the_later_year_of_the_first_flagged_interval() -> None:
    prob, _ = _stacks()
    date = derive.date_layer(prob, YEARS, 0.8)
    assert date.dtype == np.uint8
    assert date.tolist() == [[14, 16, 0], [0, 16, 14]]


def test_date_of_one_interval_flags_only_that_interval() -> None:
    prob, _ = _stacks()
    assert derive.date_layer(prob, YEARS, 0.8, band=1).tolist() == [
        [16, 16, 0],
        [0, 16, 0],
    ]


def test_type_is_the_agent_of_the_first_flagged_interval() -> None:
    prob, agent = _stacks()
    assert derive.type_layer(prob, agent, 0.8).tolist() == [[1, 3, 0], [0, 5, 5]]


def test_type_of_one_interval_is_the_exported_type_raster() -> None:
    prob, agent = _stacks()
    for band in range(3):
        expected = flagged_agents(agent[band], prob[band], 0.8)
        assert np.array_equal(derive.type_layer(prob, agent, 0.8, band=band), expected)


def test_date_and_type_flag_the_same_cells() -> None:
    rng = np.random.default_rng(0)
    prob = rng.integers(0, 101, size=(4, 20, 20)).astype(np.uint8)
    agent = rng.integers(1, 7, size=(4, 20, 20)).astype(np.uint8)
    prob[:, :3] = derive.NODATA
    agent[:, :3] = derive.NODATA
    date = derive.date_layer(prob, [2012, 2014, 2016, 2018], 0.7)
    kind = derive.type_layer(prob, agent, 0.7)
    assert np.array_equal(date > 0, kind > 0)


def test_probability_is_the_highest_valid_value() -> None:
    prob, _ = _stacks()
    assert derive.prob_layer(prob).tolist() == [[95, 99, 255], [0, 90, 100]]
    assert np.array_equal(derive.prob_layer(prob, band=2), prob[2])


def test_onset_layer_drops_never_monitored_and_filters_a_year() -> None:
    onset = np.array([[0, 14, 16], [255, 16, 25]], dtype=np.uint8)
    assert derive.onset_layer(onset).tolist() == [[0, 14, 16], [0, 16, 25]]
    assert derive.onset_layer(onset, 2016).tolist() == [[0, 0, 16], [0, 16, 0]]
