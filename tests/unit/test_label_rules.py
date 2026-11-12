"""Label rules of one interval on hand-built arrays."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config import load_label
from csdv_core.label import routing as R
from csdv_core.label.pooling import pool_agent, pool_change
from csdv_core.label.rules import (
    IGNORE,
    REASON_GROUPS,
    BaseLayers,
    CandidateLayer,
    Reason,
    interval_reason,
    reason_group,
    to_change,
)

CFG = load_label()
RULES = CFG.rules
N = 200
BAND_PRECISE = 8  # 4.8 m
BAND_ENVELOPE = 16  # 9.6 m
MARGIN = 32  # 19.2 m


def base(**changes) -> BaseLayers:
    layers = {
        "no_data": np.zeros((N, N), bool),
        "forest_before": np.ones((N, N), bool),
        "forest_after": np.ones((N, N), bool),
        "cover_before": np.full((N, N), 0.9, "float32"),
        "height_before": np.full((N, N), 18.0, "float32"),
        "prob": np.full((N, N), 0.01, "float32"),
    }
    layers.update(changes)
    return BaseLayers(**layers)


def square(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((N, N), bool)
    mask[r0:r1, c0:c1] = True
    return mask


def test_rules_without_candidates() -> None:
    before = np.ones((N, N), bool)
    after = np.ones((N, N), bool)
    cover = np.full((N, N), 0.9, "float32")
    height = np.full((N, N), 18.0, "float32")
    prob = np.full((N, N), 0.01, "float32")
    before[:20] = False  # non-forest strip
    cover[:20] = 0.0
    height[:20] = 0.5
    cover[:20, :50] = 0.3  # sparse trees by cover
    height[:20, 50:100] = 3.0  # low trees by height
    after[40:60] = False  # left the forest mask
    prob[80:100] = 0.25  # raised probability
    cover[150:, 150:] = np.nan  # no CHM
    no_data = square(120, 140, 0, 40)
    reason, agent = interval_reason(
        base(
            no_data=no_data,
            forest_before=before,
            forest_after=after,
            cover_before=cover,
            height_before=height,
            prob=prob,
        ),
        [],
        RULES,
    )
    assert reason[10, 150] == Reason.NONFOREST
    assert reason[10, 10] == Reason.SPARSE_WOODY
    assert reason[10, 70] == Reason.SPARSE_WOODY
    assert reason[50, 100] == Reason.FOREST_LOST_UNJUDGED
    assert reason[90, 100] == Reason.RAISED_PROB
    assert reason[130, 20] == Reason.NO_DATA
    assert reason[170, 170] == Reason.NO_DATA
    assert reason[110, 100] == Reason.STABLE_FOREST
    assert (agent == IGNORE).all()
    assert (reason != Reason.UNSET).all()


def test_unstable_interval_has_no_unverified_forest_background() -> None:
    reason, _ = interval_reason(base(unstable=True), [], RULES)
    assert (reason == Reason.UNSTABLE_FOREST).all()
    assert (to_change(reason, CFG.policy) == IGNORE).all()


@pytest.mark.parametrize("precise,band", [(True, BAND_PRECISE), (False, BAND_ENVELOPE)])
def test_accepted_candidate_band_and_ring(precise: bool, band: int) -> None:
    mask = square(60, 140, 60, 140)
    cand = CandidateLayer(R.LOSS_FULL, mask, core=mask, precise=precise, agent=1)
    reason, agent = interval_reason(base(), [cand], RULES)
    row = reason[100]
    assert row[100] == Reason.ACCEPTED_FULL and agent[100, 100] == 1
    # Inside the outline: a band, then loss.
    assert (row[60 : 60 + band] == Reason.BOUNDARY_INSIDE).all()
    assert row[60 + band] == Reason.ACCEPTED_FULL
    assert agent[100, 60] == IGNORE
    # Outside: a band, then the ring, then forest.
    assert (row[60 - band : 60] == Reason.BOUNDARY_OUTSIDE).all()
    assert (row[60 - MARGIN : 60 - band] == Reason.MARGIN).all()
    assert row[60 - MARGIN - 1] == Reason.STABLE_FOREST
    change = to_change(reason, CFG.policy)
    assert change[100, 100] == 2 and change[100, 60] == IGNORE
    assert change[100, 10] == 0


def test_inner_band_and_precise_ring_can_be_switched_off() -> None:
    mask = square(60, 140, 60, 140)
    rules = RULES.model_copy(
        update={"band_inside_precise_m": 0.0, "margin_precise_m": 0.0}
    )
    cand = CandidateLayer(R.LOSS_FULL, mask, core=mask, precise=True, agent=1)
    reason, agent = interval_reason(base(), [cand], rules)
    row = reason[100]
    assert row[60] == Reason.ACCEPTED_FULL and agent[100, 60] == 1  # up to the outline
    assert (row[60 - BAND_PRECISE : 60] == Reason.BOUNDARY_OUTSIDE).all()
    assert row[60 - BAND_PRECISE - 1] == Reason.STABLE_FOREST  # no ring beyond the band
    # An envelope keeps its ring, and its own inner band setting.
    loose = CandidateLayer(R.LOSS_PARTIAL, mask, core=mask, precise=False)
    reason, _ = interval_reason(base(), [loose], rules)
    assert reason[100, 60] == Reason.BOUNDARY_INSIDE
    assert reason[100, 60 - MARGIN + 1] == Reason.MARGIN


def test_diffuse_extension_is_unlabelled_and_has_no_inner_band() -> None:
    mask = square(60, 140, 60, 100)
    extension = square(60, 140, 100, 150)
    cand = CandidateLayer(
        R.LOSS_PARTIAL, mask, core=mask, extension=extension, crisp=False, agent=1
    )
    reason, _ = interval_reason(base(), [cand], RULES)
    assert reason[100, 120] == Reason.EXTENSION_IGNORED
    # The core runs up to the extension without a band: the event continues.
    assert reason[100, 99] == Reason.ACCEPTED_PARTIAL
    # The other sides of the core still have one.
    assert reason[100, 60] == Reason.BOUNDARY_INSIDE
    # The ring is measured from the extension.
    assert reason[100, 150 + MARGIN - 1] == Reason.MARGIN
    assert reason[100, 150 + MARGIN + 1] == Reason.STABLE_FOREST


def test_crisp_extension_is_loss() -> None:
    mask = square(60, 140, 60, 100)
    extension = square(60, 140, 100, 150)
    cand = CandidateLayer(
        R.LOSS_FULL, mask, core=mask, extension=extension, crisp=True, agent=5
    )
    reason, agent = interval_reason(base(), [cand], RULES)
    assert reason[100, 120] == Reason.EXTENSION_FULL and agent[100, 120] == 5
    assert reason[100, 100] == Reason.EXTENSION_FULL
    assert reason[100, 149] == Reason.BOUNDARY_INSIDE
    assert to_change(reason, CFG.policy)[100, 120] == 2


def test_trimmed_part_and_unusable_outline() -> None:
    mask = square(60, 140, 60, 140)
    core = square(60, 140, 60, 100)
    trimmed = mask & ~core
    cand = CandidateLayer(R.LOSS_FULL, mask, core=core, trimmed=trimmed)
    reason, _ = interval_reason(base(), [cand], RULES)
    assert reason[100, 80] == Reason.ACCEPTED_FULL
    assert reason[100, 120] == Reason.OUTLINE_TRIMMED
    nothing = CandidateLayer(
        R.LOSS_FULL, mask, core=np.zeros((N, N), bool), trimmed=mask
    )
    reason, _ = interval_reason(base(), [nothing], RULES)
    assert not np.isin(reason, [Reason.ACCEPTED_FULL]).any()
    assert reason[100, 100] == Reason.OUTLINE_TRIMMED


def test_loss_beats_ignore_beats_background() -> None:
    loss = CandidateLayer(R.LOSS_FULL, square(40, 100, 40, 100), precise=True)
    loss.core = loss.mask
    rejected = CandidateLayer(R.REJECTED_FLAGGED, square(80, 160, 80, 160))
    intact = CandidateLayer(R.INTACT, square(80, 190, 80, 190))
    for order in ([loss, rejected, intact], [intact, rejected, loss]):
        reason, _ = interval_reason(base(), order, RULES)
        assert reason[70, 70] == Reason.ACCEPTED_FULL
        assert reason[90, 90] == Reason.ACCEPTED_FULL  # claimed by all three
        assert reason[130, 130] == Reason.REJECTED_FLAGGED
        assert reason[175, 175] == Reason.JUDGED_INTACT


def test_judged_intact_beats_a_neighbours_ring() -> None:
    loss = CandidateLayer(R.LOSS_FULL, square(40, 100, 40, 100), precise=True)
    loss.core = loss.mask
    intact = CandidateLayer(R.INTACT, square(40, 100, 115, 170))
    reason, _ = interval_reason(base(), [loss, intact], RULES)
    assert reason[70, 105] == Reason.BOUNDARY_OUTSIDE
    assert reason[70, 112] == Reason.MARGIN
    assert reason[70, 120] == Reason.JUDGED_INTACT  # inside the ring of `loss`
    open_after = CandidateLayer(R.OPEN, square(40, 100, 115, 170))
    reason, _ = interval_reason(base(), [open_after], RULES)
    assert reason[70, 120] == Reason.POST_EVENT_OPEN
    assert reason[70, 110] == Reason.STABLE_FOREST  # no ring around background


def test_small_candidate_is_unlabelled_without_a_ring() -> None:
    small = CandidateLayer(R.UNDER_MIN, square(90, 100, 90, 100))
    reason, _ = interval_reason(base(), [small], RULES)
    assert reason[95, 95] == Reason.UNDER_MIN
    assert reason[95, 105] == Reason.STABLE_FOREST
    unjudged = CandidateLayer(R.UNJUDGED, square(90, 100, 90, 100))
    reason, _ = interval_reason(base(), [unjudged], RULES)
    assert reason[95, 95] == Reason.UNJUDGED and reason[95, 105] == Reason.MARGIN


def test_wide_ring() -> None:
    mask = square(80, 120, 80, 120)
    cand = CandidateLayer(R.LOSS_PARTIAL, mask, core=mask, wide_margin=True)
    reason, _ = interval_reason(base(), [cand], RULES)
    assert reason[100, 120 + 60] == Reason.MARGIN  # 36 m out
    assert reason[100, 120 + 66] == Reason.STABLE_FOREST


def test_policy_and_groups() -> None:
    codes = np.array([[r.value for r in Reason]], dtype=np.uint8)
    change = to_change(codes, CFG.policy)
    labelled = {Reason(c).name.lower() for c in codes[0][change[0] != IGNORE]}
    assert labelled == set(CFG.policy)
    assert to_change(codes, {"stable_forest": 255})[0].tolist().count(255) == codes.size
    with pytest.raises(KeyError):
        to_change(codes, {"not_a_reason": 0})
    groups = reason_group(codes)[0]
    grouped = {r for codes_ in REASON_GROUPS.values() for r in codes_}
    assert grouped == set(Reason) - {Reason.UNSET}
    assert groups[list(Reason).index(Reason.UNSET)] == IGNORE


def test_pooling() -> None:
    cfg = CFG.pooling
    change = np.zeros((8, 16), dtype=np.uint8)
    change[:4, 4:8] = IGNORE  # cell (0, 1): all ignore
    change[:2, 8:12] = IGNORE  # cell (0, 2): half ignore
    change[:4, 12:16] = 2  # cell (0, 3): all full
    change[4:8, 0:4].flat[:5] = 1  # cell (1, 0): 31% loss of labelled pixels
    change[4:8, 4:8].flat[:10] = 1  # cell (1, 1): 62% partial
    change[4:8, 8:12].flat[:5] = 1
    change[4:8, 8:12].flat[5:10] = 2  # cell (1, 2): tie, stand-replacing wins
    change[4:8, 12:16].flat[:1] = 1  # cell (1, 3): 6% loss
    pooled = pool_change(change, 4, cfg)
    assert pooled.tolist() == [[0, IGNORE, IGNORE, 2], [IGNORE, 1, 2, 0]]
    agent = np.full((8, 16), IGNORE, dtype=np.uint8)
    agent[4:8, 4:8].flat[:6] = 3
    agent[4:8, 4:8].flat[6:10] = 1
    agent[:4, 12:16] = 5
    out = pool_agent(agent, pooled, 4)
    assert out[1, 1] == 3 and out[0, 3] == 5 and out[0, 0] == IGNORE
    stacked = pool_change(np.stack([change, change]), 4, cfg)
    assert stacked.shape == (2, 2, 4) and (stacked[1] == pooled).all()
