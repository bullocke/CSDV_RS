"""Greenness score, diagnostics and sheets on small synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from csdv_core.config import load_label
from csdv_core.label import diagnostics as D
from csdv_core.label import routing as R
from csdv_core.label import sheets as S
from csdv_core.label.labels import JudgedCandidate, WindowLabels
from csdv_core.label.lattice import Window
from csdv_core.label.outlines import EventGeometry
from csdv_core.label.rules import Reason, to_change
from csdv_core.label.screen import cell_ndvi, off_season, residual_z
from csdv_core.label.stacks import WindowStack

CFG = load_label()
WINDOW = Window("w", "u", 26919, 977, 14646, 1, 1)


def test_cell_ndvi_and_missing_imagery() -> None:
    block = np.zeros((4, 32, 32), dtype=np.uint8)
    block[0], block[3] = 50, 150  # red, near-infrared
    block[:, :16, :16] = 0  # one cell without imagery
    ndvi = cell_ndvi(block, 16)
    assert ndvi.shape == (2, 2)
    assert np.isnan(ndvi[0, 0]) and ndvi[1, 1] == pytest.approx(0.5)


def test_residual_z_flags_loss_given_the_starting_greenness() -> None:
    rng = np.random.default_rng(0)
    pre = rng.uniform(0.2, 0.6, (60, 60)).astype("float32")
    # A flight that reads every NDVI lower, more so for greener cells.
    post = (0.8 * pre - 0.05 + rng.normal(0, 0.01, pre.shape)).astype("float32")
    post[10:14, 10:14] -= 0.25  # real loss
    reference = np.ones(pre.shape, bool)
    reference[10:14, 10:14] = False
    z = residual_z(pre, post, reference)
    assert np.nanmedian(np.abs(z[reference])) < 1.0
    assert (z[10:14, 10:14] < D.STRONG_Z).all()
    assert (z[reference] < D.STRONG_Z).mean() < 0.02
    assert residual_z(pre, post, np.zeros(pre.shape, bool)) is None


def test_off_season() -> None:
    assert off_season("2018-11-07") and off_season("2016-04-25")
    assert not off_season("2021-07-27")


def _window_view() -> tuple[S.WindowView, list[JudgedCandidate]]:
    rng = np.random.default_rng(1)
    side = 512 + 2 * 64
    data = rng.integers(60, 200, (3, 4, side, side), dtype=np.uint8)
    data[:, 3] = 200  # green forest
    data[:, 0] = 60
    x0, y0, x1, y1 = WINDOW.bounds
    cand = box(x0 + 60, y0 + 60, x0 + 180, y0 + 180)
    rows = slice(512 - 300, 512 - 100)
    cols = slice(100, 300)
    data[2, 3, 64 + rows.start : 64 + rows.stop, 64 + cols.start : 64 + cols.stop] = 70
    stack = WindowStack(
        WINDOW.stack_bounds(64),
        26919,
        [2015, 2018, 2021],
        ["2015-08-01", "2018-08-02", "2021-08-03"],
        [1.0, 0.6, 0.6],
        data,
    )
    reason = np.full((2, 512, 512), Reason.STABLE_FOREST, dtype=np.uint8)
    reason[1, rows, cols] = Reason.ACCEPTED_FULL
    reason[1, rows, cols.start : cols.start + 8] = Reason.BOUNDARY_INSIDE
    reason[0, :40] = Reason.NONFOREST
    labels = WindowLabels(
        bounds=WINDOW.bounds,
        intervals=["2015-08-01_2018-08-02", "2018-08-02_2021-08-03"],
        reason=reason,
        agent=np.full(reason.shape, 255, np.uint8),
        change=to_change(reason, CFG.policy),
        prob=np.zeros(reason.shape, np.uint8),
        forest=np.ones((3, 512, 512), np.uint8),
        unstable=[False, False],
    )
    routed = R.Routed(
        status="accepted",
        outcomes=[R.INTACT, R.LOSS_FULL],
        first_loss=1,
        severity="stand_replacing",
        change_type="clearcut",
        change_types=("clearcut", "clearcut"),
        outline_fit=("about_right", "about_right"),
        confidence=3,
    )
    event = EventGeometry(
        cand,
        box(0, 0, 0, 0).buffer(0).difference(box(-1, -1, 1, 1)),
        cand.difference(cand),
        "di_refined",
        "precise",
        True,
    )
    judged = [JudgedCandidate("u-di-2021-1", "di", "refined", cand, 1, routed, event)]
    view = S.WindowView("w", "a test window", WINDOW.bounds, stack, labels, judged)
    return view, judged


def test_interval_and_event_rows() -> None:
    view, judged = _window_view()
    rows = D.interval_rows(view.stack, WINDOW, view.labels, CFG)
    assert len(rows) == 2
    shares = [
        rows[1][name] for name in ("loss_full", "ignore_outline", "background_forest")
    ]
    assert sum(shares) == pytest.approx(1.0)
    assert rows[0]["background_nonforest"] == pytest.approx(40 / 512)
    assert rows[1]["cell9.6_loss"] > 0 and rows[0]["cell9.6_loss"] == 0
    assert rows[1]["cell9.6_full"] == 0  # one loss class in the policy
    # The cells that lost their greenness are the labelled ones, not background.
    assert rows[1]["strong_cells"] > 0
    assert rows[1]["strong_background_forest"] == 0
    labelled = rows[1]["strong_loss_full"] + rows[1]["strong_ignore_outline"]
    assert labelled == rows[1]["strong_cells"] and rows[1]["strong_loss_full"] > 100
    events = D.event_rows(WINDOW, view.labels, judged, CFG)
    assert len(events) == 1
    row = events[0]
    assert 0.9 < row["loss_share_px"] < 1.0 and row["band_share_px"] > 0
    assert row["cells9.6_loss"] <= row["cells9.6_inside"]
    assert row["cells2.4_loss"] / row["cells2.4_inside"] > 0.9
    assert D.suspect_rows(view.stack, WINDOW, view.labels) == []


def test_missed_loss_shows_up_as_a_suspect() -> None:
    view, _ = _window_view()
    view.labels.reason[1][:] = Reason.STABLE_FOREST  # nothing was labelled
    suspects = D.suspect_rows(view.stack, WINDOW, view.labels)
    assert len(suspects) == 1 and suspects[0]["k"] == 1
    assert suspects[0]["cells"] >= 100 and suspects[0]["mean_z"] < D.STRONG_Z


def test_sheets_have_the_expected_layout() -> None:
    view, _ = _window_view()
    overview = S.overview_sheet(view, panel_px=128)
    assert overview.width >= 3 * 128 and overview.height > 3 * 128
    pairs = S.choose_pairs(view.labels, WINDOW)
    assert pairs[0][1] == 1 and pairs[0][2] > 0  # the interval with loss first
    chip, k, loss_px, ignore = pairs[0]
    sheet = S.pair_sheet(
        view,
        chip.bounds,
        chip.chip_id,
        k,
        CFG.pooling,
        [S.pair_note(view.stack, view.labels, k, loss_px, ignore)],
    )
    assert sheet.width > 5 * 512 and sheet.height > 512
    red = np.asarray(S.cell_image(np.array([[2, 0], [255, 1]], np.uint8), 64))
    assert tuple(red[5, 5]) == S.CHANGE_COLOURS[2]
    assert tuple(red[60, 60]) == S.CHANGE_COLOURS[1]
