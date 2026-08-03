"""Tests for the metric, stage and rule-status panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from csdv_core.config._models import (  # noqa: E402
    TrajectoriesConfig,
    TrajectoryPredicate,
    TrajectoryRule,
)
from csdv_core.trajectories.stand import blocking_report  # noqa: E402
from csdv_core.viz.panels import (  # noqa: E402
    CHANGE_PANELS,
    ENVELOPE_PANELS,
    blocking_chart,
    change_panel,
    mark_disturbance,
    metric_panel,
    stage_strip,
    stand_series,
)
from csdv_core.viz.style import UNCLASSIFIED_COLOR, stage_color  # noqa: E402

YEARS = [2012, 2014, 2016, 2018, 2020, 2022]
RES = [1.0, 1.0, 0.6, 0.6, 0.6, 0.6]


def test_envelope_panels_cover_the_five_stage_metrics():
    names = {name for name, _, _ in ENVELOPE_PANELS}
    assert names == {
        "gap_fraction",
        "crown_cv",
        "glcm_texture",
        "shrub_fraction",
        "gap_persistence",
    }


def test_change_panels_cover_the_diagnostic_differences():
    names = {name for name, _ in CHANGE_PANELS}
    assert names == {"d_crown_fraction", "d_gap_fraction", "d_crown_p90"}


def test_metric_panel_marks_the_coarser_dates_differently():
    """A canopy height model from 1.0 m imagery is not equivalent to one from
    0.6 m, so those dates must be visually distinct."""
    fig, ax = plt.subplots()
    metric_panel(
        ax,
        YEARS,
        [0.6, 0.5, 0.4, 0.3, 0.25, 0.2],
        label="Gap fraction",
        ylim=(0, 1),
        native_res_m=RES,
    )
    # One line for the series, one marker set for fine dates, one for coarse.
    assert len(ax.lines) == 3
    coarse = ax.lines[2]
    assert coarse.get_markerfacecolor() == "white"
    assert len(coarse.get_xdata()) == 2
    plt.close(fig)


def test_metric_panel_without_resolution_draws_one_marker_set():
    fig, ax = plt.subplots()
    metric_panel(ax, YEARS, np.linspace(0.6, 0.2, 6), label="Gap fraction")
    assert len(ax.lines) == 2
    plt.close(fig)


def test_metric_panel_shades_the_disturbance_interval():
    fig, ax = plt.subplots()
    metric_panel(
        ax,
        YEARS,
        np.linspace(0.6, 0.2, 6),
        label="Gap fraction",
        last_pre=2016,
        first_post=2017,
    )
    assert len(ax.patches) == 1
    plt.close(fig)


def test_metric_panel_tolerates_all_nan():
    fig, ax = plt.subplots()
    metric_panel(ax, YEARS, np.full(6, np.nan), label="Gap persistence", ylim=(0, 1))
    plt.close(fig)


def test_change_panel_draws_only_finite_bars():
    fig, ax = plt.subplots()
    change_panel(
        ax,
        YEARS,
        [np.nan, -0.05, -0.30, 0.02, 0.06, 0.09],
        label="Change in crown fraction",
    )
    assert len(ax.patches) == 5  # the first date has no previous date
    plt.close(fig)


def test_stage_strip_draws_one_cell_per_date():
    fig, ax = plt.subplots()
    stage_strip(
        ax,
        YEARS,
        ["ESI", "LSI", "LSI", None, "ESE", "ESE"],
        n_evaluated=[4, 5, 5, 5, 5, 5],
        scores=[0.75, 1.0, 0.8, 0.4, 1.0, 0.8],
    )
    assert len(ax.patches) == len(YEARS)
    plt.close(fig)


def test_stage_strip_hatches_an_unassigned_date():
    fig, ax = plt.subplots()
    stage_strip(ax, [2018, 2020], [None, "LSE"])
    unassigned, assigned = ax.patches
    assert unassigned.get_hatch() == "///"
    assert assigned.get_hatch() is None
    plt.close(fig)


def test_stage_colors_are_distinct_and_unassigned_is_grey():
    codes = ["ESI", "LSI", "ESE", "LSE", "UR", "MA_OW", "OG"]
    colors = [stage_color(c) for c in codes]
    assert len(set(colors)) == len(codes)
    assert stage_color(None) == UNCLASSIFIED_COLOR


def _config() -> TrajectoriesConfig:
    return TrajectoriesConfig(
        trajectory_order=["OK", "NOTHRESH", "NOMETRIC"],
        trajectory_codes={"OK": 1, "NOTHRESH": 2, "NOMETRIC": 3},
        trajectories={
            "OK": TrajectoryRule(
                name="Complete",
                group="DS",
                signature=[
                    TrajectoryPredicate(
                        dim="metric",
                        var="gap_fraction",
                        reducer="max",
                        op="<=",
                        value=0.1,
                    ),
                    TrajectoryPredicate(
                        dim="metric",
                        var="crown_cv",
                        reducer="max",
                        op="<=",
                        value=0.2,
                    ),
                ],
            ),
            "NOTHRESH": TrajectoryRule(
                name="Missing a number",
                group="EF",
                signature=[
                    TrajectoryPredicate(
                        dim="metric",
                        var="gap_fraction",
                        reducer="max",
                        op="<=",
                        value=0.1,
                    ),
                    TrajectoryPredicate(
                        dim="metric",
                        var="crown_cv",
                        reducer="max",
                        op="<=",
                        value=None,
                    ),
                ],
            ),
            "NOMETRIC": TrajectoryRule(
                name="Missing a metric",
                group="LC",
                signature=[
                    TrajectoryPredicate(
                        dim="metric",
                        var="ndvi_trend",
                        reducer="max",
                        op=">=",
                        value=0.4,
                    ),
                ],
            ),
        },
    )


def test_blocking_chart_bar_length_is_the_condition_count():
    cfg = _config()
    available = ["gap_fraction", "crown_cv"]
    fig, ax = plt.subplots()
    blocking_chart(
        ax,
        list(cfg.trajectory_order),
        blocking_report(cfg, available),
        cfg.trajectories,
        available_metrics=available,
    )
    # Two bars per rule: the outline of every condition, then the solid part.
    widths = [p.get_width() for p in ax.patches]
    outlines, solids = widths[:3], widths[3:]
    assert outlines == [2, 2, 1]  # conditions per rule, top to bottom
    assert solids == [2, 1, 0]  # of those, how many are usable
    plt.close(fig)


def test_blocking_chart_labels_each_reason():
    cfg = _config()
    available = ["gap_fraction", "crown_cv"]
    fig, ax = plt.subplots()
    blocking_chart(
        ax,
        list(cfg.trajectory_order),
        blocking_report(cfg, available),
        cfg.trajectories,
        available_metrics=available,
    )
    notes = [t.get_text() for t in ax.texts]
    assert "can fire" in notes
    assert "threshold not set" in notes
    assert any("metric not available" in n for n in notes)
    plt.close(fig)


def test_stand_series_sorts_by_year():
    frame = pd.DataFrame(
        {"stand_id": ["A", "A", "B"], "year": [2022, 2018, 2018], "v": [3, 1, 2]}
    )
    out = stand_series(frame, "A")
    assert list(out["year"]) == [2018, 2022]
    assert list(out["v"]) == [1, 3]


def test_stand_series_of_an_absent_stand_is_empty():
    frame = pd.DataFrame({"stand_id": ["A"], "year": [2018]})
    assert stand_series(frame, "Z").empty


@pytest.mark.parametrize("metric", [name for name, _, _ in ENVELOPE_PANELS])
def test_every_envelope_panel_has_sane_limits(metric):
    limits = {name: ylim for name, _, ylim in ENVELOPE_PANELS}[metric]
    assert limits[0] < limits[1]
    assert limits[0] >= 0.0


def test_mark_disturbance_draws_dashed_bounds_inside_the_range():
    fig, ax = plt.subplots()
    ax.set_xlim(2011, 2023)
    assert mark_disturbance(ax, 2016, 2017) is True
    assert len(ax.patches) == 1  # the shaded interval
    assert [line.get_linestyle() for line in ax.lines] == ["--", "--"]
    plt.close(fig)


def test_mark_disturbance_collapses_a_single_year_to_one_line():
    fig, ax = plt.subplots()
    ax.set_xlim(2011, 2023)
    mark_disturbance(ax, 2017, 2017)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_mark_disturbance_skips_an_event_before_the_record():
    """A stand cut in 2007 has no before. A line jammed against the left spine
    would read as an event at the first date, which is a different claim."""
    fig, ax = plt.subplots()
    ax.set_xlim(2011, 2023)
    assert mark_disturbance(ax, 2007, 2008) is False
    assert not ax.patches and not ax.lines
    plt.close(fig)


def test_metric_panel_does_not_mark_an_event_outside_its_dates():
    fig, ax = plt.subplots()
    metric_panel(
        ax,
        YEARS,
        np.linspace(0.6, 0.2, 6),
        label="Gap fraction",
        last_pre=2007,
        first_post=2008,
    )
    assert not ax.patches
    plt.close(fig)


def test_metric_panel_names_the_series_for_a_legend():
    fig, ax = plt.subplots()
    metric_panel(
        ax, YEARS, np.linspace(0.6, 0.2, 6), label="Gap fraction", series_label="(a)"
    )
    assert ax.get_legend_handles_labels()[1] == ["(a)"]
    plt.close(fig)


def test_change_panel_marks_the_disturbance_interval():
    fig, ax = plt.subplots()
    change_panel(
        ax,
        YEARS,
        [np.nan, -0.05, -0.30, 0.02, 0.06, 0.09],
        label="Change in crown fraction",
        last_pre=2016,
        first_post=2017,
    )
    spans = [
        p for p in ax.patches if p.get_width() > 1.0 - 1e-9 and p.get_width() < 1.01
    ]
    assert len(ax.patches) == 6  # five bars plus the shaded interval
    assert len(spans) == 1
    plt.close(fig)
