"""csdv_core.viz.panels — the non-map panels of a worked example.

A worked example needs three things beyond the imagery: how the metrics moved,
what stage each date was assigned and on how much evidence, and which
trajectory classes were even available to be assigned. These are the panels
that carry those.

The rule throughout is that the figure carries the measurement and the caption
carries the interpretation. Axis labels, dates and a legend go on the figure.
Class names, thresholds and caveats go in the caption.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from csdv_core.viz.style import (
    ACCENT,
    BLOCKING_COLORS,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    UNCLASSIFIED_COLOR,
    stage_color,
)

logger = logging.getLogger(__name__)

#: Metrics that drive the stage envelopes, with display labels and axis limits.
ENVELOPE_PANELS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    ("gap_fraction", "Gap fraction", (0.0, 1.0)),
    ("crown_cv", "Crown width CV", (0.0, 1.0)),
    ("glcm_texture", "Texture entropy (bits)", (0.0, 8.0)),
    ("shrub_fraction", "Shrub fraction", (0.0, 0.6)),
    ("gap_persistence", "Gap persistence", (0.0, 1.0)),
)

#: Consecutive-date change metrics that separate events looking alike in one
#: snapshot: a clearcut takes canopy and large crowns together, a thinning
#: takes canopy while leaving the large crowns, highgrading does the reverse.
CHANGE_PANELS: tuple[tuple[str, str], ...] = (
    ("d_crown_fraction", "Change in crown fraction"),
    ("d_gap_fraction", "Change in gap fraction"),
    ("d_crown_p90", "Change in crown width P90 (m)"),
)

__all__ = [
    "CHANGE_PANELS",
    "ENVELOPE_PANELS",
    "blocking_chart",
    "change_panel",
    "mark_disturbance",
    "metric_panel",
    "stage_strip",
]


def mark_disturbance(
    ax: Axes,
    last_pre: float | None,
    first_post: float | None,
    *,
    color: str = HIGHLIGHT,
) -> bool:
    """Mark the interval in which the disturbance was first seen.

    The interval is bracketed by dashed lines and lightly shaded, so a reader
    can tell whether a change in a metric straddles the event or happened
    somewhere else in the series.

    Nothing is drawn when the interval falls outside the plotted range. A stand
    disturbed before the imagery record would otherwise get a line jammed
    against the left spine, which reads as an event at the first date rather
    than as one that happened before it. Those figures say so in their title
    instead.

    Returns:
        True when the interval was drawn.
    """
    if last_pre is None or first_post is None:
        return False
    if not (np.isfinite(last_pre) and np.isfinite(first_post)):
        return False
    left, right = ax.get_xlim()
    if float(first_post) < left or float(last_pre) > right:
        return False
    ax.axvspan(last_pre, first_post, color=GRID, alpha=0.7, lw=0, zorder=0)
    for year in dict.fromkeys([float(last_pre), float(first_post)]):
        if left <= year <= right:
            ax.axvline(year, color=color, lw=1.0, ls="--", zorder=1)
    return True


def metric_panel(
    ax: Axes,
    years: Sequence[float],
    values: Sequence[float],
    *,
    label: str,
    ylim: tuple[float, float] | None = None,
    native_res_m: Sequence[float] | None = None,
    last_pre: float | None = None,
    first_post: float | None = None,
    color: str = ACCENT,
    series_label: str | None = None,
) -> None:
    """One metric against time for one stand.

    Dates whose imagery is coarser than the rest are drawn as open markers,
    because a canopy height model derived from 1.0 m imagery is not equivalent
    to one derived from 0.6 m even after both land on a shared grid.

    ``series_label`` names the line for a legend, which is what tells two
    impact polygons of the same footprint apart when both are drawn here.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    ax.set_xlim(years.min() - 1, years.max() + 1)
    mark_disturbance(ax, last_pre, first_post)
    ax.plot(years, values, "-", color=color, lw=1.5, zorder=3, label=series_label)

    if native_res_m is None:
        ax.plot(years, values, "o", color=color, ms=5, zorder=4)
    else:
        res = np.asarray(native_res_m, dtype=float)
        fine = res <= np.nanmin(res) + 1e-9
        ax.plot(years[fine], values[fine], "o", color=color, ms=5, zorder=4)
        ax.plot(
            years[~fine],
            values[~fine],
            "o",
            mfc="white",
            mec=color,
            mew=1.4,
            ms=5,
            zorder=4,
        )
    ax.set_ylabel(label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.35)


def change_panel(
    ax: Axes,
    years: Sequence[float],
    values: Sequence[float],
    *,
    label: str,
    color: str = HIGHLIGHT,
    last_pre: float | None = None,
    first_post: float | None = None,
) -> None:
    """One consecutive-date change metric as bars, with a zero line.

    Each bar sits at the later of the two dates it was computed from, so a bar
    to the right of the marked disturbance interval is the change across it.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    ax.set_xlim(years.min() - 1, years.max() + 1)
    mark_disturbance(ax, last_pre, first_post)
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
    ax.bar(years[keep], values[keep], width=1.1, color=color, alpha=0.85, zorder=2)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.35)


def stage_strip(
    ax: Axes,
    years: Sequence[float],
    stages: Sequence[str | None],
    *,
    n_evaluated: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
) -> None:
    """One coloured cell per date showing the stage assigned.

    The count of metrics each assignment rests on is printed under its cell,
    because a stage chosen on two metrics is weaker evidence than the same
    stage chosen on five, and nothing else on the figure would show that.
    """
    years = list(years)
    for i, stage in enumerate(stages):
        ax.add_patch(plt_rectangle(i - 0.45, 0.0, 0.9, 1.0, stage))
        ax.text(
            i,
            0.5,
            stage or "--",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if stage else MUTED,
        )
        if n_evaluated is not None:
            note = f"n={int(n_evaluated[i])}"
            if scores is not None and np.isfinite(scores[i]):
                note += f"  {scores[i]:.2f}"
            ax.text(i, -0.22, note, ha="center", va="top", fontsize=7, color=MUTED)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(-0.45, 1.0)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plt_rectangle(x: float, y: float, width: float, height: float, stage: str | None):
    """A stage-coloured rectangle, hatched where nothing was assigned."""
    from matplotlib.patches import Rectangle

    if stage is None:
        return Rectangle(
            (x, y),
            width,
            height,
            facecolor=UNCLASSIFIED_COLOR,
            edgecolor=MUTED,
            hatch="///",
            linewidth=0.6,
        )
    return Rectangle(
        (x, y),
        width,
        height,
        facecolor=stage_color(stage),
        edgecolor="white",
        linewidth=0.8,
    )


def blocking_chart(
    ax: Axes,
    order: Sequence[str],
    blocked: Mapping[str, Sequence[str]],
    rules: Mapping[str, object],
    *,
    names: Mapping[str, str] | None = None,
    available_metrics: Sequence[str] = (),
) -> None:
    """Which trajectory classes can fire, and how far the rest are from it.

    Each bar is one rule, its length the number of conditions the rule sets.
    The solid part is the conditions that carry a threshold and name a metric
    that exists; the hollow part is the conditions still to be filled in. A rule
    fires only when the whole bar is solid.

    Showing it this way separates a rule that needs one number from one that
    needs a metric nobody has implemented, which are very different amounts of
    work.
    """
    names = names or {}
    have = set(available_metrics)
    positions = np.arange(len(order))[::-1]

    labels, totals, ready, colors, notes = [], [], [], [], []
    for code in order:
        rule = rules.get(code)
        predicates = list(getattr(rule, "signature", []) or [])
        n_total = len(predicates)
        n_ready = sum(
            1
            for p in predicates
            if p.value is not None and (p.var is None or not have or p.var in have)
        )
        reasons = list(blocked.get(code, ()))
        primary = reasons[0].split(":")[0] if reasons else ""
        label = f"{code}  {names.get(code, '')}"
        labels.append(label if len(label) <= 48 else label[:47] + "…")
        totals.append(n_total)
        ready.append(n_ready)
        colors.append(BLOCKING_COLORS.get(primary, MUTED) if reasons else ACCENT)
        if not reasons:
            notes.append("can fire")
        else:
            unique = {r.split(":")[0] for r in reasons}
            notes.append(" + ".join(sorted(unique)))

    ax.barh(
        positions, totals, color="white", edgecolor=MUTED, height=0.7, linewidth=0.7
    )
    ax.barh(positions, ready, color=colors, height=0.7, linewidth=0)
    span = max(totals) if totals else 1
    for pos, note in zip(positions, notes, strict=True):
        ax.text(span + 0.25, pos, note, va="center", fontsize=7.5, color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks(range(0, span + 1))
    ax.set_xlabel("Conditions in the rule")
    ax.set_xlim(0, span * 2.6)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def stand_series(frame: pd.DataFrame, stand_id: str) -> pd.DataFrame:
    """One stand's rows in ascending year order."""
    return (
        frame[frame["stand_id"] == stand_id].sort_values("year").reset_index(drop=True)
    )
