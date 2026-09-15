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

#: Landsat metrics, with display labels and axis limits. These run on their own
#: time base: the NAIP metrics have six dates over ten years, these have one
#: value a year back to the mid-1980s, which reaches disturbances that happened
#: before the aerial record on hand.
#:
#: The limits start above zero because nothing in this landscape reaches there.
#: The lowest growing-season NDVI in the Elkinsville record is 0.53, from a
#: clearcut that never regrew, so an axis running to zero would spend a third of
#: its height on values no forest stand can take. The limits are still fixed
#: rather than fitted per stand, so the same height means the same thing on
#: every figure.
SATELLITE_PANELS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    ("ndvi_mean", "Growing-season NDVI", (0.30, 1.0)),
    ("ndvi_seasonal_amplitude", "NDVI seasonal amplitude", (0.20, 1.0)),
)

#: Fallback for the hollow-marker rule when the caller gives no floor. Callers
#: should pass ``marginal_at_or_below`` from the metric's own configured
#: ``min_obs``, because that number differs by metric: a growing-season mean
#: needs three observations and a seasonal fit needs six. Applying one figure
#: to both drew nearly every mean hollow and nearly every amplitude filled,
#: which said more about the threshold than about the data.
MARGINAL_SUPPORT_OBS = 3

__all__ = [
    "CHANGE_PANELS",
    "ENVELOPE_PANELS",
    "MARGINAL_SUPPORT_OBS",
    "SATELLITE_PANELS",
    "blocking_chart",
    "change_panel",
    "mark_disturbance",
    "metric_panel",
    "satellite_panel",
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


def _connected_line(
    ax: Axes,
    years: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str | None,
    gap_label: str | None,
) -> int:
    """Draw a series that runs across the years it has no value for.

    Two paths rather than one. A segment joining two consecutive reported years
    is solid, and a segment that has to jump over one or more missing years is
    dashed and faded. The reader gets a single continuous trajectory and can
    still see which parts of it are drawn between measurements.

    Both paths are assembled with NaN separators so the whole series costs two
    ``plot`` calls rather than one per segment.

    Returns:
        The number of dashed segments, which is zero when every year reports.
    """
    reported = np.flatnonzero(np.isfinite(values))
    if reported.size == 0:
        return 0

    solid_x, solid_y, gap_x, gap_y = [], [], [], []
    for start, end in zip(reported[:-1], reported[1:], strict=True):
        # Adjacent in the year array means nothing was skipped between them.
        x_pair = [years[start], years[end], np.nan]
        y_pair = [values[start], values[end], np.nan]
        if end == start + 1:
            solid_x.extend(x_pair)
            solid_y.extend(y_pair)
        else:
            gap_x.extend(x_pair)
            gap_y.extend(y_pair)

    n_gaps = len(gap_x) // 3
    if gap_x:
        ax.plot(
            gap_x,
            gap_y,
            "--",
            color=color,
            lw=1.0,
            alpha=0.55,
            zorder=3,
            label=gap_label,
        )
    # The solid path is drawn second so it wins wherever the two meet at a
    # shared point, and it carries the series label even on a record whose
    # first segment happens to be a gap.
    ax.plot(solid_x, solid_y, "-", color=color, lw=1.4, zorder=3, label=label)
    if n_gaps:
        logger.info("Series crosses %d year(s) with no value", n_gaps)
    return n_gaps


def satellite_panel(
    ax: Axes,
    years: Sequence[float],
    values: Sequence[float],
    *,
    label: str,
    ylim: tuple[float, float] | None = None,
    n_obs: Sequence[float] | None = None,
    observations: tuple[Sequence[float], Sequence[float]] | None = None,
    last_pre: float | None = None,
    first_post: float | None = None,
    imagery_years: Sequence[float] | None = None,
    color: str = ACCENT,
    series_label: str | None = None,
    xlim: tuple[float, float] | None = None,
    marginal_at_or_below: float = MARGINAL_SUPPORT_OBS,
    gap_label: str | None = "spans a year with no value",
) -> None:
    """One Landsat metric a year, over the whole satellite record.

    Three things are drawn beyond the line itself, each earning its place.

    The individual clear observations sit behind it as faint dots, on the same
    axis where that makes sense, so a reader can see how much evidence a year's
    value rests on rather than taking the line on trust. That is the same
    argument for printing the metric count under a stage cell.

    Years resting on few observations are drawn hollow, matching how
    ``metric_panel`` marks canopy heights derived from coarser imagery.

    A year the metric could not compute breaks nothing. The line runs on across
    it, dashed and faded over the years it had to skip, so the reader still sees
    one trajectory rather than a scatter of disconnected fragments. Leaving a
    real break there reads as the end of the record instead of as a year that
    happened to be cloudy, and the Landsat record is sparse enough in the 1980s
    and 1990s that a strict line is broken more often than not.

    The NAIP dates are ticked along the bottom, because the band above this one
    is on a different and much shorter time base and the two need tying
    together.

    Args:
        ax: Target axis.
        years: Calendar years, ascending.
        values: The metric at each year. NaN where it could not be computed.
        label: Y-axis label.
        ylim: Y-axis limits.
        n_obs: Observations behind each year, for the hollow-marker rule.
        observations: ``(decimal_year, value)`` for the individual clear
            observations. Only meaningful for a metric on the same scale as the
            observations, so it is passed only for the level panel.
        last_pre: Last imagery date on which the disturbance was absent.
        first_post: First date on which it was present.
        imagery_years: NAIP years to tick.
        color: Line colour.
        series_label: Legend name, which tells two impact polygons apart.
        xlim: X-axis limits. Shared across panels so the band reads as one.
        marginal_at_or_below: Years resting on this many observations or fewer
            are drawn hollow. Pass the metric's own configured ``min_obs``, so
            a hollow marker always means the same thing: this year sits on the
            fewest observations the metric will accept.
        gap_label: Legend name for the dashed segments that cross a year with
            no value. None leaves them out of the legend.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif years.size:
        ax.set_xlim(years.min() - 1, years.max() + 1)

    if observations is not None:
        obs_x, obs_y = (np.asarray(a, dtype=float) for a in observations)
        ax.plot(
            obs_x,
            obs_y,
            ".",
            color=GRID,
            ms=2.2,
            zorder=1,
            rasterized=True,
        )

    mark_disturbance(ax, last_pre, first_post)

    if imagery_years is not None:
        bottom = (ylim or ax.get_ylim())[0]
        span = (ylim or ax.get_ylim())[1] - bottom
        for year in imagery_years:
            ax.plot(
                [year, year],
                [bottom, bottom + 0.045 * span],
                color=MUTED,
                lw=1.0,
                zorder=2,
                clip_on=False,
            )

    _connected_line(
        ax, years, values, color=color, label=series_label, gap_label=gap_label
    )
    if n_obs is None:
        ax.plot(years, values, "o", color=color, ms=3.4, zorder=4)
    else:
        counts = np.asarray(n_obs, dtype=float)
        firm = counts > marginal_at_or_below
        ax.plot(years[firm], values[firm], "o", color=color, ms=3.4, zorder=4)
        ax.plot(
            years[~firm],
            values[~firm],
            "o",
            mfc="white",
            mec=color,
            mew=1.1,
            ms=3.4,
            zorder=4,
        )
    ax.set_ylabel(label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.35)


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
        # Cells nearly touch. The gaps between them carried no information and
        # spread the strip over more of the figure than it earned.
        ax.add_patch(plt_rectangle(i - 0.48, 0.0, 0.96, 1.0, stage))
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
            ax.text(i, -0.14, note, ha="center", va="top", fontsize=7, color=MUTED)
    ax.set_xlim(-0.55, len(years) - 0.45)
    ax.set_ylim(-0.38, 1.0)
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
