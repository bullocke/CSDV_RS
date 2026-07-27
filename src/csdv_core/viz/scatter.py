"""csdv_core.viz.scatter — agreement and screening scatter plots.

Two plots live here. One compares a metric measured from imagery against the
photo interpreter's ordinal cover class, which is the only external reference a
calibration delivery carries. The other lays out the stands in a module by size
and by how much of their bounding box they fill, which is how an example gets
chosen and how the choice is made auditable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from csdv_core.io.stands import COVER_CLASS_RANGE
from csdv_core.viz.style import ACCENT, GRID, HIGHLIGHT, INK, MUTED

logger = logging.getLogger(__name__)

__all__ = ["cover_class_agreement", "screening_scatter", "spearman"]


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, int]:
    """Spearman rank correlation and the number of complete pairs.

    Rank correlation rather than Pearson, because the reference is an ordinal
    cover class rather than a measurement.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    n = int(keep.sum())
    if n < 3:
        return float("nan"), n
    from scipy.stats import spearmanr

    rho = float(spearmanr(a[keep], b[keep]).statistic)
    return rho, n


def cover_class_agreement(
    ax: Axes,
    measured: Sequence[float],
    cover_class: Sequence[float],
    *,
    label: str = "Crown fraction from the canopy height model",
    class_ranges: Mapping[int, tuple[float, float]] = COVER_CLASS_RANGE,
    color: str = ACCENT,
) -> tuple[float, int]:
    """Plot a measured fraction against the interpreter's cover class.

    Each class is drawn as the band it stands for, so a reader can see directly
    whether the measurement falls inside the interpreter's bin rather than
    having to read that off a correlation. Points are jittered along the class
    axis only, so no measured value is displaced.

    Returns:
        ``(rho, n)`` from :func:`spearman` against the class midpoints.
    """
    measured = np.asarray(measured, dtype=float)
    classes = np.asarray(cover_class, dtype=float)
    valid_codes = sorted(c for c in class_ranges if c != 9)

    for code in valid_codes:
        lo, hi = class_ranges[code]
        ax.fill_between(
            [code - 0.45, code + 0.45], lo, hi, color=GRID, alpha=0.55, lw=0
        )

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.22, 0.22, size=classes.shape)
    ax.scatter(
        classes + jitter,
        measured,
        s=22,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    midpoints = np.array(
        [
            np.mean(class_ranges.get(int(c), (np.nan, np.nan)))
            if np.isfinite(c)
            else np.nan
            for c in classes
        ]
    )
    rho, n = spearman(midpoints, measured)

    ax.set_xticks(valid_codes)
    ax.set_xticklabels(
        [
            f"{int(class_ranges[c][0] * 100)}-{int(class_ranges[c][1] * 100)}"
            for c in valid_codes
        ]
    )
    ax.set_xlabel("Interpreted tree cover class (%)")
    ax.set_ylabel(label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(valid_codes) - 0.6, max(valid_codes) + 0.6)
    ax.grid(axis="y", alpha=0.4)
    return rho, n


def screening_scatter(
    ax: Axes,
    stands: pd.DataFrame,
    *,
    highlight: Sequence[str] = (),
    highlight_labels: Mapping[str, str] | None = None,
    group_column: str = "dist_group",
    size_column: str | None = None,
    group_colors: Mapping[str, str] | None = None,
) -> None:
    """Lay out a module's stands by area and bounding-box fill.

    Area sets whether a stand holds enough crowns for a crown statistic. Fill
    sets how much of a texture or spatial metric's rectangular support is
    actually inside the stand. Both bear on which stands make usable examples,
    so plotting them together makes the choice auditable rather than asserted.

    Args:
        ax: Target axis.
        stands: Frame with ``bbox_fill``, ``stand_id``, the grouping column,
            and either ``acres`` or ``area_m2``.
        highlight: Stand identifiers to ring and label.
        highlight_labels: Optional label per highlighted stand, for example the
            letter the worked example carries elsewhere in the document.
            Defaults to the footprint part of the identifier.
        group_column: Column used to colour the points.
        size_column: Optional column scaling marker area, for example the count
            of usable post-disturbance dates.
        group_colors: Optional colour per group value.
    """
    if "acres" in stands.columns:
        acres = stands["acres"].to_numpy(dtype=float)
    else:
        acres = stands["area_m2"].to_numpy(dtype=float) / 4046.856
    fill = stands["bbox_fill"].to_numpy(dtype=float)
    groups = stands[group_column].astype(str).to_numpy()

    if size_column and size_column in stands.columns:
        raw = stands[size_column].to_numpy(dtype=float)
        sizes = 18.0 + 14.0 * np.nan_to_num(raw, nan=0.0)
    else:
        sizes = np.full(len(stands), 34.0)

    palette = dict(group_colors or {})
    for group in sorted(set(groups)):
        keep = groups == group
        ax.scatter(
            acres[keep],
            fill[keep],
            s=sizes[keep],
            color=palette.get(group, MUTED),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=group,
            zorder=3,
        )

    wanted = set(highlight)
    if wanted:
        ids = stands["stand_id"].astype(str).to_numpy()
        keep = np.isin(ids, list(wanted))
        ax.scatter(
            acres[keep],
            fill[keep],
            s=sizes[keep] + 110.0,
            facecolor="none",
            edgecolor=HIGHLIGHT,
            linewidth=1.6,
            zorder=4,
        )
        labels = dict(highlight_labels or {})
        for x, y, name in zip(acres[keep], fill[keep], ids[keep], strict=True):
            ax.annotate(
                labels.get(name, name.split("-")[1]),
                (x, y),
                textcoords="offset points",
                xytext=(9, 6),
                fontsize=7.5,
                fontweight="bold",
                color=INK,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Stand area (acres)")
    ax.set_ylabel("Share of the bounding box inside the stand")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.4)
    ax.legend(loc="lower right", ncols=2)
