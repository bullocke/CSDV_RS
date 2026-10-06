"""Typical clearcut gap closure in the Elkinsville module, and one stand that departs.

Stand-clearing harvests (clearcut and clearcut with reserves whose gap fraction
at the first post-disturbance NAIP date is at least 0.5) are aligned by years
since the first imagery that shows the cut. Every stand except the outlier
builds a median line and a min to max band in two-year bins. The band stops
where the reference data stops, with no extrapolation.

Frames, written as a reveal build for one slide:

``frame_01_normal``
    Reference stands and the typical band.
``frame_02_outlier``
    The same, plus the outlier.
``frame_03_imagery``
    The plot beside NAIP chips of the outlier and a typical clearcut.

Run:
    .micromamba/envs/CSDV/bin/python \
        scripts/figures/lab_meeting_2026_10/recovery_outlier.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _flipbook import (  # noqa: E402
    NOW_COLOR,
    REPO,
    bare_axis,
    draw_outline,
    harmonize_rgb,
    image_extent,
    load_stand_series,
)

from csdv_core.viz.style import ACCENT, GRID, INK, MUTED, setup_style  # noqa: E402

logger = logging.getLogger(__name__)

STANDS_DIR = REPO / "results" / "stands" / "ElkinsvilleNE"
OUT_DIR = (
    REPO / "Presentations" / "Lab_Meeting_2026-10" / "figures" / "recovery_outlier"
)
FIG_W, FIG_H = 13.333, 7.5

CLEARCUT_LABELS = ("Clearcut", "Clearcut with reserves")
MIN_FIRST_POST_GAP = 0.5
OUTLIER = "ELKNE-U16-0-0a"
TYPICAL = "ELKNE-U8-0-0"
CHIP_YEARS = (2012, 2016, 2022)
BIN_EDGES = (0, 2, 4, 6, 8, 10)
REFERENCE_LINE = "#b9c3c9"
X_MAX = 15


@dataclass
class RecoveryData:
    """Gap fraction by years since disturbance for the selected stands."""

    series: pd.DataFrame
    band: pd.DataFrame
    deviation: pd.Series


def short_name(stand_id: str) -> str:
    return stand_id.removeprefix("ELKNE-").removesuffix("-0-0")


def load_recovery() -> RecoveryData:
    """Select stand-clearing harvests and align them by years since the cut."""
    metrics = pd.read_parquet(STANDS_DIR / "stand_metrics.parquet")
    screen = pd.read_csv(STANDS_DIR / "screen.csv")
    data = metrics[["stand_id", "year", "gap_fraction", "support_nodata_fraction"]]
    data = data.merge(
        screen[["stand_id", "dist_label", "last_pre", "first_post"]], on="stand_id"
    )
    data = data[data["dist_label"].isin(CLEARCUT_LABELS)]
    data = data[data["year"] >= data["first_post"]].copy()
    data["tsd"] = data["year"] - data["first_post"]
    data = data.sort_values(["stand_id", "tsd"])

    first = data.groupby("stand_id")["gap_fraction"].first()
    keep = first[first >= MIN_FIRST_POST_GAP].index
    dropped = sorted(set(first.index) - set(keep))
    logger.info(
        "Dropped as partial (first post gap < %.1f): %s", MIN_FIRST_POST_GAP, dropped
    )
    data = data[data["stand_id"].isin(keep)]
    logger.info("Selected %d stands: %s", len(keep), sorted(keep))

    ref = data[data["stand_id"] != OUTLIER].copy()
    ref["bin"] = pd.cut(ref["tsd"], BIN_EDGES, labels=False, include_lowest=True)
    band = (
        ref.groupby("bin")
        .agg(
            tsd=("tsd", "mean"),
            median=("gap_fraction", "median"),
            lo=("gap_fraction", "min"),
            hi=("gap_fraction", "max"),
            n=("gap_fraction", "size"),
        )
        .reset_index()
    )
    logger.info("Typical band:\n%s", band.round(2).to_string(index=False))

    data["bin"] = pd.cut(data["tsd"], BIN_EDGES, labels=False, include_lowest=True)
    medians = band.set_index("bin")["median"]
    in_range = data.dropna(subset=["bin"])
    resid = in_range["gap_fraction"] - in_range["bin"].map(medians)
    deviation = resid.groupby(in_range["stand_id"]).mean().sort_values(ascending=False)
    logger.info(
        "Mean departure from the typical band:\n%s", deviation.round(2).to_string()
    )
    if deviation.index[0] != OUTLIER:
        raise SystemExit(
            f"{OUTLIER} is not the largest departure; revisit the selection"
        )

    nodata = data.loc[data["stand_id"] == OUTLIER, "support_nodata_fraction"]
    logger.info("%s nodata fraction by date: %s", OUTLIER, nodata.round(3).tolist())
    return RecoveryData(series=data, band=band, deviation=deviation)


def draw_plot(ax: Axes, rec: RecoveryData, *, show_outlier: bool) -> None:
    for stand_id, group in rec.series.groupby("stand_id"):
        if stand_id == OUTLIER:
            continue
        ax.plot(
            group["tsd"], group["gap_fraction"], color=REFERENCE_LINE, lw=1.2, zorder=2
        )

    band = rec.band
    ax.fill_between(
        band["tsd"], band["lo"], band["hi"], color=ACCENT, alpha=0.14, lw=0, zorder=1
    )
    ax.plot(
        band["tsd"],
        band["median"],
        color=ACCENT,
        lw=3.0,
        solid_capstyle="round",
        zorder=3,
    )
    last = band.iloc[-1]
    ax.text(
        last["tsd"] + 0.3,
        last["median"] + 0.02,
        "Typical clearcut",
        color=ACCENT,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )

    if show_outlier:
        out = rec.series[rec.series["stand_id"] == OUTLIER]
        ax.plot(
            out["tsd"],
            out["gap_fraction"],
            color=NOW_COLOR,
            lw=3.5,
            marker="o",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax.text(
            out["tsd"].iloc[-1] + 0.5,
            out["gap_fraction"].iloc[-1] - 0.05,
            "One clearcut\nthat stays open",
            color=NOW_COLOR,
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="top",
        )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, X_MAX + 1, 2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Years since harvest", fontsize=14)
    ax.set_ylabel("Stand area under 2 m (gap fraction)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_chips(fig: plt.Figure, rec: RecoveryData, *, left: float) -> None:
    """Two rows of NAIP chips: the outlier above, a typical clearcut below."""
    rows = [(OUTLIER, NOW_COLOR, "Stays open"), (TYPICAL, ACCENT, "Typical")]
    n = len(CHIP_YEARS)
    width, gap, top = (0.97 - left) / n, 0.008, 0.86
    height = width * FIG_W / FIG_H
    for r, (stand_id, color, label) in enumerate(rows):
        series = load_stand_series(stand_id)
        rgb = harmonize_rgb(series)
        first_post = int(series.first_post)
        y0 = top - (r + 1) * height - r * 0.12
        fig.text(
            left,
            y0 + height + 0.05,
            label,
            color=color,
            fontsize=13,
            fontweight="bold",
        )
        for c, year in enumerate(CHIP_YEARS):
            ax = fig.add_axes((left + c * width + gap / 2, y0, width - gap, height))
            ax.imshow(rgb[year], extent=image_extent(series), interpolation="nearest")
            draw_outline(ax, series)
            bare_axis(ax)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
            ax.set_title(
                f"{year}  ·  {year - first_post} yr",
                fontsize=11,
                color=INK if r == 0 else MUTED,
                pad=4,
            )


def build(rec: RecoveryData, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        ("frame_01_normal", False, False),
        ("frame_02_outlier", True, False),
        ("frame_03_imagery", True, True),
    ]
    for name, show_outlier, chips in frames:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        rect = (0.07, 0.12, 0.45, 0.78) if chips else (0.08, 0.12, 0.86, 0.80)
        ax = fig.add_axes(rect)
        draw_plot(ax, rec, show_outlier=show_outlier)
        if chips:
            draw_chips(fig, rec, left=0.56)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        logger.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_style()
    plt.rcParams["savefig.bbox"] = "standard"  # fixed canvas so reveal frames align
    rec = load_recovery()
    build(rec, args.out, args.dpi)


if __name__ == "__main__":
    main()
