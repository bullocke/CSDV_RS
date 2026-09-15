"""Does the metric separate disturbed stands from undisturbed ones?

Allometric realism is necessary and not sufficient. ``crown_cv`` exists to help
assign a developmental stage and a trajectory, so a parameter set that produces
beautiful crowns and no separation is useless to this project.

This scores separation directly, using the interpreter labels and dates in the
calibration geodatabase. For each labelled stand it takes the dates before the
event and the dates after it, and asks how well each crown metric tells those
two states apart. Two statistics, because they answer different questions:

    AUC
        Probability that a randomly chosen post-event stand-year scores above a
        randomly chosen pre-event one. 0.5 is no separation, and it does not
        care about the shape of the distribution.

    Cohen's d
        Difference in means over the pooled standard deviation, which says
        whether a difference is large relative to the scatter it has to be seen
        through.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/discrimination.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    NAIP_YEARS,
    configure_logging,
    elkinsville_chm,
    params_from_row,
    read_table,
    read_window,
    stands,
    write_table,
)

from csdv_core.segmentation.chm_watershed import segment_crowns  # noqa: E402
from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)
from csdv_core.zonal.crowns import crown_diameter_stats, crowns_in_stand  # noqa: E402

logger = logging.getLogger("discrimination")

CROWN_METRICS = ("crown_cv", "crown_p90", "crown_mean", "crown_median", "crown_count")

#: Stands smaller than this cannot hold enough crowns for a stable statistic,
#: so including them would measure sample noise rather than separation.
MIN_AREA_HA = 1.0


def auc(pre: np.ndarray, post: np.ndarray) -> float:
    """Area under the ROC curve, computed by rank."""
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if pre.size < 3 or post.size < 3:
        return float("nan")
    combined = np.concatenate([pre, post])
    ranks = pd.Series(combined).rank().to_numpy()
    r_post = ranks[pre.size :].sum()
    n1, n2 = post.size, pre.size
    return float((r_post - n1 * (n1 + 1) / 2) / (n1 * n2))


def cohens_d(pre: np.ndarray, post: np.ndarray) -> float:
    """Standardised mean difference."""
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if pre.size < 3 or post.size < 3:
        return float("nan")
    pooled = np.sqrt(
        ((pre.size - 1) * pre.var(ddof=1) + (post.size - 1) * post.var(ddof=1))
        / (pre.size + post.size - 2)
    )
    return float((post.mean() - pre.mean()) / pooled) if pooled > 0 else float("nan")


def stand_metrics_for(stand_gdf, params: SegmentationParams, years) -> pd.DataFrame:
    """Crown statistics per stand per year.

    The read window is padded so crowns on the stand boundary are delineated
    against their real neighbours rather than against the edge of the read.
    """
    rows = []
    pad = 60.0
    for year in years:
        chm = elkinsville_chm(year)
        for _, stand in stand_gdf.iterrows():
            minx, miny, maxx, maxy = stand.geometry.bounds
            bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
            arr, transform, crs = read_window(chm, bounds)
            if not np.isfinite(arr).any():
                continue
            crowns = segment_crowns(arr, transform, crs, params=params)
            inside = crowns_in_stand(crowns, stand.geometry)
            stats = crown_diameter_stats(inside, min_crowns=3)
            rows.append(
                {
                    "stand_id": stand["stand_id"],
                    "dist_label": stand["dist_label"],
                    "dist_group": stand["dist_group"],
                    "year": year,
                    "area_ha": stand["area_m2"] / 1e4,
                    "last_pre": stand.get("LastImageryPreDist"),
                    "first_post": stand.get("FirstImageryPostDist"),
                    **stats.as_metrics(),
                }
            )
        logger.info("%d done", year)
    return pd.DataFrame(rows)


def main() -> int:
    """Score how well each crown metric separates the two states."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sweep", action="store_true")
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    parser.add_argument("--th-cr", type=float, default=0.55)
    args = parser.parse_args()

    if args.from_sweep:
        scored = read_table("sweep_tune_scored.parquet")
        passing = scored[scored["passes"]]
        chosen = (
            params_from_row(passing.iloc[0])
            if len(passing)
            else SegmentationParams(window=WINDOW_FUNCTIONS[args.window])
        )
    else:
        chosen = SegmentationParams(
            smooth_radius_m=args.smooth_radius_m,
            window=WINDOW_FUNCTIONS[args.window],
            th_cr=args.th_cr,
            max_crown_radius_m=None,
        )
    logger.info("Chosen: %s", chosen.describe())

    stand_gdf = stands()
    stand_gdf = stand_gdf[stand_gdf["area_m2"] / 1e4 >= MIN_AREA_HA].reset_index(
        drop=True
    )
    logger.info("%d stands at or above %.1f ha", len(stand_gdf), MIN_AREA_HA)

    frame = stand_metrics_for(stand_gdf, chosen, NAIP_YEARS)
    write_table(frame, "discrimination_metrics.parquet")

    def phase(row):
        pre, post = row["last_pre"], row["first_post"]
        if pd.isna(pre) or pd.isna(post):
            return "unknown"
        if row["year"] <= float(pre):
            return "pre"
        if row["year"] >= float(post):
            return "post"
        return "during"

    frame["phase"] = frame.apply(phase, axis=1)
    usable = frame[frame["phase"].isin(["pre", "post"])]
    logger.info(
        "%d pre-event and %d post-event stand-years",
        int((usable["phase"] == "pre").sum()),
        int((usable["phase"] == "post").sum()),
    )

    rows = []
    for group in ["all"] + sorted(usable["dist_group"].dropna().unique()):
        sub = usable if group == "all" else usable[usable["dist_group"] == group]
        for metric in CROWN_METRICS:
            pre = sub[sub["phase"] == "pre"][metric].to_numpy(dtype="float64")
            post = sub[sub["phase"] == "post"][metric].to_numpy(dtype="float64")
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_pre": int(np.isfinite(pre).sum()),
                    "n_post": int(np.isfinite(post).sum()),
                    "pre_mean": float(np.nanmean(pre)) if pre.size else np.nan,
                    "post_mean": float(np.nanmean(post)) if post.size else np.nan,
                    "auc": auc(pre, post),
                    "cohens_d": cohens_d(pre, post),
                }
            )
    result = pd.DataFrame(rows)
    result["separation"] = (result["auc"] - 0.5).abs() * 2
    write_table(result, "discrimination.csv")

    print("\nSeparation of post-event from pre-event states")
    print(
        result.sort_values(["group", "separation"], ascending=[True, False]).to_string(
            index=False, float_format="%.3f"
        )
    )
    print(
        "\nAUC 0.5 is no separation. A crown metric that cannot beat 0.6 on any "
        "disturbance group is not carrying the stage signal it is assigned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
