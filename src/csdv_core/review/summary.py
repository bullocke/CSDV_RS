"""csdv_core.review.summary: scores of the detections from the review labels.

Only the random sample (``stratum = "detected"``) is scored. Per unit, and
pooled over units, the table gives:

- ``precision``: true detections (confirmed or likely) over the labelled
  objects that are not uncertain. ``uncertain_share`` is reported beside it.
- ``area_precision``: the same, weighting each object by its assessed area,
  so it estimates the share of detected area that is real disturbance.
- ``timing_match``: among true detections with a known interval, the share
  detected in the right interval, and how many were detected late or early.
- ``outline_ok``: among true detections with an outline judgement, the share
  judged about right.
- ``iou`` and ``area_ratio``: median intersection over union, and detected
  over redrawn area, of the redrawn outlines.

The pooled row weights each object by the inverse of its inclusion probability
(``1 / incl_prob``), so it estimates the scores over all detected objects of
the run tag rather than giving every unit the same weight. ``site_mean``
gives the unweighted mean of the per-unit precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from csdv_core.review.sample import STRATUM_DETECTED
from csdv_core.review.store import TRUE_VERDICTS, UNSURE, ReviewStore

logger = logging.getLogger(__name__)

__all__ = ["ReviewSummary", "labelled_sample", "summarize", "timing_of"]


def timing_of(detected: str, reference: str | None) -> str:
    """``match``, ``detected_late``, ``detected_early`` or ``unsure``."""
    if not reference or reference == UNSURE:
        return "unsure"
    try:
        det = int(str(detected).split("-")[1])
        ref = int(str(reference).split("-")[1])
    except (IndexError, ValueError):
        return "unsure"
    if det == ref:
        return "match"
    return "detected_late" if det > ref else "detected_early"


def labelled_sample(store: ReviewStore) -> gpd.GeoDataFrame:
    """Labelled rows of the random sample with derived columns.

    Adds ``is_true``, ``timing``, ``weight`` (``1 / incl_prob``), and for
    redrawn outlines ``iou`` and ``area_ratio`` (computed in each unit's UTM
    CRS).
    """
    s = store.sample[store.sample["stratum"] == STRATUM_DETECTED]
    df = s.merge(store.labels, on="sample_id", how="inner")
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=store.sample.crs)
    df["is_true"] = df["verdict"].isin(TRUE_VERDICTS)
    df["timing"] = [
        timing_of(d, r) if t else ""
        for d, r, t in zip(
            df["interval"], df["ref_interval"], df["is_true"], strict=True
        )
    ]
    df["weight"] = 1.0 / df["incl_prob"].astype(float)
    df["iou"] = np.nan
    df["area_ratio"] = np.nan
    redrawn = store.redrawn.set_index("sample_id").geometry
    for i, row in df[df["sample_id"].isin(redrawn.index)].iterrows():
        crs = f"EPSG:{int(row['utm_epsg'])}"
        det = gpd.GeoSeries([row.geometry], crs=df.crs).to_crs(crs).iloc[0]
        ref = (
            gpd.GeoSeries([redrawn.loc[row["sample_id"]]], crs=df.crs)
            .to_crs(crs)
            .iloc[0]
        )
        union = det.union(ref).area
        df.loc[i, "iou"] = det.intersection(ref).area / union if union else np.nan
        df.loc[i, "area_ratio"] = det.area / ref.area if ref.area else np.nan
    return df


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna()
    if not ok.any() or weights[ok].sum() == 0:
        return np.nan
    return float(np.average(values[ok].astype(float), weights=weights[ok]))


def _scores(df: pd.DataFrame, weighted: bool) -> dict:
    w = df["weight"] if weighted else pd.Series(1.0, index=df.index)
    decided = df["verdict"] != "uncertain"
    true = df[df["is_true"]]
    timed = true[true["timing"] != "unsure"]
    judged = true[
        true["outline"]
        .fillna("")
        .isin(["about_right", "too_large", "too_small", "wrong_shape"])
    ]
    out = {
        "n": len(df),
        "n_true": int(df["is_true"].sum()),
        "n_rejected": int((df["verdict"] == "rejected").sum()),
        "n_uncertain": int((df["verdict"] == "uncertain").sum()),
        "uncertain_share": _wmean((df["verdict"] == "uncertain").astype(float), w),
        "precision": _wmean(df.loc[decided, "is_true"].astype(float), w[decided]),
        "area_precision": _wmean(
            df.loc[decided, "is_true"].astype(float),
            (w * df["assessed_ha"])[decided],
        ),
        "timing_match": _wmean(
            (timed["timing"] == "match").astype(float), w.loc[timed.index]
        ),
        "n_late": int((true["timing"] == "detected_late").sum()),
        "n_early": int((true["timing"] == "detected_early").sum()),
        "outline_ok": _wmean(
            (judged["outline"] == "about_right").astype(float), w.loc[judged.index]
        ),
        "n_redrawn": int(df["iou"].notna().sum()),
        "iou": float(df["iou"].median()) if df["iou"].notna().any() else np.nan,
        "area_ratio": (
            float(df["area_ratio"].median())
            if df["area_ratio"].notna().any()
            else np.nan
        ),
        "seconds_median": float(pd.to_numeric(df["seconds_spent"]).median()),
    }
    return out


@dataclass
class ReviewSummary:
    """Scores per unit and pooled, and the change types."""

    table: pd.DataFrame
    change_types: pd.DataFrame
    labelled: gpd.GeoDataFrame

    def to_markdown(self) -> str:
        cols = [
            "unit",
            "site",
            "n",
            "precision",
            "uncertain_share",
            "area_precision",
            "timing_match",
            "n_late",
            "n_early",
            "outline_ok",
            "n_redrawn",
            "iou",
            "area_ratio",
        ]
        t = self.table[cols].copy()
        lines = ["# Review summary", ""]
        lines.append(
            "Precision excludes uncertain labels. The pooled row weights objects by "
            "1 / inclusion probability, and site_mean is the unweighted mean over "
            "units. Timing and outline scores are among true detections."
        )
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in t.itertuples(index=False):
            cells = []
            for v in row:
                if isinstance(v, float):
                    cells.append("" if np.isnan(v) else f"{v:.2f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "## Change types", ""]
        ct = self.change_types
        lines.append("| verdict | change_type | n |")
        lines.append("|---|---|---|")
        for row in ct.itertuples(index=False):
            lines.append(f"| {row.verdict} | {row.change_type} | {row.n} |")
        return "\n".join(lines) + "\n"

    def write(self, folder: Path, stem: str) -> list[Path]:
        folder.mkdir(parents=True, exist_ok=True)
        csv = folder / f"{stem}_summary.csv"
        md = folder / f"{stem}_summary.md"
        self.table.to_csv(csv, index=False)
        md.write_text(self.to_markdown(), encoding="utf-8")
        return [csv, md]


def summarize(store: ReviewStore) -> ReviewSummary:
    """Score the labelled random sample per unit and pooled."""
    df = labelled_sample(store)
    if df.empty:
        raise ValueError("No labelled sample polygons yet")
    rows = []
    for (unit, site), part in df.groupby(["unit", "site_label"]):
        rows.append({"unit": unit, "site": site, **_scores(part, weighted=False)})
    per_unit = pd.DataFrame(rows)
    pooled = {"unit": "pooled", "site": "weighted", **_scores(df, weighted=True)}
    site_mean = {
        "unit": "site_mean",
        "site": "unweighted",
        **{
            k: (
                per_unit[k].mean()
                if per_unit[k].dtype.kind == "f"
                else per_unit[k].sum()
            )
            for k in per_unit.columns
            if k not in ("unit", "site")
        },
    }
    table = pd.concat([per_unit, pd.DataFrame([pooled, site_mean])], ignore_index=True)
    change_types = (
        df.assign(change_type=df["change_type"].fillna("").replace("", "(none)"))
        .groupby(["verdict", "change_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["verdict", "n"], ascending=[True, False])
    )
    return ReviewSummary(table=table, change_types=change_types, labelled=df)
