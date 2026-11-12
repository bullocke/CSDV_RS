"""csdv_core.label.export: the vector labels of a label run.

Inputs: the built windows of a run (candidates with routed verdicts and
outlines).
Outputs: three layers in EPSG:5070. ``events`` has one row per accepted
candidate with agreed loss. ``reviewed_areas`` has one row per window and
interval and says whether every candidate in it was judged. ``acquisitions``
lists the imagery of each window. The vector labels are the source of
truth: chips are derived from them and can be rebuilt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from csdv_core.label.pipeline import BuiltWindow, LabelRun
from csdv_core.label.requests import POLYGON_PROMPT, VERDICT_PROMPT
from csdv_core.label.routing import LOSS_OUTCOMES

logger = logging.getLogger(__name__)

STORE_CRS = "EPSG:5070"

__all__ = [
    "STORE_CRS",
    "acquisition_rows",
    "candidate_rows",
    "event_rows",
    "reviewed_rows",
    "write_layers",
]


def candidate_rows(built: BuiltWindow) -> list[dict[str, Any]]:
    """One row per candidate of a window, with what each model said."""
    rows = []
    unit = built.parts.unit
    table = built.parts.candidates.set_index("cand_id")
    for cand in built.judged:
        r = cand.routed
        src = table.loc[cand.cand_id]
        pair = built.verdicts.get(cand.cand_id, {})
        row = {
            "window_id": built.parts.window.window_id,
            "cand_id": cand.cand_id,
            "unit": unit.unit,
            "source": cand.source,
            "interval": src["interval"],
            "area_ha": float(src["area_ha"]),
            "visible_ha": float(src["visible_ha"]),
            "judged": bool(src["judged"]),
            "status": r.status,
            "outcomes": "|".join(r.outcomes),
            "severity": r.severity,
            "change_type": r.change_type,
            "agent": r.agent,
            "types": "/".join(str(t) for t in r.change_types),
            "outline_fit": "/".join(str(t) for t in r.outline_fit),
            "affected": "/".join(str(t) for t in r.affected_share),
            "confidence": r.confidence,
            "outline": cand.event.source if cand.event else None,
            "quality": cand.event.quality if cand.event else None,
            "utm_epsg": unit.epsg,
            "geometry": cand.geometry,
        }
        for model, verdict in pair.items():
            row[f"states_{model}"] = "|".join(verdict.states)
            row[f"reason_{model}"] = verdict.reason
        rows.append(row)
    return rows


def event_rows(built: BuiltWindow, run: LabelRun) -> list[dict[str, Any]]:
    """One row per candidate with agreed loss: the labelled disturbance."""
    rows = []
    unit = built.parts.unit
    dates = built.labels.intervals
    for cand in built.judged:
        r = cand.routed
        if r.first_loss is None or cand.event is None or cand.event.core.is_empty:
            continue
        event = cand.event
        geometry = event.core.union(event.extension) if event.crisp else event.core
        pre, post = dates[r.first_loss].split("_")
        rows.append(
            {
                "event_id": f"{built.parts.window.window_id}:{cand.cand_id}",
                "unit": unit.unit,
                "window_id": built.parts.window.window_id,
                "candidate_ids": cand.cand_id,
                "candidate_source": cand.source,
                "last_pre_acq": pre,
                "first_post_acq": post,
                "loss_intervals": ",".join(
                    dates[k] for k, o in enumerate(r.outcomes) if o in LOSS_OUTCOMES
                ),
                "outcomes": "|".join(r.outcomes),
                "severity": r.severity,
                "change_type": r.change_type,
                "agent": r.agent,
                "type_agree": r.change_type is not None,
                "interval_matches_di": cand.flagged == r.first_loss,
                "geometry_source": event.source,
                "geometry_quality": event.quality,
                "confidence": r.confidence,
                "label_source": (
                    f"{'+'.join(run.models)}; {VERDICT_PROMPT}; {POLYGON_PROMPT}"
                ),
                "review_status": "auto",
                "utm_epsg": unit.epsg,
                "geometry": geometry,
            }
        )
    return rows


def reviewed_rows(built: BuiltWindow) -> list[dict[str, Any]]:
    """One row per window and interval: was every candidate in it judged?"""
    window = built.parts.window
    judged = built.parts.candidates[built.parts.candidates["judged"]]
    status = {c.cand_id: c.routed.status for c in built.judged}
    incomplete = [
        c for c in judged["cand_id"] if status.get(c) in ("unjudged", "failed")
    ]
    sources = sorted(set(built.parts.candidates["source"])) or ["none"]
    return [
        {
            "window_id": window.window_id,
            "unit": window.unit,
            "interval": interval,
            "finding": "partial" if incomplete else "complete",
            "basis": ",".join(sources),
            "utm_epsg": window.epsg,
            "geometry": box(*window.bounds),
        }
        for interval in built.labels.intervals
    ]


def acquisition_rows(run: LabelRun, window_id: str, unit: str) -> list[dict[str, Any]]:
    """Imagery of a window, from the metadata written when it was rendered."""
    meta_path = run.root / "stacks" / window_id / "meta.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    return [
        {
            "unit": unit,
            "window_id": window_id,
            "year": int(year),
            "acquisition_date": info["date"],
            "naip_source": info["source"],
            "native_m": info.get("native_m"),
            "item_ids": ",".join(info.get("item_ids", [])),
        }
        for year, info in sorted(meta["years"].items(), key=lambda kv: int(kv[0]))
    ]


def _to_store(rows: list[dict[str, Any]]) -> gpd.GeoDataFrame | None:
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    parts = [
        gpd.GeoDataFrame(group, geometry="geometry", crs=int(epsg)).to_crs(STORE_CRS)
        for epsg, group in frame.groupby("utm_epsg")
    ]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=STORE_CRS)


def write_layers(
    path: Path,
    *,
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    acquisitions: list[dict[str, Any]],
) -> None:
    """Write the vector labels as layers of one GeoPackage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for name, rows in (
        ("candidates", candidates),
        ("events", events),
        ("reviewed_areas", reviewed),
    ):
        frame = _to_store(rows)
        if frame is not None:
            frame.to_file(path, layer=name)
    if acquisitions:
        table = gpd.GeoDataFrame(pd.DataFrame(acquisitions), geometry=None)
        table.to_file(path, layer="acquisitions")
