"""csdv_core.label.judge: from the candidates of a window to routed labels.

Inputs: a unit, a window, its imagery stack and the candidates that reach
into it.
Outputs: verdict and polygon requests, the records of the answers (plain
dicts, one per call, written as JSON lines), and the list of
:class:`~csdv_core.label.labels.JudgedCandidate` that the label rules read.

Requests are built once and can be sent later, so the images can be rendered
where the imagery is and the calls made where the API keys are.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

from csdv_core.config._models import LabelConfig
from csdv_core.label.clients import Reply, ask_many
from csdv_core.label.inputs import UnitInputs
from csdv_core.label.labels import JudgedCandidate
from csdv_core.label.ledger import Ledger
from csdv_core.label.outlines import (
    choose_outline,
    needs_polygon,
    parse_polygon,
)
from csdv_core.label.requests import (
    VERDICT_PROMPT,
    Request,
    polygon_request,
    verdict_request,
)
from csdv_core.label.routing import Routed, route, unjudged
from csdv_core.label.stacks import WindowStack
from csdv_core.label.verdicts import Verdict, parse_verdict
from csdv_core.label.views import View

logger = logging.getLogger(__name__)

__all__ = [
    "append_records",
    "judged_candidates",
    "polygon_requests",
    "read_records",
    "record_key",
    "route_candidates",
    "send",
    "verdict_requests",
    "verdicts_of",
]


def verdict_requests(
    unit: UnitInputs,
    stack: WindowStack,
    candidates: gpd.GeoDataFrame,
    cfg: LabelConfig,
) -> list[Request]:
    """One verdict request for every candidate marked ``judged``."""
    return [
        verdict_request(stack, row.cand_id, row.geometry, unit.unit, cfg.views)
        for row in candidates[candidates["judged"]].itertuples()
    ]


def send(
    ledger: Ledger,
    requests: list[Request],
    model: str,
    *,
    window_id: str,
    stage: str,
    workers: int = 6,
) -> list[dict[str, Any]]:
    """Send requests to one model and return one record per request."""
    jobs = [(q.prompt, q.images, {"window_id": window_id, **q.tag}) for q in requests]
    replies: list[Reply] = ask_many(ledger, model, jobs, stage=stage, workers=workers)
    return [
        {
            "window_id": window_id,
            "cand_id": q.cand_id,
            "kind": q.kind,
            "model": model,
            "dates": q.dates,
            "view": [*q.view.bounds, q.view.px],
            "extra": q.extra,
            "answer": r.data,
            "cost": r.cost,
            "cached": r.cached,
            "error": r.error,
        }
        for q, r in zip(requests, replies, strict=True)
    ]


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Append answer records to a JSON-lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Answer records of a JSON-lines file. The last record of a call wins."""
    if not path.exists():
        return []
    latest: dict[tuple, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line:
            rec = json.loads(line)
            latest[record_key(rec)] = rec
    return list(latest.values())


def record_key(rec: dict[str, Any]) -> tuple:
    """What identifies a call: window, candidate, kind, model and its variant.

    The variant is the prompt version and, for a polygon, the pair of dates.
    Answers to an older prompt stay in the file and are never mistaken for
    answers to the current one.
    """
    variant = json.dumps(rec.get("extra") or {}, sort_keys=True)
    return (rec["window_id"], rec["cand_id"], rec["kind"], rec["model"], variant)


def verdicts_of(
    records: list[dict[str, Any]], window_id: str, prompt: str = VERDICT_PROMPT
) -> dict[str, dict[str, Verdict]]:
    """Verdicts of a window under one prompt version: candidate ID, then model."""
    out: dict[str, dict[str, Verdict]] = {}
    for rec in records:
        if rec["window_id"] != window_id or rec["kind"] != "verdict":
            continue
        if (rec.get("extra") or {}).get("prompt") != prompt or rec.get("error"):
            continue
        verdict = parse_verdict(
            rec["model"], rec["answer"], len(rec["dates"]), rec.get("cost", 0.0)
        )
        out.setdefault(rec["cand_id"], {})[rec["model"]] = verdict
    return out


def route_candidates(
    unit: UnitInputs,
    candidates: gpd.GeoDataFrame,
    verdicts: dict[str, dict[str, Verdict]],
    models: tuple[str, str],
    cfg: LabelConfig,
) -> dict[str, Routed]:
    """Routed outcome of every candidate of a window.

    A candidate that was not judged (too small, or no verdict from both
    models) is unlabelled in the interval that flagged it.
    """
    n = len(unit.years) - 1
    routed: dict[str, Routed] = {}
    for row in candidates.itertuples():
        flagged = unit.intervals.index(row.interval)
        pair = verdicts.get(row.cand_id, {})
        if row.judged and all(m in pair for m in models):
            routed[row.cand_id] = route(
                pair[models[0]], pair[models[1]], flagged, cfg.routing
            )
        else:
            small = row.area_ha < cfg.candidates.min_ha
            routed[row.cand_id] = unjudged(n, flagged, small=small)
    return routed


def polygon_requests(
    unit: UnitInputs,
    stack: WindowStack,
    candidates: gpd.GeoDataFrame,
    routed: dict[str, Routed],
    cfg: LabelConfig,
) -> list[Request]:
    """A polygon request for every accepted candidate whose outline is doubted."""
    requests = []
    for row in candidates.itertuples():
        result = routed[row.cand_id]
        if result.first_loss is None or not needs_polygon(result):
            continue
        requests.append(
            polygon_request(
                stack,
                row.cand_id,
                row.geometry,
                unit.unit,
                result.first_loss,
                result.first_loss + 1,
                cfg.views,
            )
        )
    return requests


def judged_candidates(
    unit: UnitInputs,
    candidates: gpd.GeoDataFrame,
    routed: dict[str, Routed],
    records: list[dict[str, Any]],
    window_id: str,
    cfg: LabelConfig,
) -> list[JudgedCandidate]:
    """Candidates with their outcome and, where there is loss, their outline."""
    polygons = {}
    for rec in records:
        if rec["window_id"] != window_id or rec["kind"] != "polygon":
            continue
        # A polygon belongs to the interval it was drawn for. One drawn for
        # another interval (under an older prompt) is not used.
        result = routed.get(rec["cand_id"])
        if result is None or (rec.get("extra") or {}).get("pre") != result.first_loss:
            continue
        *bounds, px = rec["view"]
        polygons[rec["cand_id"]] = parse_polygon(
            rec["answer"], View(tuple(bounds), int(px))
        )
    out = []
    for row in candidates.itertuples():
        result = routed[row.cand_id]
        event = None
        if result.has_loss:
            event = choose_outline(
                row.geometry,
                source=row.source,
                geom_src=row.geom_src,
                routed=result,
                polygon=polygons.get(row.cand_id),
                cfg=cfg.outlines,
            )
        out.append(
            JudgedCandidate(
                cand_id=row.cand_id,
                source=row.source,
                geom_src=row.geom_src,
                geometry=row.geometry,
                flagged=unit.intervals.index(row.interval),
                routed=result,
                event=event,
            )
        )
    return out
