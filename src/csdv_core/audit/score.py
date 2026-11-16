"""csdv_core.audit.score: rates and agreement from the answers of an audit.

Inputs: ``items.csv`` (the key) and the answers files of one or more
interpreters. Outputs: one row per judged item (:func:`judged_table`), rates
per interpreter and stratum with exact binomial intervals and a rate weighted
by inclusion probability (:func:`rates`), and the agreement between
interpreters on the items they share (:func:`agreement`).

The weight of an item is the inverse of its inclusion probability: the
probability that its window entered the label sample, times the share of its
sub-stratum that the interpreter judged. Items are served in random order,
so the judged items of a sub-stratum are a random subset of it at any point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from csdv_core.audit.server import latest_answers

logger = logging.getLogger(__name__)

__all__ = ["agreement", "exact_interval", "judged_table", "load_answers", "rates"]


def load_answers(paths: list[Path]) -> pd.DataFrame:
    """The answer in force of every item and interpreter.

    Args:
        paths: Answers files (JSON lines), from the server or downloaded
            from the page. A file may hold several interpreters.

    Returns:
        One row per interpreter and item, with the fields of the answer.
    """
    by_interpreter: dict[str, list[dict]] = {}
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                name = record.get("interpreter") or Path(path).stem
                by_interpreter.setdefault(name, []).append(record)
    rows = []
    for name, records in by_interpreter.items():
        records.sort(key=lambda r: r.get("saved", ""))
        for record in latest_answers(records).values():
            rows.append({**record, "interpreter": name})
    return pd.DataFrame(rows)


def exact_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval of a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    tail = (1.0 - level) / 2.0
    low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return (low, high)


def judged_table(items: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Every answer beside the key of its item.

    Adds ``is_loss`` (the interpreter saw loss, or marked a missed loss in a
    window), ``unsure``, ``same_interval`` (a named loss date is one of the
    labelled ones), ``same_type`` (the named cause is the label's agent) and
    ``weight``.
    """
    if answers.empty:
        return pd.DataFrame()
    table = answers.merge(items, on="item", how="inner", suffixes=("", "_item"))
    table["is_loss"] = table["answer"] == "loss"
    table["unsure"] = table["answer"] == "unsure"
    same_interval, same_type = [], []
    for row in table.itertuples():
        labelled = {int(n) for n in str(row.label_images).split(",") if n}
        named = set()
        if getattr(row, "loss_at", None) is not None and not pd.isna(row.loss_at):
            named.add(int(row.loss_at))
        more = getattr(row, "more_at", None)
        if isinstance(more, (list, tuple)):
            named |= {int(n) for n in more}
        same_interval.append(bool(labelled & named) if labelled else np.nan)
        agent = row.agent if isinstance(row.agent, str) else ""
        answered = getattr(row, "type", None)
        known = bool(agent) and isinstance(answered, str) and bool(answered)
        same_type.append(answered == agent if known else np.nan)
    table["same_interval"] = same_interval
    table["same_type"] = same_type
    judged = table.groupby(["interpreter", "sub"])["item"].transform("count")
    table["weight"] = 1.0 / (table["window_incl_prob"] * judged / table["n_population"])
    return table


def _rate(part: pd.DataFrame, flag: pd.Series, measure: str) -> dict:
    """One rate over the rows of ``part`` where ``flag`` is not missing."""
    known = flag.notna()
    values = flag[known].astype(bool)
    weights = part.loc[known, "weight"]
    n, k = int(known.sum()), int(values.sum())
    low, high = exact_interval(k, n)
    weighted = float((weights * values).sum() / weights.sum()) if n else float("nan")
    return {
        "measure": measure,
        "n": n,
        "k": k,
        "rate": k / n if n else float("nan"),
        "low": low,
        "high": high,
        "weighted_rate": weighted,
    }


def rates(judged: pd.DataFrame) -> pd.DataFrame:
    """Rates per interpreter and audit stratum.

    Items answered "cannot tell" are left out of a rate and counted in
    ``unsure``. For accepted events the measures are the share with loss, the
    share whose loss lies in the labelled interval, the share whose cause is
    the label's agent (among those with loss) and the share whose label
    extent holds. For rejected candidates and disagreements it is the share
    with loss. For windows it is the share with loss on ground labelled
    background. For model-only patches (block 3) it is the share with loss,
    the share whose loss lies in the interval the model flagged, and the share
    whose cause is the model's agent: ground where the model sees loss and the
    candidate set had nothing to judge.
    """
    rows = []
    for (name, stratum), part in judged.groupby(["interpreter", "stratum"]):
        sure = part[~part["unsure"]]
        loss = sure[sure["is_loss"]]
        found = [_rate(sure, sure["is_loss"], "loss seen")]
        if stratum == "accepted":
            found.append(
                _rate(sure, sure["same_interval"], "loss in labelled interval")
            )
            found.append(_rate(loss, loss["same_type"], "cause equals label agent"))
            if "outline" in loss:
                holds = loss["outline"].map(
                    lambda v: v == "good" if isinstance(v, str) else np.nan
                )
                found.append(_rate(loss, holds, "label extent holds"))
        elif stratum == "model_only":
            found.append(_rate(sure, sure["same_interval"], "loss in model interval"))
            found.append(_rate(loss, loss["same_type"], "cause equals model agent"))
        for row in found:
            rows.append(
                {
                    "interpreter": name,
                    "stratum": stratum,
                    **row,
                    "unsure": int(part["unsure"].sum()),
                    "median_seconds": float(part["seconds"].median()),
                }
            )
    return pd.DataFrame(rows)


def agreement(judged: pd.DataFrame) -> pd.DataFrame:
    """Answers of every interpreter on the items judged by more than one.

    Returns:
        One row per such item: its stratum, one column per interpreter with
        the answer (``none``, ``loss`` or ``unsure``), ``n`` interpreters and
        ``all_agree``. Items on which all agree form a reference set that is
        independent of the detection and of the vision models.
    """
    if judged.empty:
        return pd.DataFrame()
    wide = judged.pivot_table(
        index=["item", "stratum"],
        columns="interpreter",
        values="answer",
        aggfunc="first",
    )
    names = list(wide.columns)
    wide["n"] = wide[names].notna().sum(axis=1)
    wide = wide[wide["n"] > 1].copy()
    wide["all_agree"] = [
        len(set(row.dropna())) == 1 for _, row in wide[names].iterrows()
    ]
    return wide.reset_index()
