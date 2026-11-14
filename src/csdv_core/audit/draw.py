"""csdv_core.audit.draw: the seeded sample of a blind audit.

Inputs: a label run folder (``windows.csv``, ``labels/labels.gpkg``,
``requests/`` and ``stacks/``). Output: a table of audit items, written once
as ``items.csv`` and never drawn again.

Two blocks are drawn. Block 1 holds judged candidates of three audit strata
(accepted events, rejected candidates, disagreements), shuffled so that the
interpreter cannot tell them apart. Block 2 holds complete windows, whose
background is checked for missed loss. Every unit of the population gets a
key from a hash of the seed and its name. A stratum takes its smallest keys,
so a larger sample of the same seed contains the smaller one.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from csdv_core.disturbance.sampling import allocate_by_stratum
from csdv_core.label.routing import LOSS_OUTCOMES

log = logging.getLogger(__name__)

#: Change types whose loss is spread through a stand, not a clean opening.
DIFFUSE_TYPES = frozenset(
    {"partial_harvest", "thinning", "wind", "insect_disease", "gap_mortality"}
)
#: Routed statuses of each audit stratum of block 1.
STRATUM_STATUSES = {
    "accepted": ("accepted",),
    "rejected": ("rejected", "rejected_weak"),
    "disagree": ("disagree", "date_disputed"),
}
SEEDED_STRATA = ("di", "other", "quiet")
ITEM_COLUMNS = [
    "item",
    "block",
    "order",
    "kind",
    "stratum",
    "sub",
    "window_id",
    "cand_id",
    "unit",
    "status",
    "change_type",
    "agent",
    "label_images",
    "window_incl_prob",
    "n_population",
    "rank",
    "reserve",
    "shared",
]

__all__ = [
    "DIFFUSE_TYPES",
    "ITEM_COLUMNS",
    "AuditDesign",
    "draw_items",
    "extend_items",
    "label_images",
    "read_items",
]


@dataclass(frozen=True)
class AuditDesign:
    """Size of each audit stratum.

    Attributes:
        n_accepted: Accepted events.
        diffuse_share: Share of the accepted events taken from diffuse types.
            Each half is shared among the agent groups in proportion.
        min_per_agent: Fewest accepted events of an agent group in each half,
            so that every agent is seen by the interpreter.
        n_rejected: Candidates both models rejected.
        n_disagree: Candidates the models disagreed about (loss or its date).
        n_windows: Complete windows whose background is checked.
        reserve: Further items per stratum, as a share of its size, placed
            after the target items so the sample can grow without a redraw.
        n_shared: Accepted events, and windows, that every interpreter judges.
        seed: Seed of the keys. Keep it for every extension.
    """

    n_accepted: int = 60
    diffuse_share: float = 0.5
    min_per_agent: int = 3
    n_rejected: int = 40
    n_disagree: int = 20
    n_windows: int = 60
    reserve: float = 0.5
    n_shared: int = 30
    seed: int = 0


def _key(seed: int, name: str) -> float:
    digest = hashlib.sha256(f"{seed}:audit:{name}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def label_images(outcomes: str) -> str:
    """Image numbers (1-based, later date) of the intervals labelled loss.

    Interval *k* (0-based) lies between image *k + 1* and image *k + 2*.
    """
    numbers = [
        str(k + 2) for k, o in enumerate(str(outcomes).split("|")) if o in LOSS_OUTCOMES
    ]
    return ",".join(numbers)


def _windows(label_dir: Path, exclude: set[str]) -> pd.DataFrame:
    """Seeded windows that may enter the audit."""
    windows = pd.read_csv(label_dir / "windows.csv")
    keep = (
        windows["stratum"].isin(SEEDED_STRATA)
        & windows["incl_prob"].notna()
        & ~windows["window_id"].isin(exclude)
    )
    return windows[keep].reset_index(drop=True)


def _candidate_population(
    label_dir: Path, windows: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Judged candidates of the eligible windows, one row per candidate.

    A candidate seen through several windows was judged once per window. One
    of those views is kept at random, so no ground is shown twice.
    """
    cands = pyogrio.read_dataframe(
        label_dir / "labels" / "labels.gpkg", layer="candidates", read_geometry=False
    )
    cands = cands[cands["judged"].astype(bool)]
    cands = cands.merge(
        windows[["window_id", "incl_prob"]], on="window_id", how="inner"
    )
    rendered = [
        (label_dir / "requests" / w / "requests.jsonl").exists()
        for w in cands["window_id"]
    ]
    cands = cands[rendered].copy()
    cands["view_key"] = [
        _key(seed, f"{w}:{c}")
        for w, c in zip(cands["window_id"], cands["cand_id"], strict=True)
    ]
    cands = cands.sort_values("view_key").drop_duplicates("cand_id")
    cands["key"] = [_key(seed, c) for c in cands["cand_id"]]
    return cands.reset_index(drop=True)


def _sub(stratum: str, row: pd.Series) -> str:
    """Sub-stratum of a candidate: its status, or for an accepted event
    whether its type is diffuse and its agent group."""
    if stratum == "accepted":
        spread = "diffuse" if row["change_type"] in DIFFUSE_TYPES else "other"
        agent = row["agent"] if isinstance(row["agent"], str) and row["agent"] else ""
        return f"accepted_{spread}_{agent or 'mixed'}"
    return str(row["status"])


def _take(frame: pd.DataFrame, shares: dict[str, int], reserve: float) -> pd.DataFrame:
    """The smallest keys of each sub-stratum, target items then reserve."""
    parts = []
    for sub, n in shares.items():
        pool = frame[frame["sub"] == sub].sort_values("key")
        total = min(len(pool), n + int(np.ceil(n * reserve)))
        chosen = pool.head(total).copy()
        chosen["n_population"] = len(pool)
        chosen["rank"] = range(1, total + 1)
        chosen["reserve"] = chosen["rank"] > n
        parts.append(chosen)
    return pd.concat(parts) if parts else frame.head(0)


def _candidate_items(pop: pd.DataFrame, design: AuditDesign) -> pd.DataFrame:
    parts = []
    for stratum, statuses in STRATUM_STATUSES.items():
        frame = pop[pop["status"].isin(statuses)].copy()
        if frame.empty:
            continue
        frame["stratum"] = stratum
        frame["sub"] = [_sub(stratum, row) for _, row in frame.iterrows()]
        counts = frame["sub"].value_counts().to_dict()
        if stratum == "accepted":
            n_diffuse = int(round(design.n_accepted * design.diffuse_share))
            shares = {}
            for spread, n in (
                ("accepted_diffuse_", n_diffuse),
                ("accepted_other_", design.n_accepted - n_diffuse),
            ):
                part = {s: c for s, c in counts.items() if s.startswith(spread)}
                shares.update(
                    allocate_by_stratum(part, n, min_per_stratum=design.min_per_agent)
                )
        else:
            n = design.n_rejected if stratum == "rejected" else design.n_disagree
            shares = allocate_by_stratum(counts, n, min_per_stratum=1)
        parts.append(_take(frame, shares, design.reserve))
    items = pd.concat(parts)
    items["kind"] = "cand"
    items["block"] = 1
    items["label_images"] = [
        label_images(o) if s == "accepted" else ""
        for o, s in zip(items["outcomes"], items["status"], strict=True)
    ]
    return items.rename(columns={"incl_prob": "window_incl_prob"})


def _window_items(
    label_dir: Path, windows: pd.DataFrame, design: AuditDesign
) -> pd.DataFrame:
    """Complete, rendered windows, shared among the sample strata."""
    areas = pyogrio.read_dataframe(
        label_dir / "labels" / "labels.gpkg",
        layer="reviewed_areas",
        read_geometry=False,
    )
    complete = areas.groupby("window_id")["finding"].agg(
        lambda s: bool((s == "complete").all())
    )
    frame = windows[windows["window_id"].map(complete).fillna(False).astype(bool)]
    frame = frame.copy()
    frame["sub"] = frame["stratum"]
    frame["key"] = [_key(design.seed, w) for w in frame["window_id"]]
    counts = frame["sub"].value_counts().to_dict()
    shares = allocate_by_stratum(counts, design.n_windows, min_per_stratum=1)
    items = _take(frame, shares, design.reserve)
    items["stratum"] = "background"
    items["kind"] = "window"
    items["block"] = 2
    for column in ("cand_id", "status", "change_type", "agent", "label_images"):
        items[column] = ""
    return items.rename(columns={"incl_prob": "window_incl_prob"})


def _ordered(items: pd.DataFrame, design: AuditDesign) -> pd.DataFrame:
    """Shuffle each block, target items first, and name the items."""
    name = items["cand_id"].where(items["kind"] == "cand", items["window_id"])
    items = items.assign(
        order_key=[_key(design.seed, f"order:{n}") for n in name]
    ).sort_values(["block", "reserve", "order_key"])
    out = []
    for block, part in items.groupby("block", sort=True):
        part = part.copy()
        part["order"] = range(1, len(part) + 1)
        letter = "C" if block == 1 else "B"
        part["item"] = [f"{letter}{i:03d}" for i in part["order"]]
        pool = part["stratum"].isin(["accepted", "background"]) & ~part["reserve"]
        part["shared"] = False
        part.loc[part[pool].index[: design.n_shared], "shared"] = True
        out.append(part)
    result = pd.concat(out).reset_index(drop=True)
    for column in ("change_type", "agent"):
        result[column] = result[column].fillna("")
    return result[ITEM_COLUMNS]


def draw_items(
    label_dir: Path,
    design: AuditDesign | None = None,
    exclude_windows: set[str] | None = None,
) -> pd.DataFrame:
    """Draw the audit items of a label run.

    Args:
        label_dir: Folder of the label run.
        design: Sizes and seed. None: the default design.
        exclude_windows: Windows the interpreter has already seen with the
            model answers (for example on check-in sheets).

    Returns:
        One row per item with the columns of :data:`ITEM_COLUMNS`. The table
        is the key of the audit and is kept away from the page.
    """
    label_dir = Path(label_dir)
    design = design or AuditDesign()
    windows = _windows(label_dir, exclude_windows or set())
    pop = _candidate_population(label_dir, windows, design.seed)
    items = pd.concat(
        [
            _candidate_items(pop, design),
            _window_items(label_dir, windows, design),
        ]
    )
    result = _ordered(items, design)
    counts = result.groupby(["stratum", "reserve"]).size().to_dict()
    log.info("Drew %d audit items: %s", len(result), counts)
    return result


def extend_items(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append the items of a larger draw to an existing table.

    Existing items keep their names, order and flags. Items that are new are
    placed after them, in the order of the larger draw.
    """

    def names(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
        cand = frame["cand_id"].fillna("")
        return list(zip(frame["kind"], frame["window_id"], cand, strict=True))

    seen = set(names(old))
    fresh = new[[name not in seen for name in names(new)]].copy()
    parts = [old]
    for block, part in fresh.groupby("block", sort=True):
        part = part.copy()
        start = int(old.loc[old["block"] == block, "order"].max() or 0)
        part["order"] = range(start + 1, start + len(part) + 1)
        letter = "C" if block == 1 else "B"
        part["item"] = [f"{letter}{i:03d}" for i in part["order"]]
        part["shared"] = False
        parts.append(part)
    return pd.concat(parts).reset_index(drop=True)[ITEM_COLUMNS]


def read_items(path: Path) -> pd.DataFrame:
    """Read ``items.csv`` with text columns kept as text."""
    text = ["cand_id", "status", "change_type", "agent", "label_images"]
    items = pd.read_csv(path, dtype={c: str for c in text}, keep_default_na=False)
    for column in ("reserve", "shared"):
        items[column] = items[column].astype(str).str.lower().eq("true")
    return items
