"""csdv_core.label.sample: a probability sample of windows to label.

Inputs: the candidates and cell layers of every unit of a label run.
Outputs: a table of windows with the stratum, the seed candidate and its
inclusion probability.

Three strata. ``di`` windows are seeded by a detected object, ``other``
windows by a weak-probability or CHM-loss object, and ``quiet`` windows are
forest chips with no candidate and low disturbance probability throughout.
Every unit gets the same number of seeded windows, shared among its
sub-strata (last interval or not, probability class, size class) close to
proportionally. Each candidate has a random key fixed by the seed and its
ID, and the smallest keys are taken, so a larger sample contains the
smaller one.

The inclusion probability belongs to the seed. Other candidates in a window
were not drawn with a known probability, and windows can overlap, so rates
for a paper are estimated from seeds only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from csdv_core.disturbance.sampling import allocate_by_stratum
from csdv_core.label.inputs import fill_forest_dips
from csdv_core.label.lattice import GSD, Window, window_for
from csdv_core.label.pipeline import LabelRun

logger = logging.getLogger(__name__)

SIZE_EDGES_HA = (1.0, 5.0)
PROB_EDGE = 0.9

__all__ = ["SampleDesign", "draw_windows", "seed_window", "sub_stratum"]


@dataclass(frozen=True)
class SampleDesign:
    """Size of each stratum.

    Attributes:
        n_di: Windows seeded by a detected object.
        n_other: Windows seeded by a weak-probability or CHM-loss object.
        n_quiet: Forest windows without any candidate.
        seed: Seed of the candidate keys. Keep it for every batch of a sample.
        quiet_forest_share: Smallest forest share of a quiet chip.
        quiet_raised_share: Largest share of a quiet chip's cells with raised
            disturbance probability in any interval.
    """

    n_di: int = 650
    n_other: int = 150
    n_quiet: int = 150
    seed: int = 0
    quiet_forest_share: float = 0.8
    quiet_raised_share: float = 0.02


def _key(seed: int, name: str) -> float:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def sub_stratum(row: pd.Series, last_interval: str) -> str:
    """Sub-stratum of a candidate: last interval or not, probability, size."""
    when = "last" if row["interval"] == last_interval else "earlier"
    prob = "high" if row["mean_prob"] >= PROB_EDGE else "low"
    size = "small"
    if row["area_ha"] >= SIZE_EDGES_HA[1]:
        size = "large"
    elif row["area_ha"] >= SIZE_EDGES_HA[0]:
        size = "medium"
    return f"{when}-{prob}-{size}"


def seed_window(run: LabelRun, window_id: str, cand: pd.Series, seed: int) -> Window:
    """The window around a seed candidate.

    A candidate larger than the largest window is seen through a window
    placed on a random point inside it, fixed by the seed and its ID.
    """
    cfg = run.cfg.chips
    geom = cand["geometry"]
    minx, miny, maxx, maxy = geom.bounds
    side = cfg.chip_px * GSD * cfg.max_window_chips
    anchor = None
    if maxx - minx > side or maxy - miny > side:
        rng = np.random.default_rng(int(_key(seed, cand["cand_id"]) * 2**32))
        point = geom.representative_point()
        anchor = (point.x, point.y)
        for _ in range(200):
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if geom.contains(Point(x, y)):
                anchor = (x, y)
                break
    unit = run.unit(cand["unit"])
    return window_for(
        geom.bounds,
        window_id=window_id,
        unit=unit.unit,
        epsg=unit.epsg,
        margin_m=cfg.window_margin_m,
        max_chips=cfg.max_window_chips,
        chip_px=cfg.chip_px,
        anchor=anchor,
    )


def _seeded(
    run: LabelRun, unit_id: str, frame: gpd.GeoDataFrame, n: int, seed: int
) -> list[dict]:
    """Draw ``n`` seeds of one unit and stratum, shared among sub-strata."""
    if frame.empty or n <= 0:
        return []
    unit = run.unit(unit_id)
    frame = frame.copy()
    frame["sub"] = [sub_stratum(r, unit.intervals[-1]) for _, r in frame.iterrows()]
    frame["key"] = [_key(seed, c) for c in frame["cand_id"]]
    counts = frame["sub"].value_counts().to_dict()
    share = allocate_by_stratum(counts, n, min_per_stratum=1)
    rows = []
    for sub, take in share.items():
        chosen = frame[frame["sub"] == sub].nsmallest(take, "key")
        for _, cand in chosen.iterrows():
            rows.append(
                {
                    "unit": unit_id,
                    "seed_cand": cand["cand_id"],
                    "sub_stratum": sub,
                    "incl_prob": take / counts[sub],
                    "key": cand["key"],
                    "_cand": cand,
                }
            )
    return rows


def _quiet_chips(run: LabelRun, unit_id: str, design: SampleDesign) -> pd.DataFrame:
    """Every lattice chip of a unit that is quiet forest, with a random key."""
    unit = run.unit(unit_id)
    cfg = run.cfg
    side = cfg.chips.chip_px * GSD
    x0, y0, x1, y1 = unit.grid.bounds
    forest = fill_forest_dips(unit.layer("forest"))[0]
    with np.errstate(invalid="ignore"):
        raised = (
            np.nanmax(np.nan_to_num(unit.layer("probability")[1:], nan=0.0), axis=0)
            >= cfg.candidates.weak_prob
        )
    cands = run.candidates(unit_id)
    index = cands.sindex
    t = unit.grid.cell_transform
    rows = []
    for ix in range(int(np.ceil(x0 / side)), int(np.floor(x1 / side))):
        for iy in range(int(np.ceil(y0 / side)), int(np.floor(y1 / side))):
            wx0, wy0 = ix * side, iy * side
            c0 = int((wx0 - t.c) // t.a)
            c1 = int(np.ceil((wx0 + side - t.c) / t.a))
            r0 = int((t.f - (wy0 + side)) // -t.e)
            r1 = int(np.ceil((t.f - wy0) / -t.e))
            if r0 < 0 or c0 < 0 or r1 > forest.shape[0] or c1 > forest.shape[1]:
                continue
            if forest[r0:r1, c0:c1].mean() < design.quiet_forest_share:
                continue
            if raised[r0:r1, c0:c1].mean() > design.quiet_raised_share:
                continue
            pad = cfg.rules.margin_wide_m
            box_ = (wx0 - pad, wy0 - pad, wx0 + side + pad, wy0 + side + pad)
            if len(index.query(box(*box_), predicate="intersects")):
                continue
            rows.append(
                {
                    "ix0": ix,
                    "iy0": iy,
                    "key": _key(design.seed, f"{unit_id}:quiet:{ix}:{iy}"),
                }
            )
    return pd.DataFrame(rows, columns=["ix0", "iy0", "key"])


def _equal_allocation(counts: dict[str, int], n_total: int) -> dict[str, int]:
    """The same number per unit, as far as each unit has that many.

    What units with few candidates cannot take is handed to the others, one
    at a time, starting with those that have the least.
    """
    alloc = {name: 0 for name in counts}
    left = min(n_total, sum(counts.values()))
    while left > 0:
        room = sorted(
            (n for n in counts if alloc[n] < counts[n]), key=lambda n: alloc[n]
        )
        if not room:
            break
        for name in room:
            if left == 0:
                break
            alloc[name] += 1
            left -= 1
    return alloc


def draw_windows(run: LabelRun, units: list[str], design: SampleDesign) -> pd.DataFrame:
    """Draw the sample of windows over ``units``.

    Returns:
        One row per window: ``window_id``, ``unit``, ``epsg``, ``ix0``,
        ``iy0``, ``nx``, ``ny``, ``stratum``, ``sub_stratum``, ``seed_cand``
        and ``incl_prob``. Two seeds that give the same window keep one row,
        that of the smaller key.
    """
    min_ha = run.cfg.candidates.min_ha
    counts_di, counts_other = {}, {}
    for unit_id in units:
        cands = run.candidates(unit_id)
        big = cands[cands["area_ha"] >= min_ha]
        counts_di[unit_id] = int((big["source"] == "di").sum())
        counts_other[unit_id] = int((big["source"] != "di").sum())
    # Equal allocation among units for detected objects (units with fewer
    # objects give all of them). Other objects follow their number per unit,
    # which puts more windows where landscape-scale loss is.
    alloc_di = _equal_allocation(counts_di, design.n_di)
    alloc_other = allocate_by_stratum(counts_other, design.n_other, min_per_stratum=2)
    rows: list[dict] = []
    for unit_id in units:
        cands = run.candidates(unit_id)
        big = cands[cands["area_ha"] >= min_ha]
        for stratum, frame, n in (
            ("di", big[big["source"] == "di"], alloc_di[unit_id]),
            ("other", big[big["source"] != "di"], alloc_other[unit_id]),
        ):
            for row in _seeded(run, unit_id, frame, n, design.seed):
                window = seed_window(run, "", row.pop("_cand"), design.seed)
                rows.append(
                    {
                        **row,
                        "stratum": stratum,
                        "epsg": window.epsg,
                        "ix0": window.ix0,
                        "iy0": window.iy0,
                        "nx": window.nx,
                        "ny": window.ny,
                    }
                )
    n_quiet_unit = -(-design.n_quiet // max(len(units), 1)) if design.n_quiet else 0
    for unit_id in units:
        if not n_quiet_unit:
            break
        chips = _quiet_chips(run, unit_id, design)
        if chips.empty:
            logger.info("%s: no quiet forest chip", unit_id)
            continue
        take = chips.astype({"key": float}).nsmallest(n_quiet_unit, "key")
        for chip in take.itertuples():
            rows.append(
                {
                    "unit": unit_id,
                    "seed_cand": "",
                    "sub_stratum": "",
                    "incl_prob": len(take) / len(chips),
                    "key": chip.key,
                    "stratum": "quiet",
                    "epsg": run.unit(unit_id).epsg,
                    "ix0": int(chip.ix0),
                    "iy0": int(chip.iy0),
                    "nx": 1,
                    "ny": 1,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = {"di": 0, "other": 1, "quiet": 2}
    table["_rank"] = table["stratum"].map(order)
    table = table.sort_values(["_rank", "unit", "key"])
    table = table.drop_duplicates(["unit", "ix0", "iy0", "nx", "ny"], keep="first")
    quiet = table["stratum"] == "quiet"
    table = pd.concat([table[~quiet], table[quiet].head(design.n_quiet)])
    table = table.reset_index(drop=True)
    table.insert(0, "window_id", [f"W{i + 1:04d}" for i in range(len(table))])
    return table.drop(columns=["_rank", "key"])
