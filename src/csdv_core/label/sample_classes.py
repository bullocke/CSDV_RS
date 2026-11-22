"""csdv_core.label.sample_classes: a sample that favours disagreement.

Inputs: the candidates of every unit of a label run, with ``model_share``
where the unit has a map of the pair model, and a table of the run's sites.
Outputs: a table of windows drawn larger than any budget, in a fixed order
of priority within each stratum, and the list of candidates to judge in each
window.

The first sampler (:mod:`csdv_core.label.sample`) spreads windows over what
the detection found. This one spreads them over how the pair model and the
detection relate, because a paid verdict teaches the model most where the
two disagree:

``model_only``
    A ``model`` object: the pair model flags it and no candidate touches it.
    Loss the canopy height missed, or a false alarm of the model.
``cand_only``
    A detected, weak or CHM-loss object of which the model flags under a
    tenth. Loss the model misses, or a false alarm of the detection.
``agree``
    The model flags half or more of the candidate. Mostly loss the model
    already finds, kept as clean positives of new types and places.
``partly``
    In between.
``plain``
    Units without a map, and units set aside for testing, where the model
    must have no say in what is judged (a test set chosen by one model's
    errors flatters every other model). Their classes are the sources.

Within a unit each class is split by whether the candidate lies in the
unit's priority interval, the pair of dates around the event the place was
chosen for. Every candidate has a random key fixed by the seed and its ID,
and a stratum's candidates are taken in the order of their keys. The draw
takes the first ``draw_per_stratum`` of each stratum, so a later, larger
allocation contains the earlier one and no window is ever drawn again.
``allocate`` then says how many of each stratum are judged in a stage.

A window is judged in part: its seed and, at most, a few more candidates of
the seed's interval, each candidate in one window only (``judged.csv``).
Windows may overlap or coincide. The labels of a window then use every
verdict on a candidate it holds, whichever window asked for it. The
inclusion probability belongs to the seed and is that of the stratum:
windows taken so far over candidates in the stratum.

A later stage can also steer by season. A plan row that names a season
(``nov-apr`` or ``may-oct``, by the month of the later image of the seed's
interval) splits its stratum in two. The windows drawn in a stratum are its
first candidates in key order, so those of one season are the first of that
season, and taking them in the order of rank is still a random sample of the
season's candidates. The inclusion probability of a split stratum is then
windows taken in the season over candidates of the season (``season_n``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import box

from csdv_core.label.pipeline import LabelRun
from csdv_core.label.sample import (
    SampleDesign,
    _key,
    _quiet_allocation,
    _quiet_chips,
    seed_window,
)

logger = logging.getLogger(__name__)

#: Order in which the classes of a unit give their first windows to stage 1.
CLASS_ORDER = ("model_only", "cand_only", "agree", "partly", "di", "chm", "weak")
SILENT_BELOW = 0.1
FLAGS_FROM = 0.5
SEASON_MONTHS = range(5, 10)
#: Seasons a plan can name, by the month of the later image of an interval.
OFF_SEASON = "nov-apr"
ON_SEASON = "may-oct"

__all__ = [
    "ClassDesign",
    "allocate_stage",
    "candidate_class",
    "class_counts",
    "draw_class_windows",
    "image_season",
    "offseason_intervals",
    "seed_seasons",
]


@dataclass(frozen=True)
class ClassDesign:
    """How the windows of a sample by class are drawn.

    Attributes:
        draw_per_stratum: Seeds drawn in each stratum (unit, class, priority
            interval or not). More than will be judged: the rest waits for a
            later allocation.
        stage1_per_unit: Windows per unit that form stage 1, one per stratum
            in the order of :data:`CLASS_ORDER`, priority intervals first.
        judged_per_window: Candidates judged in a window, the seed included.
        edge_m: A seed must lie mostly this far inside its unit's grid,
            which is the extent of the mosaic (the box and the 150 m the
            download adds around it). Canopy heights within 260 m of a
            mosaic's edge depend on the mosaic: in the round-2 pilot a box
            and its quad differed by 0.3 to 0.9 m RMSE at 150 to 260 m and
            not at all beyond.
        plain_units: Units sampled by source, without the model's map.
        priority: Per unit, the interval the place was chosen for.
        seed: Seed of the candidate keys.
        quiet: Number and rule of the quiet windows, as in the first sampler.
            Only ``n_quiet``, the quiet shares, ``quiet_max_per_unit`` and
            ``quiet_intervals`` are read.
    """

    draw_per_stratum: int = 8
    stage1_per_unit: int = 4
    judged_per_window: int = 2
    edge_m: float = 260.0
    plain_units: frozenset[str] = frozenset()
    priority: Mapping[str, str] = field(default_factory=dict)
    seed: int = 0
    quiet: SampleDesign = field(
        default_factory=lambda: SampleDesign(n_di=0, n_other=0, n_quiet=0)
    )


def candidate_class(source: str, model_share: float, plain: bool) -> str | None:
    """Class of a candidate, or None when it is not a seed in its unit.

    A unit sampled without the model (``plain``) has no ``model`` seeds and
    uses the source as the class. So does a candidate without a
    ``model_share``, which is one of a unit that has no map.
    """
    if source == "model":
        return None if plain else "model_only"
    if plain or model_share is None or np.isnan(model_share):
        return source
    if model_share < SILENT_BELOW:
        return "cand_only"
    if model_share >= FLAGS_FROM:
        return "agree"
    return "partly"


def image_season(date: str) -> str:
    """``nov-apr`` or ``may-oct``, by the month of an image date."""
    month = int(str(date)[5:7])
    return ON_SEASON if 5 <= month <= 10 else OFF_SEASON


def offseason_intervals(run: LabelRun, units: Sequence[str]) -> dict[str, tuple]:
    """Per unit, the intervals a quiet chip is judged in.

    These are the intervals with a date outside May to September, where the
    unit has any: stable forest on such images is what a model trained on
    summer pairs lacks. A unit flown in summer only gets all its intervals,
    so that a chip needs to be quiet in one of them and not in every one,
    which few chips are where forest is harvested on short rotations.
    """
    out = {}
    for unit_id in units:
        unit = run.unit(unit_id)
        months = [int(str(unit.dates[y])[5:7]) for y in unit.years]
        off = [
            name
            for k, name in enumerate(unit.intervals)
            if months[k] not in SEASON_MONTHS or months[k + 1] not in SEASON_MONTHS
        ]
        out[unit_id] = tuple(off or unit.intervals)
    return out


def _inner_box(run: LabelRun, unit_id: str, edge_m: float):
    x0, y0, x1, y1 = run.unit(unit_id).grid.bounds
    return box(x0 + edge_m, y0 + edge_m, x1 - edge_m, y1 - edge_m)


def _unit_seeds(run: LabelRun, unit_id: str, design: ClassDesign) -> pd.DataFrame:
    """Every candidate of a unit that can seed a window, with its stratum."""
    cands = run.candidates(unit_id)
    cands = cands[cands["area_ha"] >= run.cfg.candidates.min_ha].copy()
    if cands.empty:
        return cands
    if "model_share" not in cands:
        cands["model_share"] = np.nan
    plain = unit_id in design.plain_units
    cands["class"] = [
        candidate_class(s, m, plain)
        for s, m in zip(cands["source"], cands["model_share"], strict=True)
    ]
    cands = cands[cands["class"].notna()]
    inner = _inner_box(run, unit_id, design.edge_m)
    inside = cands.geometry.intersection(inner).area / cands.geometry.area
    cands = cands[inside >= 0.5].copy()
    wanted = design.priority.get(unit_id)
    cands["when"] = np.where(cands["interval"] == wanted, "priority", "other")
    cands["sub"] = cands["class"] + "-" + cands["when"]
    unit = run.unit(unit_id)
    later = {
        name: unit.dates[unit.years[k + 1]] for k, name in enumerate(unit.intervals)
    }
    cands["season"] = [image_season(later[i]) for i in cands["interval"]]
    cands["key"] = [_key(design.seed, c) for c in cands["cand_id"]]
    return cands


def _stratum_order(sub: str) -> tuple[int, int]:
    name, when = sub.rsplit("-", 1)
    return (0 if when == "priority" else 1, CLASS_ORDER.index(name))


def draw_class_windows(
    run: LabelRun, units: list[str], design: ClassDesign
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the windows of every unit and the candidates to judge in each.

    Returns:
        ``(windows, judged)``. ``windows`` has one row per window with the
        columns of the first sampler plus ``stratum_n`` (candidates in the
        seed's stratum) and ``rank`` (order of the seed within it). ``stage``
        is 1 for the first windows of a unit and for quiet windows, and 0
        for a window that waits for an allocation. ``judged`` lists the
        candidates to judge: ``window_id``, ``cand_id``, ``role`` (``seed``
        or ``extra``).
    """
    rows: list[dict] = []
    min_ha = run.cfg.candidates.min_ha
    unseen = 0
    for unit_id in units:
        seeds = _unit_seeds(run, unit_id, design)
        if seeds.empty:
            logger.info("%s: no candidate to seed a window", unit_id)
            continue
        first = 0
        for sub in sorted(seeds["sub"].unique(), key=_stratum_order):
            group = seeds[seeds["sub"] == sub].sort_values("key")
            rank = 0
            for _, cand in group.iterrows():
                if rank >= design.draw_per_stratum:
                    break
                window = seed_window(run, "", cand, design.seed)
                # A seed is judged through its window, which needs enough of
                # it inside. A long or awkwardly placed one is passed over.
                seen = cand["geometry"].intersection(box(*window.bounds)).area / 1e4
                if seen < min_ha:
                    unseen += 1
                    continue
                rank += 1
                stage = 0
                if rank == 1 and first < design.stage1_per_unit:
                    stage, first = 1, first + 1
                rows.append(
                    {
                        "unit": unit_id,
                        "epsg": window.epsg,
                        "ix0": window.ix0,
                        "iy0": window.iy0,
                        "nx": window.nx,
                        "ny": window.ny,
                        "stratum": "di" if cand["source"] == "di" else "other",
                        "sub_stratum": sub,
                        "seed_cand": cand["cand_id"],
                        "stratum_n": len(group),
                        "rank": rank,
                        "incl_prob": np.nan,
                        "stage": stage,
                        "_key": cand["key"],
                    }
                )
    if unseen:
        logger.info(
            "%d candidates passed over: their window shows under %.1f ha of them",
            unseen,
            min_ha,
        )
    quiet = design.quiet
    quiet_chips: dict[str, pd.DataFrame] = {}
    if quiet.n_quiet:
        for unit_id in units:
            chips = _quiet_chips(run, unit_id, quiet)
            if not chips.empty:
                quiet_chips[unit_id] = chips.astype({"key": float})
    share = _quiet_allocation(
        {u: len(c) for u, c in quiet_chips.items()},
        quiet.n_quiet,
        quiet.quiet_max_per_unit,
    )
    for unit_id, chips in quiet_chips.items():
        take = chips.nsmallest(share[unit_id], "key")
        for rank, chip in enumerate(take.itertuples(), start=1):
            rows.append(
                {
                    "unit": unit_id,
                    "epsg": run.unit(unit_id).epsg,
                    "ix0": int(chip.ix0),
                    "iy0": int(chip.iy0),
                    "nx": 1,
                    "ny": 1,
                    "stratum": "quiet",
                    "sub_stratum": getattr(chip, "intervals", "") or "",
                    "seed_cand": "",
                    "stratum_n": len(chips),
                    "rank": rank,
                    "incl_prob": len(take) / len(chips),
                    "stage": 1,
                    "_key": chip.key,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, pd.DataFrame(columns=["window_id", "cand_id", "role"])
    # Seeded windows of a unit in the order they are judged, quiet ones last.
    # Two seeds can give the same window. Both rows are kept: each seed is
    # judged at its own stage, and the labels of shared ground use every
    # verdict whichever window asked for it (pipeline.build_window).
    table["_quiet"] = (table["stratum"] == "quiet").astype(int)
    table["_late"] = (table["stage"] != 1).astype(int)
    table = table.sort_values(["_quiet", "unit", "_late", "rank", "_key"])
    table = table.reset_index(drop=True)
    table.insert(0, "window_id", [f"W{i + 1:05d}" for i in range(len(table))])
    table = table.drop(columns=["_quiet", "_late", "_key"])
    judged = _judged_list(run, table, design)
    _set_inclusion(table)
    return table, judged


def _judged_list(
    run: LabelRun, table: pd.DataFrame, design: ClassDesign
) -> pd.DataFrame:
    """The seed of each window and a few more candidates of its interval.

    Windows are gone through in the order they would be judged. A candidate
    is judged in the first window that takes it and in no other, so that no
    call is paid twice for one candidate (the first sample judged 4,344
    candidates of which 3,822 were different ones).
    """
    from csdv_core.label.candidates import window_candidates

    min_ha = run.cfg.candidates.min_ha
    taken: set[str] = set()
    rows = []
    seeded = table[table["stratum"] != "quiet"]
    order = seeded.sort_values(["stage", "rank"], ascending=[False, True])
    # Every seed is spoken for before any extra is chosen.
    all_seeds = set(seeded["seed_cand"])
    for row in order.itertuples():
        if row.seed_cand in taken:
            continue
        taken.add(row.seed_cand)
        rows.append(
            {"window_id": row.window_id, "cand_id": row.seed_cand, "role": "seed"}
        )
        if design.judged_per_window <= 1:
            continue
        cands = run.candidates(row.unit)
        seed = cands[cands["cand_id"] == row.seed_cand].iloc[0]
        inside = window_candidates(cands, run.window(row).bounds, min_ha)
        more = inside[
            inside["judged"]
            & (inside["interval"] == seed["interval"])
            & ~inside["cand_id"].isin(taken | all_seeds)
        ]
        if row.unit in design.plain_units:
            more = more[more["source"] != "model"]
        # The largest visible ones: most likely the rest of the seed's event.
        for cand_id in more.nlargest(design.judged_per_window - 1, "visible_ha")[
            "cand_id"
        ]:
            taken.add(cand_id)
            rows.append(
                {"window_id": row.window_id, "cand_id": cand_id, "role": "extra"}
            )
    return pd.DataFrame(rows, columns=["window_id", "cand_id", "role"])


def _set_inclusion(table: pd.DataFrame) -> None:
    """Inclusion probability of the seeds: windows staged over the stratum.

    A stratum split by season (``season_n`` is set) counts per season.
    """
    seeded = table["stratum"] != "quiet"
    if "season_n" in table:
        split = table["season_n"].notna()
        part = table["season"].where(split, "")
        size = table["season_n"].where(split, table["stratum_n"])
    else:
        part = pd.Series("", index=table.index)
        size = table["stratum_n"]
    keyed = table.assign(_part=part)
    staged = keyed[seeded & (keyed["stage"] > 0)]
    counts = staged.groupby(["unit", "sub_stratum", "_part"]).size()
    keys = list(zip(table["unit"], table["sub_stratum"], part, strict=True))
    taken = pd.Series([counts.get(k, 0) for k in keys], index=table.index)
    prob = (taken / size).where(table["stage"] > 0)
    table.loc[seeded, "incl_prob"] = prob[seeded]


def seed_seasons(run: LabelRun, units: Sequence[str], design: ClassDesign):
    """Every candidate that can seed a window, with its stratum and season.

    Returns:
        A table with the columns ``unit``, ``cand_id``, ``sub_stratum`` and
        ``season``. :func:`allocate_stage` needs it for a plan that names
        seasons. The design has to be that of the draw (the units sampled
        without the model, the priority intervals, the edge).
    """
    rows = []
    for unit_id in units:
        seeds = _unit_seeds(run, unit_id, design)
        if seeds.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "unit": unit_id,
                    "cand_id": seeds["cand_id"].to_numpy(),
                    "sub_stratum": seeds["sub"].to_numpy(),
                    "season": seeds["season"].to_numpy(),
                }
            )
        )
    columns = ["unit", "cand_id", "sub_stratum", "season"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _split_by_season(out: pd.DataFrame, named: pd.DataFrame, seeds: pd.DataFrame):
    """Split the strata a plan names with a season, in place.

    Raises:
        ValueError: If a stratum is not in the sample, or holds another
            number of candidates than when the sample was drawn (the seeds
            were then listed with another design).
    """
    for unit_id, sub in (
        named[["unit", "sub_stratum"]].drop_duplicates().itertuples(index=False)
    ):
        rows = out[
            (out["unit"] == unit_id)
            & (out["sub_stratum"] == sub)
            & (out["stratum"] != "quiet")
        ]
        if rows.empty:
            raise ValueError(f"No stratum {sub} in {unit_id}")
        pool = seeds[(seeds["unit"] == unit_id) & (seeds["sub_stratum"] == sub)]
        drawn = int(rows["stratum_n"].iloc[0])
        season_of = pool.set_index("cand_id")["season"]
        if len(pool) != drawn or not rows["seed_cand"].isin(season_of.index).all():
            raise ValueError(
                f"{unit_id} {sub}: {len(pool)} candidates listed, {drawn} when "
                "the sample was drawn. Give the sites file and the edge of the draw."
            )
        season = rows["seed_cand"].map(season_of)
        out.loc[rows.index, "season"] = season
        out.loc[rows.index, "season_n"] = season.map(pool["season"].value_counts())


def allocate_stage(
    table: pd.DataFrame,
    plan: pd.DataFrame,
    stage: int,
    judged_windows: set[str] | None = None,
    seeds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Give the next windows of each stratum to a stage.

    Args:
        table: ``windows.csv`` of a sample by class.
        plan: Columns ``unit``, ``sub_stratum`` and ``n``: how many windows
            of the stratum the stage adds. Strata that are not named get
            none. An optional column ``season`` (``nov-apr`` or ``may-oct``)
            limits a row to the windows whose seed ends on an image of that
            season, and splits the stratum by season from then on.
        stage: The stage to fill (2 or more).
        judged_windows: Windows that hold answers. They keep their stage.
        seeds: The table of :func:`seed_seasons`. Needed when the plan
            names a season.

    Returns:
        The table with ``stage`` and ``incl_prob`` updated. Windows are
        taken in the order of ``rank``, after those of earlier stages. With
        seasons in the plan the table also has the columns ``season`` and
        ``season_n`` (candidates of the season in the stratum), set for the
        strata that are split.

    Raises:
        ValueError: If the plan asks for more windows than a stratum has
            left, names a stratum or a season the sample does not have, or
            the stage is not above 1.
    """
    if stage < 2:
        raise ValueError("Stage 1 is set when the sample is drawn")
    missing = {"unit", "sub_stratum", "n"} - set(plan.columns)
    if missing:
        raise ValueError(f"The plan lacks the columns {sorted(missing)}")
    out = table.copy()
    plan = plan.copy()
    if "season" not in plan:
        plan["season"] = ""
    plan["season"] = plan["season"].fillna("").astype(str).str.strip()
    named = plan[(plan["season"] != "") & (plan["n"] > 0)]
    unknown = sorted(set(named["season"]) - {OFF_SEASON, ON_SEASON})
    if unknown:
        raise ValueError(f"The plan names the seasons {unknown}")
    if "season_n" not in out:
        out["season"], out["season_n"] = "", np.nan
    out["season"] = out["season"].fillna("").astype(str)
    if len(named):
        if seeds is None:
            raise ValueError("The plan names seasons: the seeds are needed")
        _split_by_season(out, named, seeds)
    held = set(judged_windows or ())
    # Windows of this stage that are not judged yet go back to the pool, so
    # that the command can be given again with another plan.
    back = (out["stage"] == stage) & ~out["window_id"].isin(held)
    out.loc[back, "stage"] = 0
    short = []
    for row in plan.itertuples():
        n = int(row.n)
        if n <= 0:
            continue
        group = out[
            (out["unit"] == row.unit)
            & (out["sub_stratum"] == row.sub_stratum)
            & (out["stratum"] != "quiet")
        ]
        if group.empty:
            raise ValueError(f"No stratum {row.sub_stratum} in {row.unit}")
        name = row.sub_stratum
        if row.season:
            group = group[group["season"] == row.season]
            name = f"{name} {row.season}"
        free = group[group["stage"] == 0].sort_values("rank")
        if len(free) < n:
            short.append(f"{row.unit} {name}: {n} asked, {len(free)} left")
            continue
        out.loc[free.index[:n], "stage"] = stage
    if short:
        raise ValueError(
            "The plan asks for more windows than were drawn: " + "; ".join(short)
        )
    _set_inclusion(out)
    return out


def class_counts(run: LabelRun, units: list[str], design: ClassDesign) -> pd.DataFrame:
    """Candidates per unit and stratum that can seed a window."""
    rows = []
    for unit_id in units:
        seeds = _unit_seeds(run, unit_id, design)
        for sub, group in seeds.groupby("sub") if len(seeds) else []:
            rows.append(
                {
                    "unit": unit_id,
                    "sub_stratum": sub,
                    "candidates": len(group),
                    "area_ha": round(float(group["area_ha"].sum()), 1),
                }
            )
    return pd.DataFrame(rows, columns=["unit", "sub_stratum", "candidates", "area_ha"])
