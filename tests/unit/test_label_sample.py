"""The window sample: allocation among units, quiet windows and stages."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from csdv_core.label import sample as S
from csdv_core.label.pipeline import LabelRun
from csdv_core.label.sample import SampleDesign, draw_windows

UNIT = "q4407159"


@pytest.fixture()
def run(label_export: Path, tmp_path: Path) -> LabelRun:
    root = tmp_path / "label"
    root.mkdir()
    return LabelRun(root=root, runs_dir=label_export / "runs" / "default")


def test_equal_allocation_hands_the_shortfall_to_units_with_room() -> None:
    alloc = S._equal_allocation({"a": 1, "b": 30, "c": 30}, 11)
    assert alloc == {"a": 1, "b": 5, "c": 5}
    assert S._equal_allocation({"a": 2, "b": 3}, 50) == {"a": 2, "b": 3}


def test_quiet_windows_go_to_units_with_room_up_to_the_cap() -> None:
    available = {"a": 903, "b": 804, "c": 262, "d": 12, "e": 3, "f": 1}
    alloc = S._quiet_allocation(available, 150, 25)
    assert alloc == {"a": 25, "b": 25, "c": 25, "d": 12, "e": 3, "f": 1}
    # Without a cap the units with room take the whole shortfall.
    free = S._quiet_allocation(available, 150, None)
    assert sum(free.values()) == 150 and free["d"] == 12 and free["a"] == 45
    # A small sample is spread evenly and takes nothing a unit lacks.
    assert S._quiet_allocation(available, 6, 25) == dict.fromkeys(available, 1)


def test_quiet_windows_carry_their_units_inclusion_probability(
    run: LabelRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    chips = pd.DataFrame(
        {"ix0": range(100, 110), "iy0": 200, "key": [i / 10 for i in range(10)]}
    )
    monkeypatch.setattr(S, "_quiet_chips", lambda *args: chips)
    design = SampleDesign(n_di=0, n_other=0, n_quiet=8, quiet_max_per_unit=3)
    table = draw_windows(run, [UNIT], design)
    assert list(table["stratum"]) == ["quiet"] * 3
    assert list(table["ix0"]) == [100, 101, 102]  # the three smallest keys
    assert list(table["incl_prob"]) == pytest.approx([0.3] * 3)
    assert (table["stage"] == 1).all()
    more = draw_windows(
        run, [UNIT], SampleDesign(n_di=0, n_other=0, n_quiet=8, quiet_max_per_unit=None)
    )
    assert len(more) == 8 and list(more["incl_prob"]) == pytest.approx([0.8] * 8)


def _drawn(units: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for unit, strata in units.items():
        for stratum, n in strata.items():
            for i in range(n):
                seed = "" if stratum == "quiet" else f"{unit}-{stratum}-2018-{i}"
                rows.append({"unit": unit, "stratum": stratum, "seed_cand": seed})
    return pd.DataFrame(rows)


def test_stage_one_holds_reviewed_quiet_and_a_few_windows_per_unit() -> None:
    table = _drawn(
        {
            "a": {"reviewed": 1, "di": 6, "other": 3, "quiet": 2},
            "b": {"di": 1, "other": 4},
        }
    )
    design = SampleDesign(stage1_di=2, stage1_other=1)
    table["stage"] = S._stages(table, design)
    first = table[table["stage"] == 1]
    counts = first.groupby(["unit", "stratum"]).size().to_dict()
    assert counts == {
        ("a", "reviewed"): 1,
        ("a", "di"): 2,
        ("a", "other"): 1,
        ("a", "quiet"): 2,
        ("b", "di"): 1,  # a unit with fewer windows gives all of them
        ("b", "other"): 1,
    }
    assert set(table["stage"]) == {1, 2}
    assert S._stages(table, design) == list(table["stage"])  # fixed by the seed


def test_stage_one_grows_with_its_size() -> None:
    table = _drawn({"a": {"di": 8, "other": 5}})
    small = pd.Series(S._stages(table, SampleDesign(stage1_di=2, stage1_other=1)))
    large = pd.Series(S._stages(table, SampleDesign(stage1_di=4, stage1_other=2)))
    assert (small == 1).sum() == 3 and (large == 1).sum() == 6
    assert (large[small == 1] == 1).all()
    none = S._stages(table, SampleDesign(stage1_di=0, stage1_other=0))
    assert set(none) == {2}


def test_stage_keys_differ_from_sampling_keys() -> None:
    """Stage 1 is not simply the candidates with the smallest sampling keys."""
    table = _drawn({"a": {"di": 40}})
    table["key"] = [S._key(0, c) for c in table["seed_cand"]]
    stage = pd.Series(S._stages(table, SampleDesign(stage1_di=10)))
    by_sampling_key = set(table["key"].nsmallest(10).index)
    assert set(stage[stage == 1].index) != by_sampling_key


def test_a_larger_draw_holds_the_smaller(run: LabelRun) -> None:
    """Seeds are the smallest keys of a sub-stratum, so a sample can grow."""
    unit = run.unit(UNIT)
    frame = gpd.GeoDataFrame(
        {
            "cand_id": [f"{UNIT}-di-2018-{i}" for i in range(60)],
            "unit": UNIT,
            "interval": unit.intervals[0],
            "mean_prob": 0.95,
            "area_ha": 0.5,
        },
        geometry=[box(0, 0, 10, 10)] * 60,
        crs=f"EPSG:{unit.epsg}",
    )
    small = {r["seed_cand"] for r in S._seeded(run, UNIT, frame, 5, 0)}
    large = {r["seed_cand"] for r in S._seeded(run, UNIT, frame, 20, 0)}
    assert len(small) == 5 and len(large) == 20 and small < large
    other_seed = {r["seed_cand"] for r in S._seeded(run, UNIT, frame, 5, 1)}
    assert other_seed != small


def test_the_draw_is_fixed_by_the_seed_and_numbers_its_windows(run: LabelRun) -> None:
    design = SampleDesign(n_di=5, n_other=5, n_quiet=2, seed=1)
    table = draw_windows(run, [UNIT], design)
    assert table.equals(draw_windows(run, [UNIT], design))
    assert list(table["window_id"]) == [f"W{i + 1:04d}" for i in range(len(table))]
    assert "stage" in table and set(table["stage"]) <= {1, 2}
