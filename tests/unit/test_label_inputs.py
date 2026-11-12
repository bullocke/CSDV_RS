"""Unit inputs, candidates, imagery and label rasters on a synthetic run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.config import load_label
from csdv_core.label import candidates as C
from csdv_core.label import routing as R
from csdv_core.label.chips import read_chip, write_window_chips
from csdv_core.label.inputs import (
    fill_forest_dips,
    list_units,
    load_unit_inputs,
    unstable_intervals,
)
from csdv_core.label.labels import JudgedCandidate, window_labels
from csdv_core.label.lattice import GSD, Window
from csdv_core.label.naip import read_naip, warp_sources
from csdv_core.label.outlines import choose_outline
from csdv_core.label.rules import Reason
from csdv_core.label.stacks import WindowStack, build_stack, load_stack
from csdv_core.label.verdicts import Verdict

CFG = load_label()
UNIT = "q4407159"
#: Chips 977-978 east, 14646-14647 north lie inside the synthetic grid.
WINDOW = Window("w1", UNIT, 26919, 977, 14646, 2, 2)


@pytest.fixture()
def unit(label_export: Path):
    return load_unit_inputs(label_export / "runs" / "default" / UNIT)


def test_unit_inputs(unit, label_export: Path) -> None:
    assert unit.unit == UNIT and unit.run_tag == "default" and unit.epsg == 26919
    assert unit.years == [2015, 2018, 2021]
    assert unit.intervals == ["2015-2018", "2018-2021"]
    assert unit.dates[2018] == "2018-08-02" and unit.sources[2021] == "pc"
    assert unit.has_layers
    assert unit.layer("forest").dtype == bool
    assert unit.layer("probability").shape == (3, 40, 40)
    assert np.isnan(unit.layer("probability")[0]).all()
    assert [u.unit for u in list_units(label_export)] == [UNIT]


def test_layers_missing_is_an_error_not_a_default(label_export: Path) -> None:
    run = label_export / "runs" / "default" / UNIT
    (label_export / "run_layers" / UNIT / "default" / "layers" / "forest.tif").unlink()
    unit = load_unit_inputs(run)
    assert not unit.has_layers
    with pytest.raises(FileNotFoundError):
        unit.layer("forest")


def test_forest_dips_and_unstable_intervals() -> None:
    forest = np.ones((4, 2, 2), dtype=bool)
    forest[1, 0, 0] = False  # one-date dip
    forest[2:, 1, 1] = False  # real loss
    filled = fill_forest_dips(forest)
    assert filled[1, 0, 0] and not filled[2, 1, 1]
    assert unstable_intervals(filled, 0.15) == [False, True, False]


def test_candidates(unit) -> None:
    di = C.di_candidates(unit)
    assert list(di.cand_id) == [f"{UNIT}-di-2018-1", f"{UNIT}-di-2021-2"]
    assert list(di.geom_src) == ["refined", "cells"]
    other = C.other_candidates(unit, CFG.candidates)
    by_source = {row.source: row for row in other.itertuples()}
    assert set(by_source) == {"chm", "weak"} and len(other) == 2
    assert by_source["chm"].interval == "2018-2021"
    assert by_source["chm"].area_ha == pytest.approx(25 * 0.036864)
    assert by_source["weak"].interval == "2015-2018"
    assert by_source["weak"].area_ha == pytest.approx(16 * 0.036864)
    assert by_source["weak"].mean_prob == pytest.approx(0.4)
    every = C.unit_candidates(unit, CFG.candidates)
    assert len(every) == 4 and every.crs.to_epsg() == 26919
    seen = C.window_candidates(every, WINDOW.bounds, CFG.candidates.min_ha)
    flags = seen.set_index("cand_id")
    assert not flags.loc[f"{UNIT}-di-2021-2", "judged"]  # two cells
    assert flags.loc[f"{UNIT}-di-2018-1", "judged"]
    assert flags.loc[by_source["chm"].cand_id, "clipped"]  # crosses the west edge


def _mosaic(path: Path, res: float, value: int, x0: float, y1: float, n: int) -> None:
    # rasterio marks the fourth band of such a file as alpha, as it does for
    # the mosaics written by `csdv download`. It must still be read as data.
    data = np.full((4, n, n), value, dtype="uint8")
    if res < 0.5:  # a 0.3 m checkerboard, to tell averaging from picking
        data[:, ::2, ::2] = 0
        data[:, 1::2, 1::2] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=4,
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(x0, y1, res, res),
    ) as dst:
        dst.write(data)


def test_warp_averages_fine_imagery_and_fills_in_order(tmp_path: Path) -> None:
    bounds = (300000.0, 4499940.0, 300060.0, 4500000.0)
    fine = tmp_path / "fine.tif"
    coarse = tmp_path / "coarse.tif"
    _mosaic(fine, 0.3, 200, 300000.0, 4500000.0, 100)  # the north-west quarter
    _mosaic(coarse, 1.0, 80, 299990.0, 4500010.0, 90)  # everything
    data, filled = warp_sources([str(fine), str(coarse)], 26919, bounds)
    assert data.shape == (4, 100, 100)
    assert filled == [50 * 50, 100 * 100 - 50 * 50]
    # Each 0.6 m pixel holds two dark and two bright 0.3 m pixels.
    assert np.all(np.abs(data[:, 5:45, 5:45].astype(int) - 100) <= 2)
    assert np.all(data[:, :, 55:] == 80) and np.all(data[:, 55:, :] == 80)
    with rasterio.open(fine) as src:
        assert src.colorinterp[3].name == "alpha"  # the trap this guards against


def test_stack_from_local_mosaics(unit, tmp_path: Path) -> None:
    root = tmp_path / "data"
    x0, _, _, y1 = WINDOW.stack_bounds(64)
    for year, res, tag in ((2015, 1.0, "1m"), (2018, 0.6, "0p6m"), (2021, 0.3, "0p3m")):
        folder = root / "naip" / UNIT / str(year)
        folder.mkdir(parents=True)
        _mosaic(
            folder / f"NAIP_{UNIT}_{year}0801_{tag}.tif",
            res,
            90 + year % 10,
            x0 - 30,
            y1 + 30,
            int(820 / res),
        )
        (folder / f"NAIP_{UNIT}_{year}0801_{tag}_chm.tif").write_bytes(b"")
    meta = build_stack(unit, WINDOW, tmp_path / "stack", source="local", data_root=root)
    assert meta["failed"] == {} and sorted(meta["years"]) == ["2015", "2018", "2021"]
    assert meta["years"]["2021"]["native_m"] == 0.3
    assert meta["years"]["2015"]["date"] == "2015-08-01"
    stack = load_stack(tmp_path / "stack")
    side = 2 * 512 + 2 * 64
    assert stack.data.shape == (3, 4, side, side) and stack.years == unit.years
    with rasterio.open(tmp_path / "stack" / "2018.tif") as src:
        assert src.descriptions == ("R", "G", "B", "NIR")
        assert src.colorinterp[3].name != "alpha"
        assert src.transform.c == pytest.approx(x0)
    assert stack.crop(WINDOW.bounds).shape == (3, 4, 1024, 1024)
    # A unit without a mosaic for the year fails. It never downloads.
    with pytest.raises(FileNotFoundError):
        read_naip(
            "local",
            unit=UNIT,
            year=2030,
            epsg=26919,
            bounds=WINDOW.bounds,
            data_root=root,
        )


def _verdict(states: list[str], **kw) -> Verdict:
    return Verdict(model="m", states=states, confidence=3, valid=True, **kw)


def test_window_labels_and_chips(unit, tmp_path: Path) -> None:
    every = C.unit_candidates(unit, CFG.candidates).set_index("cand_id")
    side = 2 * 512 + 2 * 64
    rng = np.random.default_rng(0)
    stack = WindowStack(
        bounds=WINDOW.stack_bounds(64),
        epsg=26919,
        years=unit.years,
        dates=["2015-08-01", "2018-08-02", "2021-08-03"],
        native_m=[1.0, 0.6, 0.6],
        data=rng.integers(1, 255, (3, 4, side, side), dtype=np.uint8),
    )
    clearcut = dict(
        change_type="clearcut", outline_fit="about_right", affected_share="over_75"
    )
    accepted = R.route(
        _verdict(["forest", "full_loss", "open"], **clearcut),
        _verdict(["forest", "full_loss", "open"], **clearcut),
        0,
        CFG.routing,
    )
    rejected = R.route(
        _verdict(["forest"] * 3), _verdict(["forest"] * 3), 1, CFG.routing
    )
    di = every.loc[f"{UNIT}-di-2018-1"]
    chm = every[every.source == "chm"].iloc[0]
    small = every.loc[f"{UNIT}-di-2021-2"]
    judged = [
        JudgedCandidate(
            di.name,
            "di",
            "refined",
            di.geometry,
            0,
            accepted,
            choose_outline(
                di.geometry,
                source="di",
                geom_src="refined",
                routed=accepted,
                polygon=None,
                cfg=CFG.outlines,
            ),
        ),
        JudgedCandidate(chm.name, "chm", "cells", chm.geometry, 1, rejected),
        JudgedCandidate(
            small.name, "di", "cells", small.geometry, 1, R.unjudged(2, 1, small=True)
        ),
    ]
    labels = window_labels(unit, WINDOW, stack, judged, CFG)
    assert labels.reason.shape == (2, 1024, 1024)
    assert labels.intervals == ["2015-08-01_2018-08-02", "2018-08-02_2021-08-03"]
    assert (labels.reason != Reason.UNSET).all()

    grid = unit.grid.cell_transform
    x0, _, _, y1 = WINDOW.bounds

    def at(k: int, row: float, col: float) -> int:
        """Reason at the centre of a cell position of the unit grid."""
        x = grid.c + (col + 0.5) * 19.2
        y = grid.f - (row + 0.5) * 19.2
        return int(labels.reason[k, int((y1 - y) / GSD), int((x - x0) / GSD)])

    assert at(0, 12, 12) == Reason.ACCEPTED_FULL
    assert at(1, 12, 12) == Reason.POST_EVENT_OPEN
    assert labels.agent[0][labels.reason[0] == Reason.ACCEPTED_FULL].tolist()[0] == 1
    assert at(0, 27, 8) == Reason.JUDGED_INTACT
    assert at(1, 27, 8) == Reason.REJECTED_FLAGGED
    assert at(1, 20, 30) == Reason.UNDER_MIN and at(0, 20, 30) == Reason.STABLE_FOREST
    assert at(0, 31.5, 26.5) == Reason.RAISED_PROB  # weak block, never judged
    assert at(1, 31.5, 26.5) == Reason.STABLE_FOREST
    assert at(0, 36, 32) == Reason.NONFOREST and at(0, 36, 36) == Reason.SPARSE_WOODY
    assert at(0, 20, 20) == Reason.STABLE_FOREST
    assert labels.change[0][labels.reason[0] == Reason.ACCEPTED_FULL].max() == 1
    assert labels.prob.max() <= 100 or labels.prob.max() == 255
    assert labels.forest.shape == (3, 1024, 1024)

    rows = write_window_chips(WINDOW, stack, labels, tmp_path / "chips")
    assert len(rows) == 4 and rows[0]["chip_id"] == "e26919_x977_y14647"
    chip = read_chip(tmp_path / "chips" / rows[0]["chip_id"])
    assert chip["naip"].shape == (12, 512, 512)
    assert chip["naip_names"][:5] == [
        "2015-08-01_R",
        "2015-08-01_G",
        "2015-08-01_B",
        "2015-08-01_NIR",
        "2018-08-02_R",
    ]
    assert chip["change"].shape == (2, 512, 512)
    assert chip["aux_names"][0] == "prob_2015-08-01_2018-08-02"
    assert chip["aux_names"][-1] == "forest_2021-08-03"
    assert chip["transform"].c == pytest.approx(977 * 307.2)
    assert chip["transform"].f == pytest.approx(14648 * 307.2)
    assert (chip["reason"] == labels.reason[:, :512, :512]).all()
    assert set(np.unique(chip["change"])) <= {0, 1, 2, 255}


def test_labels_need_the_units_years(unit) -> None:
    stack = WindowStack(
        WINDOW.stack_bounds(64),
        26919,
        [2015, 2021],
        ["2015-08-01", "2021-08-03"],
        [None, None],
        np.zeros((2, 4, 1152, 1152), np.uint8),
    )
    with pytest.raises(ValueError, match="years"):
        window_labels(unit, WINDOW, stack, [], CFG)
