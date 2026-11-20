"""Tests for a batch of units sent through one work queue.

No network and no Slurm: ``sbatch`` is replaced, so what is checked is which
jobs are submitted for what is on disk, their dependencies, the row files
and the checks a unit must pass before it is finished.
"""

from __future__ import annotations

import pytest

from csdv_core.grid import queue as queue_mod
from csdv_core.grid.queue import (
    QueueOptions,
    chm_rows,
    gpu_pools,
    priority_interval_ok,
    queue_dir,
    submit_queue,
    sync_unit_years,
    years_on_disk,
)
from csdv_core.grid.submit import SubmitError
from csdv_core.grid.units import GridPaths, Unit, load_unit

POOLS = (
    "--account=a1 --partition=p1 --qos=q1 --gres=gpu:1",
    "--account=a2 --partition=p2 --qos=q2 --gres=gpu:1",
)
CPU = "--account=c --partition=c --qos=c"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    paths = GridPaths(
        grid_dir=tmp_path / "grid",
        export_dir=tmp_path / "exports",
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )
    for name in ("CSDV_REPO", "CSDV_LOG_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    calls: list[list[str]] = []

    def fake_sbatch(command):
        calls.append(command)
        return str(2000 + len(calls))

    monkeypatch.setattr(queue_mod, "_sbatch", fake_sbatch)
    return paths, calls


def _units() -> list[Unit]:
    return [
        Unit.point(-83.0, 35.0, 2.4, label="wind_a"),
        Unit.point(-90.0, 46.0, 2.4, label="snow_b"),
    ]


def _opts(**more) -> QueueOptions:
    return QueueOptions(name="round2", pools=POOLS, cpu=CPU, **more)


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _naip(paths: GridPaths, unit: str, year: int) -> None:
    _touch(paths.naip_year_dir(unit, year) / f"NAIP_{unit}_{year}0801_0p6m.tif")


def _chm(paths: GridPaths, unit: str, year: int) -> None:
    _touch(paths.chm_year_dir(unit, year) / f"NAIP_{unit}_{year}0801_0p6m_chm.tif")


def test_pools_are_read_from_one_setting() -> None:
    assert gpu_pools(f"{POOLS[0]} ; {POOLS[1]};") == POOLS
    with pytest.raises(SubmitError, match="CSDV_GPU_POOLS"):
        gpu_pools("")


def test_a_new_batch_submits_download_workers_on_every_pool_and_finish(env) -> None:
    paths, calls = env
    plan = submit_queue(paths, _units(), _opts(workers_per_pool=2, min_years=4))
    assert [j.step for j in plan.jobs] == ["download", *["worker"] * 4, "finish"]
    download, *workers, finish = (j.command for j in plan.jobs)

    assert "--array=1-2%6" in download and "--min-years" in download
    rows = plan.download_rows.read_text().splitlines()
    assert rows[0] == "unit,west,south,east,north" and rows[1].startswith("p_wind_a,")

    # Workers wait for the downloads, whatever their outcome, and carry
    # their pool's options and nothing of another pool.
    assert all("--dependency=afterany:2001" in w for w in workers)
    assert sum("--partition=p1" in w for w in workers) == 2
    assert sum("--partition=p2" in w for w in workers) == 2
    assert all("--time=04:00:00" in w and "--mem=24G" in w for w in workers)
    assert all(w[-1] == str(queue_dir(paths, "round2")) for w in workers)

    # The finish array waits for the downloads and for every worker.
    deps = next(a for a in finish if a.startswith("--dependency="))
    assert deps == "--dependency=" + ",".join(
        f"afterany:{job}" for job in (2001, 2002, 2003, 2004, 2005)
    )
    assert "--array=1-2%20" in finish and "--partition=c" in finish
    assert plan.finish_rows.read_text().splitlines() == ["unit", "p_wind_a", "p_snow_b"]

    folder = queue_dir(paths, "round2")
    assert (folder / "workers.txt").read_text().split() == [
        "2002",
        "2003",
        "2004",
        "2005",
    ]
    assert (folder / "units.csv").read_text().count("\n") == 3
    assert load_unit(paths, "p_wind_a").buffer_km == 2.4
    assert len(calls) == 6


def test_the_same_command_again_submits_only_what_is_missing(env) -> None:
    paths, calls = env
    units = _units()
    folder = queue_dir(paths, "round2")
    # Unit one is downloaded with one CHM made and one to make. Unit two is
    # exported.
    for year in (2018, 2021):
        _naip(paths, "p_wind_a", year)
    _chm(paths, "p_wind_a", 2018)
    _touch(folder / "downloaded" / "p_wind_a")
    _touch(folder / "downloaded" / "p_snow_b")
    _touch(paths.export_run_dir("p_snow_b", "default") / "export.json")

    plan = submit_queue(paths, units, _opts(workers_per_pool=1))
    assert plan.to_download == [] and plan.rows == [("p_wind_a", 2021)]
    assert plan.to_finish == ["p_wind_a"]
    assert [j.step for j in plan.jobs] == ["worker", "worker", "finish"]
    assert not any("--dependency" in a for a in plan.jobs[0].command)
    assert "--array=1-1%20" in plan.jobs[-1].command

    # Everything exported: nothing is submitted.
    _touch(paths.export_run_dir("p_wind_a", "default") / "export.json")
    calls.clear()
    plan = submit_queue(paths, units, _opts())
    assert plan.jobs == [] and calls == []


def test_a_dry_run_writes_and_submits_nothing(env) -> None:
    paths, calls = env
    plan = submit_queue(paths, _units(), _opts(dry_run=True, checkpoint="best.pt"))
    assert calls == [] and not queue_dir(paths, "round2").exists()
    finish = plan.jobs[-1].command
    assert "--dependency=afterany:<download>,afterany:<workers>" in finish
    assert finish[finish.index("--checkpoint") + 1] == "best.pt"


def test_rows_are_read_from_disk_and_leave_out_years_that_failed_twice(env) -> None:
    paths, _ = env
    folder = queue_dir(paths, "round2")
    for year in (2016, 2018, 2021):
        _naip(paths, "u", year)
    _chm(paths, "u", 2016)
    # A folder with two mosaics is not a year the queue can use.
    _touch(paths.naip_year_dir("u", 2023) / "a.tif")
    _touch(paths.naip_year_dir("u", 2023) / "b.tif")
    assert years_on_disk(paths, "u") == [2016, 2018, 2021]
    assert chm_rows(paths, folder, ["u"]) == [("u", 2018), ("u", 2021)]
    (folder / "failed").mkdir(parents=True)
    (folder / "failed" / "u_2021").write_text("job1\n")
    assert chm_rows(paths, folder, ["u"]) == [("u", 2018), ("u", 2021)]
    (folder / "failed" / "u_2021").write_text("job1\njob2\n")
    assert chm_rows(paths, folder, ["u"]) == [("u", 2018)]


def test_priority_interval_needs_two_consecutive_years() -> None:
    years = [2016, 2018, 2021]
    assert priority_interval_ok(years, "2018-2021")
    assert not priority_interval_ok(years, "2016-2021")  # a date lies between
    assert not priority_interval_ok(years, "2018-2020")
    assert not priority_interval_ok(years, "spring")


def test_a_unit_is_finished_only_when_its_years_are_ready(env, tmp_path) -> None:
    paths, _ = env
    folder = queue_dir(paths, "round2")
    unit = Unit.point(-83.0, 35.0, 2.4, label="wind_a")
    uid = unit.unit_id
    for year in (2014, 2016, 2018, 2021):
        _naip(paths, uid, year)
        _chm(paths, uid, year)
    _naip(paths, uid, 2023)

    with pytest.raises(SubmitError, match="download has not finished"):
        sync_unit_years(paths, folder, unit)
    _touch(folder / "downloaded" / uid)
    with pytest.raises(SubmitError, match=r"no CHM yet for \[2023\]"):
        sync_unit_years(paths, folder, unit)

    # After two failed tries the year is left out and the unit goes on.
    (folder / "failed").mkdir(parents=True)
    (folder / "failed" / f"{uid}_2023").write_text("a\nb\n")
    assert sync_unit_years(paths, folder, unit, min_years=4) == [2014, 2016, 2018, 2021]
    saved = load_unit(paths, uid)
    assert sorted(saved.years) == [2014, 2016, 2018, 2021]
    assert saved.skipped_years[2023] == "CHM inference failed"
    with pytest.raises(SubmitError, match="fewer than 5"):
        sync_unit_years(paths, folder, unit, min_years=5)

    # The unit was sited on an event between 2021 and 2023. Without 2023 it
    # has no pair of dates around it, and is not finished.
    wanted = tmp_path / "priority.csv"
    wanted.write_text(
        f"# sited events\nunit,interval\n{uid},2021-2023\nother,2012-2014\n"
    )
    with pytest.raises(SubmitError, match="priority interval 2021-2023"):
        sync_unit_years(paths, folder, unit, priority_intervals=wanted)
    wanted.write_text(f"unit,interval\n{uid},2018-2021\n")
    assert len(sync_unit_years(paths, folder, unit, priority_intervals=wanted)) == 4
    # A background place names no interval: nothing to check.
    wanted.write_text(f"unit,role,interval\n{uid},train,\n")
    assert len(sync_unit_years(paths, folder, unit, priority_intervals=wanted)) == 4


def test_a_download_with_a_failed_year_counts_when_enough_arrive(
    tmp_path, monkeypatch
) -> None:
    from click.testing import CliRunner

    from csdv_core.download import naip_auto

    monkeypatch.setattr(
        naip_auto,
        "discover_years",
        lambda bbox: ({y: 1.0 for y in (2014, 2016, 2018, 2021)}, {}),
    )

    def download(bbox, plan, **kwargs):
        paths = [tmp_path / f"{p.year}.tif" for p in plan.years if p.year != 2021]
        return paths, [2021]

    monkeypatch.setattr(naip_auto, "download_planned_years", download)
    args = ["--site", "u", "--bbox", "-83", "35", "-82.9", "35.1"]
    args += ["--out-root", str(tmp_path)]
    cli = CliRunner()
    assert cli.invoke(naip_auto.cli, args).exit_code != 0
    assert cli.invoke(naip_auto.cli, [*args, "--min-years", "3"]).exit_code == 0
    result = cli.invoke(naip_auto.cli, [*args, "--min-years", "4"])
    assert result.exit_code != 0 and "No mosaic written for [2021]" in result.output


def test_export_copies_the_layers_and_the_model_map(env) -> None:
    from csdv_core.grid.export import export_layers

    paths, _ = env
    run_dir = paths.run_dir("u", "default")
    for rel in ("detection/probability.tif", "layers/forest.tif"):
        _touch(run_dir / rel)
    for name in ("prob_u.tif", "prob_forest_u.tif", "type_u_2018_2021.tif"):
        _touch(run_dir / "model" / name)
    written = export_layers(paths, "u", "default")
    out = paths.export_dir / "run_layers" / "u" / "default"
    assert sorted(str(p.relative_to(out)) for p in written) == [
        "detection/probability.tif",
        "layers/forest.tif",
        "model/prob_forest_u.tif",
        "model/prob_u.tif",
    ]


def test_a_box_is_compared_with_its_quad_by_distance_from_its_edge(env) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from csdv_core.grid.compare import compare_units

    paths, _ = env
    rng = np.random.default_rng(0)
    quad = rng.uniform(0, 30, (1500, 1500)).astype("float32")
    # The box is the quad's pixels 200 to 1200 (600 m on a side). Its outer
    # 60 m (100 pixels) are 1 m higher, as an edge effect would be, and a
    # block away from the edge has no height in the box.
    box = quad[200:1200, 200:1200].copy()
    edge = np.ones(box.shape, dtype=bool)
    edge[100:-100, 100:-100] = False
    box[edge] += 1.0
    box[500:520, 500:520] = np.nan

    def write(unit: str, year: int, date: str, data, x0: float, y1: float) -> None:
        folder = paths.chm_year_dir(unit, year)
        folder.mkdir(parents=True)
        with rasterio.open(
            folder / f"NAIP_{unit}_{date}_0p6m_chm.tif",
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:26917",
            transform=from_origin(x0, y1, 0.6, 0.6),
            nodata=float("nan"),
        ) as dst:
            dst.write(data, 1)

    for year, date in ((2018, "20180801"), (2021, "20210715")):
        write("q1", year, date, quad, 300000.0, 4500000.0)
        write("p_box", year, date, box, 300000.0 + 120.0, 4500000.0 - 120.0)
    # A year the quad was flown on another day, and one only the quad has.
    write("q1", 2023, "20230601", quad, 300000.0, 4500000.0)
    write("p_box", 2023, "20230905", box, 300120.0, 4499880.0)
    write("q1", 2016, "20160801", quad, 300000.0, 4500000.0)

    rows = compare_units(paths, "p_box", "q1", edges_m=(30.0, 60.0, 120.0))
    assert sorted({r.year for r in rows}) == [2018, 2021, 2023]
    by_band = {r.band: r for r in rows if r.year == 2018}
    assert list(by_band) == ["0-30 m", "30-60 m", "60-120 m", "120 m and more"]
    assert by_band["0-30 m"].rmse_m == pytest.approx(1.0, abs=1e-4)
    assert by_band["30-60 m"].mean_m == pytest.approx(1.0, abs=1e-4)
    assert by_band["60-120 m"].rmse_m == pytest.approx(0.0, abs=1e-6)
    inner = by_band["120 m and more"]
    assert inner.rmse_m == pytest.approx(0.0, abs=1e-6) and inner.share_over_half_m == 0
    assert inner.pixels == 600 * 600 - 20 * 20
    other_day = next(r for r in rows if r.year == 2023)
    assert (
        other_day.date_unit == "2023-09-05" and other_day.date_against == "2023-06-01"
    )
    with pytest.raises(FileNotFoundError):
        compare_units(paths, "p_box", "p_box_missing")
