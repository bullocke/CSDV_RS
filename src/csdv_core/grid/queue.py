"""csdv_core.grid.queue: a batch of units through one work queue of GPU jobs.

``csdv grid submit`` gives every unit its own chain, with one GPU array task
per year. Each task waits in the GPU queue on its own, loads the model and
stages the conditioning rasters again, and one failed download year cancels
the unit's chain. That suits a few quads. For many small units the queue
here submits three things for the whole batch:

1. ``download``: one array on the data transfer nodes, a task per unit.
   A task plans the unit's years itself, drops a year that fails when
   enough remain, and leaves a marker when it is done.
2. ``workers``: a few GPU jobs on every pool named in ``$CSDV_GPU_POOLS``.
   A worker stages the conditioning rasters once, then takes one unit-year
   after another that has a NAIP mosaic and no CHM, claiming each with a
   folder that only one job can create. Whichever pool frees a GPU first
   does the work, and a worker that finds nothing left stops at once.
3. ``finish``: one CPU array, a task per unit, after every worker has ended:
   detection, the pair model's map on CPU (optional), export.

Nothing is planned on the login node, and no list of unit-years is written:
what is left to do is read from disk every time. Giving the same command
again therefore submits only what is missing, which is also how a batch
recovers from a preempted worker or a failed download.

Submit from a CHPC login node after ``source scripts/chpc/env.sh``.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from csdv_core.grid.status import chm_done
from csdv_core.grid.submit import Job, SubmitError, _env, _sbatch
from csdv_core.grid.units import GridPaths, Unit, save_unit

logger = logging.getLogger(__name__)

__all__ = [
    "QueueOptions",
    "QueuePlan",
    "chm_rows",
    "cpu_options",
    "gpu_pools",
    "plan_queue",
    "priority_interval_ok",
    "queue_dir",
    "submit_queue",
    "sync_unit_years",
    "years_on_disk",
]

#: A unit-year whose inference failed this often is left out of its unit.
MAX_ROW_FAILURES = 2


def gpu_pools(text: str | None) -> tuple[str, ...]:
    """The GPU pools of ``$CSDV_GPU_POOLS``: sbatch options, ``;`` between pools.

    Each pool is one string of options for a single GPU, for example
    ``--account=A --partition=B --qos=C --gres=gpu:1``. Allocation names
    live in the gitignored ``scripts/chpc/env.sh`` and nowhere in the code.

    Raises:
        SubmitError: If no pool is named.
    """
    pools = tuple(part.strip() for part in (text or "").split(";") if part.strip())
    if not pools:
        raise SubmitError(
            "$CSDV_GPU_POOLS names no GPU pool. Set it in scripts/chpc/env.sh "
            "(see env.example.sh)."
        )
    return pools


@dataclass(frozen=True)
class QueueOptions:
    """Options of one queue submission.

    Attributes:
        name: Name of the queue. Its folder is ``<grid dir>/queues/<name>``.
        pools: sbatch option strings, one per GPU pool (:func:`gpu_pools`).
        cpu: sbatch options of the CPU allocation (``$CSDV_SBATCH_CPU``).
        run_tag: Name of the detection run.
        start_year: First NAIP year a download considers.
        min_coverage: Smallest covered share of the box for a year.
        min_years: Fewest years a unit needs to be downloaded and finished.
        workers_per_pool: GPU jobs submitted to each pool.
        worker_hours: Walltime of a worker. It stops taking unit-years a
            quarter of an hour before.
        worker_mem: Memory of a worker. A 4.8 km box needs far less than the
            48G of a whole quad, and a small request waits less.
        exclude_nodes: Nodes the workers must not run on.
        download_throttle: Download tasks running at one time.
        finish_throttle: Finish tasks running at one time.
        checkpoint: Pair-model checkpoint. With it, every unit gets the
            model's map (``<run>/model/``), made on the CPU of its finish task.
        priority_intervals: CSV with columns ``unit`` and ``interval``. A
            unit whose interval lost a date is not finished.
        quicklooks: Write 3 m quicklooks in the export.
        detect_args: Further ``detect-disturbance`` options.
        dry_run: Print the jobs, submit and write nothing.
    """

    name: str
    pools: tuple[str, ...]
    cpu: str
    run_tag: str = "default"
    start_year: int = 2010
    min_coverage: float = 0.9
    min_years: int = 4
    workers_per_pool: int = 2
    worker_hours: float = 4.0
    worker_mem: str = "24G"
    exclude_nodes: tuple[str, ...] = ()
    download_throttle: int = 6
    finish_throttle: int = 20
    checkpoint: Path | None = None
    priority_intervals: Path | None = None
    quicklooks: bool = True
    detect_args: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass
class QueuePlan:
    """What one submission of a queue does."""

    folder: Path
    units: list[Unit]
    to_download: list[str] = field(default_factory=list)
    rows: list[tuple[str, int]] = field(default_factory=list)
    to_finish: list[str] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    download_rows: Path | None = None
    finish_rows: Path | None = None


def queue_dir(paths: GridPaths, name: str) -> Path:
    return paths.grid_dir / "queues" / name


def years_on_disk(paths: GridPaths, unit_id: str) -> list[int]:
    """Years of a unit that have exactly one NAIP mosaic."""
    root = paths.data_root / "naip" / unit_id
    years = []
    for folder in sorted(root.glob("*")):
        if folder.is_dir() and folder.name.isdigit():
            if len(list(folder.glob("*.tif"))) == 1:
                years.append(int(folder.name))
    return years


def _failures(folder: Path, unit_id: str, year: int) -> int:
    path = folder / "failed" / f"{unit_id}_{year}"
    if not path.is_file():
        return 0
    return len([line for line in path.read_text().splitlines() if line.strip()])


def chm_rows(
    paths: GridPaths, folder: Path, unit_ids: list[str]
) -> list[tuple[str, int]]:
    """Unit-years that have a mosaic, no CHM, and have not failed twice."""
    return [
        (unit_id, year)
        for unit_id in unit_ids
        for year in years_on_disk(paths, unit_id)
        if not chm_done(paths, unit_id, year)
        and _failures(folder, unit_id, year) < MAX_ROW_FAILURES
    ]


def priority_interval_ok(years: list[int], interval: str) -> bool:
    """Whether ``interval`` (``2018-2021``) is a pair of consecutive ``years``."""
    try:
        first, last = (int(part) for part in interval.split("-"))
    except ValueError:
        return False
    years = sorted(years)
    return (
        first in years and last in years and years.index(last) == years.index(first) + 1
    )


def sync_unit_years(
    paths: GridPaths,
    folder: Path,
    unit: Unit,
    *,
    min_years: int = 2,
    priority_intervals: Path | None = None,
) -> list[int]:
    """The years of a unit that can go into detection, checked and recorded.

    A year counts when it has a mosaic and a CHM. The unit's record takes
    these years, so that later steps and ``csdv grid status`` see what was
    processed and not what a catalogue promised.

    Raises:
        SubmitError: If the download has not finished, a CHM is still to be
            made, too few years are left, or the unit's priority interval
            lost one of its dates.
    """
    uid = unit.unit_id
    if not (folder / "downloaded" / uid).exists():
        raise SubmitError(f"{uid}: the download has not finished")
    on_disk = years_on_disk(paths, uid)
    years = [y for y in on_disk if chm_done(paths, uid, y)]
    waiting = [
        y
        for y in on_disk
        if y not in years and _failures(folder, uid, y) < MAX_ROW_FAILURES
    ]
    if waiting:
        raise SubmitError(
            f"{uid}: no CHM yet for {waiting}. Give the queue command again."
        )
    dropped = sorted(set(on_disk) - set(years))
    if dropped:
        logger.warning("%s: left out after failed inference: %s", uid, dropped)
    if len(years) < max(2, min_years):
        raise SubmitError(f"{uid}: {len(years)} usable years, fewer than {min_years}")
    if priority_intervals is not None:
        import csv

        with Path(priority_intervals).open(newline="", encoding="utf-8") as fh:
            lines = [line for line in fh if not line.lstrip().startswith("#")]
        wanted = [
            (row.get("interval") or "").strip()
            for row in csv.DictReader(lines)
            if (row.get("unit") or "").strip() == uid
        ]
        for interval in wanted:
            if not priority_interval_ok(years, interval):
                raise SubmitError(
                    f"{uid}: its priority interval {interval} is not a pair of "
                    f"consecutive years of {years}"
                )
    unit.years = {
        y: unit.years.get(y, {"source": "local", "coverage": 1.0}) for y in years
    }
    for year in dropped:
        unit.skipped_years[year] = "CHM inference failed"
    save_unit(paths, unit)
    return years


def _clock(hours: float) -> str:
    minutes = round(hours * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def plan_queue(paths: GridPaths, units: list[Unit], opts: QueueOptions) -> QueuePlan:
    """Decide what a submission of the queue has to do, from what is on disk."""
    folder = queue_dir(paths, opts.name)
    plan = QueuePlan(folder, units)
    ids = [u.unit_id for u in units]
    # A unit that is exported needs nothing more, whatever else is on disk.
    plan.to_finish = [
        u
        for u in ids
        if not (paths.export_run_dir(u, opts.run_tag) / "export.json").exists()
    ]
    plan.to_download = [
        u for u in plan.to_finish if not (folder / "downloaded" / u).exists()
    ]
    plan.rows = chm_rows(paths, folder, plan.to_finish)

    repo = _env("CSDV_REPO", opts.dry_run)
    log_dir = _env("CSDV_LOG_DIR", opts.dry_run)
    slurm = f"{repo}/scripts/slurm"
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    one = f"--output={log_dir}/%x-%j.out"
    many = f"--output={log_dir}/%x-%A_%a.out"

    if plan.to_download:
        n = len(plan.to_download)
        cmd = ["sbatch", "--parsable", f"--job-name=csdv_qdl_{opts.name}"]
        cmd += [f"--array=1-{n}%{opts.download_throttle}", many]
        plan.download_rows = folder / f"download_{stamp}.csv"
        cmd += [f"{slurm}/run_grid_download.sbatch", str(folder)]
        cmd += [str(plan.download_rows)]
        cmd += ["--start-year", str(opts.start_year)]
        cmd += ["--min-coverage", str(opts.min_coverage)]
        cmd += ["--min-years", str(opts.min_years)]
        plan.jobs.append(Job("download", cmd))

    if plan.to_download or plan.rows:
        seconds = round(opts.worker_hours * 3600)
        for pool in opts.pools:
            for _ in range(opts.workers_per_pool):
                cmd = ["sbatch", "--parsable", f"--job-name=csdv_chmq_{opts.name}"]
                cmd += [*shlex.split(pool), f"--time={_clock(opts.worker_hours)}"]
                cmd += [f"--mem={opts.worker_mem}", one]
                cmd += [f"--export=ALL,CSDV_WORKER_SECONDS={seconds}"]
                if opts.exclude_nodes:
                    cmd.append(f"--exclude={','.join(opts.exclude_nodes)}")
                cmd += ["@@download", f"{slurm}/run_chm_worker.sbatch", str(folder)]
                plan.jobs.append(Job("worker", cmd))

    if plan.to_finish:
        n = len(plan.to_finish)
        cmd = ["sbatch", "--parsable", f"--job-name=csdv_fin_{opts.name}"]
        cmd += [*shlex.split(opts.cpu), f"--array=1-{n}%{opts.finish_throttle}", many]
        plan.finish_rows = folder / f"finish_{stamp}.csv"
        cmd += ["@@download", "@@worker", f"{slurm}/run_grid_finish.sbatch"]
        cmd += [str(folder), str(plan.finish_rows), opts.run_tag]
        cmd += ["--min-years", str(opts.min_years)]
        if opts.checkpoint is not None:
            cmd += ["--checkpoint", str(opts.checkpoint)]
        if opts.priority_intervals is not None:
            cmd += ["--priority-intervals", str(opts.priority_intervals)]
        if not opts.quicklooks:
            cmd.append("--no-quicklooks")
        if opts.detect_args:
            cmd += ["--", *opts.detect_args]
        plan.jobs.append(Job("finish", cmd))
    return plan


def _resolve(command: list[str], ids: dict[str, list[str]]) -> list[str]:
    """Replace the ``@@step`` markers by one ``afterany`` dependency."""
    deps = []
    for arg in command:
        if arg.startswith("@@"):
            deps += [f"afterany:{job}" for job in ids.get(arg[2:], [])]
    out = [arg for arg in command if not arg.startswith("@@")]
    if deps:
        out[2:2] = [f"--dependency={','.join(deps)}"]
    return out


def submit_queue(paths: GridPaths, units: list[Unit], opts: QueueOptions) -> QueuePlan:
    """Plan the queue and submit its jobs (or, with ``dry_run``, only plan).

    Raises:
        SubmitError: If sbatch fails, or a unit exists with another box.
    """
    plan = plan_queue(paths, units, opts)
    ids: dict[str, list[str]] = {}
    if opts.dry_run:
        placeholders = {"download": ["<download>"], "worker": ["<workers>"]}
        present = {job.step for job in plan.jobs}
        shown = {step: jobs for step, jobs in placeholders.items() if step in present}
        for job in plan.jobs:
            job.command = _resolve(job.command, shown)
        return plan

    folder = plan.folder
    for sub in ("downloaded", "claims", "failed"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    for unit in units:
        try:
            save_unit(paths, unit)
        except ValueError as exc:
            raise SubmitError(str(exc)) from exc
    by_id = {u.unit_id: u for u in units}
    lines = ["unit,west,south,east,north"]
    for unit in units:
        box = ",".join(f"{v:.6f}" for v in unit.bbox_4326)
        lines.append(f"{unit.unit_id},{box}")
    (folder / "units.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for job in plan.jobs:
        # The row files are written just before the job that reads them.
        if job.step == "download":
            rows = [lines[0]]
            for uid in plan.to_download:
                box = ",".join(f"{v:.6f}" for v in by_id[uid].bbox_4326)
                rows.append(f"{uid},{box}")
            plan.download_rows.write_text("\n".join(rows) + "\n", encoding="utf-8")
        elif job.step == "finish":
            plan.finish_rows.write_text(
                "unit\n" + "".join(f"{u}\n" for u in plan.to_finish), encoding="utf-8"
            )
        job.command = _resolve(job.command, ids)
        job.job_id = _sbatch(job.command)
        ids.setdefault(job.step, []).append(job.job_id)
        logger.info("%s: job %s", job.step, job.job_id)
    if ids.get("worker"):
        # A worker that finds nothing left cancels its pending siblings, so
        # that the finish array does not wait on a pool that never frees up.
        (folder / "workers.txt").write_text(
            "\n".join(ids["worker"]) + "\n", encoding="utf-8"
        )
    record = folder / "submissions.csv"
    new = not record.exists()
    with record.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("submitted,step,job,units_or_rows\n")
        now = datetime.now().isoformat(timespec="seconds")
        counts = {
            "download": len(plan.to_download),
            "worker": len(plan.rows),
            "finish": len(plan.to_finish),
        }
        for job in plan.jobs:
            fh.write(f"{now},{job.step},{job.job_id},{counts[job.step]}\n")
    return plan


def cpu_options() -> str:
    """``$CSDV_SBATCH_CPU``, the sbatch options of the CPU allocation."""
    value = os.environ.get("CSDV_SBATCH_CPU", "").strip()
    if not value:
        raise SubmitError(
            "$CSDV_SBATCH_CPU is not set. Source scripts/chpc/env.sh first."
        )
    return value
