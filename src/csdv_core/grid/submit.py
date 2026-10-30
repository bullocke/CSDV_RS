"""csdv_core.grid.submit — submit a unit's whole chain of Slurm jobs.

For one unit the chain is

1. ``download``: ``csdv download naip-auto`` on a data transfer node
   (``run_download_dtn.sbatch``), for the planned years without a mosaic;
2. ``chm``: NAIP-CHM inference, one GPU array task per year without a CHM
   (``run_chm_inference.sbatch``);
3. ``detect``: ``csdv detect-disturbance`` into
   ``disturbance/<unit>/<run_tag>/`` (``run_detect_disturbance.sbatch``);
4. ``chips``: inspection chips into the run folder
   (``run_disturbance_chips.sbatch``);
5. ``export``: the small download set (``run_grid_export.sbatch``).

Each job waits for the one before it (``afterok``), and Slurm cancels the rest
of the chain when a job fails. Steps whose outputs exist are left out, so a
second run tag for a unit submits only steps 3 to 5. Every submission is
appended to the ledger, ``<grid dir>/submissions.csv``.

Submit from a CHPC login node after ``source scripts/chpc/env.sh``; the
catalogue queries that plan the years need outbound internet.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from csdv_core.download.naip_auto import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_START_YEAR,
    discover_years,
    plan_years,
)
from csdv_core.grid.status import (
    ACTIVE_STATES,
    JOB_PREFIX,
    STEPS,
    append_ledger,
    chm_done,
    naip_done,
    read_ledger,
    sacct_states,
)
from csdv_core.grid.units import GridPaths, Unit, load_unit, save_unit

logger = logging.getLogger(__name__)

#: Where the CHM array can run, as sbatch options added to the script's own.
#: ``notchpeak`` keeps the script's ``notchpeak-gpu`` lines. The general
#: kingspeak (titanx) and lonepeak (1080ti) GPUs are older but run the cu126
#: build, and the preemptable routes requeue a task that is cancelled. Not
#: ``granite-gpu-guest``: its Blackwell cards are too new for cu126.
GPU_ROUTES: dict[str, tuple[str, ...]] = {
    "notchpeak": (),
    "kingspeak": (
        "--partition=kingspeak-gpu",
        "--qos=kingspeak-gpu",
        "--account=kingspeak-gpu",
    ),
    "lonepeak": (
        "--partition=lonepeak-gpu",
        "--qos=lonepeak-gpu",
        "--account=lonepeak-gpu",
    ),
    "notchpeak-guest": (
        "--partition=notchpeak-gpu-guest",
        "--qos=notchpeak-gpu-guest",
        "--account=owner-gpu-guest",
        "--requeue",
    ),
    "granite": (
        "--partition=granite-gpu",
        "--qos=granite-gpu-freecycle",
        "--account=wangj",
        "--requeue",
    ),
}

__all__ = [
    "GPU_ROUTES",
    "Job",
    "SubmitError",
    "SubmitOptions",
    "UnitSubmission",
    "plan_unit_years",
    "submit_unit",
]


class SubmitError(RuntimeError):
    """A unit cannot be submitted as asked."""


@dataclass(frozen=True)
class SubmitOptions:
    """Options shared by every unit of one submission.

    Attributes:
        run_tag: Name of the detection run, ``disturbance/<unit>/<run_tag>/``.
        steps: Steps to consider; done steps are still skipped.
        years: Years wanted. Default: the years already planned for the unit,
            or every available year from ``start_year`` on.
        start_year: First year considered when planning.
        min_coverage: Smallest covered share of the box for a year.
        replan: Ask the catalogues again even if the unit has planned years.
        config: ``--config`` YAML for ``detect-disturbance``.
        detect_args: Further ``detect-disturbance`` options.
        quicklooks: Write NAIP and CHM quicklooks in the export step.
        gpu: Key of :data:`GPU_ROUTES` naming where CHM inference runs.
        exclude_nodes: Nodes the CHM tasks must not run on, for example a
            node with a faulty GPU.
        dry_run: Print the jobs without submitting or writing anything.
    """

    run_tag: str = "default"
    steps: tuple[str, ...] = STEPS
    years: tuple[int, ...] | None = None
    start_year: int = DEFAULT_START_YEAR
    min_coverage: float = DEFAULT_MIN_COVERAGE
    replan: bool = False
    config: Path | None = None
    detect_args: tuple[str, ...] = ()
    quicklooks: bool = True
    gpu: str = "notchpeak"
    exclude_nodes: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass
class Job:
    """One sbatch call of a chain."""

    step: str
    command: list[str]
    job_id: str = ""


@dataclass
class UnitSubmission:
    """The jobs of one unit and why other steps were left out."""

    unit: Unit
    run_tag: str
    jobs: list[Job] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    manifest: Path | None = None


def plan_unit_years(unit: Unit, opts: SubmitOptions) -> list[int]:
    """Return the years of this submission, planning them if needed.

    The catalogues are asked only when the unit has no planned years, when
    ``opts.replan`` is set, or when ``opts.years`` names a year not yet
    planned. The plan is kept in ``unit.years``; a run restricted with
    ``opts.years`` adds to it rather than replacing it.

    Raises:
        SubmitError: If a year asked for by name has no usable imagery, or
            fewer than two years are usable.
    """
    wanted = set(opts.years) if opts.years else None
    have = set(unit.years)
    if have and not opts.replan and (wanted is None or wanted <= have):
        years = sorted(wanted or have)
    else:
        pc, rest = discover_years(unit.bbox_4326)
        plan = plan_years(
            pc,
            rest,
            years=opts.years,
            start_year=opts.start_year,
            min_coverage=opts.min_coverage,
        )
        planned = {
            p.year: {"source": p.source, "coverage": round(p.coverage, 3)}
            for p in plan.years
        }
        for year, reason in plan.skipped.items():
            logger.warning("%s %d: skipped, %s", unit.unit_id, year, reason)
        if wanted is None:
            unit.years, unit.skipped_years = planned, dict(plan.skipped)
        else:
            missing = sorted(wanted & set(plan.skipped))
            if missing:
                raise SubmitError(
                    f"{unit.unit_id}: no usable NAIP for {missing} "
                    f"({'; '.join(plan.skipped[y] for y in missing)})"
                )
            unit.years.update(planned)
        years = sorted(planned)
    if len(years) < 2:
        raise SubmitError(
            f"{unit.unit_id}: {len(years)} usable NAIP year(s); detection needs 2"
        )
    return years


def _env(name: str, dry_run: bool) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if dry_run:
        return f"${name}"
    raise SubmitError(f"${name} is not set; source scripts/chpc/env.sh first")


def _dependency(deps: Sequence[str]) -> list[str]:
    deps = [d for d in deps if d]
    if not deps:
        return []
    return [f"--dependency={','.join(deps)}", "--kill-on-invalid-dep=yes"]


def _detect_running(paths: GridPaths, unit_id: str, run_tag: str) -> str | None:
    """Job ID of a queued or running detection for this unit and tag, if any."""
    jobs = [
        r["detect_job"]
        for r in read_ledger(paths)
        if r["unit"] == unit_id and r["run_tag"] == run_tag and r.get("detect_job")
    ]
    if not jobs:
        return None
    states = sacct_states(jobs)
    for job in jobs:
        if set(states.get(job, [])) & ACTIVE_STATES:
            return job
    return None


def _git_commit(repo: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def _sbatch(command: list[str]) -> str:
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0:
        raise SubmitError(
            f"sbatch failed: {out.stderr.strip()}\n  {shlex.join(command)}"
        )
    return out.stdout.strip().split(";")[0]


def submit_unit(paths: GridPaths, unit: Unit, opts: SubmitOptions) -> UnitSubmission:
    """Plan and submit (or with ``dry_run``, only plan) one unit's chain.

    Raises:
        SubmitError: When the run tag exists, a detection with the same tag is
            queued, fewer than two years are usable, or sbatch fails.
    """
    unknown = set(opts.steps) - set(STEPS)
    if unknown:
        raise SubmitError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}")
    existing = load_unit(paths, unit.unit_id)
    if existing is not None:
        if [round(v, 6) for v in existing.bbox_4326] != [
            round(v, 6) for v in unit.bbox_4326
        ]:
            raise SubmitError(
                f"{unit.unit_id} already exists with box {list(existing.bbox_4326)}; "
                "give this area another label"
            )
        unit.years, unit.skipped_years = existing.years, existing.skipped_years
        unit.label = unit.label or existing.label
        unit.notes = unit.notes or existing.notes
    years = plan_unit_years(unit, opts)

    repo = _env("CSDV_REPO", opts.dry_run)
    log_dir = _env("CSDV_LOG_DIR", opts.dry_run)
    slurm = f"{repo}/scripts/slurm"
    uid, tag = unit.unit_id, opts.run_tag
    run_dir = paths.run_dir(uid, tag)
    sub = UnitSubmission(unit, tag)
    wanted = set(opts.steps)

    if "detect" in wanted:
        if run_dir.exists():
            raise SubmitError(
                f"{uid}: run {tag!r} exists ({run_dir}). Pick a new --run-tag, or "
                "use --steps chips,export to redo the later steps."
            )
        queued = None if opts.dry_run else _detect_running(paths, uid, tag)
        if queued:
            raise SubmitError(f"{uid}: run {tag!r} is already queued as job {queued}")
    elif {"chips", "export"} & wanted and not run_dir.exists():
        raise SubmitError(f"{uid}: run {tag!r} does not exist yet ({run_dir})")

    def name(step: str) -> str:
        return f"--job-name={JOB_PREFIX[step]}_{uid}"

    def log(array: bool = False) -> str:
        return f"--output={log_dir}/%x-{'%A_%a' if array else '%j'}.out"

    # 1. NAIP download.
    need_naip = [y for y in years if not naip_done(paths, uid, y)]
    if "download" not in wanted:
        sub.skipped["download"] = "not requested"
    elif not need_naip:
        sub.skipped["download"] = "all years present"
    else:
        west, south, east, north = (f"{v:.6f}" for v in unit.bbox_4326)
        cmd = ["sbatch", "--parsable", name("download"), log()]
        cmd += [f"{slurm}/run_download_dtn.sbatch", "naip-auto", "--site", uid]
        cmd += ["--bbox", west, south, east, north]
        cmd += ["--years", ",".join(map(str, need_naip))]
        cmd += ["--min-coverage", str(opts.min_coverage)]
        sub.jobs.append(Job("download", cmd))

    # 2. CHM inference, one array task per year.
    need_chm = [y for y in years if not chm_done(paths, uid, y)]
    if "chm" not in wanted:
        sub.skipped["chm"] = "not requested"
    elif not need_chm:
        sub.skipped["chm"] = "all years present"
    else:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        sub.manifest = Path(_env("CSDV_MANIFEST_DIR", opts.dry_run)) / (
            f"grid_{uid}_{stamp}.csv"
        )
        cmd = ["sbatch", "--parsable", f"--array=1-{len(need_chm)}", name("chm")]
        cmd += [log(array=True), *GPU_ROUTES[opts.gpu]]
        if opts.exclude_nodes:
            cmd.append(f"--exclude={','.join(opts.exclude_nodes)}")
        cmd += ["@download", f"{slurm}/run_chm_inference.sbatch", str(sub.manifest)]
        sub.jobs.append(Job("chm", cmd))

    # 3. Detection.
    if "detect" in wanted:
        args = list(opts.detect_args)
        if not any(a == "--years" or a.startswith("--years=") for a in args):
            args = ["--years", ",".join(map(str, years)), *args]
        if opts.config is not None:
            args = ["--config", str(opts.config), *args]
        cmd = ["sbatch", "--parsable", name("detect"), log(), "@chm"]
        cmd += [f"{slurm}/run_detect_disturbance.sbatch", uid]
        cmd += ["--output-dir", str(run_dir), *args]
        sub.jobs.append(Job("detect", cmd))
    else:
        sub.skipped["detect"] = "not requested"

    # 4. Chips.
    if "chips" not in wanted:
        sub.skipped["chips"] = "not requested"
    elif (run_dir / "inspection").exists():
        sub.skipped["chips"] = "inspection/ exists"
    else:
        cmd = ["sbatch", "--parsable", name("chips"), log(), "@detect"]
        cmd += [f"{slurm}/run_disturbance_chips.sbatch", str(run_dir)]
        sub.jobs.append(Job("chips", cmd))

    # 5. Export. It runs whether or not the chips succeed.
    if "export" in wanted:
        cmd = ["sbatch", "--parsable", name("export"), log(), "@detect", "@@chips"]
        cmd += [f"{slurm}/run_grid_export.sbatch", uid, tag]
        if not opts.quicklooks:
            cmd.append("--no-quicklooks")
        sub.jobs.append(Job("export", cmd))
    else:
        sub.skipped["export"] = "not requested"

    _resolve_dependencies(sub.jobs, placeholders=opts.dry_run)
    if opts.dry_run:
        return sub

    save_unit(paths, unit)
    if sub.manifest is not None:
        sub.manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = [f"{uid},{y},{paths.naip_year_dir(uid, y)}," for y in need_chm]
        sub.manifest.write_text(
            "site,year,naip_path,crowns_name\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    submitted: dict[str, str] = {}
    try:
        for job in sub.jobs:
            job.command = _with_ids(job.command, submitted)
            job.job_id = _sbatch(job.command)
            submitted[job.step] = job.job_id
            logger.info("%s %s: job %s", uid, job.step, job.job_id)
    finally:
        # Record whatever reached the queue, even when a later sbatch failed.
        if submitted:
            detect_extra = [
                *(["--config", str(opts.config)] if opts.config else []),
                *opts.detect_args,
            ]
            append_ledger(
                paths,
                {
                    "submitted": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "unit": uid,
                    "label": unit.label,
                    "run_tag": tag,
                    "years": ",".join(map(str, years)),
                    "steps": ",".join(s for s in STEPS if s in submitted),
                    **{f"{step}_job": job_id for step, job_id in submitted.items()},
                    "detect_args": (
                        shlex.join(detect_extra) if "detect" in submitted else ""
                    ),
                    "git_commit": _git_commit(repo),
                    "manifest": str(sub.manifest or ""),
                },
            )
    return sub


# Dependency markers in a planned command: "@step" waits for that step's job
# to succeed (afterok), "@@step" only for it to end (afterany). Markers of
# steps that are not in the chain are dropped.
def _resolve_dependencies(jobs: list[Job], *, placeholders: bool) -> None:
    present = {j.step for j in jobs}
    for job in jobs:
        markers = [a for a in job.command if a.startswith("@")]
        kept = [m for m in markers if m.lstrip("@") in present]
        command = [a for a in job.command if not a.startswith("@")]
        if kept:
            command[2:2] = [f"<deps:{' '.join(kept)}>"]
        job.command = command
        if placeholders:
            job.command = _with_ids(job.command, {s: f"<{s}>" for s in present})


def _with_ids(command: list[str], ids: dict[str, str]) -> list[str]:
    out = []
    for arg in command:
        if arg.startswith("<deps:"):
            deps = []
            for marker in arg[len("<deps:") : -1].split():
                kind = "afterany" if marker.startswith("@@") else "afterok"
                deps.append(f"{kind}:{ids[marker.lstrip('@')]}")
            out.extend(_dependency(deps))
        else:
            out.append(arg)
    return out
