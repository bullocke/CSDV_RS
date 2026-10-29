"""csdv_core.grid.status — what has been submitted, what is done, what failed.

Progress is read from disk first: a NAIP year is done when its folder holds a
GeoTIFF, a CHM year when its folder holds a ``*chm*.tif``, a run when its
``summary.json`` exists, chips when ``inspection/index.md`` exists and an
export when its ``export.json`` (written last) exists. Slurm is asked
(``sacct``) only about the jobs, taken from the submission ledger, of the first
step that is not done.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from csdv_core.grid.units import GridPaths, Unit, load_unit

logger = logging.getLogger(__name__)

#: Steps of a unit's chain, in order.
STEPS = ("download", "chm", "detect", "chips", "export")

#: Slurm job name prefix of each step; the job name is ``<prefix>_<unit>``.
JOB_PREFIX = {
    "download": "csdv_dl",
    "chm": "csdv_chm",
    "detect": "csdv_dist",
    "chips": "csdv_chips",
    "export": "csdv_export",
}

#: Columns of the submission ledger.
LEDGER_COLUMNS = (
    "submitted",
    "unit",
    "label",
    "run_tag",
    "years",
    "steps",
    *(f"{step}_job" for step in STEPS),
    "detect_args",
    "git_commit",
    "manifest",
)

#: Slurm states of a job that may still finish.
ACTIVE_STATES = frozenset(
    {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED"}
)

__all__ = [
    "ACTIVE_STATES",
    "JOB_PREFIX",
    "LEDGER_COLUMNS",
    "STEPS",
    "StatusRow",
    "append_ledger",
    "chm_done",
    "collect_status",
    "naip_done",
    "read_ledger",
    "sacct_states",
    "write_status",
]


def naip_done(paths: GridPaths, unit_id: str, year: int) -> bool:
    """Whether the unit's NAIP mosaic for ``year`` exists."""
    return any(paths.naip_year_dir(unit_id, year).glob("*.tif"))


def chm_done(paths: GridPaths, unit_id: str, year: int) -> bool:
    """Whether the unit's CHM for ``year`` exists."""
    return any(paths.chm_year_dir(unit_id, year).glob("*chm*.tif"))


def read_ledger(paths: GridPaths) -> list[dict[str, str]]:
    """Rows of the submission ledger, oldest first (empty if none)."""
    if not paths.ledger.exists():
        return []
    with paths.ledger.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_ledger(paths: GridPaths, row: dict[str, str]) -> None:
    """Append one submission to the ledger, writing the header if new."""
    paths.ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not paths.ledger.exists()
    with paths.ledger.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def sacct_states(job_ids: Iterable[str]) -> dict[str, list[str]]:
    """Slurm states of jobs, ``{job id: [state of each task]}``.

    An array job has one state per task. Returns an empty mapping when
    ``sacct`` is not available (e.g. off CHPC) or fails.
    """
    ids = sorted({j for j in job_ids if j and j[0].isdigit()})
    if not ids or shutil.which("sacct") is None:
        return {}
    cmd = ["sacct", "-X", "-n", "-P", "--format=JobID,State", "-j", ",".join(ids)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("sacct failed: %s", exc)
        return {}
    states: dict[str, list[str]] = {}
    for line in out.splitlines():
        job, _, state = line.partition("|")
        # "123_4" is task 4 of array 123, "123_[5-9]" its pending tasks, and
        # "CANCELLED by 456" a cancellation.
        base = job.split("_")[0]
        state = state.split()[0] if state else "UNKNOWN"
        count = 1
        if "_[" in job:
            count = _array_range_size(job.split("_[", 1)[1].rstrip("]"))
        states.setdefault(base, []).extend([state] * count)
    return states


def _array_range_size(spec: str) -> int:
    """Number of tasks in an array spec such as ``2-5,7%4``."""
    total = 0
    for part in spec.split("%")[0].split(","):
        lo, _, hi = part.partition("-")
        total += int(hi) - int(lo) + 1 if hi else 1
    return total


def _summarize(states: Sequence[str]) -> str:
    counts = Counter(states)
    if len(counts) == 1:
        state, n = next(iter(counts.items()))
        return state if n == 1 else f"{n} {state}"
    return ", ".join(f"{n} {s}" for s, n in counts.most_common())


@dataclass
class StatusRow:
    """Progress of one unit and run tag."""

    unit: str
    label: str
    years: int
    naip: str
    chm: str
    run_tag: str
    detect: str
    chips: str
    export: str
    state: str
    run_dir: str


def _latest(rows: Sequence[dict[str, str]], key: str) -> dict[str, str] | None:
    for row in reversed(rows):
        if row.get(key):
            return row
    return None


def _log_hint(step: str, unit_id: str, job_id: str) -> str:
    log_dir = os.environ.get("CSDV_LOG_DIR", "$CSDV_LOG_DIR")
    task = "_<task>" if step == "chm" else ""
    return f"{log_dir}/{JOB_PREFIX[step]}_{unit_id}-{job_id}{task}.out"


def _unit_rows(
    paths: GridPaths,
    unit: Unit,
    ledger: Sequence[dict[str, str]],
    run_tag: str | None,
) -> list[tuple[StatusRow, str | None, str | None]]:
    """Status rows of one unit, each with the step and job to ask Slurm about."""
    years = unit.year_list
    n_naip = sum(naip_done(paths, unit.unit_id, y) for y in years)
    n_chm = sum(chm_done(paths, unit.unit_id, y) for y in years)
    unit_ledger = [r for r in ledger if r["unit"] == unit.unit_id]

    tags = {r["run_tag"] for r in unit_ledger if r.get("run_tag")}
    run_root = paths.results_root / "disturbance" / unit.unit_id
    if run_root.is_dir():
        tags |= {d.name for d in run_root.iterdir() if (d / "run_params.json").exists()}
    if run_tag is not None:
        tags &= {run_tag}
    out = []
    for tag in sorted(tags) or ["-"]:
        tag_ledger = [r for r in unit_ledger if r["run_tag"] == tag]
        run_dir = paths.run_dir(unit.unit_id, tag)
        done = {
            "download": bool(years) and n_naip == len(years),
            "chm": bool(years) and n_chm == len(years),
            "detect": (run_dir / "summary.json").exists(),
            "chips": (run_dir / "inspection" / "index.md").exists(),
            "export": (
                paths.export_run_dir(unit.unit_id, tag) / "export.json"
            ).exists(),
        }
        mark = {s: "done" if done[s] else "-" for s in STEPS}
        if (
            tag_ledger
            and not done["chips"]
            and not _latest(tag_ledger, "chips_job")
            and "chips" not in tag_ledger[-1]["steps"].split(",")
        ):
            # Chips were left out of the submission (--no-chips).
            done["chips"], mark["chips"] = True, "skipped"
        pending = next((s for s in STEPS if not done[s]), None)
        if tag == "-":
            pending = next((s for s in ("download", "chm") if not done[s]), None)
        job = None
        if pending is not None:
            source = unit_ledger if pending in ("download", "chm") else tag_ledger
            row = _latest(source, f"{pending}_job")
            job = row[f"{pending}_job"] if row else None
        out.append(
            (
                StatusRow(
                    unit=unit.unit_id,
                    label=unit.label,
                    years=len(years),
                    naip=f"{n_naip}/{len(years)}",
                    chm=f"{n_chm}/{len(years)}",
                    run_tag=tag,
                    detect=mark["detect"],
                    chips=mark["chips"],
                    export=mark["export"],
                    state="done" if pending is None else f"{pending}: not submitted",
                    run_dir=str(run_dir) if tag != "-" else "",
                ),
                pending,
                job,
            )
        )
    return out


def collect_status(
    paths: GridPaths,
    *,
    unit_ids: Iterable[str] | None = None,
    run_tag: str | None = None,
    use_sacct: bool = True,
) -> list[StatusRow]:
    """Progress of every unit with a record (or of ``unit_ids``)."""
    if unit_ids is None:
        units_dir = paths.grid_dir / "units"
        unit_ids = (
            sorted(d.name for d in units_dir.iterdir()) if units_dir.is_dir() else []
        )
    ledger = read_ledger(paths)
    staged = []
    for unit_id in unit_ids:
        unit = load_unit(paths, unit_id)
        if unit is None:
            logger.warning("%s has no unit record yet", unit_id)
            continue
        staged.extend(_unit_rows(paths, unit, ledger, run_tag))

    states = sacct_states(job for _, _, job in staged) if use_sacct else {}
    rows = []
    for row, step, job in staged:
        if step is not None and job:
            task_states = states.get(job)
            if task_states is None:
                row.state = f"{step}: job {job}"
            else:
                row.state = f"{step}: {_summarize(task_states)}"
                if not set(task_states) <= ACTIVE_STATES | {"COMPLETED"}:
                    row.state += f" (log {_log_hint(step, row.unit, job)})"
                elif set(task_states) == {"COMPLETED"}:
                    row.state += ", output missing"
        rows.append(row)
    return rows


def write_status(paths: GridPaths, rows: Sequence[StatusRow]) -> Path:
    """Write ``status.csv`` to the grid folder and copy it to the exports."""
    paths.status_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.status_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(StatusRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    paths.export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.status_csv, paths.export_dir / "status.csv")
    return paths.status_csv
