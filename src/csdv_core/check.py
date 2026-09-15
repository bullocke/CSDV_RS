"""``csdv check`` preflight subcommand.

Validates the environment, paths, configs, and (optionally) per-site
pipeline state. Designed to run in seconds on a CHPC login node so the
e2e dispatch script can call it as its first step.

Output is a sectioned table. Each row is one of:

    [OK]    ...   passed
    [WARN]  ...   non-fatal issue
    [FAIL]  ...   blocking issue
    [INFO]  ...   informational (e.g. optional import not present)

Exit code is 0 unless at least one ``[FAIL]`` row is emitted, or
``--strict`` is set and any ``[WARN]`` row is emitted.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

logger = logging.getLogger(__name__)

Level = Literal["OK", "WARN", "FAIL", "INFO"]


@dataclass
class CheckRow:
    level: Level
    section: str
    label: str
    detail: str = ""


def _fmt(rows: list[CheckRow]) -> str:
    width = max((len(r.label) for r in rows), default=10)
    lines: list[str] = []
    current_section = ""
    for r in rows:
        if r.section != current_section:
            lines.append("")
            lines.append(f"# {r.section}")
            current_section = r.section
        prefix = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[
            r.level
        ]
        line = f"  {prefix}  {r.label.ljust(width)}"
        if r.detail:
            line += f"  {r.detail}"
        lines.append(line)
    return "\n".join(lines).lstrip()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_python(rows: list[CheckRow]) -> None:
    v = sys.version_info
    label = "python>=3.10"
    detail = f"{v.major}.{v.minor}.{v.micro}"
    level: Level = "OK" if (v.major, v.minor) >= (3, 10) else "FAIL"
    rows.append(CheckRow(level, "environment", label, detail))


_REQUIRED_IMPORTS = (
    "numpy",
    "rasterio",
    "geopandas",
    "skimage",
    "click",
    "pydantic",
    "yaml",
    "scipy",
    # The stand metric and satellite observation caches are parquet. Without an
    # engine the pipeline runs to completion and then fails on the write, which
    # is the worst place to find out.
    "pyarrow",
)
_OPTIONAL_IMPORTS = ("torch", "ee", "wxee", "rpy2", "deepforest")


def _check_imports(rows: list[CheckRow]) -> None:
    for mod in _REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
            rows.append(CheckRow("OK", "environment", f"import {mod}"))
        except Exception as exc:
            rows.append(CheckRow("FAIL", "environment", f"import {mod}", repr(exc)))
    for mod in _OPTIONAL_IMPORTS:
        try:
            importlib.import_module(mod)
            rows.append(CheckRow("OK", "environment", f"import {mod} (optional)"))
        except Exception:
            rows.append(
                CheckRow(
                    "INFO", "environment", f"import {mod} (optional)", "not installed"
                )
            )


def _check_paths(rows: list[CheckRow]) -> None:
    from csdv_core.io.paths import project_paths

    paths = project_paths()
    for label, p in (
        ("data_root", paths.data_root),
        ("results_root", paths.results_root),
        ("cache_root", paths.cache_root),
    ):
        env_var = "CSDV_" + label.upper()
        from_env = os.environ.get(env_var) is not None
        detail = str(p)
        if p.is_symlink():
            detail += f"  -> {os.readlink(p)}"
        if not from_env:
            detail += "  (default; CSDV_" + label.upper() + " not set)"
        if not p.exists():
            rows.append(CheckRow("FAIL", "paths", label, detail + "  MISSING"))
            continue
        if not os.access(p, os.W_OK):
            rows.append(CheckRow("FAIL", "paths", label, detail + "  NOT WRITABLE"))
            continue
        rows.append(CheckRow("OK", "paths", label, detail))


def _check_configs(rows: list[CheckRow]) -> None:
    from csdv_core.config import (
        load_metrics,
        load_site_types,
        load_sites,
        load_stages,
        load_trajectories,
    )

    loaders = (
        ("sites.yaml", load_sites),
        ("site_types.yaml", load_site_types),
        ("metrics.yaml", load_metrics),
        ("stages.yaml", load_stages),
        ("trajectories.yaml", load_trajectories),
    )
    for name, fn in loaders:
        try:
            cfg = fn()
        except Exception as exc:
            rows.append(CheckRow("FAIL", "configs", name, repr(exc)))
            continue
        if name == "sites.yaml":
            detail = f"{len(cfg.sites)} sites"
        elif name == "site_types.yaml":
            detail = f"{len(cfg.site_types)} site types"
        elif name == "metrics.yaml":
            detail = (
                f"{len(cfg.metrics)} metrics, windows={cfg.defaults.window_sizes_m}"
            )
        elif name == "stages.yaml":
            detail = f"{len(cfg.stages)} stages"
        elif name == "trajectories.yaml":
            detail = f"{len(cfg.trajectories)} trajectories"
        else:  # pragma: no cover
            detail = ""
        rows.append(CheckRow("OK", "configs", name, detail))


def _check_site(rows: list[CheckRow], site: str) -> None:
    from csdv_core.config import load_sites

    try:
        sites = load_sites()
    except Exception as exc:
        rows.append(
            CheckRow("FAIL", "site", site, f"sites.yaml failed to load: {exc!r}")
        )
        return
    if site not in sites.sites:
        rows.append(
            CheckRow(
                "FAIL",
                "site",
                site,
                f"not in sites.yaml; known: {sorted(sites.sites)}",
            )
        )
        return
    entry = sites.sites[site]
    detail = (
        f"category={entry.category} state={entry.state} naip.years={entry.naip.years}"
    )
    rows.append(CheckRow("OK", "site", site, detail))


def _check_years(rows: list[CheckRow], site: str, years: list[int]) -> None:
    from csdv_core.config import load_sites

    try:
        entry = load_sites().sites.get(site)
    except Exception:
        return
    if entry is None:
        return
    known = set(entry.naip.years)
    for y in years:
        if y in known:
            rows.append(CheckRow("OK", "years", f"{site}/{y}"))
        else:
            rows.append(
                CheckRow(
                    "WARN",
                    "years",
                    f"{site}/{y}",
                    f"not in sites.yaml naip.years={sorted(known)}",
                )
            )


def _check_window(rows: list[CheckRow], window_m: float) -> None:
    from csdv_core.config import load_metrics

    try:
        ws = list(load_metrics().defaults.window_sizes_m)
    except Exception as exc:
        rows.append(CheckRow("FAIL", "window", str(window_m), repr(exc)))
        return
    if float(window_m) in [float(w) for w in ws]:
        rows.append(CheckRow("OK", "window", f"{window_m} m", f"in defaults={ws}"))
    else:
        rows.append(
            CheckRow(
                "WARN",
                "window",
                f"{window_m} m",
                f"not in metrics.defaults.window_sizes_m={ws}",
            )
        )


def _pipeline_steps(site: str, year: int, window_m: float) -> list[tuple[str, Path]]:
    from csdv_core.io.paths import project_paths

    p = project_paths()
    return [
        ("NAIP", p.naip_dir(site, year)),
        ("NAIP-CHM", p.naip_chm_dir(site, year)),
        ("crowns", p.crowns_dir(site, year) / "crowns.gpkg"),
        ("metrics", p.metrics_dir(site, year, window_m) / "manifest.yaml"),
        (
            "stratification",
            p.stratification_dir(site, window_m) / "site_type.tif",
        ),
        ("stages", p.stages_dir(site, year, window_m) / "stage.tif"),
        (
            "trajectories",
            p.trajectories_dir(site, window_m) / "trajectory.tif",
        ),
    ]


def _check_pipeline_state(
    rows: list[CheckRow], site: str, years: list[int], window_m: float
) -> None:
    for year in years:
        for label, path in _pipeline_steps(site, year, window_m):
            if label in {"stratification", "trajectories"} and year != years[0]:
                # Site-level outputs only reported once.
                continue
            tag = f"{label} {site}/{year}/{int(window_m)}m"
            if label in {"stratification", "trajectories"}:
                tag = f"{label} {site}/{int(window_m)}m"
            if path.exists():
                rows.append(CheckRow("OK", "pipeline-state", tag, str(path)))
            else:
                rows.append(CheckRow("WARN", "pipeline-state", tag, f"missing: {path}"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command("check")
@click.option("--site", default=None, help="Site code to inspect (e.g. SCBI).")
@click.option(
    "--years",
    "years_csv",
    default=None,
    help="Comma-separated NAIP years (e.g. 2014,2018).",
)
@click.option(
    "--window-m",
    type=float,
    default=None,
    help="Analysis window size in meters (e.g. 50).",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Treat WARN rows as failures.",
)
def cli(
    site: str | None,
    years_csv: str | None,
    window_m: float | None,
    strict: bool,
) -> None:
    """Preflight check for CSDV runs."""
    rows: list[CheckRow] = []

    _check_python(rows)
    _check_imports(rows)
    _check_paths(rows)
    _check_configs(rows)

    if site is not None:
        _check_site(rows, site)
    years: list[int] = []
    if years_csv is not None:
        try:
            years = [int(s.strip()) for s in years_csv.split(",") if s.strip()]
        except ValueError as exc:
            rows.append(CheckRow("FAIL", "years", years_csv, repr(exc)))
        else:
            if site is not None:
                _check_years(rows, site, years)
    if window_m is not None:
        _check_window(rows, window_m)
    if site is not None and years and window_m is not None:
        _check_pipeline_state(rows, site, years, window_m)

    click.echo(_fmt(rows))

    n_fail = sum(1 for r in rows if r.level == "FAIL")
    n_warn = sum(1 for r in rows if r.level == "WARN")
    summary = f"\nsummary: {n_fail} FAIL, {n_warn} WARN, {len(rows)} rows"
    click.echo(summary)

    if n_fail or (strict and n_warn):
        raise SystemExit(1)


__all__ = ["cli"]
