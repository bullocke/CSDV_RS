"""csdv_core.download.naip_auto — NAIP from whichever source holds each year.

The Planetary Computer is the default source, but its archive stops a year or
more behind the NAIP programme. The USDA FPAC image service holds only the
newest collection for each area, which is often the year the Planetary
Computer lacks. This module asks both catalogues which years cover a box and
downloads every year from the Planetary Computer when it has the year, and
from the USDA service otherwise. Both downloaders write the same grid and file
name, so the years line up whichever source each came from. The source of
every year is recorded in its ``.provenance.json`` sidecar.

A year that covers too little of the box is skipped rather than downloaded.
USDA did not finish some states in their scheduled year (Maine, Montana and
Idaho in 2023), and a mosaic that is mostly fill makes a poor CHM year.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import click

from csdv_core.download import naip_pc, naip_rest
from csdv_core.download.naip_pc import _resolve_aoi

logger = logging.getLogger(__name__)

#: Source labels used in plans and logs.
SOURCE_PC = "pc"
SOURCE_REST = "rest"

#: Earliest year considered when no years are given.
DEFAULT_START_YEAR = 2010

#: Smallest covered share of the box for a year to be used. Maine 2023, 52%
#: of the q4506921 cell, is left out at this setting.
DEFAULT_MIN_COVERAGE = 0.9

__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_START_YEAR",
    "SOURCE_PC",
    "SOURCE_REST",
    "YearPlan",
    "YearsPlan",
    "discover_years",
    "download_planned_years",
    "plan_years",
]


@dataclass(frozen=True)
class YearPlan:
    """Where one NAIP year comes from.

    Attributes:
        year: NAIP year.
        source: ``"pc"`` (Planetary Computer) or ``"rest"`` (USDA service).
        coverage: Share of the box covered by the source's quads.
    """

    year: int
    source: str
    coverage: float


@dataclass(frozen=True)
class YearsPlan:
    """Planned and skipped years for one box.

    Attributes:
        years: Years to download, in order.
        skipped: Years left out, with the reason.
    """

    years: tuple[YearPlan, ...]
    skipped: Mapping[int, str]

    @property
    def year_list(self) -> list[int]:
        return [plan.year for plan in self.years]


def discover_years(
    bbox_4326: tuple[float, float, float, float],
) -> tuple[dict[int, float], dict[int, float]]:
    """Ask both catalogues which years cover ``bbox_4326``.

    Returns:
        ``(planetary_computer, usda_service)``, each ``{year: covered share}``.
    """
    pc = naip_pc.available_years(bbox_4326)
    rest = naip_rest.available_years(bbox_4326)
    logger.info(
        "Planetary Computer years: %s",
        ", ".join(f"{y} ({c:.0%})" for y, c in pc.items()) or "none",
    )
    logger.info(
        "USDA service years: %s",
        ", ".join(f"{y} ({c:.0%})" for y, c in rest.items()) or "none",
    )
    return pc, rest


def plan_years(
    pc: Mapping[int, float],
    rest: Mapping[int, float],
    *,
    years: Iterable[int] | None = None,
    start_year: int = DEFAULT_START_YEAR,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> YearsPlan:
    """Choose a source for each year.

    The Planetary Computer is used when it covers at least ``min_coverage`` of
    the box, the USDA service when only it does. Other years are skipped.

    Args:
        pc: Planetary Computer ``{year: covered share}``.
        rest: USDA service ``{year: covered share}``.
        years: Years wanted. Default: every year from ``start_year`` on that
            either catalogue lists.
        start_year: First year considered when ``years`` is None.
        min_coverage: Smallest covered share accepted.

    Returns:
        The plan, with a reason for every skipped year.
    """
    if years is None:
        wanted = sorted(y for y in set(pc) | set(rest) if y >= start_year)
    else:
        wanted = sorted(set(years))
    planned: list[YearPlan] = []
    skipped: dict[int, str] = {}
    for year in wanted:
        pc_cov = pc.get(year, 0.0)
        rest_cov = rest.get(year, 0.0)
        if pc_cov >= min_coverage:
            planned.append(YearPlan(year, SOURCE_PC, pc_cov))
        elif rest_cov >= min_coverage:
            planned.append(YearPlan(year, SOURCE_REST, rest_cov))
        elif year in pc or year in rest:
            best = max(pc_cov, rest_cov)
            skipped[year] = f"covers {best:.0%} of the box (< {min_coverage:.0%})"
        else:
            skipped[year] = "no imagery in either catalogue"
    return YearsPlan(tuple(planned), skipped)


def download_planned_years(
    bbox_4326: tuple[float, float, float, float],
    plan: YearsPlan,
    *,
    site: str,
    out_root: Path | str,
    pad_m: float = 150.0,
    overwrite: bool = False,
) -> tuple[list[Path], list[int]]:
    """Download every planned year from its source.

    Returns:
        ``(written or existing GeoTIFFs, years that failed)``.
    """
    paths: list[Path] = []
    failed: list[int] = []
    for year_plan in plan.years:
        download = (
            naip_pc.download_naip_year
            if year_plan.source == SOURCE_PC
            else naip_rest.download_naip_rest_year
        )
        logger.info(
            "%d: from %s (%.0f%% of the box)",
            year_plan.year,
            "the Planetary Computer" if year_plan.source == SOURCE_PC else "USDA",
            100 * year_plan.coverage,
        )
        try:
            paths.append(
                download(
                    bbox_4326,
                    site=site,
                    year=year_plan.year,
                    out_root=out_root,
                    pad_m=pad_m,
                    overwrite=overwrite,
                )
            )
        except ValueError as exc:
            logger.error("%d: %s", year_plan.year, exc)
            failed.append(year_plan.year)
    return paths, failed


@click.command("naip-auto")
@click.option("--site", required=True, help="Site label; output goes to naip/<site>/.")
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=None,
    help="W S E N in EPSG:4326. Default: sites.yaml center +/- --half-km.",
)
@click.option(
    "--years",
    default=None,
    help="Comma-separated. Default: every available year from --start-year on.",
)
@click.option("--start-year", default=DEFAULT_START_YEAR, show_default=True, type=int)
@click.option(
    "--min-coverage",
    default=DEFAULT_MIN_COVERAGE,
    show_default=True,
    type=float,
    help="Skip years whose quads cover less of the box than this share.",
)
@click.option("--half-km", default=2.5, show_default=True, type=float)
@click.option(
    "--pad-m",
    default=150.0,
    show_default=True,
    type=float,
    help="Buffer added before snapping; covers the CHM edge artefact.",
)
@click.option(
    "--out-root",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Default: $CSDV_DATA_ROOT/naip.",
)
@click.option("--overwrite", is_flag=True, default=False)
def cli(
    site: str,
    bbox: tuple[float, float, float, float] | None,
    years: str | None,
    start_year: int,
    min_coverage: float,
    half_km: float,
    pad_m: float,
    out_root: Path | None,
    overwrite: bool,
) -> None:
    """Download NAIP one mosaic per year, choosing the source for each year.

    Years come from the Planetary Computer when it has them and from the USDA
    image service otherwise (typically the newest year). Needs outbound
    internet. On CHPC run it on a data transfer node.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    bbox_4326 = _resolve_aoi(site, bbox, half_km)
    year_list = (
        [int(token) for token in years.split(",") if token.strip()] if years else None
    )
    if out_root is None:
        from csdv_core.io.paths import project_paths

        out_root = project_paths().data_root / "naip"

    pc, rest = discover_years(bbox_4326)
    plan = plan_years(
        pc, rest, years=year_list, start_year=start_year, min_coverage=min_coverage
    )
    for year, reason in plan.skipped.items():
        logger.warning("%d: skipped, %s", year, reason)
    if not plan.years:
        raise click.ClickException("No NAIP year to download over this box")

    paths, failed = download_planned_years(
        bbox_4326, plan, site=site, out_root=out_root, pad_m=pad_m, overwrite=overwrite
    )
    for path in paths:
        click.echo(str(path))
    # Years asked for by name must all arrive; discovered years may be skipped.
    missing = failed + (sorted(plan.skipped) if year_list else [])
    if missing:
        raise click.ClickException(f"No mosaic written for {missing}")
