"""On-disk contract for the satellite products.

Two tables, in two places, because they are two different kinds of thing.

The observation table caches an external archive: one row per stand per scene,
under ``data_root/satellite/<site>/``, beside a JSON manifest recording what was
asked for, what came back, and under which rules. It is written once and read
many times, and appending a year is cheap, so a run that dies at 2007 resumes at
2007 rather than refetching two decades.

The annual table is derived: one row per stand per year, under
``results_root/stands/<site>/`` with the rest of the per-stand products. Its
rows for the NAIP years are joined into the stand metric table so the stage
classifier sees them without needing to know where they came from.

Parquet for both, matching ``stand_metrics.parquet``. JSON for the manifest,
matching ``data/naip_chm/<site>/manifest.json``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from csdv_core.satellite.annual import REASON_COLUMN

logger = logging.getLogger(__name__)

OBSERVATIONS_NAME = "landsat_observations.parquet"
MANIFEST_NAME = "landsat_observations.manifest.json"
ANNUAL_NAME = "satellite_annual.parquet"

#: Two observations are the same when they are the same stand in the same
#: scene. Used to make appends idempotent.
_OBSERVATION_KEY = ["stand_id", "image_id"]

__all__ = [
    "ANNUAL_NAME",
    "MANIFEST_NAME",
    "OBSERVATIONS_NAME",
    "join_satellite_metrics",
    "read_annual",
    "read_manifest",
    "read_observations",
    "write_annual",
    "write_observations",
]


def write_observations(
    frame: pd.DataFrame,
    out_dir: Path | str,
    provenance: Mapping[str, Any],
    *,
    append: bool = True,
) -> tuple[Path, Path]:
    """Write the observation cache and its manifest.

    Args:
        frame: Observations from
            :func:`csdv_core.satellite.extract.fetch_observations`.
        out_dir: Destination, usually ``paths.satellite_dir(site)``.
        provenance: The dict returned alongside the frame.
        append: Merge with an existing cache rather than replacing it. Rows are
            deduplicated on ``(stand_id, image_id)``, keeping the new ones, so
            refetching a year that partly failed is safe.

    Returns:
        ``(parquet_path, manifest_path)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / OBSERVATIONS_NAME
    manifest_path = out_dir / MANIFEST_NAME

    combined = frame
    merged_provenance = dict(provenance)
    if append and parquet_path.exists():
        existing = read_observations(parquet_path)
        combined = (
            pd.concat([existing, frame], ignore_index=True)
            .drop_duplicates(subset=_OBSERVATION_KEY, keep="last")
            .sort_values(["stand_id", "time_ms"])
            .reset_index(drop=True)
        )
        logger.info(
            "Appended %d observations to %d existing, %d after deduplication",
            len(frame),
            len(existing),
            len(combined),
        )
        if manifest_path.exists():
            previous = read_manifest(manifest_path)
            merged_provenance = _merge_provenance(previous, merged_provenance)

    merged_provenance["n_observations"] = int(len(combined))
    combined.to_parquet(parquet_path, index=False)
    manifest_path.write_text(
        json.dumps(merged_provenance, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote %d observations to %s", len(combined), parquet_path)
    return parquet_path, manifest_path


def _merge_provenance(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Union the chunk bookkeeping so the manifest describes the whole cache."""
    merged = dict(current)
    merged["chunks_requested"] = sorted(
        set(previous.get("chunks_requested", []))
        | set(current.get("chunks_requested", []))
    )
    # A chunk that succeeded this time is no longer failed.
    failed = {**previous.get("chunks_failed", {}), **current.get("chunks_failed", {})}
    merged["chunks_failed"] = {
        label: reason
        for label, reason in failed.items()
        if label in current.get("chunks_failed", {})
        or label not in current.get("chunks_requested", [])
    }
    merged["start_year"] = min(
        previous.get("start_year", current["start_year"]), current["start_year"]
    )
    merged["end_year"] = max(
        previous.get("end_year", current["end_year"]), current["end_year"]
    )
    return merged


def read_observations(path: Path | str) -> pd.DataFrame:
    """Read the observation cache, accepting a directory or the file itself."""
    path = Path(path)
    if path.is_dir():
        path = path / OBSERVATIONS_NAME
    return pd.read_parquet(path)


def read_manifest(path: Path | str) -> dict[str, Any]:
    """Read the observation manifest, accepting a directory or the file itself."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def write_annual(frame: pd.DataFrame, out_dir: Path | str) -> Path:
    """Write the per-stand-per-year metric table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ANNUAL_NAME
    frame.to_parquet(path, index=False)
    logger.info("Wrote %d stand-year records to %s", len(frame), path)
    return path


def read_annual(path: Path | str) -> pd.DataFrame:
    """Read the per-stand-per-year metric table."""
    path = Path(path)
    if path.is_dir():
        path = path / ANNUAL_NAME
    return pd.read_parquet(path)


def join_satellite_metrics(
    stand_metrics: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Attach the satellite metrics for each imagery year to the stand metrics.

    A left join on ``(stand_id, year)``, so no stand-date is dropped and a year
    with no satellite value simply arrives as NaN. The stage classifier already
    treats a missing metric and a NaN metric identically, lowering the count of
    metrics evaluated rather than counting as a failure, so a gap degrades the
    score's confidence rather than the classification.

    The two reason columns are concatenated, not overwritten. A stand-date can
    legitimately be missing a crown statistic and a satellite metric for
    unrelated causes, and losing either explanation would make a blank
    untraceable.

    Args:
        stand_metrics: The zonal table, one row per stand per imagery date.
        annual: The satellite table, one row per stand per year.
        metrics: Metric columns to bring across. Defaults to every column that
            is neither a key nor support.

    Returns:
        A copy of ``stand_metrics`` with the satellite columns attached.

    Raises:
        ValueError: If the two frames would collide on a metric column, which
            means a metric has been defined twice under one name.
    """
    if annual.empty:
        logger.warning("No satellite metrics to join; returning the stand metrics")
        return stand_metrics.copy()

    keys = ["stand_id", "year"]
    if metrics is None:
        skip = {*keys, "area_m2", REASON_COLUMN}
        metrics = [
            c for c in annual.columns if c not in skip and not c.startswith("sat_")
        ]

    collisions = (set(metrics) & set(stand_metrics.columns)) - set(keys)
    if collisions:
        raise ValueError(
            f"Satellite and stand metric tables both define {sorted(collisions)}. "
            "One metric name must mean one thing."
        )

    carried = [*keys, *metrics, REASON_COLUMN]
    carried += [c for c in annual.columns if c.startswith("sat_")]
    right = annual[[c for c in dict.fromkeys(carried) if c in annual]].copy()

    merged = stand_metrics.copy()
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int64")
    right["year"] = pd.to_numeric(right["year"], errors="coerce").astype("int64")
    merged = merged.merge(right, on=keys, how="left", validate="many_to_one")

    if "unavailable" in merged and REASON_COLUMN in merged:
        merged["unavailable"] = [
            "; ".join(part for part in (a, b) if isinstance(part, str) and part)
            for a, b in zip(
                merged["unavailable"].fillna(""),
                merged[REASON_COLUMN].fillna(""),
                strict=True,
            )
        ]
        merged = merged.drop(columns=[REASON_COLUMN])

    matched = int(merged[metrics[0]].notna().sum()) if metrics else 0
    logger.info(
        "Joined %s to %d stand-dates; %d have a value for %s",
        ", ".join(metrics),
        len(merged),
        matched,
        metrics[0] if metrics else "-",
    )
    return merged
