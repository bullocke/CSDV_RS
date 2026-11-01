"""csdv_core.review.store: the review GeoPackage (sample, labels, redrawn outlines).

One GeoPackage per run tag, by default
``<data_root>/reference/grid_pilot/review_<run_tag>.gpkg``, holds three layers:

``sample``
    The sample frame (:mod:`csdv_core.review.sample`), EPSG:5070.
``interpretations``
    One row per interpreted sample ID (:class:`Interpretation`), no geometry.
``redrawn``
    The interpreter's outline of a sample polygon, where the detected one was
    wrong, keyed by ``sample_id``, EPSG:5070.

The file is small, so every save rewrites it whole into a temporary file that
then replaces the old one. A save first reloads the file if another process
(``csdv review extend`` while the app runs) changed it. Every save is also
appended to ``review_log.csv`` next to the GeoPackage, with the redrawn outline
as WKT, so no label is lost if a write fails.

The label vocabulary follows the ElkinsvilleNE adjudication
(``data/reference/ElkinsvilleNE/supplemental_disturbance/adjudication.csv``),
so the two reference sets can be pooled.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely import make_valid, union_all
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from csdv_core.review.sample import STORE_CRS, STRATUM_DETECTED

logger = logging.getLogger(__name__)

SAMPLE_LAYER = "sample"
LABELS_LAYER = "interpretations"
REDRAWN_LAYER = "redrawn"

#: Is the object a real disturbance? Same values as the ElkinsvilleNE verdicts.
VERDICTS: tuple[str, ...] = ("confirmed", "likely", "uncertain", "rejected")
#: Verdicts that count as a true detection.
TRUE_VERDICTS: tuple[str, ...] = ("confirmed", "likely")
#: Agent or artefact. The artefact classes go with ``rejected``.
CHANGE_TYPES: tuple[str, ...] = (
    "clearcut",
    "partial_harvest",
    "thinning",
    "gap_mortality",
    "wind",
    "fire",
    "insect_disease",
    "flood",
    "clearing_infrastructure",
    "mining",
    "other",
    "artefact_chm_noise",
    "artefact_seam",
    "artefact_shade",
    "phenology_illumination",
)
#: How well the assessed outline matches the disturbance.
OUTLINE_CHOICES: tuple[str, ...] = (
    "about_right",
    "too_large",
    "too_small",
    "wrong_shape",
    "unsure",
)
#: Context flags (any number).
CONTEXT_FLAGS: tuple[str, ...] = (
    "part_of_larger_event",
    "overlaps_other_interval",
    "seam_or_season_problem",
    "unit_edge",
)
#: ``ref_interval`` when the interval cannot be told.
UNSURE = "unsure"

__all__ = [
    "CHANGE_TYPES",
    "CONTEXT_FLAGS",
    "OUTLINE_CHOICES",
    "TRUE_VERDICTS",
    "UNSURE",
    "VERDICTS",
    "Interpretation",
    "ReviewStore",
    "default_store_path",
]


def default_store_path(run_tag: str) -> Path:
    """``<data_root>/reference/grid_pilot/review_<run_tag>.gpkg``."""
    from csdv_core.io.paths import project_paths

    root = project_paths().data_root / "reference" / "grid_pilot"
    return root / f"review_{run_tag}.gpkg"


@dataclass
class Interpretation:
    """The interpreter's reading of one sample polygon.

    Attributes:
        sample_id: Sample ID (``R001``, ``X001``).
        verdict: One of :data:`VERDICTS`.
        change_type: One of :data:`CHANGE_TYPES`, or empty.
        ref_interval: Interval in which the disturbance happened
            (``"2015-2017"``), or :data:`UNSURE`. Empty for rejected objects.
        outline: One of :data:`OUTLINE_CHOICES`, or empty.
        redrawn: True when a redrawn outline is stored for the sample.
        context: Comma-separated :data:`CONTEXT_FLAGS`.
        confidence: 1 (low) to 3 (high), or None.
        notes: Free text.
        interpreter: Who interpreted it.
        saved_at: ISO time of the save (filled in by :meth:`ReviewStore.save`).
        seconds_spent: Time from opening the polygon to saving.
    """

    sample_id: str
    verdict: str
    change_type: str = ""
    ref_interval: str = ""
    outline: str = ""
    redrawn: bool = False
    context: str = ""
    confidence: int | None = None
    notes: str = ""
    interpreter: str = ""
    saved_at: str = ""
    seconds_spent: float | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, not {self.verdict!r}")
        if self.change_type and self.change_type not in CHANGE_TYPES:
            raise ValueError(f"unknown change_type {self.change_type!r}")
        if self.outline and self.outline not in OUTLINE_CHOICES:
            raise ValueError(f"unknown outline {self.outline!r}")
        flags = [f for f in self.context.split(",") if f]
        unknown = set(flags) - set(CONTEXT_FLAGS)
        if unknown:
            raise ValueError(f"unknown context flags {sorted(unknown)}")

    @classmethod
    def from_row(cls, row: pd.Series) -> Interpretation:
        """Rebuild a record from a row of the labels table."""
        values = {}
        for f in fields(cls):
            v = row.get(f.name)
            if v is not None and not isinstance(v, str) and pd.isna(v):
                v = None
            if f.name == "confidence" and v is not None:
                v = int(v)
            elif f.name == "redrawn":
                v = bool(v)
            elif f.name == "seconds_spent" and v is not None:
                v = float(v)
            elif v is None and f.type == "str":
                v = ""
            values[f.name] = v if v is not None else f.default
        return cls(**values)


LABEL_COLUMNS: list[str] = [f.name for f in fields(Interpretation)]


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LABEL_COLUMNS}).astype(
        {"redrawn": "bool"}
    )


def _empty_redrawn() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"sample_id": []}, geometry=[], crs=STORE_CRS)


def _as_polygonal(geometry: BaseGeometry) -> MultiPolygon:
    """Valid MultiPolygon of the polygon parts of a geometry."""
    geometry = make_valid(geometry)
    parts = [
        g
        for g in getattr(geometry, "geoms", [geometry])
        if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty
    ]
    merged = union_all(parts) if parts else Polygon()
    if isinstance(merged, Polygon):
        merged = MultiPolygon([merged]) if not merged.is_empty else MultiPolygon()
    return merged


class ReviewStore:
    """Read and write the review GeoPackage of one run tag."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.log_path = self.path.with_name("review_log.csv")
        self.sample = gpd.GeoDataFrame()
        self.labels = _empty_labels()
        self.redrawn = _empty_redrawn()
        self._mtime: float | None = None
        if self.path.exists():
            self.load()

    # ---- file I/O -------------------------------------------------------

    @classmethod
    def create(cls, path: Path, sample: gpd.GeoDataFrame) -> ReviewStore:
        """Start a store with a sample. Refuses to overwrite an existing file."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists; extend it instead")
        store = cls(path)
        store.sample = sample
        store.write()
        return store

    def load(self) -> None:
        """Read all layers from the file."""
        layers = {name for name, _ in pyogrio.list_layers(self.path)}
        self.sample = gpd.read_file(self.path, layer=SAMPLE_LAYER)
        if LABELS_LAYER in layers:
            labels = pyogrio.read_dataframe(self.path, layer=LABELS_LAYER)
            for c in LABEL_COLUMNS:
                if c not in labels:
                    labels[c] = None
            labels["redrawn"] = labels["redrawn"].fillna(False).astype(bool)
            self.labels = labels[LABEL_COLUMNS]
        else:
            self.labels = _empty_labels()
        if REDRAWN_LAYER in layers:
            self.redrawn = gpd.read_file(self.path, layer=REDRAWN_LAYER)
        else:
            self.redrawn = _empty_redrawn()
        self._mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        """Reload when another process wrote the file since the last read."""
        if self.path.exists() and self.path.stat().st_mtime != self._mtime:
            logger.info("%s changed on disk, reloading", self.path.name)
            self.load()
            return True
        return False

    def write(self) -> None:
        """Write all layers to a temporary file and move it into place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.gpkg")
        tmp.unlink(missing_ok=True)
        self.sample.to_file(tmp, layer=SAMPLE_LAYER, driver="GPKG")
        labels = self.labels.copy()
        labels["confidence"] = pd.array(labels["confidence"], dtype="Int64")
        labels["seconds_spent"] = pd.to_numeric(labels["seconds_spent"])
        pyogrio.write_dataframe(labels, tmp, layer=LABELS_LAYER, driver="GPKG")
        redrawn = self.redrawn
        if redrawn.empty:
            # An empty layer still records the schema and CRS.
            redrawn = gpd.GeoDataFrame(
                {"sample_id": pd.Series(dtype="str")},
                geometry=gpd.GeoSeries([], crs=STORE_CRS),
            )
        redrawn.to_file(
            tmp, layer=REDRAWN_LAYER, driver="GPKG", geometry_type="MultiPolygon"
        )
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    def _log(self, rec: Interpretation, wkt: str) -> None:
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS + ["redrawn_wkt"])
            if new:
                writer.writeheader()
            writer.writerow({**asdict(rec), "redrawn_wkt": wkt})

    # ---- sample ---------------------------------------------------------

    def set_sample(self, sample: gpd.GeoDataFrame) -> None:
        """Replace the sample (after an extension) and write."""
        self.reload_if_changed()
        self.sample = sample
        self.write()

    def add_sample_rows(self, rows: gpd.GeoDataFrame) -> None:
        """Append rows (an opportunistic pick) and write."""
        self.reload_if_changed()
        self.sample = gpd.GeoDataFrame(
            pd.concat([self.sample, rows.to_crs(self.sample.crs)], ignore_index=True),
            geometry="geometry",
            crs=self.sample.crs,
        )
        self.write()

    def ordered_ids(self, include_opportunistic: bool = True) -> list[str]:
        """Sample IDs in review order (``R`` rows first, then ``X`` rows)."""
        s = self.sample
        if not include_opportunistic:
            s = s[s["stratum"] == STRATUM_DETECTED]
        return s.sort_values("order")["sample_id"].tolist()

    def row(self, sample_id: str) -> pd.Series:
        """Sample row of an ID. Raises KeyError for an unknown ID."""
        hit = self.sample[self.sample["sample_id"] == sample_id]
        if hit.empty:
            raise KeyError(sample_id)
        return hit.iloc[0]

    def done_ids(self) -> set[str]:
        return set(self.labels["sample_id"])

    def first_open(self) -> str | None:
        """First ID of the random sample without a saved label, or None."""
        done = self.done_ids()
        for sid in self.ordered_ids(include_opportunistic=False):
            if sid not in done:
                return sid
        return None

    def next_open(self, after: str) -> str | None:
        """First unlabelled ID after ``after`` in review order, wrapping round."""
        ids = self.ordered_ids()
        done = self.done_ids()
        start = ids.index(after) + 1 if after in ids else 0
        for sid in ids[start:] + ids[:start]:
            if sid not in done:
                return sid
        return None

    def progress(self) -> tuple[int, int]:
        """Labelled and total IDs of the random sample."""
        ids = self.ordered_ids(include_opportunistic=False)
        done = self.done_ids()
        return sum(i in done for i in ids), len(ids)

    def progress_by_unit(self) -> pd.DataFrame:
        """Labelled and total sample IDs per unit."""
        s = self.sample[self.sample["stratum"] == STRATUM_DETECTED]
        done = s["sample_id"].isin(self.done_ids())
        table = (
            pd.DataFrame({"unit": s["unit"], "label": s["site_label"], "done": done})
            .groupby(["unit", "label"])["done"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "done", "count": "total"})
        )
        return table.reset_index()

    # ---- labels ---------------------------------------------------------

    def get(self, sample_id: str) -> Interpretation | None:
        hit = self.labels[self.labels["sample_id"] == sample_id]
        return None if hit.empty else Interpretation.from_row(hit.iloc[-1])

    def redrawn_geometry(self, sample_id: str, crs: str = "EPSG:4326"):
        """Redrawn outline of a sample in ``crs``, or None."""
        hit = self.redrawn[self.redrawn["sample_id"] == sample_id]
        if hit.empty:
            return None
        return hit.to_crs(crs).geometry.iloc[0]

    def save(
        self,
        rec: Interpretation,
        geometry: BaseGeometry | None = None,
        geometry_crs: str = "EPSG:4326",
        *,
        clear_geometry: bool = False,
    ) -> None:
        """Save a label, and a redrawn outline when given, then write the file.

        Args:
            rec: The interpretation. ``saved_at`` is set here.
            geometry: Redrawn outline, or None to keep the stored one.
            geometry_crs: CRS of ``geometry``.
            clear_geometry: Remove a stored redrawn outline.
        """
        self.reload_if_changed()
        if rec.sample_id not in set(self.sample["sample_id"]):
            raise KeyError(f"{rec.sample_id} is not in the sample")
        rec.saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        wkt = ""
        keep = self.redrawn["sample_id"] != rec.sample_id
        if geometry is not None and not geometry.is_empty:
            geom = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(STORE_CRS)
            poly = _as_polygonal(geom.iloc[0])
            if not poly.is_empty:
                new = gpd.GeoDataFrame(
                    {"sample_id": [rec.sample_id]}, geometry=[poly], crs=STORE_CRS
                )
                self.redrawn = gpd.GeoDataFrame(
                    pd.concat([self.redrawn[keep], new], ignore_index=True),
                    crs=STORE_CRS,
                )
                wkt = poly.wkt
        elif clear_geometry:
            self.redrawn = self.redrawn[keep].reset_index(drop=True)
        rec.redrawn = bool((self.redrawn["sample_id"] == rec.sample_id).any())
        row = pd.DataFrame([asdict(rec)])[LABEL_COLUMNS]
        others = self.labels[self.labels["sample_id"] != rec.sample_id]
        self.labels = (
            pd.concat([others, row], ignore_index=True) if len(others) else row
        )
        self._log(rec, wkt)
        self.write()
        logger.info("Saved %s (%s)", rec.sample_id, rec.verdict)
