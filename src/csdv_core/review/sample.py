"""csdv_core.review.sample: the review sample of detected disturbance objects.

The sample is drawn from the exported grid runs of one run tag
(``<export_dir>/runs/<run_tag>/<unit>/``). Every unit contributes the same
number of objects (equal allocation), drawn at random from all of its objects.
Each object gets a random key from a generator seeded by the seed and the unit
ID, and the objects with the smallest keys are taken. Growing the sample to a
larger ``n_per_site`` therefore takes the next keys and gives the same set as a
single draw of the larger size.

The rows of each batch are shuffled across units and numbered after the rows
already in the sample (``R001``, ``R002``, ...), so earlier IDs, and the labels
saved against them, never change. ``incl_prob`` is the share of the unit's
objects in the sample (``n_sampled_unit / n_objects_unit``) and is updated for every row of a
unit when the unit's sample grows.

Rows added by hand from the map have ``stratum = "opportunistic"``, IDs
``X001``, ... and no inclusion probability. They are kept as reference
polygons but left out of the scores.

The assessed geometry is the refined 0.6 m outline where the run has one, else
the 19.2 m object (``geom_src``). The sample is stored in EPSG:5070, because
one GeoPackage layer needs one CRS and the units span seven UTM zones.
Areas are computed in each unit's own UTM CRS (``utm_epsg``).
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

logger = logging.getLogger(__name__)

#: CRS of every stored review geometry (CONUS Albers, NAD83).
STORE_CRS = "EPSG:5070"
#: Stratum of the random sample of detected objects.
STRATUM_DETECTED = "detected"
#: Stratum of objects added by hand from the map.
STRATUM_OPPORTUNISTIC = "opportunistic"
#: GeoPackage layers of an exported run.
OBJECTS_LAYER = "objects"
REFINED_LAYER = "refined_outlines"

SAMPLE_COLUMNS: list[str] = [
    "sample_id",
    "order",
    "batch",
    "stratum",
    "run_tag",
    "unit",
    "site_label",
    "object_id",
    "interval",
    "year_before",
    "onset_year",
    "area_ha",
    "assessed_ha",
    "geom_src",
    "growth",
    "follow_up",
    "earlier_share",
    "mean_prob",
    "n_objects_unit",
    "n_sampled_unit",
    "incl_prob",
    "utm_epsg",
]

#: Column types of the sample table. Nullable integers for the columns that
#: opportunistic rows leave empty.
SAMPLE_DTYPES: dict[str, str] = {
    "sample_id": "str",
    "order": "int64",
    "batch": "int64",
    "stratum": "str",
    "run_tag": "str",
    "unit": "str",
    "site_label": "str",
    "object_id": "int64",
    "interval": "str",
    "year_before": "int64",
    "onset_year": "int64",
    "area_ha": "float64",
    "assessed_ha": "float64",
    "geom_src": "str",
    "growth": "bool",
    "follow_up": "bool",
    "earlier_share": "float64",
    "mean_prob": "float64",
    "n_objects_unit": "Int64",
    "n_sampled_unit": "Int64",
    "incl_prob": "float64",
    "utm_epsg": "int64",
}

__all__ = [
    "SAMPLE_COLUMNS",
    "STORE_CRS",
    "STRATUM_DETECTED",
    "STRATUM_OPPORTUNISTIC",
    "ExportedRun",
    "draw_sample",
    "extend_sample",
    "list_runs",
    "opportunistic_row",
    "read_object_geometries",
    "typed",
]


@dataclass
class ExportedRun:
    """One exported unit run and what the review needs to know about it."""

    unit: str
    run_tag: str
    run_dir: Path
    label: str
    years: list[int]
    crs: str

    @property
    def gpkg(self) -> Path:
        return self.run_dir / "disturbances.gpkg"

    @property
    def intervals(self) -> list[str]:
        return [
            f"{a}-{b}" for a, b in zip(self.years[:-1], self.years[1:], strict=True)
        ]

    def objects(self) -> pd.DataFrame:
        """``objects.csv`` of the run."""
        return pd.read_csv(self.run_dir / "objects.csv")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _status_labels(path: Path) -> dict[str, str]:
    """Unit labels from the export's ``status.csv`` (older unit records lack one)."""
    if not path.exists():
        return {}
    table = pd.read_csv(path, usecols=["unit", "label"]).dropna()
    return dict(zip(table["unit"].astype(str), table["label"].astype(str), strict=True))


def list_runs(export_dir: Path, run_tag: str) -> list[ExportedRun]:
    """Every exported unit of ``run_tag`` that has objects, sorted by unit."""
    runs = []
    tag_dir = Path(export_dir) / "runs" / run_tag
    labels = _status_labels(Path(export_dir) / "status.csv")
    for run_dir in sorted(p for p in tag_dir.glob("*") if p.is_dir()):
        gpkg = run_dir / "disturbances.gpkg"
        if not gpkg.exists() or not (run_dir / "objects.csv").exists():
            logger.warning("%s: no disturbances.gpkg or objects.csv, skipped", run_dir)
            continue
        params = _read_json(run_dir / "run_params.json")
        unit_rec = _read_json(run_dir / "unit.json")
        years = params.get("params", {}).get("inputs", {}).get("years") or sorted(
            int(y) for y in unit_rec.get("years", {})
        )
        crs = pyogrio.read_info(gpkg, layer=OBJECTS_LAYER)["crs"]
        runs.append(
            ExportedRun(
                unit=run_dir.name,
                run_tag=run_tag,
                run_dir=run_dir,
                label=unit_rec.get("label") or labels.get(run_dir.name, ""),
                years=[int(y) for y in years],
                crs=crs,
            )
        )
    return runs


def _unit_keys(unit: str, object_ids: np.ndarray, seed: int) -> pd.Series:
    """Random key per object, fixed by the seed, the unit and the object ID."""
    rng = np.random.default_rng([seed, zlib.crc32(unit.encode())])
    order = np.sort(object_ids)
    keys = rng.random(len(order))
    return pd.Series(keys, index=order)


def read_object_geometries(
    run: ExportedRun, object_ids: list[int]
) -> tuple[gpd.GeoSeries, pd.Series]:
    """Assessed geometry of objects: the refined outline, else the 19.2 m object.

    Returns:
        The geometries in the run's UTM CRS and the source of each
        (``"refined"`` or ``"object"``), both indexed by object ID.
    """
    ids = ",".join(str(int(i)) for i in object_ids)
    where = f"object_id IN ({ids})"
    objs = gpd.read_file(run.gpkg, layer=OBJECTS_LAYER, where=where)
    geom = objs.set_index("object_id").geometry
    src = pd.Series("object", index=geom.index)
    layers = {name for name, _ in pyogrio.list_layers(run.gpkg)}
    if REFINED_LAYER in layers:
        ref = gpd.read_file(run.gpkg, layer=REFINED_LAYER, where=where)
        ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
        ref = ref.set_index("object_id").geometry
        geom.loc[ref.index] = ref.values
        src.loc[ref.index] = "refined"
    return geom, src


def _rows_for(
    run: ExportedRun, objects: pd.DataFrame, ids: list[int]
) -> gpd.GeoDataFrame:
    """Sample rows (without IDs) of the given objects of one unit."""
    rows = objects.set_index("object_id").loc[ids].reset_index()
    geom, src = read_object_geometries(run, ids)
    utm = gpd.GeoSeries(geom.loc[ids].values, crs=run.crs)
    return gpd.GeoDataFrame(
        {
            "stratum": STRATUM_DETECTED,
            "run_tag": run.run_tag,
            "unit": run.unit,
            "site_label": run.label,
            "object_id": rows["object_id"].astype(int),
            "interval": rows["interval"].astype(str),
            "year_before": rows["year_before"].astype(int),
            "onset_year": rows["onset_year"].astype(int),
            "area_ha": rows["area_ha"].astype(float),
            "assessed_ha": utm.area.to_numpy() / 1e4,
            "geom_src": src.loc[ids].to_numpy(),
            "growth": rows.get("growth", False),
            "follow_up": rows.get("follow_up", True),
            "earlier_share": rows.get("earlier_share", 0.0),
            "mean_prob": rows.get("mean_prob", np.nan),
            "utm_epsg": utm.crs.to_epsg(),
        },
        geometry=utm.to_crs(STORE_CRS).values,
        crs=STORE_CRS,
    )


def _unit_rows(
    run: ExportedRun,
    n_target: int,
    seed: int,
    exclude_growth: bool,
    already: set[int],
) -> tuple[gpd.GeoDataFrame, int]:
    """New sample rows of one unit up to ``n_target`` objects, and ``n_objects_unit``."""
    objects = run.objects()
    if exclude_growth and "growth" in objects:
        objects = objects[~objects["growth"].astype(bool)]
    keys = _unit_keys(run.unit, objects["object_id"].to_numpy(), seed)
    chosen = keys.sort_values().index[:n_target]
    new_ids = [int(i) for i in chosen if int(i) not in already]
    if not new_ids:
        empty = gpd.GeoDataFrame(columns=SAMPLE_COLUMNS, geometry=[], crs=STORE_CRS)
        return empty, len(objects)
    return _rows_for(run, objects, new_ids), len(objects)


def _finish(
    sample: gpd.GeoDataFrame, new: gpd.GeoDataFrame, totals: dict[str, int], seed: int
) -> gpd.GeoDataFrame:
    """Shuffle a new batch, number it after the sample and update ``incl_prob``."""
    batch = int(sample["batch"].max()) + 1 if len(sample) else 1
    rng = np.random.default_rng([seed, batch])
    new = new.iloc[rng.permutation(len(new))].reset_index(drop=True)
    detected = sample[sample["stratum"] == STRATUM_DETECTED] if len(sample) else sample
    start = len(detected) + 1
    new["sample_id"] = [f"R{i:03d}" for i in range(start, start + len(new))]
    new["order"] = np.arange(start, start + len(new))
    new["batch"] = batch
    out = pd.concat([sample, new], ignore_index=True) if len(sample) else new
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=STORE_CRS)
    det = out["stratum"] == STRATUM_DETECTED
    n_unit = out[det].groupby("unit")["sample_id"].transform("count")
    out.loc[det, "n_sampled_unit"] = n_unit
    out.loc[det, "n_objects_unit"] = out.loc[det, "unit"].map(totals)
    out.loc[det, "incl_prob"] = (
        out.loc[det, "n_sampled_unit"] / out.loc[det, "n_objects_unit"]
    )
    return typed(out)


def typed(sample: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample columns in order with :data:`SAMPLE_DTYPES`."""
    out = sample[SAMPLE_COLUMNS + ["geometry"]].copy()
    for col, dtype in SAMPLE_DTYPES.items():
        if dtype == "Int64":
            out[col] = pd.to_numeric(out[col]).astype("Int64")
        else:
            out[col] = out[col].astype(dtype)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=STORE_CRS)


def extend_sample(
    sample: gpd.GeoDataFrame | None,
    export_dir: Path,
    run_tag: str,
    n_per_site: int,
    *,
    seed: int = 0,
    exclude_growth: bool = False,
) -> gpd.GeoDataFrame:
    """Grow the sample to ``n_per_site`` detected objects per unit.

    Args:
        sample: Current sample, or None to start one.
        export_dir: Export folder with ``runs/<run_tag>/<unit>/``.
        run_tag: Run tag to sample.
        n_per_site: Objects per unit after the call (units with fewer objects
            contribute all of them).
        seed: Seed of the object keys and of the batch shuffle. Keep it the
            same for every batch of one sample.
        exclude_growth: Leave out growth objects (below the MMU, kept only
            because they touch an earlier detection).

    Returns:
        The sample with any new batch appended. Unchanged when every unit
        already has ``n_per_site`` objects.
    """
    if sample is None:
        sample = gpd.GeoDataFrame(columns=SAMPLE_COLUMNS, geometry=[], crs=STORE_CRS)
    runs = list_runs(export_dir, run_tag)
    if not runs:
        raise FileNotFoundError(f"No exported runs under {export_dir}/runs/{run_tag}")
    parts, totals = [], {}
    for run in runs:
        mine = sample[
            (sample["unit"] == run.unit) & (sample["stratum"] == STRATUM_DETECTED)
        ]
        already = {int(i) for i in mine["object_id"]}
        rows, totals[run.unit] = _unit_rows(
            run, n_per_site, seed, exclude_growth, already
        )
        if len(rows):
            parts.append(rows)
        logger.info(
            "%s (%s): %d objects, %d new in sample",
            run.unit,
            run.label,
            totals[run.unit],
            len(rows),
        )
    if not parts:
        logger.info("Every unit already has %d sampled objects", n_per_site)
        return sample
    new = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=STORE_CRS)
    return _finish(sample, new, totals, seed)


def draw_sample(
    export_dir: Path,
    run_tag: str,
    n_per_site: int = 6,
    *,
    seed: int = 0,
    exclude_growth: bool = False,
) -> gpd.GeoDataFrame:
    """Draw a new sample of ``n_per_site`` detected objects per unit."""
    return extend_sample(
        None,
        export_dir,
        run_tag,
        n_per_site,
        seed=seed,
        exclude_growth=exclude_growth,
    )


def opportunistic_row(
    sample: gpd.GeoDataFrame, run: ExportedRun, object_id: int
) -> gpd.GeoDataFrame:
    """Sample row for an object picked by hand (``X001``, ...), not yet added."""
    taken = sample[sample["stratum"] == STRATUM_OPPORTUNISTIC]
    objects = run.objects()
    if int(object_id) not in set(objects["object_id"]):
        raise KeyError(f"{run.unit}: no object {object_id}")
    rows = _rows_for(run, objects, [int(object_id)])
    rows["stratum"] = STRATUM_OPPORTUNISTIC
    rows["sample_id"] = f"X{len(taken) + 1:03d}"
    rows["order"] = 10**6 + len(taken) + 1
    rows["batch"] = 0
    for col in ("n_objects_unit", "n_sampled_unit", "incl_prob"):
        rows[col] = None
    return typed(rows)
