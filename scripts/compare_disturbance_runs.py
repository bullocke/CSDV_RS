"""Compare detect-disturbance runs of one site, with variant overlay chips.

Each run is a variant of the same site and years (rule, p_boundary, smoothing
scales, minimum mapping unit). The first ``--run`` is the baseline. There is no
reference data, so the comparison is between variants: how much each maps,
where they agree, how large the events are, and chips that draw every
variant's outline over the same NAIP images.

Runs written since detections are kept per interval have
``detection/detected_by_interval.tif``, and a cell can be detected in more
than one interval. Older runs have one detection per cell, read from
``onset_year.tif``. Agreement between variants is judged on the area each maps
in any interval, because a variant that dates the same ground to another
interval has still mapped it.

Outputs, in ``--out``:

- ``summary.csv``: per variant, objects, area detected at least once
  (``detected_ha``) and summed over intervals (``detections_ha``), refined
  0.6 m envelope and opening area, event sizes, and agreement with the
  baseline. ``area_by_interval.csv``: detected area per interval.
- ``overlap_cells.csv``, ``overlap_cells_same_interval.csv`` and
  ``overlap_refined.csv``: share of the row variant's area that the column
  variant also maps (cells in any interval, cells in the same interval,
  refined envelopes).
- ``agreement.csv`` and ``agreement.tif``: area, and a map, of the number of
  variants detecting each cell (any interval).
- ``locations.csv`` and ``locations.gpkg``: places, the union of all
  variants' detections split by interval, with the area each variant maps
  there in any interval and in the place's own interval. Strata follow the
  area mapped in any interval.
- ``chips/<id>.png``: before, after and next NAIP RGB of sampled places, then
  one panel per variant with every outline of every interval, coloured by
  interval, on the last NAIP image.
- ``report.md``: the tables and links to the chips.

NAIP comes from ``$CSDV_DATA_ROOT/naip/<site>/<year>/`` (``--naip-source
local``, on CHPC) or as small windows of the Planetary Computer quads
(``--naip-source pc``, needs internet), cached in ``<out>/naip_cache``.

Example::

    R=analysis/NAIP_disturbance_detection/2_Disturbance_Index/chpc
    python scripts/compare_disturbance_runs.py \\
        --run D6=$R/attempt_1/LincolnNH --run D6s=$R/attempt_2/LincolnNH_D6s \\
        --run p09=$R/attempt_2/LincolnNH_p09 --run sig10=$R/attempt_2/LincolnNH_sig10 \\
        --run mmu1=$R/attempt_2/LincolnNH_mmu1 \\
        --out $R/attempt_2/comparison/LincolnNH --naip-source pc \\
        --point S06=-71.52193,44.18478
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GDAL_CACHEMAX", "256")

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from csdv_core.disturbance._figure_helpers import (
    bare,
    interval_colors,
    transform_extent,
)
from csdv_core.viz.maps import stretch_rgb
from csdv_core.viz.style import INK, MUTED, save_fig, setup_style

logger = logging.getLogger(__name__)

M2_PER_HA = 10_000.0
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
#: Strata of the union places, judged on the area each variant maps there in
#: any interval: every variant, the baseline and some others, the baseline
#: alone, and variants without the baseline.
STRATA = ("all", "some", "baseline_only", "not_baseline")
#: Dark halo behind outlines and labels drawn on NAIP.
HALO_COLOR = "#111111"
#: Outline of the sampled place: white, dashed, with the dark halo.
PLACE_COLOR = "#ffffff"
#: Opacity of the interval fills in the variant panels.
FILL_ALPHA = 0.35
#: Polygons covering at least this share of a panel get an interval label.
LABEL_MIN_SHARE = 0.03
#: Small variant panels in the bottom row of a chip.
MAX_VARIANTS = 5
RGB_PERCENTILES = (2.0, 98.0)
NO_IMAGE_COLOR = "#e3e6e8"
#: Re-search the Planetary Computer before its signed links expire (~1 h).
PC_SEARCH_MAX_AGE_S = 30 * 60


@dataclass
class Variant:
    """One detect-disturbance run.

    Attributes:
        label: Short name used in tables and legends.
        run_dir: Run folder.
        site: Site name.
        params: Run parameters (``run_params.json``).
        years: NAIP years.
        onset: int16 first-detection raster (0 = not detected, -1 = not
            monitored).
        detected: bool ``(T, rows, cols)`` detections per interval (index 0
            is all False). Runs without ``detected_by_interval.tif`` have one
            detection per cell, taken from ``onset``.
        transform: Cell grid transform.
        crs: Grid CRS.
        cell_ha: Cell area (ha).
        objects: Detected objects (cell outlines).
        refined: Refined 0.6 m envelopes (empty when the run did not refine).
        openings: Refined canopy openings (empty when the run did not refine).
        acquisition: Acquisition date per year.
    """

    label: str
    run_dir: Path
    site: str
    params: dict[str, Any]
    years: list[int]
    onset: np.ndarray
    detected: np.ndarray
    transform: Any
    crs: Any
    cell_ha: float
    objects: gpd.GeoDataFrame
    refined: gpd.GeoDataFrame
    openings: gpd.GeoDataFrame
    acquisition: dict[int, str]

    @property
    def ever(self) -> np.ndarray:
        """Cells detected in at least one interval."""
        return self.detected.any(axis=0)

    def interval(self, year: int) -> str:
        """Interval label of the NAIP year that closes it, e.g. ``2016-2018``."""
        index = self.years.index(year)
        return f"{self.years[index - 1]}-{year}"

    def all_outlines(self) -> gpd.GeoDataFrame:
        """Refined envelopes of every interval, with cell outlines for objects
        the run could not refine (column ``refined`` False)."""
        refined_ids = set(self.refined["object_id"])
        cells = self.objects[~self.objects["object_id"].isin(refined_ids)]
        columns = ["object_id", "onset_year", "geometry"]
        parts = [
            self.refined[columns].assign(refined=True),
            cells[columns].assign(refined=False),
        ]
        frame = pd.concat(parts, ignore_index=True)
        frame["onset_year"] = frame["onset_year"].astype(int)
        return gpd.GeoDataFrame(frame, geometry="geometry", crs=self.crs)


def _read_layer(gpkg: Path, layer: str, crs) -> gpd.GeoDataFrame:
    """A GeoPackage layer, or an empty frame when the run did not write it."""
    if layer in set(gpd.list_layers(gpkg)["name"]):
        return gpd.read_file(gpkg, layer=layer)
    columns = ["object_id", "onset_year", "geometry"]
    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)


def _read_detected(run_dir: Path, years: list[int], onset: np.ndarray) -> np.ndarray:
    """Detections per interval, from ``detected_by_interval.tif`` or, for runs
    written before it existed, one detection per cell from the onset map."""
    path = run_dir / "detection" / "detected_by_interval.tif"
    if not path.exists():
        return onset[np.newaxis] == np.asarray(years).reshape(-1, 1, 1)
    with rasterio.open(path) as src:
        stack = src.read().astype(bool)
    if stack.shape[0] != len(years):
        raise SystemExit(f"{path} has {stack.shape[0]} bands for {len(years)} years")
    return stack


def load_variant(label: str, run_dir: Path) -> Variant:
    """Read a run folder's detection rasters, GeoPackage layers and metadata."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "run_params.json").read_text(encoding="utf-8"))
    with rasterio.open(run_dir / "detection" / "onset_year.tif") as src:
        onset = src.read(1)
        transform, crs = src.transform, src.crs
    years = [int(year) for year in summary["years"]]
    gpkg = run_dir / "detection" / "disturbances.gpkg"
    acquisition = {}
    for entry in summary["inputs"]:
        acquisition.setdefault(int(entry["year"]), entry.get("acquisition_date"))
    return Variant(
        label=label,
        run_dir=run_dir,
        site=summary["site"],
        params=params["params"],
        years=years,
        onset=onset,
        detected=_read_detected(run_dir, years, onset),
        transform=transform,
        crs=crs,
        cell_ha=abs(transform.a * transform.e) / M2_PER_HA,
        objects=gpd.read_file(gpkg, layer="objects"),
        refined=_read_layer(gpkg, "refined_outlines", crs),
        openings=_read_layer(gpkg, "refined_openings", crs),
        acquisition=acquisition,
    )


def check_comparable(variants: list[Variant]) -> None:
    """Refuse runs of different sites, years or grids."""
    base = variants[0]
    for variant in variants[1:]:
        problems = []
        if variant.site != base.site:
            problems.append(f"site {variant.site} vs {base.site}")
        if variant.years != base.years:
            problems.append(f"years {variant.years} vs {base.years}")
        if (
            variant.onset.shape != base.onset.shape
            or not variant.transform.almost_equals(base.transform)
        ):
            problems.append("grid")
        if problems:
            raise SystemExit(
                f"{variant.label} is not comparable with {base.label}: "
                + ", ".join(problems)
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _share(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def object_labels(variant: Variant) -> list[tuple[int, np.ndarray, int]]:
    """``(year, labels, n)`` per interval: the 8-connected objects of the
    cells the run detects in that interval."""
    result = []
    for index, year in enumerate(variant.years[1:], start=1):
        labels, count = ndimage.label(variant.detected[index], EIGHT_CONNECTED)
        result.append((year, labels, count))
    return result


def objects_kept(base_labels, other: Variant) -> tuple[int, int]:
    """Baseline objects with at least one cell the other run detects in the
    same interval, and the baseline object count."""
    kept = total = 0
    for index, (_, labels, count) in enumerate(base_labels, start=1):
        if count == 0:
            continue
        hit = np.unique(labels[other.detected[index] & (labels > 0)])
        kept += hit.size
        total += count
    return kept, total


def summarize(variants: list[Variant]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-variant summary table and detected area per interval."""
    base = variants[0]
    base_ever = base.ever
    base_labels = object_labels(base)
    rows, interval_rows = [], []
    for variant in variants:
        ever = variant.ever
        both = ever & base_ever
        same_interval = (variant.detected & base.detected).sum()
        kept, total = objects_kept(base_labels, variant)
        n_objects = sum(count for _, _, count in object_labels(variant))
        event_ha = variant.objects["area_ha"]
        envelope_ha = variant.refined.geometry.area / M2_PER_HA
        rows.append(
            {
                "variant": variant.label,
                "objects": n_objects,
                "detected_ha": ever.sum() * variant.cell_ha,
                "detections_ha": variant.detected.sum() * variant.cell_ha,
                "refined_objects": len(variant.refined),
                "envelope_ha": envelope_ha.sum(),
                "openings_ha": variant.openings.geometry.area.sum() / M2_PER_HA,
                "event_ha_mean": event_ha.mean(),
                "event_ha_median": event_ha.median(),
                "event_ha_p90": event_ha.quantile(0.9),
                "event_ha_max": event_ha.max(),
                "envelope_ha_median": envelope_ha.median(),
                "baseline_area_kept_pct": _share(both.sum(), base_ever.sum()),
                "baseline_area_kept_same_interval_pct": _share(
                    same_interval, base.detected.sum()
                ),
                "area_outside_baseline_pct": _share(
                    (ever & ~base_ever).sum(), ever.sum()
                ),
                "jaccard_vs_baseline": both.sum() / (ever | base_ever).sum(),
                "baseline_objects_kept_pct": _share(kept, total),
            }
        )
        interval_row = {"variant": variant.label}
        for index, year in enumerate(variant.years[1:], start=1):
            interval_row[variant.interval(year)] = (
                variant.detected[index].sum() * variant.cell_ha
            )
        interval_rows.append(interval_row)
    return (
        pd.DataFrame(rows).set_index("variant"),
        pd.DataFrame(interval_rows).set_index("variant"),
    )


def overlap_cells(variants: list[Variant], *, same_interval: bool) -> pd.DataFrame:
    """Share (%) of the row variant's detections the column variant also has.

    Any interval compares the cells each variant detects at least once. Same
    interval compares cell-interval pairs, so a cell detected twice counts
    twice.
    """
    table = {}
    for row in variants:
        table[row.label] = {}
        for column in variants:
            if same_interval:
                both = (row.detected & column.detected).sum()
                total = row.detected.sum()
            else:
                both = (row.ever & column.ever).sum()
                total = row.ever.sum()
            table[row.label][column.label] = _share(both, total)
    return pd.DataFrame(table).T


def overlap_refined(variants: list[Variant]) -> pd.DataFrame:
    """Share (%) of the row variant's refined envelope area inside the column
    variant's envelopes."""
    unions = {v.label: unary_union(list(v.refined.geometry)) for v in variants}
    table = {}
    for row, row_union in unions.items():
        table[row] = {
            column: _share(row_union.intersection(column_union).area, row_union.area)
            for column, column_union in unions.items()
        }
    return pd.DataFrame(table).T


def agreement(variants: list[Variant]) -> tuple[np.ndarray, pd.DataFrame]:
    """Number of variants detecting each cell (any interval), and area per count."""
    count = np.zeros(variants[0].onset.shape, dtype="uint8")
    for variant in variants:
        count += variant.ever.astype("uint8")
    cell_ha = variants[0].cell_ha
    union_cells = (count > 0).sum()
    rows = []
    for n in range(1, len(variants) + 1):
        cells = (count == n).sum()
        rows.append(
            {
                "variants_detecting": n,
                "area_ha": cells * cell_ha,
                "share_of_union_pct": _share(cells, union_cells),
            }
        )
    return count, pd.DataFrame(rows).set_index("variants_detecting")


def write_agreement_raster(count: np.ndarray, variant: Variant, path: Path) -> None:
    """Write the agreement count as a uint8 GeoTIFF (0 = no variant)."""
    profile = {
        "driver": "GTiff",
        "height": count.shape[0],
        "width": count.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": variant.crs,
        "transform": variant.transform,
        "compress": "DEFLATE",
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(count, 1)


# ---------------------------------------------------------------------------
# Places: the union of all variants' detections
# ---------------------------------------------------------------------------


def union_locations(variants: list[Variant]) -> gpd.GeoDataFrame:
    """Places detected by any variant, one per 8-connected union per interval.

    Each place carries the area every variant maps in it in any interval
    (``ha_<label>``) and in the place's own interval (``same_ha_<label>``),
    the interval holding most of the variant's cells there
    (``main_interval_<label>``), the number of variants mapping it in any
    interval and its stratum. A variant that dates the same ground to another
    interval has still mapped it, so the strata follow ``ha_<label>``.
    """
    base = variants[0]
    n_variants = len(variants)
    detection_years = base.years[1:]
    rows = []
    for index, year in enumerate(detection_years, start=1):
        flags = [variant.detected[index] for variant in variants]
        union = np.logical_or.reduce(flags)
        labels, count = ndimage.label(union, EIGHT_CONNECTED)
        if count == 0:
            continue
        cells = np.bincount(labels.ravel(), minlength=count + 1)
        same = np.stack(
            [np.bincount(labels[flag], minlength=count + 1) for flag in flags]
        )
        anywhere = np.stack(
            [np.bincount(labels[v.ever], minlength=count + 1) for v in variants]
        )
        # Cells of each place each variant detects in each interval:
        # (variants, intervals, places + 1).
        by_interval = np.stack(
            [
                np.stack(
                    [
                        np.bincount(labels[v.detected[other]], minlength=count + 1)
                        for other in range(1, len(base.years))
                    ]
                )
                for v in variants
            ]
        )
        parts: dict[int, list] = {}
        for geometry, value in shapes(
            labels.astype("int32"), mask=union, transform=base.transform
        ):
            parts.setdefault(int(value), []).append(shape(geometry))
        for label in range(1, count + 1):
            present = anywhere[:, label] > 0
            n_present = int(present.sum())
            if n_present == n_variants:
                stratum = "all"
            elif not present[0]:
                stratum = "not_baseline"
            elif n_present == 1:
                stratum = "baseline_only"
            else:
                stratum = "some"
            row = {
                "onset_year": year,
                "interval": base.interval(year),
                "area_ha": cells[label] * base.cell_ha,
                "n_variants": n_present,
                "stratum": stratum,
                "geometry": unary_union(parts[label]),
            }
            for v_index, variant in enumerate(variants):
                row[f"ha_{variant.label}"] = anywhere[v_index, label] * base.cell_ha
                row[f"same_ha_{variant.label}"] = same[v_index, label] * base.cell_ha
                counts = by_interval[v_index, :, label]
                row[f"main_interval_{variant.label}"] = (
                    base.interval(detection_years[int(counts.argmax())])
                    if counts.any()
                    else ""
                )
            rows.append(row)
    locations = gpd.GeoDataFrame(rows, crs=base.crs)
    locations = locations.sort_values(
        ["onset_year", "area_ha"], ascending=[True, False]
    )
    locations.index = [f"L{i:04d}" for i in range(1, len(locations) + 1)]
    locations.index.name = "loc_id"
    centroids = locations.geometry.centroid.to_crs("EPSG:4326")
    locations["lon"] = centroids.x
    locations["lat"] = centroids.y
    return locations


def select_locations(
    locations: gpd.GeoDataFrame,
    *,
    n_per_stratum: int,
    min_ha: float,
    seed: int,
    points: dict[str, tuple[float, float]],
) -> gpd.GeoDataFrame:
    """Sample places per stratum, round-robin over intervals, plus named points.

    A named point (lon, lat) selects every place within 30 m of it.
    """
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for stratum in STRATA:
        pool = locations[
            (locations["stratum"] == stratum) & (locations["area_ha"] >= min_ha)
        ]
        queues = {
            interval: list(rng.permutation(group.index))
            for interval, group in pool.groupby("interval")
        }
        while queues and len([p for p in picked if p in pool.index]) < n_per_stratum:
            for interval in sorted(queues):
                if queues[interval]:
                    picked.append(queues[interval].pop())
                if len([p for p in picked if p in pool.index]) >= n_per_stratum:
                    break
            queues = {k: v for k, v in queues.items() if v}
    point_names: dict[str, list[str]] = {}
    for name, (lon, lat) in points.items():
        xy = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(locations.crs)[0]
        near = locations[locations.distance(xy) <= 30.0]
        if near.empty:
            logger.warning("No variant detects anything within 30 m of %s", name)
        for loc_id in near.index:
            point_names.setdefault(loc_id, []).append(name)
    chosen = list(dict.fromkeys(picked + list(point_names)))
    selected = locations.loc[chosen].copy()
    selected["point"] = [", ".join(point_names.get(loc_id, [])) for loc_id in chosen]
    selected["_rank"] = selected["stratum"].map(STRATA.index)
    selected = selected.sort_values(["_rank", "onset_year"])
    return selected.drop(columns="_rank")


# ---------------------------------------------------------------------------
# NAIP readers
# ---------------------------------------------------------------------------


class LocalNaip:
    """NAIP RGB from the site's mosaics under ``$CSDV_DATA_ROOT`` (CHPC)."""

    def __init__(self, run_dir: Path):
        from csdv_core.disturbance._pipeline_inputs import discover_with_saved_grid
        from csdv_core.disturbance.chips import read_rgb_chip
        from csdv_core.disturbance.store import load_run

        saved = load_run(run_dir)
        self._inputs = discover_with_saved_grid(
            saved.site, saved.params, saved.grid, need_naip=True
        )
        self._read = read_rgb_chip

    def rgb(self, year: int, bounds) -> tuple[np.ndarray, tuple]:
        return self._read(self._inputs, year, bounds)


class PlanetaryComputerNaip:
    """NAIP RGB windows from the Planetary Computer quads, cached as GeoTIFFs.

    Each chip is mosaicked onto a 0.6 m grid aligned with the run's pixel grid
    by :func:`csdv_core.download.naip_pc.naip_mosaic`, which reads only the
    part of each quad the chip covers.
    """

    def __init__(self, base: Variant, cache_dir: Path, pixel_m: float = 0.6):
        self._crs = base.crs
        self._origin = (base.transform.c, base.transform.f)
        self._pixel_m = pixel_m
        self._cache_dir = cache_dir
        height, width = base.onset.shape
        right = base.transform.c + width * base.transform.a
        bottom = base.transform.f + height * base.transform.e
        self._bbox_4326 = transform_bounds(
            base.crs, "EPSG:4326", base.transform.c, bottom, right, base.transform.f
        )
        self._items: dict[int, tuple[float, list]] = {}
        logging.getLogger("csdv_core.download.naip_pc").setLevel(logging.WARNING)

    def _year_items(self, year: int) -> list:
        from csdv_core.download.naip_pc import search_naip_items

        cached = self._items.get(year)
        if cached is None or time.monotonic() - cached[0] > PC_SEARCH_MAX_AGE_S:
            items = search_naip_items(self._bbox_4326, year=year)
            self._items[year] = (time.monotonic(), items)
        return self._items[year][1]

    def rgb(self, year: int, bounds) -> tuple[np.ndarray, tuple]:
        from csdv_core.download.naip_pc import naip_mosaic

        x0, y0 = self._origin
        px = self._pixel_m
        left = x0 + np.floor((bounds[0] - x0) / px) * px
        top = y0 - np.floor((y0 - bounds[3]) / px) * px
        width = int(np.ceil((bounds[2] - left) / px))
        height = int(np.ceil((top - bounds[1]) / px))
        path = self._cache_dir / f"{year}_{left:.0f}_{top:.0f}_{width}x{height}.tif"
        if not path.exists():
            chip_4326 = box(
                *transform_bounds(
                    self._crs,
                    "EPSG:4326",
                    left,
                    top - height * px,
                    left + width * px,
                    top,
                )
            )
            items = [
                i for i in self._year_items(year) if box(*i.bbox).intersects(chip_4326)
            ]
            naip_mosaic(
                items,
                path,
                dst_crs=str(self._crs),
                dst_transform=from_origin(left, top, px, px),
                dst_width=width,
                dst_height=height,
            )
        with rasterio.open(path) as src:
            rgb = np.moveaxis(src.read([1, 2, 3]).astype("float32"), 0, -1)
            rgb[src.dataset_mask() == 0] = np.nan
            extent = transform_extent(src.transform, rgb.shape)
        return stretch_rgb(rgb, percentiles=RGB_PERCENTILES), extent


# ---------------------------------------------------------------------------
# Chips
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChipLook:
    """Chip size and resolution.

    Attributes:
        pad_m: Margin around the place on every side (m).
        min_m: Smallest chip side (m).
        max_m: Largest chip side (m); larger places are framed on their centre.
        dpi: PNG resolution.
    """

    pad_m: float = 40.0
    min_m: float = 200.0
    max_m: float = 700.0
    dpi: int = 150


def chip_bounds(geometry, look: ChipLook) -> tuple[float, float, float, float]:
    """Square bounds around a place, between ``min_m`` and ``max_m`` wide."""
    min_x, min_y, max_x, max_y = geometry.bounds
    side = max(max_x - min_x, max_y - min_y) + 2 * look.pad_m
    side = min(max(side, look.min_m), look.max_m)
    centre_x, centre_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    half = side / 2
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)


def _rings(geometry, *, holes: bool = True) -> list[np.ndarray]:
    """Exterior (and, with ``holes``, interior) rings of a polygon or
    multipolygon as xy arrays."""
    polygons = getattr(geometry, "geoms", [geometry])
    rings = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        rings.append(np.asarray(polygon.exterior.coords))
        if holes:
            rings.extend(np.asarray(ring.coords) for ring in polygon.interiors)
    return rings


def _hole_rings(geometry) -> list[np.ndarray]:
    """Interior rings of a polygon or multipolygon as xy arrays."""
    polygons = getattr(geometry, "geoms", [geometry])
    return [
        np.asarray(ring.coords)
        for polygon in polygons
        if polygon.geom_type == "Polygon"
        for ring in polygon.interiors
    ]


def draw_outline(
    ax,
    geometry,
    *,
    color: str,
    width: float,
    dashed: bool,
    halo: bool,
    holes: bool = True,
):
    """Draw a polygon outline, dashed for cell outlines of unrefined objects.
    ``holes=False`` draws the outer edge only."""
    effects = None
    if halo:
        effects = [
            path_effects.Stroke(linewidth=width + 1.8, foreground=HALO_COLOR),
            path_effects.Normal(),
        ]
    style = (0, (3.0, 2.0)) if dashed else "-"
    for ring in _rings(geometry, holes=holes):
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color=color,
            linewidth=width,
            linestyle=style,
            path_effects=effects,
            solid_joinstyle="round",
            solid_capstyle="round",
        )


def fill_polygon(ax, geometry, *, color: str, dashed: bool) -> None:
    """Fill a polygon in a translucent interval colour and outline it with a
    dark halo (dashed for cell outlines of unrefined objects). Holes stay
    open."""
    polygons = getattr(geometry, "geoms", [geometry])
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        rings = [np.asarray(polygon.exterior.coords)]
        rings.extend(np.asarray(ring.coords) for ring in polygon.interiors)
        codes = np.concatenate(
            [
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
                for ring in rings
            ]
        )
        path = MplPath(np.concatenate(rings), codes)
        ax.add_patch(PathPatch(path, facecolor=color, alpha=FILL_ALPHA, linewidth=0))
    draw_outline(
        ax, geometry, color=color, width=1.3, dashed=dashed, halo=True, holes=False
    )
    # Holes (cells or openings left out) get a thin line without a halo, so a
    # patchy object does not bury the image under outlines.
    for ring in _hole_rings(geometry):
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=0.5)


def draw_interval_labels(ax, labels: list[tuple[float, float, str]], side: float):
    """Write the interval labels, stacking those that would overlap (closer
    than about one label width across or one line height down) so
    overlapping intervals stay readable."""
    merged: list[list] = []
    for x, y, text in labels:
        for entry in merged:
            if abs(entry[0] - x) < 0.15 * side and abs(entry[1] - y) < 0.07 * side:
                entry[2] += f"\n{text}"
                break
        else:
            merged.append([x, y, text])
    for x, y, text in merged:
        ax.text(
            x,
            y,
            text,
            color="white",
            fontsize=6.5,
            ha="center",
            va="center",
            linespacing=1.1,
            path_effects=[
                path_effects.withStroke(linewidth=2.0, foreground=HALO_COLOR)
            ],
        )


def short_interval(interval: str) -> str:
    """``2011-2013`` as ``11-13`` for labels inside small panels."""
    first, last = interval.split("-")
    return f"{first[2:]}-{last[2:]}"


def place_status(place: pd.Series, label: str) -> str:
    """What one variant maps in a place: its area there in any interval and
    the part of it in the place's own interval."""
    hectares = place[f"ha_{label}"]
    same_ha = place[f"same_ha_{label}"]
    if hectares <= 0:
        return "place: not mapped"
    if same_ha >= hectares - 1e-9:
        return f"place: {hectares:.2f} ha, all this interval"
    return f"place: {hectares:.2f} ha, {same_ha:.2f} this interval"


def _year_title(role: str, variant: Variant, year: int | None) -> str:
    if year is None:
        return f"{role} · no later image"
    date = variant.acquisition.get(year)
    return f"{role} · {year}" + (f" ({date})" if date else "")


def _show_rgb(ax, reader, year: int | None, bounds) -> None:
    ax.set_facecolor(NO_IMAGE_COLOR)
    if year is not None:
        rgb, extent = reader.rgb(year, bounds)
        ax.imshow(rgb, extent=extent, interpolation="nearest")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    bare(ax, frame=True)


def draw_chip(
    loc_id: str,
    place: pd.Series,
    variants: list[Variant],
    reader,
    out_path: Path,
    look: ChipLook,
) -> Path:
    """One figure: before, after and next RGB of the place's interval with
    the place outlined, then one panel per variant with every outline of
    every interval, coloured by interval, on the last NAIP image."""
    base = variants[0]
    year = int(place["onset_year"])
    index = base.years.index(year)
    before = base.years[index - 1]
    after_next = base.years[index + 1] if index + 1 < len(base.years) else None
    last_year = base.years[-1]
    bounds = chip_bounds(place.geometry, look)
    frame = box(*bounds)
    detection_years = base.years[1:]
    palette = interval_colors(len(detection_years))
    colors = dict(zip(detection_years, palette, strict=True))

    n_variants = len(variants)
    # Panels are square: a big panel spans 5 of 15 columns and a small one 3,
    # so the row heights follow 5.4 : 3.2 and the figure height fits them.
    fig = plt.figure(figsize=(11.0, 7.1))
    grid = fig.add_gridspec(
        2,
        15,
        height_ratios=(5.4, 3.2),
        hspace=0.16,
        wspace=0.1,
        left=0.015,
        right=0.985,
        top=0.865,
        bottom=0.01,
    )
    big_axes = [fig.add_subplot(grid[0, 5 * i : 5 * (i + 1)]) for i in range(3)]
    # Small panels keep one size (3 of 15 columns), filled from the left.
    small_axes = [
        fig.add_subplot(grid[1, 3 * i : 3 * (i + 1)]) for i in range(n_variants)
    ]

    for ax, role, panel_year in zip(
        big_axes, ("Before", "After", "Next"), (before, year, after_next), strict=True
    ):
        _show_rgb(ax, reader, panel_year, bounds)
        ax.set_title(_year_title(role, base, panel_year), fontsize=9, color=INK)
        draw_outline(
            ax,
            place.geometry,
            color=PLACE_COLOR,
            width=1.4,
            dashed=True,
            halo=True,
            holes=False,
        )

    label_area = LABEL_MIN_SHARE * frame.area
    for variant_index, (ax, variant) in enumerate(
        zip(small_axes, variants, strict=True)
    ):
        _show_rgb(ax, reader, last_year, bounds)
        outlines = variant.all_outlines()
        outlines = outlines[outlines.intersects(frame)].sort_values("onset_year")
        mapped = []
        largest: dict[int, Any] = {}
        for _, row in outlines.iterrows():
            geometry = row.geometry.intersection(frame)
            if geometry.is_empty:
                continue
            onset_year = int(row["onset_year"])
            mapped.append(geometry)
            fill_polygon(
                ax, geometry, color=colors[onset_year], dashed=not row["refined"]
            )
            if geometry.area > getattr(largest.get(onset_year), "area", 0.0):
                largest[onset_year] = geometry
        # One label per interval, on its largest polygon in the frame.
        labels = [
            (point.x, point.y, short_interval(variant.interval(onset_year)))
            for onset_year, geometry in sorted(largest.items())
            if geometry.area >= label_area
            for point in [geometry.representative_point()]
        ]
        draw_interval_labels(ax, labels, bounds[2] - bounds[0])
        draw_outline(
            ax,
            place.geometry,
            color=PLACE_COLOR,
            width=0.9,
            dashed=True,
            halo=True,
            holes=False,
        )
        in_view_ha = unary_union(mapped).area / M2_PER_HA if mapped else 0.0
        ax.set_title(
            f"{variant.label}{' (baseline)' if variant_index == 0 else ''}: "
            f"{in_view_ha:.1f} ha in view\n" + place_status(place, variant.label),
            fontsize=7.5,
            color=INK,
        )

    handles = [
        Patch(
            facecolor=color,
            edgecolor=HALO_COLOR,
            linewidth=0.6,
            label=base.interval(detection_year),
        )
        for detection_year, color in colors.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=PLACE_COLOR,
            linewidth=1.4,
            linestyle=(0, (3.0, 2.0)),
            path_effects=[
                path_effects.Stroke(linewidth=3.2, foreground=HALO_COLOR),
                path_effects.Normal(),
            ],
            label="place",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.2,
            linestyle=(0, (3.0, 2.0)),
            label="cell outline (not refined)",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handlelength=2.2,
    )
    title = [
        loc_id,
        place["stratum"].replace("_", " "),
        f"interval {place['interval']}",
        f"union {place['area_ha']:.2f} ha",
        f"{int(place['n_variants'])} of {n_variants} variants",
        f"{place['lat']:.5f}, {place['lon']:.5f}",
    ]
    if place.get("point"):
        title.append(f"point {place['point']}")
    caption = (
        "Bottom row: every mapped polygon of every interval, coloured by "
        f"interval, on NAIP {last_year}"
    )
    fig.suptitle("  |  ".join(title) + "\n" + caption, fontsize=9.5, color=INK, y=0.998)
    save_fig(fig, out_path, dpi=look.dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def md_table(frame: pd.DataFrame, digits: int = 1) -> str:
    """A pipe table, floats rounded to ``digits``, index as the first column.

    Rows are read with ``itertuples`` so integer columns stay integers.
    """

    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:,.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,d}"
        return str(value)

    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, *values in frame.itertuples(name=None):
        lines.append("| " + " | ".join([str(index)] + [fmt(v) for v in values]) + " |")
    return "\n".join(lines)


def stratum_table(locations: gpd.GeoDataFrame, base_label: str) -> pd.DataFrame:
    """Places and union area per stratum, with the places where the baseline
    maps some of the cells in another interval."""
    rows = []
    for stratum in STRATA:
        group = locations[locations["stratum"] == stratum]
        rows.append(
            {
                "stratum": stratum,
                "places": len(group),
                "union_ha": group["area_ha"].sum(),
                "baseline_other_interval": int(
                    (
                        group[f"ha_{base_label}"]
                        > group[f"same_ha_{base_label}"] + 1e-9
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("stratum")


def write_report(
    out_dir: Path,
    variants: list[Variant],
    summary: pd.DataFrame,
    by_interval: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    agreement_table: pd.DataFrame,
    locations: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    chip_paths: dict[str, Path],
) -> Path:
    base = variants[0]
    lines = [
        f"# Variant comparison: {base.site}",
        "",
        f"Years {', '.join(map(str, base.years))}. Baseline: **{base.label}**. "
        "No reference data: every figure compares variants with each other.",
        "",
        "| variant | rule | p_boundary | spatial_sigma_m | mmu_ha | run |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants:
        decide = variant.params["decide"]
        lines.append(
            f"| {variant.label} | {decide['rule']} | {decide['p_boundary']} | "
            f"{decide['spatial_sigma_m']} | {variant.params['mmu']['mmu_ha']} | "
            f"`{variant.run_dir}` |"
        )
    size_cols = [
        "objects",
        "detected_ha",
        "detections_ha",
        "refined_objects",
        "envelope_ha",
        "openings_ha",
        "event_ha_mean",
        "event_ha_median",
        "event_ha_p90",
        "event_ha_max",
        "envelope_ha_median",
    ]
    agree_cols = [
        "baseline_area_kept_pct",
        "baseline_area_kept_same_interval_pct",
        "area_outside_baseline_pct",
        "jaccard_vs_baseline",
        "baseline_objects_kept_pct",
    ]
    lines += [
        "",
        "## Area and event size",
        "",
        "`detected_ha` counts 19.2 m cells detected in at least one interval "
        "after the MMU, and `detections_ha` sums the area of every interval, so "
        "a cell detected twice (for example a strip cut followed by an "
        "overstory removal) counts twice. Runs written before detections were "
        "kept per interval have one detection per cell, so the two are equal. "
        "`envelope_ha` is the refined 0.6 m outline area and `openings_ha` the "
        "canopy openings inside it. Event sizes are per object (cell area).",
        "",
        md_table(summary[size_cols], 2),
        "",
        "## Detected area per interval (ha)",
        "",
        md_table(by_interval, 1),
        "",
        "## Agreement with the baseline",
        "",
        "Area kept: share of the baseline's detected cells the variant also "
        "detects in any interval, then share of the baseline's cell-interval "
        "detections the variant has in the same interval. Outside: share of the "
        "variant's cells the baseline never detects. Objects kept: baseline "
        "objects with at least one cell detected in the same interval.",
        "",
        md_table(summary[agree_cols], 2),
        "",
        "## Pairwise overlap (%)",
        "",
        "Share of the row variant's area that the column variant also maps.",
        "",
    ]
    for title, table in tables.items():
        lines += [f"### {title}", "", md_table(table, 1), ""]
    lines += [
        "## Cells by number of variants detecting them",
        "",
        md_table(agreement_table, 1),
        "",
        "## Places by stratum",
        "",
        "A place is an 8-connected patch of the union of all variants' "
        "detections in one interval. Strata follow the area each variant maps in "
        "the place in any interval, because a variant that dates the same "
        "ground to another interval has still mapped it. "
        "`baseline_other_interval` counts places where the baseline maps some "
        "of the cells in another interval.",
        "",
        md_table(stratum_table(locations, base.label), 1),
        "",
        "## Chips",
        "",
        "Places are the union of all variants' detections, split by interval. "
        "Strata: mapped (in any interval) by all variants, by the baseline and "
        "some others, by the baseline alone, or only by variants other than the "
        "baseline. The top row shows the place's interval with the place "
        "outlined. The bottom row shows, for each variant, every polygon of every "
        "interval on the last NAIP image, filled in the interval's colour and "
        "labelled with its years: refined envelopes, or 19.2 m cells (dashed) "
        "where the run could not refine the object. Each panel title gives the "
        "area the variant maps in the frame (any interval, overlaps counted "
        "once) and the area it maps in the place, in any interval and in the "
        "place's own interval.",
        "",
        "| loc_id | stratum | interval | union ha | variants | point | chip |",
        "|---|---|---|---|---|---|---|",
    ]
    for loc_id, place in selected.iterrows():
        chip = chip_paths.get(loc_id)
        link = f"[png]({chip.relative_to(out_dir)})" if chip else ""
        lines.append(
            f"| {loc_id} | {place['stratum']} | {place['interval']} | "
            f"{place['area_ha']:.2f} | {int(place['n_variants'])} | {place['point']} | {link} |"
        )
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_pairs(values: list[str], what: str) -> dict[str, str]:
    pairs = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            raise SystemExit(f"{what} must look like NAME=VALUE, got {value!r}")
        pairs[name] = rest
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=RUN_DIR, repeated; the first is the baseline.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--naip-source", choices=("pc", "local", "none"), default="pc")
    parser.add_argument("--n-per-stratum", type=int, default=5)
    parser.add_argument(
        "--min-location-ha",
        type=float,
        default=None,
        help="Smallest place to sample. Default: the baseline's MMU.",
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="NAME=LON,LAT: also chip every place within 30 m of this point.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-m", type=float, default=ChipLook.pad_m)
    parser.add_argument("--min-chip-m", type=float, default=ChipLook.min_m)
    parser.add_argument("--max-chip-m", type=float, default=ChipLook.max_m)
    parser.add_argument("--dpi", type=int, default=ChipLook.dpi)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )

    runs = _parse_pairs(args.run, "--run")
    if len(runs) > MAX_VARIANTS:
        raise SystemExit(f"At most {MAX_VARIANTS} runs, got {len(runs)}")
    points = {
        name: tuple(float(v) for v in value.split(","))
        for name, value in _parse_pairs(args.point, "--point").items()
    }
    variants = [load_variant(label, Path(path)) for label, path in runs.items()]
    check_comparable(variants)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, by_interval = summarize(variants)
    summary.to_csv(out_dir / "summary.csv", float_format="%.4f")
    by_interval.to_csv(out_dir / "area_by_interval.csv", float_format="%.4f")
    tables = {
        "Cells, any interval": overlap_cells(variants, same_interval=False),
        "Cells, same interval": overlap_cells(variants, same_interval=True),
        "Refined envelopes": overlap_refined(variants),
    }
    for table, name in zip(
        tables.values(),
        ("overlap_cells", "overlap_cells_same_interval", "overlap_refined"),
        strict=True,
    ):
        table.to_csv(out_dir / f"{name}.csv", float_format="%.2f")
    count, agreement_table = agreement(variants)
    agreement_table.to_csv(out_dir / "agreement.csv", float_format="%.4f")
    write_agreement_raster(count, variants[0], out_dir / "agreement.tif")

    locations = union_locations(variants)
    locations.drop(columns="geometry").to_csv(
        out_dir / "locations.csv", float_format="%.5f"
    )
    locations.to_file(out_dir / "locations.gpkg", layer="locations", driver="GPKG")
    min_ha = args.min_location_ha
    if min_ha is None:
        min_ha = variants[0].params["mmu"]["mmu_ha"]
    selected = select_locations(
        locations,
        n_per_stratum=args.n_per_stratum,
        min_ha=min_ha,
        seed=args.seed,
        points=points,
    )
    logger.info(
        "%d places in the union, %d selected for chips: %s",
        len(locations),
        len(selected),
        selected["stratum"].value_counts().to_dict(),
    )

    chip_paths: dict[str, Path] = {}
    if args.naip_source != "none":
        setup_style()
        look = ChipLook(args.pad_m, args.min_chip_m, args.max_chip_m, args.dpi)
        if args.naip_source == "local":
            reader = LocalNaip(variants[0].run_dir)
        else:
            cache = out_dir / "naip_cache"
            cache.mkdir(exist_ok=True)
            reader = PlanetaryComputerNaip(variants[0], cache)
        chips_dir = out_dir / "chips"
        chips_dir.mkdir(exist_ok=True)
        for number, (loc_id, place) in enumerate(selected.iterrows(), start=1):
            name = f"{loc_id}_{place['stratum']}_{place['interval']}.png"
            chip_paths[loc_id] = draw_chip(
                loc_id, place, variants, reader, chips_dir / name, look
            )
            logger.info("Chip %d of %d: %s", number, len(selected), name)
    report = write_report(
        out_dir,
        variants,
        summary,
        by_interval,
        tables,
        agreement_table,
        locations,
        selected,
        chip_paths,
    )
    logger.info("Report: %s", report)


if __name__ == "__main__":
    main()
