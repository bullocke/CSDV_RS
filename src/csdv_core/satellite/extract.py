"""csdv_core.satellite.extract — reduce satellite archives over stand polygons.

This is the only module in the package that touches the network. It turns a set
of stand polygons and a date range into a tidy table with one row per stand per
scene, which every later step reads from disk.

**The pixel rule differs from the rest of the project, deliberately.**
:mod:`csdv_core.zonal.mask` uses the pixel-centre rule, the one Section 2 of the
classification document specifies: a pixel counts when its centre falls inside
the stand. Earth Engine does not offer that. Its region reduction rasterises the
polygon into per-pixel coverage fractions and uses them as reducer weights, so
``mean()`` is an area-weighted mean and ``mean().unweighted()`` counts every
pixel the polygon touches at all, which is ``all_touched=True``.

At NAIP's 0.6 m, where an acre holds eleven thousand pixels, all three rules
agree to within a tenth of a percent and the choice is cosmetic. At Landsat's
30 m, where a stand in this project can be five pixels, it is not. A 90 by 90 m
stand covers nine pixels of area but touches sixteen; the unweighted mean of
those sixteen is roughly half surrounding forest. So the area-weighted mean is
used, because it is the estimator the polygon actually asks for, and the
unweighted count and the weight sum are carried alongside so that every row
shows what it rests on. The manifest records ``pixel_rule`` so any product on
disk says which rule made it.

One physical limit is worth stating plainly and is not a code problem: Landsat's
effective resolution is roughly one and a half times its 30 m grid, so a
five-pixel stand carries real signal from outside its own boundary no matter how
the reduction is weighted. That is what ``qa.min_area_m2`` is for.

Work is chunked by calendar year. A whole-record request exceeds the interactive
compute deadline, and a year is also the natural cache, resume and retry unit
because the derived product is per year. A year that fails after its retries is
recorded and the run continues; one bad year never kills a forty-year job.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd

from csdv_core.satellite.registry import get_sensor

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    import ee

    from csdv_core.config import SatelliteConfig

logger = logging.getLogger(__name__)

#: Ground area of one Landsat pixel, square metres.
PIXEL_AREA_M2 = 900.0

#: Observation table schema. Declared rather than inferred because Earth Engine
#: returns the union of the properties that happen to be present, so a page in
#: which every stand was fully cloud-masked comes back without the index column
#: at all. Reindexing to this makes that a row of NaN instead of a KeyError.
OBSERVATION_COLUMNS: dict[str, str] = {
    "stand_id": "string",
    "image_id": "string",
    "sensor": "string",
    "date": "string",
    "year": "int16",
    "doy": "int16",
    "time_ms": "int64",
    "wrs_path": "int16",
    "wrs_row": "int16",
    "scene_cloud_cover": "float32",
    "n_pixels": "int32",
    "pixel_weight_sum": "float32",
    "area_m2": "float32",
    "expected_pixels": "float32",
    "coverage_fraction": "float32",
}

#: Error text that means "try again". Anything else is a bug or a config
#: problem, and retrying it just burns four deadlines.
_TRANSIENT_PATTERNS = (
    "computation timed out",
    "deadline",
    "user memory limit exceeded",
    "too many concurrent aggregations",
    "backend error",
    "internal error",
    "service unavailable",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    " 429",
    " 500",
    " 502",
    " 503",
    " 504",
)

#: Failures that mean subdividing the request might help, as opposed to just
#: waiting.
_SUBDIVIDE_PATTERNS = (
    "computation timed out",
    "user memory limit exceeded",
    "deadline",
)

_CHUNK_LADDER = {"year": "half_year", "half_year": "month", "month": None}

__all__ = [
    "OBSERVATION_COLUMNS",
    "PIXEL_AREA_M2",
    "aoi_geometry",
    "date_chunks",
    "fetch_observations",
    "reduce_chunk",
    "stands_to_fc",
]


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def date_chunks(
    start_year: int, end_year: int, *, granularity: str = "year"
) -> list[tuple[str, str, str]]:
    """Split a year range into request chunks.

    Args:
        start_year: First calendar year, inclusive.
        end_year: Last calendar year, inclusive.
        granularity: ``"year"``, ``"half_year"`` or ``"month"``.

    Returns:
        ``(label, start_date, end_date)`` triples, end exclusive.

    Raises:
        ValueError: On an unknown granularity or a reversed range.
    """
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} precedes start_year {start_year}")
    if granularity not in _CHUNK_LADDER:
        raise ValueError(
            f"Unknown granularity {granularity!r}. Known: {sorted(_CHUNK_LADDER)}"
        )

    out: list[tuple[str, str, str]] = []
    for year in range(int(start_year), int(end_year) + 1):
        if granularity == "year":
            out.append((str(year), f"{year}-01-01", f"{year + 1}-01-01"))
        elif granularity == "half_year":
            out.append((f"{year}H1", f"{year}-01-01", f"{year}-07-01"))
            out.append((f"{year}H2", f"{year}-07-01", f"{year + 1}-01-01"))
        else:
            for month in range(1, 13):
                nxt = (year + 1, 1) if month == 12 else (year, month + 1)
                out.append(
                    (
                        f"{year}-{month:02d}",
                        f"{year}-{month:02d}-01",
                        f"{nxt[0]}-{nxt[1]:02d}-01",
                    )
                )
    return out


def _subdivide(granularity: str) -> str | None:
    """Return the next finer chunk granularity, or None at the floor."""
    return _CHUNK_LADDER.get(granularity)


def _is_transient(message: str) -> bool:
    """True when an Earth Engine error is worth retrying."""
    text = f" {message.lower()} "
    if "not signed up" in text or "permission" in text or "credential" in text:
        return False
    return any(pattern in text for pattern in _TRANSIENT_PATTERNS)


def _with_retry(
    fn: Callable[[], Any],
    *,
    label: str,
    max_attempts: int = 4,
    base: float = 2.0,
    cap: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn``, retrying transient Earth Engine failures with backoff.

    Raises:
        Exception: The last error, once attempts are exhausted or the failure
            is not transient. Auth and malformed-expression errors are re-raised
            immediately, because retrying them only wastes deadlines.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below, then re-raised
            last = exc
            if not _is_transient(str(exc)):
                raise
            if attempt == max_attempts:
                break
            delay = min(cap, base**attempt) + random.random()
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1f s",
                label,
                attempt,
                max_attempts,
                str(exc)[:200],
                delay,
            )
            sleep(delay)
    assert last is not None
    raise last


def _normalize_observations(
    frame: pd.DataFrame, index: str, areas: dict[str, float]
) -> pd.DataFrame:
    """Coerce one raw Earth Engine page into the declared observation schema.

    Handles three things the raw page gets wrong for our purposes. The reducer
    names its outputs ``<band>_mean`` and ``<band>_count``, so an ``ndvi`` band
    produces a column literally called ``ndvi_mean`` — the exact name of one of
    the annual metrics. It is renamed to ``ndvi`` here, and the count to
    ``ndvi_n_pixels``, so a per-scene observation can never be mistaken for a
    per-year metric. Missing columns are filled rather than raising. And stand
    area is joined from the geometry rather than trusted from the server.
    """
    frame = frame.copy()
    renames = {
        f"{index}_mean": index,
        f"{index}_count": f"{index}_n_pixels",
        "valid_count": "n_pixels",
        "valid_sum": "pixel_weight_sum",
    }
    frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame})

    columns = {**OBSERVATION_COLUMNS, index: "float32", f"{index}_n_pixels": "int32"}
    for name in columns:
        if name not in frame:
            frame[name] = np.nan

    if "time_ms" in frame:
        stamps = pd.to_datetime(
            pd.to_numeric(frame["time_ms"], errors="coerce"), unit="ms", errors="coerce"
        )
        frame["date"] = stamps.dt.strftime("%Y-%m-%d")
        frame["year"] = stamps.dt.year
        frame["doy"] = stamps.dt.dayofyear

    frame["area_m2"] = frame["stand_id"].map(areas)
    frame["expected_pixels"] = frame["area_m2"] / PIXEL_AREA_M2
    weights = pd.to_numeric(frame["pixel_weight_sum"], errors="coerce")
    frame["coverage_fraction"] = weights / frame["expected_pixels"].replace(0.0, np.nan)

    for name, dtype in columns.items():
        series = frame[name]
        if dtype.startswith(("int", "float")):
            series = pd.to_numeric(series, errors="coerce")
            if dtype.startswith("int"):
                series = series.fillna(0)
        frame[name] = series.astype(dtype)

    frame = frame[list(columns)].reset_index(drop=True)
    check_mask_propagation(frame, index)
    return frame


def check_mask_propagation(frame: pd.DataFrame, index: str) -> int:
    """Warn when the coverage band did not inherit the index's mask.

    A scene on which every pixel was cloud must report zero surviving pixels.
    If it reports the stand's full area instead, the band the coverage fraction
    is summed from is unmasked, every quality gate downstream is inert, and a
    stand glimpsed through a hole in the clouds scores as well as a clear one.

    This is cheap and it has caught the bug once already, so it stays.

    Returns:
        The number of inconsistent rows.
    """
    if frame.empty:
        return 0
    null_index = pd.to_numeric(frame[index], errors="coerce").isna()
    claims_pixels = pd.to_numeric(frame["n_pixels"], errors="coerce").fillna(0) > 0
    bad = int((null_index & claims_pixels).sum())
    if bad:
        logger.error(
            "%d of %d observations have no index value but claim surviving pixels. "
            "The coverage band is not inheriting the index mask, so every quality "
            "gate is inert. See satellite.sensors._with_valid_band.",
            bad,
            len(frame),
        )
    return bad


# --------------------------------------------------------------------------
# Earth Engine side
# --------------------------------------------------------------------------
def stands_to_fc(
    stands: gpd.GeoDataFrame,
    *,
    id_column: str = "stand_id",
    simplify_tolerance_m: float = 1.0,
) -> ee.FeatureCollection:
    """Build an Earth Engine FeatureCollection carrying only the stand id.

    Geometries are handed over in their native projected CRS with
    ``geodesic=False``, so the polygon edges Earth Engine rasterises are the
    same straight lines the shapefile draws. Reprojecting the polygon client
    side would be lossless but pointless; letting Earth Engine treat projected
    coordinates as geodesic would not.

    Only ``id_column`` rides along. Anything else would come back on every row
    of the result for no benefit.

    Args:
        stands: Stand polygons in a projected CRS.
        id_column: Column holding the stand identifier.
        simplify_tolerance_m: Vertex simplification, one thirtieth of a Landsat
            pixel by default. Keeps the serialised expression small. Past
            roughly a thousand stands, upload an Earth Engine asset instead of
            inlining geometry.

    Raises:
        ValueError: If the frame is empty or its CRS is geographic.
    """
    import ee

    if stands.empty:
        raise ValueError("No stands to reduce over")
    if stands.crs is None or stands.crs.is_geographic:
        raise ValueError(
            f"Stands need a projected CRS so the simplification tolerance is in "
            f"metres, got {stands.crs}"
        )

    epsg = f"EPSG:{stands.crs.to_epsg()}"
    features = []
    for _, stand in stands.iterrows():
        geometry = stand.geometry
        if simplify_tolerance_m > 0:
            geometry = geometry.simplify(simplify_tolerance_m, preserve_topology=True)
        features.append(
            ee.Feature(
                ee.Geometry(geometry.__geo_interface__, proj=epsg, geodesic=False),
                {id_column: str(stand[id_column])},
            )
        )
    logger.info("Built a FeatureCollection of %d stands in %s", len(features), epsg)
    return ee.FeatureCollection(features)


def aoi_geometry(stands: gpd.GeoDataFrame, *, buffer_m: float = 1000.0) -> ee.Geometry:
    """Return the buffered bounding rectangle of the stands, for scene filtering."""
    import ee

    minx, miny, maxx, maxy = stands.total_bounds
    epsg = f"EPSG:{stands.crs.to_epsg()}"
    return ee.Geometry.Rectangle(
        [minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m],
        proj=epsg,
        geodesic=False,
    )


def _reducer(index: str) -> ee.Reducer:
    """Mean, count and weight-sum in one pass over each scene.

    ``mean()`` is weighted by pixel coverage, which is the estimator wanted.
    ``count().unweighted()`` gives an honest integer pixel tally. ``sum()`` over
    a constant band of ones gives the effective pixel count, whose ratio to the
    stand's area in pixels is exactly the fraction that survived masking, and
    that one number serves as the cloud gate, the SLC-off gate and the audit of
    the weighting all at once.
    """
    import ee

    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.count().unweighted(), sharedInputs=True)
        .combine(ee.Reducer.sum(), sharedInputs=True)
    )


def reduce_chunk(
    stands_fc: ee.FeatureCollection,
    aoi: ee.Geometry,
    start: str,
    end: str,
    *,
    cfg: SatelliteConfig,
    sensors: Sequence[str],
    indices: Sequence[str],
    tile_scale: int | None = None,
) -> pd.DataFrame:
    """Reduce every scene in a date range over every stand, in one request.

    Rows where the stand was fully masked are kept rather than filtered server
    side, so a year that vanished to cloud says so instead of quietly not being
    there.
    """
    import ee

    from csdv_core.satellite.sensors import build_collection

    collection = build_collection(
        aoi, start, end, cfg=cfg, sensors=sensors, indices=indices
    )
    reducer = _reducer(indices[0])
    scale = float(cfg.extraction.scale_m)
    tiles = int(tile_scale if tile_scale is not None else cfg.extraction.tile_scale)

    def _per_image(image: ee.Image) -> ee.FeatureCollection:
        reduced = image.reduceRegions(
            collection=stands_fc, reducer=reducer, scale=scale, tileScale=tiles
        )
        properties = {
            "image_id": image.get("system:index"),
            "sensor": image.get("sensor"),
            "time_ms": image.get("system:time_start"),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "scene_cloud_cover": image.get("scene_cloud_cover"),
        }
        return reduced.map(lambda feature: feature.set(properties))

    table = ee.FeatureCollection(collection.map(_per_image)).flatten()
    return ee.data.computeFeatures(
        {
            "expression": table,
            "fileFormat": "PANDAS_DATAFRAME",
            "pageSize": int(cfg.extraction.page_size),
            "workloadTag": cfg.earth_engine.workload_tag,
        }
    )


def fetch_observations(
    stands: gpd.GeoDataFrame,
    *,
    cfg: SatelliteConfig | None = None,
    sensors: Sequence[str] | None = None,
    indices: Sequence[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    max_workers: int | None = None,
    initialize: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch per-stand-per-scene observations for a whole date range.

    Args:
        stands: Stand polygons with a ``stand_id`` column, in a projected CRS.
        cfg: Loaded ``satellite.yaml``. Read from config when omitted.
        sensors: Sensor names. Defaults to the configured set.
        indices: Index names. Defaults to the configured set.
        start_year: First year. Defaults to the configured start.
        end_year: Last year. Defaults to the configured end.
        max_workers: Concurrent chunk requests. Defaults to the configured
            value. Set to 1 when debugging a failing chunk.
        initialize: Initialize Earth Engine before the first request.

    Returns:
        ``(observations, provenance)``. Chunks that failed after their retries
        are named in ``provenance["chunks_failed"]`` and are simply absent from
        the frame, so the annual step reports those years as having no
        observations rather than the whole run dying.
    """
    from csdv_core.config import load_satellite

    cfg = cfg if cfg is not None else load_satellite()
    sensors = list(sensors or cfg.extraction.sensors)
    indices = list(indices or cfg.extraction.indices)
    start_year = int(
        start_year if start_year is not None else cfg.extraction.start_year
    )
    end_year = int(end_year if end_year is not None else cfg.extraction.end_year)
    workers = int(max_workers or cfg.earth_engine.max_workers)

    if initialize:
        import ee

        from csdv_core.download._ee import initialize_ee

        initialize_ee(project=cfg.earth_engine.project)
        ee.data.setDeadline(int(cfg.earth_engine.deadline_ms))

    stands_fc = stands_to_fc(
        stands, simplify_tolerance_m=cfg.extraction.simplify_tolerance_m
    )
    aoi = aoi_geometry(stands, buffer_m=cfg.extraction.aoi_buffer_m)
    areas = {
        str(row["stand_id"]): float(row.geometry.area) for _, row in stands.iterrows()
    }

    chunks = date_chunks(start_year, end_year, granularity=cfg.extraction.chunk)
    frames: list[pd.DataFrame] = []
    failed: dict[str, str] = {}

    def _run(chunk: tuple[str, str, str]) -> tuple[str, pd.DataFrame | None, str]:
        label, start, end = chunk
        granularity = cfg.extraction.chunk
        tile_scale = cfg.extraction.tile_scale
        while True:
            try:
                raw = _with_retry(
                    # tile_scale is bound at call time rather than captured, so
                    # a retry after subdivision uses the escalated value that
                    # the except branch below set, not the original.
                    lambda ts=tile_scale: reduce_chunk(
                        stands_fc,
                        aoi,
                        start,
                        end,
                        cfg=cfg,
                        sensors=sensors,
                        indices=indices,
                        tile_scale=ts,
                    ),
                    label=label,
                    max_attempts=cfg.extraction.max_attempts,
                    base=cfg.extraction.backoff_base_s,
                    cap=cfg.extraction.backoff_max_s,
                )
                return label, _normalize_observations(raw, indices[0], areas), ""
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                message = str(exc)
                finer = _subdivide(granularity)
                retriable = any(p in message.lower() for p in _SUBDIVIDE_PATTERNS)
                if not (retriable and finer):
                    logger.error("Chunk %s failed: %s", label, message[:300])
                    return label, None, message
                logger.warning(
                    "Chunk %s exhausted retries at %s granularity; subdividing to %s",
                    label,
                    granularity,
                    finer,
                )
                granularity, tile_scale = finer, min(16, tile_scale * 2)
                sub = date_chunks(
                    int(start[:4]), int(start[:4]), granularity=granularity
                )
                parts = [part for part in sub if start <= part[1] < end]
                pieces = []
                for part in parts:
                    piece_label, piece_start, piece_end = part
                    try:
                        pieces.append(
                            _normalize_observations(
                                _with_retry(
                                    lambda s=piece_start, e=piece_end, ts=tile_scale: (
                                        reduce_chunk(
                                            stands_fc,
                                            aoi,
                                            s,
                                            e,
                                            cfg=cfg,
                                            sensors=sensors,
                                            indices=indices,
                                            tile_scale=ts,
                                        )
                                    ),
                                    label=piece_label,
                                    max_attempts=cfg.extraction.max_attempts,
                                    base=cfg.extraction.backoff_base_s,
                                    cap=cfg.extraction.backoff_max_s,
                                ),
                                indices[0],
                                areas,
                            )
                        )
                    except Exception as sub_exc:  # noqa: BLE001
                        logger.error(
                            "Sub-chunk %s failed: %s", piece_label, str(sub_exc)[:200]
                        )
                if pieces:
                    return label, pd.concat(pieces, ignore_index=True), ""
                return label, None, message

    logger.info(
        "Reducing %d sensors over %d stands in %d %s chunks (%d-%d), %d workers",
        len(sensors),
        len(stands),
        len(chunks),
        cfg.extraction.chunk,
        start_year,
        end_year,
        workers,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for label, frame, error in pool.map(_run, chunks):
            if frame is None:
                failed[label] = error
            elif not frame.empty:
                frames.append(frame)

    observations = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["stand_id", "time_ms"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    )

    provenance = {
        "sensors": sensors,
        "sensor_collections": {
            name: get_sensor(name).collection_id for name in sensors
        },
        "indices": indices,
        "start_year": start_year,
        "end_year": end_year,
        "chunks_requested": [label for label, _, _ in chunks],
        "chunks_failed": failed,
        "n_stands": int(len(stands)),
        "n_observations": int(len(observations)),
        "scale_m": float(cfg.extraction.scale_m),
        "crs": "native scene projection (no reprojection)",
        # Recorded because it differs from csdv_core.zonal.mask's pixel-centre
        # rule. See the module docstring.
        "pixel_rule": "ee_area_weighted",
        "simplify_tolerance_m": float(cfg.extraction.simplify_tolerance_m),
        "stand_crs": str(stands.crs),
        "qa": cfg.qa.model_dump(),
        "earth_engine_project": cfg.earth_engine.project,
    }
    if failed:
        logger.warning(
            "%d of %d chunks failed: %s",
            len(failed),
            len(chunks),
            ", ".join(sorted(failed)),
        )
    logger.info(
        "Fetched %d observations over %d stands", len(observations), len(stands)
    )
    return observations, provenance
