"""csdv_core.satellite.sensors — Landsat Collection 2 sensor definitions.

Five sensors cover 1984 to the present: Landsat 4, 5 and 7 carry TM or ETM+,
Landsat 8 and 9 carry OLI. The band numbering shifts between them. TM and ETM+
put red in ``SR_B3`` and near infrared in ``SR_B4``; OLI puts red in ``SR_B4``
and near infrared in ``SR_B5``. Getting that wrong does not raise anything. It
returns a green-over-red ratio that varies plausibly with season and land
cover, so the whole point of holding the mapping as data in a
:class:`~csdv_core.satellite.registry.SensorSpec` is that a unit test can assert
it directly.

Two decisions here shape every number downstream.

Tier 1 only. Tier 2 scenes have relaxed geometric tolerance, and a stand in
this project can be five Landsat pixels across, so a scene registered a pixel
off samples the wrong ground.

Landsat 7 is kept after the scan line corrector failed in May 2003, even though
roughly a fifth of each scene is striped nodata. Landsat 5 Collection 2 ends in
May 2012 and Landsat 8 begins in March 2013, so between them the archive is
Landsat 7 alone. Excluding the SLC-off era would put a hole through 2012, which
is the first NAIP date at Elkinsville. The gaps carry the fill bit, so they mask
out cleanly, and what protects against a stand sampled from a sliver is the
per-observation coverage gate rather than a per-sensor exclusion.

All ``ee`` imports are lazy, so this module imports on a machine with no
``earthengine-api``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from csdv_core.satellite.registry import SensorSpec, get_sensor, register_sensor

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    import ee

    from csdv_core.config import SatelliteConfig

logger = logging.getLogger(__name__)

#: QA_PIXEL bit positions, Landsat Collection 2. The layout is identical across
#: Landsat 4 to 9. ``cirrus`` is defined for all of them but is always 0 on TM
#: and ETM+, which have no cirrus band.
QA_PIXEL_BITS: dict[str, int] = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "clear": 6,
    "water": 7,
}

#: Bits 8-9 of QA_PIXEL. 0 none, 1 low, 2 medium, 3 high.
CLOUD_CONFIDENCE_SHIFT = 8

#: Collection 2 Level 2 optical surface reflectance scaling. A digital number of
#: 7273 is reflectance 0.0 and 43636 is 1.0.
REFLECTANCE_SCALE = 2.75e-05
REFLECTANCE_OFFSET = -0.2

__all__ = [
    "CLOUD_CONFIDENCE_SHIFT",
    "LANDSAT_4",
    "LANDSAT_5",
    "LANDSAT_7",
    "LANDSAT_8",
    "LANDSAT_9",
    "QA_PIXEL_BITS",
    "REFLECTANCE_OFFSET",
    "REFLECTANCE_SCALE",
    "build_collection",
    "prepare_image",
    "qa_bitmask",
    "scale_reflectance",
    "sensor_collection",
    "sensors_for_year",
]


# --------------------------------------------------------------------------
# The five Landsat sensors
# --------------------------------------------------------------------------
# TM and ETM+ share a band layout; OLI shifts the optical bands up by one.
# Saturation bit index is the native band number minus one, but it is written
# out rather than derived because that rule does not survive contact with
# Sentinel-2.
_TM_BANDS = {
    "blue": "SR_B1",
    "green": "SR_B2",
    "red": "SR_B3",
    "nir": "SR_B4",
    "swir1": "SR_B5",
    "swir2": "SR_B7",
}
_TM_SATURATION = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir1": 4, "swir2": 6}

_OLI_BANDS = {
    "blue": "SR_B2",
    "green": "SR_B3",
    "red": "SR_B4",
    "nir": "SR_B5",
    "swir1": "SR_B6",
    "swir2": "SR_B7",
}
_OLI_SATURATION = {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6}


LANDSAT_4 = register_sensor(
    SensorSpec(
        name="L4",
        platform="Landsat 4 TM",
        collection_id="LANDSAT/LT04/C02/T1_L2",
        bands=_TM_BANDS,
        saturation_bits=_TM_SATURATION,
        first_year=1982,
        last_year=1993,
    )
)

LANDSAT_5 = register_sensor(
    SensorSpec(
        name="L5",
        platform="Landsat 5 TM",
        collection_id="LANDSAT/LT05/C02/T1_L2",
        bands=_TM_BANDS,
        saturation_bits=_TM_SATURATION,
        first_year=1984,
        last_year=2012,
        notes="Collection 2 ends 2012-05-05, so 2012 is covered by Landsat 7 alone.",
    )
)

LANDSAT_7 = register_sensor(
    SensorSpec(
        name="L7",
        platform="Landsat 7 ETM+",
        collection_id="LANDSAT/LE07/C02/T1_L2",
        bands=_TM_BANDS,
        saturation_bits=_TM_SATURATION,
        first_year=1999,
        last_year=2024,
        notes=(
            "Scan line corrector failed 2003-05-31. Roughly 22 percent of each "
            "scene is striped fill, which carries QA_PIXEL bit 0 and masks out "
            "cleanly. Partial coverage of a stand is caught by the coverage "
            "gate, not by excluding the sensor."
        ),
    )
)

LANDSAT_8 = register_sensor(
    SensorSpec(
        name="L8",
        platform="Landsat 8 OLI",
        collection_id="LANDSAT/LC08/C02/T1_L2",
        bands=_OLI_BANDS,
        saturation_bits=_OLI_SATURATION,
        first_year=2013,
        last_year=None,
    )
)

LANDSAT_9 = register_sensor(
    SensorSpec(
        name="L9",
        platform="Landsat 9 OLI-2",
        collection_id="LANDSAT/LC09/C02/T1_L2",
        bands=_OLI_BANDS,
        saturation_bits=_OLI_SATURATION,
        first_year=2021,
        last_year=None,
    )
)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def qa_bitmask(names: Sequence[str]) -> int:
    """Return the QA_PIXEL bitmask selecting every named condition.

    A pixel is usable when ``qa & mask == 0``.

    Args:
        names: QA_PIXEL bit names, keys of :data:`QA_PIXEL_BITS`.

    Returns:
        The combined mask. Masking fill, dilated cloud, cirrus, cloud, cloud
        shadow and snow gives 63.

    Raises:
        KeyError: On an unknown bit name.
    """
    mask = 0
    for name in names:
        if name not in QA_PIXEL_BITS:
            raise KeyError(
                f"Unknown QA_PIXEL bit {name!r}. Known: {sorted(QA_PIXEL_BITS)}"
            )
        mask |= 1 << QA_PIXEL_BITS[name]
    return mask


def scale_reflectance(dn: Any, spec: SensorSpec) -> Any:
    """Apply a sensor's reflectance scale and offset to raw digital numbers.

    Works on NumPy arrays, so the constants used by the Earth Engine path are
    exercised by the unit suite even though the Earth Engine call is not.
    """
    return np.asarray(dn, dtype=np.float64) * spec.reflectance_scale + (
        spec.reflectance_offset
    )


def sensors_for_year(year: int, *, names: Sequence[str] | None = None) -> list[str]:
    """Return the sensors acquiring in ``year``, in registration order.

    Args:
        year: Calendar year.
        names: Restrict to these sensors. Defaults to every registered sensor.

    Returns:
        Sorted sensor names. Empty when nothing was flying, which is a valid
        answer, not an error.
    """
    from csdv_core.satellite.registry import list_sensors

    candidates = list(names) if names is not None else list_sensors()
    out = []
    for name in candidates:
        spec = get_sensor(name)
        if spec.first_year <= year and (
            spec.last_year is None or year <= spec.last_year
        ):
            out.append(name)
    return sorted(out)


# --------------------------------------------------------------------------
# Earth Engine side
# --------------------------------------------------------------------------
def prepare_image(image: ee.Image, spec: SensorSpec, cfg: SatelliteConfig) -> ee.Image:
    """Mask and harmonize one scene into named surface reflectance bands.

    Masking runs in three passes: the QA_PIXEL conditions named in
    ``cfg.qa.mask_bits``, optionally a cloud-confidence guard, and per-band
    saturation from QA_RADSAT. Reflectance outside the physical range is then
    dropped, which catches the residual scaling artefacts Collection 2 leaves at
    the extremes.

    Args:
        image: A raw Collection 2 Level 2 scene.
        spec: The sensor it came from.
        cfg: Loaded ``satellite.yaml``.

    Returns:
        An image whose bands carry harmonized names (``red``, ``nir``, ...),
        masked, scaled, and tagged with ``sensor`` and the scene properties the
        observation table records.
    """
    import ee

    qa = image.select(spec.qa_band)
    mask = qa.bitwiseAnd(qa_bitmask(cfg.qa.mask_bits)).eq(0)
    if cfg.qa.mask_cloud_confidence_medium:
        mask = mask.And(qa.rightShift(CLOUD_CONFIDENCE_SHIFT).bitwiseAnd(3).lt(2))

    harmonized_names = sorted(spec.bands)
    native_names = [spec.bands[name] for name in harmonized_names]
    optical = (
        image.select(native_names, harmonized_names)
        .multiply(spec.reflectance_scale)
        .add(spec.reflectance_offset)
    )

    if cfg.qa.mask_saturated_bands and spec.saturation_band:
        radsat = image.select(spec.saturation_band)
        for name in harmonized_names:
            bit = spec.saturation_bits.get(name)
            if bit is not None:
                mask = mask.And(radsat.bitwiseAnd(1 << bit).eq(0))

    in_range = optical.gte(cfg.qa.reflectance_min).And(
        optical.lte(cfg.qa.reflectance_max)
    )
    optical = optical.updateMask(mask).updateMask(in_range.reduce(ee.Reducer.min()))

    return optical.copyProperties(
        image, ["system:time_start", "system:index", "WRS_PATH", "WRS_ROW"]
    ).set(
        {
            "sensor": spec.name,
            "collection_id": spec.collection_id,
            "scene_cloud_cover": image.get(spec.cloud_property),
        }
    )


def sensor_collection(
    spec: SensorSpec,
    geometry: ee.Geometry,
    start: str,
    end: str,
    cfg: SatelliteConfig,
) -> ee.ImageCollection:
    """Filter one sensor's archive to an area and date range and prepare it."""
    import ee

    return (
        ee.ImageCollection(spec.collection_id)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt(spec.cloud_property, cfg.extraction.max_scene_cloud_cover))
        .map(lambda image: prepare_image(image, spec, cfg))
    )


def _with_valid_band(indexed: ee.Image, index_name: str) -> ee.Image:
    """Append a band of ones carrying the index's own mask.

    Summing this band over a stand gives the effective pixel count that survived
    cloud, shadow and scan-line masking, and its ratio to the stand's area in
    pixels is the coverage fraction the quality gates read.

    It has to be derived from the index band rather than from
    ``ee.Image.constant(1)``. A constant image carries no mask, so summing it
    returns the full stand area on every scene, including one where every pixel
    was cloud. The coverage gate then never fires and a stand sampled through a
    hole in the clouds looks as trustworthy as a clear one. Multiplying the
    index by zero and adding one propagates the mask, which is the whole point.
    """
    return indexed.addBands(
        indexed.select(index_name).multiply(0).add(1).rename("valid")
    )


def build_collection(
    geometry: ee.Geometry,
    start: str,
    end: str,
    *,
    cfg: SatelliteConfig,
    sensors: Sequence[str],
    indices: Sequence[str],
) -> ee.ImageCollection:
    """Merge every sensor over a date range into one index-valued collection.

    Sensors whose operating span does not overlap the range are skipped rather
    than queried, so a 1990 chunk never touches the Landsat 9 archive.

    Returns:
        A collection whose images carry one band per requested index plus a
        constant ``valid`` band. The constant band is what makes the reduction
        able to report how much of a stand survived masking independently of
        which index is being read.
    """
    import ee

    from csdv_core.satellite.indices import apply_indices

    start_year, end_year = int(start[:4]), int(end[:4])
    parts: list[ee.ImageCollection] = []
    for name in sorted(sensors):
        spec = get_sensor(name)
        overlaps = spec.first_year <= end_year and (
            spec.last_year is None or start_year <= spec.last_year
        )
        if not overlaps:
            continue
        harmonized = sorted(spec.bands)
        parts.append(
            sensor_collection(spec, geometry, start, end, cfg).map(
                lambda image, bands=harmonized: _with_valid_band(
                    apply_indices(image, indices, bands), indices[0]
                ).copyProperties(
                    image,
                    [
                        "system:time_start",
                        "system:index",
                        "WRS_PATH",
                        "WRS_ROW",
                        "sensor",
                        "collection_id",
                        "scene_cloud_cover",
                    ],
                )
            )
        )

    if not parts:
        logger.warning("No sensor overlaps %s to %s", start, end)
        return ee.ImageCollection([])

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part)
    return ee.ImageCollection(merged).sort("system:time_start")
