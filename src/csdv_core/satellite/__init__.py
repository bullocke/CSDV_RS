"""csdv_core.satellite — per-stand spectral series from satellite archives.

This package sits beside :mod:`csdv_core.zonal` rather than inside it, for the
same reason ``zonal`` sits beside ``metrics``: the unit of work is different.
A zonal metric reads one raster on the analysis grid and returns one number per
stand per NAIP date. A satellite metric reads an archive of hundreds of scenes
on a foreign grid at fifty times the pixel size, reduces each one over the stand
to make an observation, and only then derives one number per stand per *year*.
Nothing in ``zonal`` can dispatch that, and forcing it to would mean giving
every zonal metric a time axis it does not have.

The pipeline is three stages, and each has its own registry so that new work
is a data change rather than a restructuring:

    sensor archive  --sensors.py-->  harmonized reflectance
                    --indices.py-->  per-pixel index bands
                    --extract.py-->  one row per stand per scene   (observations)
                    --annual.py -->  one row per stand per year    (metrics)

Adding Sentinel-2 means adding a :class:`~csdv_core.satellite.registry.SensorSpec`.
Adding NBR means the index is already registered and the config needs one more
entry. Adding a phenology metric means one function decorated with
``register_annual``.

Two products land on disk. The observation table is a cache of an external
archive and lives under ``data_root``; the annual table is derived and lives
under ``results_root`` beside the other per-stand products. Both go through
:mod:`csdv_core.io.satellite_io`.
"""

from __future__ import annotations

from csdv_core.satellite import annual, indices, sensors  # noqa: F401  (registration)
from csdv_core.satellite.annual import (
    AnnualResult,
    HarmonicFit,
    StandYearRecord,
    annual_table,
    filter_observations,
    fit_single_harmonic,
    theil_sen_slope,
)
from csdv_core.satellite.registry import (
    AnnualMetricSpec,
    IndexSpec,
    SensorSpec,
    get_annual,
    get_index,
    get_sensor,
    list_annual_metrics,
    list_indices,
    list_sensors,
    register_annual,
    register_index,
    register_sensor,
)

__all__ = [
    "AnnualMetricSpec",
    "AnnualResult",
    "HarmonicFit",
    "IndexSpec",
    "SensorSpec",
    "StandYearRecord",
    "annual_table",
    "filter_observations",
    "fit_single_harmonic",
    "get_annual",
    "get_index",
    "get_sensor",
    "list_annual_metrics",
    "list_indices",
    "list_sensors",
    "register_annual",
    "register_index",
    "register_sensor",
    "theil_sen_slope",
]
