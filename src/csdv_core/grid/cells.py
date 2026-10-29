"""csdv_core.grid.cells — the CONUS processing grid.

A grid cell is one USGS 7.5 minute quadrangle (0.125 by 0.125 degrees), the
same cell that NAIP digital orthophoto quarter quads are cut from. Its ID is
``q`` plus the quad code that starts every NAIP DOQQ name: two digits of
latitude and three of longitude for the south-east corner of the 1 degree
block, then the quad's number in that block, 01 at the north-west corner and
counting eastward row by row to 64. ``q4407159`` is the quad whose NAIP quarter
quads are named ``m_4407159_ne`` and so on.

A cell is about 9 to 13 km east to west and 13.9 km north to south, 13,000 to
17,500 ha across CONUS. That is smaller than the NorthMaine site (19,052 ha),
which ran through download, CHM inference and detection without memory
problems. A cell never straddles a UTM zone, because zone edges fall on
multiples of 6 degrees.

The grid is computed, not stored. Bounds are nominal NAD83 degrees, used here
as EPSG:4326 (the two datums differ by about a metre in CONUS).
"""

from __future__ import annotations

import math
import re

#: Side of a grid cell in degrees (7.5 minutes).
QUAD_DEG = 0.125

#: Quads per row (and per column) of a 1 degree block.
QUADS_PER_ROW = 8

#: Largest half-width of a point unit. A square of this half-width is about
#: 25,600 ha, a little larger than NorthMaine.
MAX_BUFFER_KM = 8.0

_KM_PER_DEG_LAT = 111.32

_QUAD_ID_RE = re.compile(r"^q?(?P<lat>\d{2})(?P<lon>\d{3})(?P<cell>\d{2})$")

__all__ = [
    "MAX_BUFFER_KM",
    "QUAD_DEG",
    "point_bounds",
    "quad_bounds",
    "quad_id_at",
]


def quad_id_at(lon: float, lat: float) -> str:
    """Return the ID of the grid cell that contains a point.

    A point exactly on a shared edge goes to one of the two cells.

    Args:
        lon: Longitude in degrees, negative west.
        lat: Latitude in degrees.

    Returns:
        Cell ID such as ``q4407159``.

    Raises:
        ValueError: Outside the north-western quadrant the quad code covers.
    """
    if not (lon < 0 < lat):
        raise ValueError(f"({lon}, {lat}) is not west of Greenwich and north")
    lat0 = math.floor(lat)
    lon0 = math.floor(-lon)
    # Rows count down from the block's north edge, columns east from its west.
    row = min(int((lat0 + 1 - lat) / QUAD_DEG), QUADS_PER_ROW - 1)
    col = min(int((lon + lon0 + 1) / QUAD_DEG), QUADS_PER_ROW - 1)
    cell = row * QUADS_PER_ROW + col + 1
    return f"q{lat0:02d}{lon0:03d}{cell:02d}"


def quad_bounds(quad_id: str) -> tuple[float, float, float, float]:
    """Return the ``(west, south, east, north)`` bounds of a grid cell.

    Args:
        quad_id: Cell ID with or without the ``q`` prefix (``q4407159`` or
            ``4407159``).
    """
    match = _QUAD_ID_RE.match(quad_id)
    if match is None:
        raise ValueError(f"Not a 7.5 minute quad ID: {quad_id!r}")
    cell = int(match["cell"])
    if not 1 <= cell <= QUADS_PER_ROW**2:
        raise ValueError(f"Quad number must be 01 to 64: {quad_id!r}")
    row, col = divmod(cell - 1, QUADS_PER_ROW)
    north = int(match["lat"]) + 1 - row * QUAD_DEG
    west = -(int(match["lon"]) + 1) + col * QUAD_DEG
    return (west, north - QUAD_DEG, west + QUAD_DEG, north)


def point_bounds(
    lon: float, lat: float, buffer_km: float
) -> tuple[float, float, float, float]:
    """Return the box of half-width ``buffer_km`` around a point, in degrees.

    Raises:
        ValueError: If ``buffer_km`` is not positive or exceeds
            :data:`MAX_BUFFER_KM`.
    """
    if not 0 < buffer_km <= MAX_BUFFER_KM:
        raise ValueError(
            f"buffer_km must be in (0, {MAX_BUFFER_KM}] km, got {buffer_km}. "
            "Larger units risk running out of memory; split the area into cells."
        )
    dlat = buffer_km / _KM_PER_DEG_LAT
    dlon = buffer_km / (_KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return (
        round(lon - dlon, 6),
        round(lat - dlat, 6),
        round(lon + dlon, 6),
        round(lat + dlat, 6),
    )
