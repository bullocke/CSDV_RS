"""Soils variables for site-type stratification.

Pure functions: take soil source rasters plus a target grid and return a
dict of variables on that grid. No I/O. The caller (CLI) handles raster
opening and reprojection driver dispatch.

Variables produced (per V5 Section 2 environmental drivers):

- ``hydric_pct``     hydric soil percentage [0, 100]
- ``awc_total_cm``   rooting-zone available water capacity (cm)
- ``parmat_kind``    categorical: 0=unknown, 1=residuum, 2=till,
                     3=outwash, 4=alluvium, 5=loess, 6=lacustrine
- ``texture_class``  categorical: 0=unknown, 1=sand, 2=loam, 3=silt, 4=clay
- ``bedrock_kind``   categorical: 0=unknown, 1=carbonate, 2=sandstone,
                     3=shale, 4=igneous

For PoC, the caller supplies pre-clipped per-site source rasters (gNATSGO
reprojected to the site CRS). A higher-level fetch helper is documented in
``docs/data_layout.md``.
"""

from __future__ import annotations

import logging

import numpy as np

from csdv_core.io.grids import GridSpec

logger = logging.getLogger(__name__)

PARMAT_KIND = {
    "unknown": 0,
    "residuum": 1,
    "till": 2,
    "outwash": 3,
    "alluvium": 4,
    "loess": 5,
    "lacustrine": 6,
}

TEXTURE_CLASS = {
    "unknown": 0,
    "sand": 1,
    "loam": 2,
    "silt": 3,
    "clay": 4,
}

BEDROCK_KIND = {
    "unknown": 0,
    "carbonate": 1,
    "sandstone": 2,
    "shale": 3,
    "igneous": 4,
}

REQUIRED_SOIL_VARS = (
    "hydric_pct",
    "awc_total_cm",
    "parmat_kind",
    "texture_class",
    "bedrock_kind",
)


def stack_soil_variables(
    rasters: dict[str, np.ndarray],
    grid: GridSpec,
) -> dict[str, np.ndarray]:
    """Validate and return a soils variable stack on ``grid``.

    Args:
        rasters: Dict mapping each name in :data:`REQUIRED_SOIL_VARS` to a
            2-D array already aligned to ``grid``. Categorical variables
            should be uint8 with codes from :data:`PARMAT_KIND`,
            :data:`TEXTURE_CLASS`, :data:`BEDROCK_KIND`. Continuous
            variables should be float32.
        grid: Target :class:`GridSpec`. Carried through unchanged; included
            so callers cannot forget to pass it.

    Returns:
        A new dict with the same keys, dtype-normalised. Missing variables
        are filled with their ``unknown``/NaN sentinels.
    """
    if not rasters:
        raise ValueError("No soil rasters provided")

    shapes = {name: arr.shape for name, arr in rasters.items()}
    ref_shape = next(iter(shapes.values()))
    for name, shp in shapes.items():
        if shp != ref_shape:
            raise ValueError(f"Soil raster {name} shape {shp} != reference {ref_shape}")

    out: dict[str, np.ndarray] = {}
    for name in REQUIRED_SOIL_VARS:
        arr = rasters.get(name)
        if arr is None:
            logger.warning("soil variable %s missing; filling with unknown", name)
            if name in {"parmat_kind", "texture_class", "bedrock_kind"}:
                arr = np.zeros(ref_shape, dtype="uint8")
            else:
                arr = np.full(ref_shape, np.nan, dtype="float32")
        else:
            if name in {"parmat_kind", "texture_class", "bedrock_kind"}:
                arr = arr.astype("uint8", copy=False)
            else:
                arr = arr.astype("float32", copy=False)
        out[name] = arr

    # ``grid`` is accepted for symmetry with other stratification helpers
    # and to make caller intent explicit; the variables are already on it.
    del grid
    return out


def encode_categorical(
    labels: np.ndarray,
    mapping: dict[str, int],
) -> np.ndarray:
    """Encode a 2-D string-label array using ``mapping``.

    Unknown labels (including empty strings and ``None``) map to 0.
    """
    flat = labels.ravel()
    out = np.zeros(flat.shape[0], dtype="uint8")
    for key, code in mapping.items():
        if key == "unknown":
            continue
        out[np.asarray(flat == key)] = code
    return out.reshape(labels.shape)
