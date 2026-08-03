"""csdv_core.satellite.indices — spectral indices as backend-agnostic algebra.

Each index is written once and runs in two places: on the Earth Engine side,
where it is computed per pixel before the polygon reduction, and locally on
NumPy arrays in the unit tests. That matters because the arithmetic is the part
most likely to be wrong in a way nobody notices. A near-infrared band swapped
for a red one still produces a number between -1 and 1 that looks like a
vegetation index.

``ee.Image`` has no Python arithmetic operators, so ``(nir - red) / (nir + red)``
cannot be written once and run on both backends. What both backends do share is
the method surface ``subtract``, ``add``, ``divide`` and ``multiply``, so the
formulas are written against that and :class:`NumpyBand` supplies it for arrays.

One behavioural difference between the backends is worth knowing. Dividing by
zero produces a masked pixel in Earth Engine and NaN in NumPy. Both read as "no
value" downstream, but the Earth Engine path drops the pixel from the count
while the NumPy path keeps it. Nothing in the pipeline depends on the
difference, because a zero-sum red plus near-infrared is masked before the
index is computed anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from csdv_core.satellite.registry import get_index, register_index

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    import ee

logger = logging.getLogger(__name__)

__all__ = [
    "BandAlgebra",
    "NumpyBand",
    "apply_indices",
    "nbr",
    "ndmi",
    "ndvi",
    "normalized_difference",
]


@runtime_checkable
class BandAlgebra(Protocol):
    """The four methods every index formula is allowed to call.

    ``ee.Image`` satisfies this already. :class:`NumpyBand` supplies it for
    arrays. Restricting formulas to this surface is what keeps them testable
    without Earth Engine credentials.
    """

    def subtract(self, other: Any) -> Any: ...

    def add(self, other: Any) -> Any: ...

    def divide(self, other: Any) -> Any: ...

    def multiply(self, other: Any) -> Any: ...


class NumpyBand:
    """A NumPy array wearing the :class:`BandAlgebra` surface.

    Division by zero yields NaN rather than raising, matching how a masked
    Earth Engine pixel reads downstream.
    """

    __slots__ = ("values",)

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def _wrap(self, other: Any) -> np.ndarray:
        return other.values if isinstance(other, NumpyBand) else np.asarray(other)

    def subtract(self, other: Any) -> NumpyBand:
        return NumpyBand(self.values - self._wrap(other))

    def add(self, other: Any) -> NumpyBand:
        return NumpyBand(self.values + self._wrap(other))

    def multiply(self, other: Any) -> NumpyBand:
        return NumpyBand(self.values * self._wrap(other))

    def divide(self, other: Any) -> NumpyBand:
        denominator = self._wrap(other)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.divide(
                self.values,
                denominator,
                out=np.full(np.broadcast(self.values, denominator).shape, np.nan),
                where=denominator != 0,
            )
        return NumpyBand(out)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.values if dtype is None else self.values.astype(dtype)
        return arr.copy() if copy else arr

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NumpyBand({self.values!r})"


def normalized_difference(a: Any, b: Any) -> Any:
    """Return ``(a - b) / (a + b)`` using only the band-algebra surface."""
    return a.subtract(b).divide(a.add(b))


@register_index(
    "ndvi",
    bands=("nir", "red"),
    description="Normalized difference vegetation index",
)
def ndvi(bands: Mapping[str, Any]) -> Any:
    """Normalized difference vegetation index, (NIR - red) / (NIR + red)."""
    return normalized_difference(bands["nir"], bands["red"])


@register_index(
    "nbr",
    bands=("nir", "swir2"),
    description="Normalized burn ratio",
)
def nbr(bands: Mapping[str, Any]) -> Any:
    """Normalized burn ratio, (NIR - SWIR2) / (NIR + SWIR2)."""
    return normalized_difference(bands["nir"], bands["swir2"])


@register_index(
    "ndmi",
    bands=("nir", "swir1"),
    description="Normalized difference moisture index",
)
def ndmi(bands: Mapping[str, Any]) -> Any:
    """Normalized difference moisture index, (NIR - SWIR1) / (NIR + SWIR1)."""
    return normalized_difference(bands["nir"], bands["swir1"])


def apply_indices(
    image: ee.Image,
    names: Sequence[str],
    available_bands: Sequence[str],
) -> ee.Image:
    """Compute every named index on a harmonized image and return them as bands.

    Args:
        image: Image whose bands are already harmonized, so a band is selected
            by its harmonized name (``"red"``, ``"nir"``, ...).
        names: Index names to compute.
        available_bands: Harmonized band names the image actually carries.

    Returns:
        An image with one band per requested index, named after the index. The
        source reflectance bands are dropped, because only the indices are
        reduced and carrying reflectance through the reduction would triple the
        row width for no gain.

    Raises:
        KeyError: If an index needs a band the sensor does not provide. Failing
            here rather than returning an all-null column is the difference
            between a config error and a silent hole in the record.
    """
    have = set(available_bands)
    outputs = []
    for name in names:
        spec = get_index(name)
        missing = [band for band in spec.bands if band not in have]
        if missing:
            raise KeyError(
                f"Index {name!r} needs band(s) {missing} which this sensor does "
                f"not provide. Available: {sorted(have)}"
            )
        band_map = {band: image.select(band) for band in spec.bands}
        outputs.append(spec.fn(band_map).rename(name))

    first, rest = outputs[0], outputs[1:]
    return first.addBands(rest) if rest else first
