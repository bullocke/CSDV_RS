"""Tests for csdv_core.satellite.indices.

The point of the band-algebra indirection is that the arithmetic Earth Engine
runs can be exercised here, on NumPy, with no credentials and no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.satellite.indices import (
    BandAlgebra,
    NumpyBand,
    apply_indices,
    normalized_difference,
)
from csdv_core.satellite.registry import get_index, list_indices


def _run(name: str, **bands: np.ndarray) -> np.ndarray:
    spec = get_index(name)
    return np.asarray(spec.fn({k: NumpyBand(v) for k, v in bands.items()}))


def test_ndvi_hand_computed() -> None:
    out = _run("ndvi", nir=np.array([0.4]), red=np.array([0.1]))
    assert out[0] == pytest.approx(0.6)


def test_ndvi_is_zero_when_bands_are_equal() -> None:
    out = _run("ndvi", nir=np.array([0.3]), red=np.array([0.3]))
    assert out[0] == pytest.approx(0.0)


def test_zero_denominator_gives_nan_not_an_exception() -> None:
    out = _run("ndvi", nir=np.array([0.0]), red=np.array([0.0]))
    assert np.isnan(out[0])


def test_ndvi_matches_the_plain_formula_on_random_input() -> None:
    rng = np.random.default_rng(0)
    nir = rng.uniform(0.05, 0.6, size=500)
    red = rng.uniform(0.01, 0.3, size=500)
    np.testing.assert_allclose(
        _run("ndvi", nir=nir, red=red), (nir - red) / (nir + red), rtol=1e-9
    )


def test_every_index_is_the_same_normalized_difference_kernel() -> None:
    rng = np.random.default_rng(1)
    a = rng.uniform(0.05, 0.6, size=64)
    b = rng.uniform(0.01, 0.3, size=64)
    expected = np.asarray(normalized_difference(NumpyBand(a), NumpyBand(b)))
    np.testing.assert_allclose(_run("ndvi", nir=a, red=b), expected)
    np.testing.assert_allclose(_run("nbr", nir=a, swir2=b), expected)
    np.testing.assert_allclose(_run("ndmi", nir=a, swir1=b), expected)


def test_numpy_band_satisfies_the_protocol_every_index_is_written_against() -> None:
    assert isinstance(NumpyBand(np.zeros(1)), BandAlgebra)


def test_indices_touch_no_method_the_adapter_lacks() -> None:
    """An index calling an ee.Image method NumpyBand lacks fails here, not in production."""

    class Recorder:
        def __init__(self) -> None:
            self.calls: set[str] = set()

        def __getattr__(self, name: str) -> object:
            if not hasattr(NumpyBand, name):
                raise AssertionError(
                    f"index called {name!r}, which the NumpyBand adapter does not "
                    "implement, so this formula is untestable without Earth Engine"
                )
            self.calls.add(name)
            return lambda *_args, **_kwargs: self

    for name in list_indices():
        spec = get_index(name)
        recorder = Recorder()
        spec.fn(dict.fromkeys(spec.bands, recorder))
        assert recorder.calls  # the formula did something


def test_apply_indices_names_the_missing_band() -> None:
    class Stub:
        def select(self, name: str) -> object:  # pragma: no cover - never reached
            raise AssertionError("should have failed before selecting a band")

    with pytest.raises(KeyError, match="swir2"):
        apply_indices(Stub(), ["nbr"], available_bands=["red", "nir"])
