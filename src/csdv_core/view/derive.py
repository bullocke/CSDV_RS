"""csdv_core.view.derive: map layers from the per-interval rasters of a run.

A pair-model run writes loss probability (percent, 255 where a date has no
imagery) and the agent code (1 to 6, 255 where there is none) with one band
per interval. It writes no date raster: the date of a loss is the interval
whose band reaches the threshold. The functions here reduce those stacks to
one uint8 layer each.

- Date: the later year of the first flagged interval, as ``year - 2000``,
  which is also the encoding of the Disturbance Index onset raster. 0 is no
  loss.
- Type: the agent code of the first flagged interval. 0 is no loss.
- Probability: percent, 255 where there is no imagery.

Each function takes ``band`` to look at one interval instead of all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NODATA = 255
NO_LOSS = 0
#: Date codes are ``year - YEAR_BASE`` (csdv_core.grid.export.ONSET_YEAR_BASE).
YEAR_BASE = 2000

__all__ = [
    "NODATA",
    "NO_LOSS",
    "YEAR_BASE",
    "date_layer",
    "loss_mask",
    "onset_layer",
    "prob_layer",
    "type_layer",
]


def loss_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Cells whose loss probability reaches a threshold.

    Args:
        prob: Loss probability in percent, 255 where there is no imagery.
        threshold: Probability (0 to 1), rounded to a whole percent as the
            exported type rasters are.
    """
    cut = int(round(threshold * 100))
    return (prob != NODATA) & (prob >= cut)


def _year_codes(end_years: Sequence[int]) -> np.ndarray:
    codes = np.asarray(end_years, dtype=np.int64) - YEAR_BASE
    return np.clip(codes, 1, NODATA - 1).astype(np.uint8)


def date_layer(
    prob: np.ndarray,
    end_years: Sequence[int],
    threshold: float,
    band: int | None = None,
) -> np.ndarray:
    """Year code of the first flagged interval, 0 where none is flagged.

    Args:
        prob: ``(intervals, rows, cols)`` loss probability in percent.
        end_years: Later year of every interval.
        threshold: Probability from which a cell counts as loss.
        band: Index of one interval. Then only that interval is flagged.
    """
    codes = _year_codes(end_years)
    mask = loss_mask(prob, threshold)
    if band is not None:
        return np.where(mask[band], codes[band], NO_LOSS).astype(np.uint8)
    first = mask.argmax(axis=0)
    return np.where(mask.any(axis=0), codes[first], NO_LOSS).astype(np.uint8)


def type_layer(
    prob: np.ndarray,
    agent: np.ndarray,
    threshold: float,
    band: int | None = None,
) -> np.ndarray:
    """Agent code of the first flagged interval, 0 where none is flagged.

    With ``band`` the result equals the exported ``type_<unit>_<interval>.tif``
    when ``threshold`` is the threshold of the run.
    """
    from csdv_core.pairmodel.predict import flagged_agents

    if band is not None:
        return flagged_agents(agent[band], prob[band], threshold)
    types = flagged_agents(agent, prob, threshold)
    first = (types != NO_LOSS).argmax(axis=0)
    return np.take_along_axis(types, first[np.newaxis], axis=0)[0]


def prob_layer(prob: np.ndarray, band: int | None = None) -> np.ndarray:
    """Loss probability of one interval, or the highest over all of them."""
    if band is not None:
        return prob[band]
    valid = prob != NODATA
    highest = np.where(valid, prob, 0).max(axis=0)
    return np.where(valid.any(axis=0), highest, NODATA).astype(np.uint8)


def onset_layer(onset: np.ndarray, end_year: int | None = None) -> np.ndarray:
    """The Disturbance Index onset raster as a date layer.

    Args:
        onset: ``year - 2000``, 0 no detection, 255 never monitored.
        end_year: Keep only the cells whose onset is this year.
    """
    out = np.where(onset == NODATA, NO_LOSS, onset).astype(np.uint8)
    if end_year is not None:
        out = np.where(out == end_year - YEAR_BASE, out, NO_LOSS).astype(np.uint8)
    return out
