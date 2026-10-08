"""Forest masks from NAIP-derived canopy height and NAIP NDVI.

Pure array functions for building binary forest masks at the native 0.6 m
pixel size or aggregated to square blocks of pixels (16 px = 9.6 m cells).
The CHM rule matches the cell definition used in the ElkinsvilleNE disturbance
exploration (share of pixels above 5 m at least 0.5). The NDVI rule supports a
fixed raw threshold, a robust z-score against a reference forest sample, or a
per-image Otsu threshold.

Inputs: CHM (metres, NaN = nodata) and red/NIR arrays. Outputs: ``uint8``
masks coded 1 = forest, 0 = non-forest, 255 = nodata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

FOREST = 1
NONFOREST = 0
NODATA = 255
MAD_SCALE = 1.4826

NdviMode = Literal["raw", "z", "otsu"]
CombineRule = Literal["chm", "ndvi", "and", "or"]


@dataclass(frozen=True)
class RobustStats:
    """Median and scaled MAD of a reference sample."""

    median: float
    scale: float
    n: int


@dataclass
class ForestMaskResult:
    """A coded forest mask and the parameters that produced it."""

    mask: np.ndarray
    params: dict[str, float | int | str | None] = field(default_factory=dict)

    @property
    def forest_fraction(self) -> float:
        valid = self.mask != NODATA
        return float((self.mask[valid] == FOREST).mean()) if valid.any() else np.nan


def block_view(arr: np.ndarray, block_px: int) -> np.ndarray:
    """Reshape ``(H, W)`` to ``(H // b, W // b, b * b)``, dropping partial blocks."""
    h, w = arr.shape
    r, c = h // block_px, w // block_px
    out = arr[: r * block_px, : c * block_px].reshape(r, block_px, c, block_px)
    return np.moveaxis(out, 1, 2).reshape(r, c, block_px * block_px)


def block_fraction(
    flag: np.ndarray, valid: np.ndarray, block_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Share of valid pixels that are flagged, per block.

    Returns:
        ``(fraction, any_valid)``. Blocks with no valid pixels get fraction 0.
    """
    if block_px == 1:
        # Boolean at pixel size: a float copy of a full 0.6 m scene is ~0.4 GB.
        return flag & valid, valid.copy()
    f = block_view(flag & valid, block_px).sum(-1)
    n = block_view(valid, block_px).sum(-1)
    return (f / np.maximum(n, 1)).astype("float32"), n > 0


def block_mean(values: np.ndarray, block_px: int) -> np.ndarray:
    """NaN-aware block mean (NaN where a block has no finite values)."""
    if block_px == 1:
        return values.astype("float32", copy=False)
    bv = block_view(values, block_px)
    ok = np.isfinite(bv)
    n = ok.sum(-1)
    s = np.where(ok, bv, 0.0).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype("float32")


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI from uint8 DNs, as in the cell builder: ``(nir-red)/max(nir+red, 1)``."""
    r = red.astype("float32")
    n = nir.astype("float32")
    return (n - r) / np.maximum(n + r, 1.0)


def robust_stats(values: np.ndarray, reference: np.ndarray) -> RobustStats:
    """Median and ``1.4826 * MAD`` of ``values`` where ``reference`` is true."""
    v = values[reference & np.isfinite(values)]
    if v.size == 0:
        raise ValueError("Reference sample is empty")
    med = float(np.median(v))
    scale = float(MAD_SCALE * np.median(np.abs(v - med)))
    return RobustStats(med, scale if scale > 0 else 1.0, int(v.size))


def otsu_threshold(values: np.ndarray, sample: int = 2_000_000, seed: int = 0) -> float:
    """Otsu threshold over the finite values (randomly subsampled if large)."""
    v = values[np.isfinite(values)]
    if v.size > sample:
        v = np.random.default_rng(seed).choice(v, sample, replace=False)
    return float(threshold_otsu(v))


def clean_mask(
    mask: np.ndarray, min_patch_px: int = 0, fill_holes_px: int = 0
) -> np.ndarray:
    """Drop forest patches below ``min_patch_px`` and fill holes below ``fill_holes_px``."""
    out = mask.astype(bool)
    if min_patch_px > 1:
        out = remove_small_objects(out, max_size=min_patch_px - 1)
    if fill_holes_px > 1:
        out = remove_small_holes(out, max_size=fill_holes_px - 1)
    return out


def encode(forest: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Code a boolean mask as uint8 (1 forest, 0 non-forest, 255 nodata)."""
    # FOREST = 1 and NONFOREST = 0, so a direct cast avoids an int64 temporary.
    out = forest.astype("uint8")
    out[~valid] = NODATA
    return out


def chm_forest_mask(
    chm: np.ndarray,
    *,
    height_m: float = 5.0,
    block_px: int = 16,
    min_cover: float = 0.5,
    min_patch_px: int = 0,
    fill_holes_px: int = 0,
) -> ForestMaskResult:
    """Forest where the share of CHM pixels above ``height_m`` is at least ``min_cover``.

    With ``block_px=1`` this is the pixel rule ``chm > height_m`` (``min_cover``
    has no effect). With ``block_px=16`` it reproduces the 9.6 m cell rule
    ``cov5 >= 0.5`` of the disturbance exploration.

    Args:
        chm: Canopy height in metres, NaN for nodata.
        height_m: Height that counts as tree canopy.
        block_px: Source pixels per output cell side.
        min_cover: Minimum canopy share per block.
        min_patch_px: Remove forest patches smaller than this (output cells).
        fill_holes_px: Fill non-forest holes smaller than this (output cells).
    """
    valid = np.isfinite(chm)
    cover, any_valid = block_fraction(np.nan_to_num(chm) > height_m, valid, block_px)
    forest = cover >= min_cover if block_px > 1 else cover > 0
    forest = clean_mask(forest & any_valid, min_patch_px, fill_holes_px)
    params = dict(
        source="chm",
        height_m=height_m,
        block_px=block_px,
        min_cover=min_cover if block_px > 1 else None,
        min_patch_px=min_patch_px,
        fill_holes_px=fill_holes_px,
    )
    return ForestMaskResult(encode(forest, any_valid), params)


def ndvi_forest_mask(
    ndvi_values: np.ndarray,
    *,
    threshold: float,
    mode: NdviMode = "raw",
    stats: RobustStats | None = None,
    block_px: int = 16,
    min_patch_px: int = 0,
    fill_holes_px: int = 0,
) -> ForestMaskResult:
    """Forest where block-mean NDVI (or its robust z-score) exceeds ``threshold``.

    Args:
        ndvi_values: Pixel NDVI, NaN for nodata.
        threshold: Raw NDVI (``mode="raw"`` or ``"otsu"``) or z-score (``"z"``).
            For ``"otsu"`` pass the threshold from :func:`otsu_threshold`.
        mode: ``"raw"``, ``"z"`` (needs ``stats``) or ``"otsu"``.
        stats: Reference median and scale for ``mode="z"``, computed at the
            same ``block_px`` with :func:`robust_stats`.
        block_px: Source pixels per output cell side.
    """
    v = block_mean(ndvi_values, block_px)
    valid = np.isfinite(v)
    if mode == "z":
        if stats is None:
            raise ValueError("mode='z' needs reference stats")
        v = (v - stats.median) / stats.scale
    forest = clean_mask(
        np.where(valid, v > threshold, False), min_patch_px, fill_holes_px
    )
    params = dict(
        source="ndvi",
        ndvi_mode=mode,
        ndvi_threshold=round(float(threshold), 6),
        ref_median=None if stats is None else round(stats.median, 6),
        ref_scale=None if stats is None else round(stats.scale, 6),
        block_px=block_px,
        min_patch_px=min_patch_px,
        fill_holes_px=fill_holes_px,
    )
    return ForestMaskResult(encode(forest, valid), params)


def combine_masks(
    chm: np.ndarray, ndvi_mask: np.ndarray, rule: CombineRule
) -> np.ndarray:
    """Combine two coded masks. Nodata in either input is nodata in the output."""
    if rule == "chm":
        return chm.copy()
    if rule == "ndvi":
        return ndvi_mask.copy()
    valid = (chm != NODATA) & (ndvi_mask != NODATA)
    a, b = chm == FOREST, ndvi_mask == FOREST
    forest = a & b if rule == "and" else a | b
    return encode(forest, valid)


def params_tag(params: dict) -> str:
    """Short stable hash of a parameter dict, used to name exports."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
