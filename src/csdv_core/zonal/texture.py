"""csdv_core.zonal.texture — grey level co-occurrence texture inside a stand.

Texture needs a rectangular support because a co-occurrence matrix counts pairs
of neighbouring pixels on a grid. A stand is not rectangular, so the metric runs
over the stand's bounding box with everything outside the stand excluded, and
the in-stand share of that box is reported alongside the value.

Excluding is the operative word. :func:`csdv_core.metrics.texture.glcm_texture`
sets invalid pixels to grey level 0 before building the matrix, which is correct
for the occasional nodata pixel it was written for but wrong for a masked
polygon: every pixel outside the stand lands in the same bin, the matrix gains a
spike at (0, 0) proportional to one minus the fill fraction, and the resulting
entropy tracks the shape of the polygon rather than the structure of the canopy.
Stands here fill between a quarter and four fifths of their bounding box, so the
error is large and varies stand to stand. Raising or lowering a minimum-coverage
guard does not help, because the number that comes back is wrong rather than
missing.

:func:`masked_glcm` instead counts only those neighbour pairs whose members are
both inside the stand. Pixels outside contribute nothing at all. On a fully
valid rectangle it reproduces :func:`skimage.feature.graycomatrix` exactly, and
unlike the windowed version its result does not change when the surrounding
padding does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = 16
DEFAULT_DISTANCES: tuple[int, ...] = (1,)
DEFAULT_ANGLES: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

#: Minimum count of valid pixels before entropy is reported. This replaces the
#: "at least half the tile must be finite" rule of the windowed metric, which
#: is a statement about the bounding box rather than about the sample. A
#: one-acre stand, the minimum mapping unit, holds about 11,000 pixels at 0.6 m,
#: so this guard only catches genuinely tiny stands.
MIN_VALID_PIXELS = 256

__all__ = [
    "DEFAULT_ANGLES",
    "DEFAULT_DISTANCES",
    "DEFAULT_LEVELS",
    "MIN_VALID_PIXELS",
    "TextureResult",
    "glcm_entropy",
    "masked_glcm",
    "quantize_masked",
    "texture_entropy",
]


@dataclass(frozen=True)
class TextureResult:
    """Texture entropy for one stand, with the support needed to read it.

    Attributes:
        entropy_bits: Shannon entropy of the co-occurrence matrix in bits,
            averaged over offsets. NaN when not computed. The theoretical
            maximum is ``2 * log2(levels)``, so 8.0 bits at 16 levels.
        levels: Grey levels used for quantization.
        n_valid: Valid in-stand pixels the matrix was built from.
        support_fraction: In-stand share of the bounding box.
        vmin: Lower bound of the value stretch applied before quantization.
        vmax: Upper bound of the stretch.
        reason: Empty when computed, otherwise why the value is NaN.
    """

    entropy_bits: float
    levels: int
    n_valid: int
    support_fraction: float
    vmin: float
    vmax: float
    reason: str = ""


def quantize_masked(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Quantize the valid pixels of ``values`` to ``[0, levels - 1]``.

    The stretch defaults to the minimum and maximum of the valid pixels, which
    matches the windowed metric and keeps entropy on the same 0 to
    ``2 * log2(levels)`` scale as the stage envelopes. Pass explicit bounds to
    compare stands or dates against a common stretch.

    Invalid pixels are set to 0 in the returned array purely so it has a dtype
    that :func:`masked_glcm` can index with. They are excluded from every pair
    count, so their value is never used.

    Returns:
        ``(quantized, vmin, vmax)``.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != valid.shape:
        raise ValueError(f"Array shape {arr.shape} does not match mask {valid.shape}")
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")
    sample = arr[valid]
    if sample.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8), float("nan"), float("nan")
    lo = float(np.min(sample)) if vmin is None else float(vmin)
    hi = float(np.max(sample)) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8), lo, hi
    scaled = (arr - lo) / (hi - lo) * (levels - 1)
    quant = np.clip(np.nan_to_num(scaled, nan=0.0), 0, levels - 1).astype(np.uint8)
    quant[~valid] = 0
    return quant, lo, hi


def _offsets(
    distances: tuple[int, ...],
    angles: tuple[float, ...],
) -> list[tuple[int, int, int, int]]:
    """Return ``(distance_index, angle_index, row_offset, col_offset)`` tuples.

    The rounding convention matches :func:`skimage.feature.graycomatrix`, so a
    fully valid rectangle reproduces its matrix exactly.
    """
    out: list[tuple[int, int, int, int]] = []
    for di, distance in enumerate(distances):
        for ai, angle in enumerate(angles):
            row = int(round(np.sin(angle) * distance))
            col = int(round(np.cos(angle) * distance))
            out.append((di, ai, row, col))
    return out


def masked_glcm(
    quant: np.ndarray,
    valid: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    distances: tuple[int, ...] = DEFAULT_DISTANCES,
    angles: tuple[float, ...] = DEFAULT_ANGLES,
) -> np.ndarray:
    """Build a symmetric, normalized co-occurrence matrix over in-mask pairs only.

    Args:
        quant: Quantized image with values in ``[0, levels - 1]``.
        valid: Boolean array, True where a pixel may take part in a pair.
        levels: Grey levels.
        distances: Pixel offsets.
        angles: Offset angles in radians.

    Returns:
        Array of shape ``(levels, levels, len(distances), len(angles))``, each
        offset plane summing to 1. A plane with no admissible pair is all zero.
    """
    quant = np.asarray(quant)
    if quant.shape != valid.shape:
        raise ValueError(f"Array shape {quant.shape} does not match mask {valid.shape}")
    rows, cols = quant.shape
    out = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    flat = quant.astype(np.int64)

    for di, ai, dr, dc in _offsets(distances, angles):
        r0, r1 = max(0, -dr), min(rows, rows - dr)
        c0, c1 = max(0, -dc), min(cols, cols - dc)
        if r1 <= r0 or c1 <= c0:
            continue
        head = flat[r0:r1, c0:c1]
        tail = flat[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        both = valid[r0:r1, c0:c1] & valid[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        if not both.any():
            continue
        pairs = head[both] * levels + tail[both]
        counts = np.bincount(pairs, minlength=levels * levels).reshape(levels, levels)
        counts = counts + counts.T  # symmetric, as skimage does
        total = counts.sum()
        if total > 0:
            out[:, :, di, ai] = counts / total
    return out


def glcm_entropy(glcm: np.ndarray) -> float:
    """Mean Shannon entropy in bits over the offset planes of ``glcm``."""
    values: list[float] = []
    for di in range(glcm.shape[2]):
        for ai in range(glcm.shape[3]):
            plane = glcm[:, :, di, ai].ravel()
            plane = plane[plane > 0]
            values.append(
                0.0 if plane.size == 0 else float(-np.sum(plane * np.log2(plane)))
            )
    return float(np.mean(values)) if values else float("nan")


def texture_entropy(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    distances: tuple[int, ...] = DEFAULT_DISTANCES,
    angles: tuple[float, ...] = DEFAULT_ANGLES,
    vmin: float | None = None,
    vmax: float | None = None,
    min_valid_pixels: int = MIN_VALID_PIXELS,
) -> TextureResult:
    """Texture entropy inside a stand, in bits.

    Args:
        image: 2-D array over the stand's bounding box. The pipeline passes the
            NAIP near infrared band. Non-finite pixels are treated as invalid
            in addition to those outside ``mask``.
        mask: Boolean in-stand mask of the same shape.
        levels: Grey levels for quantization.
        distances: GLCM pixel offsets.
        angles: GLCM offset angles in radians.
        vmin: Optional lower bound of the stretch, for cross-date comparison.
        vmax: Optional upper bound.
        min_valid_pixels: Below this many valid pixels no value is reported.

    Returns:
        A :class:`TextureResult`. ``entropy_bits`` is NaN with a stated reason
        when the stand holds too few valid pixels or has no tonal variation.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape != mask.shape:
        raise ValueError(f"Array shape {arr.shape} does not match mask {mask.shape}")
    valid = mask & np.isfinite(arr)
    n_valid = int(valid.sum())
    support = float(mask.sum()) / float(mask.size) if mask.size else float("nan")

    if n_valid < min_valid_pixels:
        return TextureResult(
            entropy_bits=float("nan"),
            levels=levels,
            n_valid=n_valid,
            support_fraction=support,
            vmin=float("nan"),
            vmax=float("nan"),
            reason=f"n_valid={n_valid} < min_valid_pixels={min_valid_pixels}",
        )

    quant, lo, hi = quantize_masked(arr, valid, levels=levels, vmin=vmin, vmax=vmax)
    if not np.isfinite(hi) or hi <= lo:
        return TextureResult(
            entropy_bits=float("nan"),
            levels=levels,
            n_valid=n_valid,
            support_fraction=support,
            vmin=lo,
            vmax=hi,
            reason="no tonal variation inside the stand",
        )
    glcm = masked_glcm(quant, valid, levels=levels, distances=distances, angles=angles)
    return TextureResult(
        entropy_bits=glcm_entropy(glcm),
        levels=levels,
        n_valid=n_valid,
        support_fraction=support,
        vmin=lo,
        vmax=hi,
    )
