"""csdv_core.segmentation.flood — seeded crown growth with a per-crown floor.

A priority flood (Meyer 1991 ordering) that grows every crown from its tree top
at once, highest pixel first. A pixel joins the first crown that reaches it
*and* whose admission test it passes. A pixel one crown rejects stays free, so
a shorter neighbour can still claim it.

That last property is the point of this module. The earlier engine ran an
unconstrained watershed first and trimmed each basin to ``th_cr`` times its
own apex afterwards. The tallest tree in a clump floods down into the saddles
it shares with shorter trees, the trim then removes those pixels against the
tall tree's floor, and nothing offers them to the shorter tree whose floor
they pass. Across twelve NAIP-CHM strata that left 12 to 38 percent of the
canopy unassigned for this reason alone.

Two admission rules are available.

``apex`` (default)
    A pixel joins crown L when its height is at or above a floor fixed by L's
    apex height: ``max(th_seed * apex, min(th_cr * apex, apex - min_depth))``,
    raised to ``apex - max_depth`` when a maximum depth is set. The floor
    depends only on the crown, so the result does not depend on growth order
    beyond the tie-break. This is the rule Dalponte and Coomes (2016) describe
    in prose.

``mean``
    lidR's ``dalponte2016`` rule: the pixel must exceed ``th_seed * apex`` and
    ``th_cr`` times the crown's current mean height. Provided so the
    difference between the two rules can be measured with everything else
    held fixed. It is not recommended for production.

Under both rules a pixel must lie within the circular radius cap of the
crown's tree top and may not exceed ``CEILING_FACTOR`` times the apex height,
as in lidR. Growth is 4-connected, so every crown is one connected piece that
contains its tree top.

Ties in height are broken by insertion order, which makes the output
deterministic. Each pixel is labelled once and pushed at most four times per
neighbouring crown, so the cost is O(N log N) in the canopy pixel count and
does not depend on crown radius.
"""

from __future__ import annotations

import heapq
import logging

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

__all__ = [
    "CEILING_FACTOR",
    "RULES",
    "crown_floors",
    "grow_crowns",
    "grow_crowns_reference",
]

#: A crown never grows onto pixels higher than this multiple of its apex.
#: Matches ``hSeed + hSeed * 0.05`` in lidR's ``C_dalponte2016``.
CEILING_FACTOR = 1.05

#: Admission rules and their integer codes in the compiled kernel.
RULES = {"apex": 0, "mean": 1}

_DR = np.array([-1, 0, 0, 1], dtype=np.int64)
_DC = np.array([0, -1, 1, 0], dtype=np.int64)


def crown_floors(
    apex: np.ndarray,
    *,
    th_cr: float,
    th_seed: float = 0.0,
    min_crown_depth_m: float | None = None,
    max_crown_depth_m: float | None = None,
    rule: str = "apex",
) -> np.ndarray:
    """Static height floor per crown.

    Under ``apex`` this is the whole admission test. Under ``mean`` it is only
    the ``th_seed`` part, and the mean-height test is applied during growth.

    ``min_crown_depth_m`` guarantees a short crown at least that much vertical
    extent: a 6 m tree at ``th_cr`` 0.7 keeps only 1.8 m of crown, so a 2.5 m
    minimum depth lowers its floor to 3.5 m. ``max_crown_depth_m`` does the
    opposite for tall trees.
    """
    apex = np.asarray(apex, dtype="float64")
    seed_floor = float(th_seed) * apex
    if rule == "mean":
        return seed_floor
    floor = float(th_cr) * apex
    if min_crown_depth_m is not None:
        floor = np.minimum(floor, apex - float(min_crown_depth_m))
    if max_crown_depth_m is not None:
        floor = np.maximum(floor, apex - float(max_crown_depth_m))
    return np.maximum(floor, seed_floor)


@njit(cache=True)
def _before(pri, tie, i, j):
    """Heap order: higher pixel first, then earlier insertion."""
    if pri[i] != pri[j]:
        return pri[i] > pri[j]
    return tie[i] < tie[j]


@njit(cache=True)
def _swap(pri, tie, pix, lab, i, j):
    pri[i], pri[j] = pri[j], pri[i]
    tie[i], tie[j] = tie[j], tie[i]
    pix[i], pix[j] = pix[j], pix[i]
    lab[i], lab[j] = lab[j], lab[i]


@njit(cache=True)
def _sift_up(pri, tie, pix, lab, i):
    while i > 0:
        parent = (i - 1) // 2
        if _before(pri, tie, i, parent):
            _swap(pri, tie, pix, lab, i, parent)
            i = parent
        else:
            break


@njit(cache=True)
def _sift_down(pri, tie, pix, lab, size):
    i = 0
    while True:
        left = 2 * i + 1
        if left >= size:
            break
        best = left
        right = left + 1
        if right < size and _before(pri, tie, right, left):
            best = right
        if _before(pri, tie, best, i):
            _swap(pri, tie, pix, lab, i, best)
            i = best
        else:
            break


@njit(cache=True)
def _flood_kernel(heights, seed_rows, seed_cols, floor, ceiling, max_r2, rule, th_cr):
    nr, nc = heights.shape
    n = seed_rows.size
    labels = np.zeros((nr, nc), dtype=np.int32)
    refused = np.zeros(n, dtype=np.int64)
    contacts = np.zeros(n, dtype=np.int64)
    touched_by = np.zeros((nr, nc), dtype=np.int32)
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    cap = max(1024, 4 * n)
    pri = np.empty(cap, dtype=np.float64)
    tie = np.empty(cap, dtype=np.int64)
    pix = np.empty(cap, dtype=np.int64)
    lab = np.empty(cap, dtype=np.int32)
    size = 0
    counter = 0

    for s in range(n):
        r = seed_rows[s]
        c = seed_cols[s]
        pri[size] = heights[r, c]
        tie[size] = counter
        pix[size] = r * nc + c
        lab[size] = s + 1
        _sift_up(pri, tie, pix, lab, size)
        size += 1
        counter += 1

    while size > 0:
        idx = pix[0]
        label = lab[0]
        size -= 1
        if size > 0:
            pri[0] = pri[size]
            tie[0] = tie[size]
            pix[0] = pix[size]
            lab[0] = lab[size]
            _sift_down(pri, tie, pix, lab, size)
        r = idx // nc
        c = idx % nc
        if labels[r, c] != 0:
            continue
        li = label - 1
        labels[r, c] = label
        sums[li] += heights[r, c]
        counts[li] += 1

        for k in range(4):
            rr = r + _DR[k]
            cc = c + _DC[k]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            owner = labels[rr, cc]
            if owner == label:
                continue
            # Every other outcome is a boundary pixel of this crown. Count it
            # once per crown, and count separately the ones the cap set.
            first = touched_by[rr, cc] != label
            if first:
                touched_by[rr, cc] = label
            if owner != 0:
                contacts[li] += first
                continue
            hq = heights[rr, cc]
            ok = hq == hq and hq <= ceiling[li]  # NaN is outside the canopy
            if ok:
                if rule == 0:
                    ok = hq >= floor[li]
                else:
                    ok = hq > floor[li] and hq > th_cr * sums[li] / counts[li]
            if not ok:
                contacts[li] += first
                continue
            dr = rr - seed_rows[li]
            dc = cc - seed_cols[li]
            if dr * dr + dc * dc > max_r2:
                contacts[li] += first
                refused[li] += first
                continue
            if size == cap:
                cap *= 2
                pri2 = np.empty(cap, dtype=np.float64)
                tie2 = np.empty(cap, dtype=np.int64)
                pix2 = np.empty(cap, dtype=np.int64)
                lab2 = np.empty(cap, dtype=np.int32)
                pri2[:size] = pri[:size]
                tie2[:size] = tie[:size]
                pix2[:size] = pix[:size]
                lab2[:size] = lab[:size]
                pri, tie, pix, lab = pri2, tie2, pix2, lab2
            pri[size] = hq
            tie[size] = counter
            pix[size] = rr * nc + cc
            lab[size] = label
            _sift_up(pri, tie, pix, lab, size)
            size += 1
            counter += 1

    return labels, refused, contacts


def _prepare(
    heights: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    apex: np.ndarray,
    th_cr: float,
    th_seed: float,
    min_crown_depth_m: float | None,
    max_crown_depth_m: float | None,
    max_radius_px: float | None,
    rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule!r}; expected one of {sorted(RULES)}")
    h = np.ascontiguousarray(heights, dtype=np.float64)
    rows = np.ascontiguousarray(seed_rows, dtype=np.int64)
    cols = np.ascontiguousarray(seed_cols, dtype=np.int64)
    apex = np.asarray(apex, dtype=np.float64)
    floor = crown_floors(
        apex,
        th_cr=th_cr,
        th_seed=th_seed,
        min_crown_depth_m=min_crown_depth_m,
        max_crown_depth_m=max_crown_depth_m,
        rule=rule,
    )
    ceiling = CEILING_FACTOR * apex
    max_r2 = np.inf if max_radius_px is None else float(max_radius_px) ** 2
    return h, rows, cols, floor, ceiling, max_r2


def _capped_fraction(refused: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(contacts > 0, refused / np.maximum(contacts, 1), 0.0)
    return frac.astype("float32")


def grow_crowns(
    heights: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    apex: np.ndarray,
    *,
    th_cr: float,
    th_seed: float = 0.0,
    min_crown_depth_m: float | None = None,
    max_crown_depth_m: float | None = None,
    max_radius_px: float | None = None,
    rule: str = "apex",
) -> tuple[np.ndarray, np.ndarray]:
    """Grow one crown from each tree top.

    Args:
        heights: Smoothed CHM in metres, NaN outside the canopy.
        seed_rows: Tree top rows.
        seed_cols: Tree top columns.
        apex: Height at each tree top.
        th_cr: Crown bound, relative to the apex (``apex`` rule) or to the
            crown's running mean height (``mean`` rule).
        th_seed: Floor relative to the apex, applied under both rules.
        min_crown_depth_m: Minimum vertical extent kept for every crown.
        max_crown_depth_m: Maximum vertical extent allowed for any crown.
        max_radius_px: Circular radius cap in pixels. None disables it.
        rule: ``"apex"`` or ``"mean"``.

    Returns:
        ``(labels, capped_frac)``. Labels are 1..n in seed order, 0 elsewhere.
        ``capped_frac[i]`` is the share of crown i's boundary set by the radius
        cap rather than by its floor, a gap or a neighbour: 1 for a crown the
        cap shaped all round, near 0 for one it merely grazed. A crown with a
        large value carries a censored diameter.
    """
    h, rows, cols, floor, ceiling, max_r2 = _prepare(
        heights,
        seed_rows,
        seed_cols,
        apex,
        th_cr,
        th_seed,
        min_crown_depth_m,
        max_crown_depth_m,
        max_radius_px,
        rule,
    )
    if rows.size == 0:
        return np.zeros(h.shape, dtype=np.int32), np.zeros(0, dtype="float32")
    labels, refused, contacts = _flood_kernel(
        h, rows, cols, floor, ceiling, max_r2, RULES[rule], float(th_cr)
    )
    return labels, _capped_fraction(refused, contacts)


def grow_crowns_reference(
    heights: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    apex: np.ndarray,
    *,
    th_cr: float,
    th_seed: float = 0.0,
    min_crown_depth_m: float | None = None,
    max_crown_depth_m: float | None = None,
    max_radius_px: float | None = None,
    rule: str = "apex",
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python twin of :func:`grow_crowns`, kept to test the compiled kernel.

    Same arguments and return values. Roughly two orders of magnitude slower.
    """
    h, rows, cols, floor, ceiling, max_r2 = _prepare(
        heights,
        seed_rows,
        seed_cols,
        apex,
        th_cr,
        th_seed,
        min_crown_depth_m,
        max_crown_depth_m,
        max_radius_px,
        rule,
    )
    nr, nc = h.shape
    n = rows.size
    labels = np.zeros((nr, nc), dtype=np.int32)
    refused = np.zeros(n, dtype=np.int64)
    contacts = np.zeros(n, dtype=np.int64)
    touched_by = np.zeros((nr, nc), dtype=np.int32)
    sums = np.zeros(n)
    counts = np.zeros(n, dtype=np.int64)
    heap: list[tuple[float, int, int, int, int]] = []
    counter = 0
    for s in range(n):
        r0, c0 = int(rows[s]), int(cols[s])
        heapq.heappush(heap, (-float(h[r0, c0]), counter, r0, c0, s + 1))
        counter += 1
    while heap:
        _, _, r, c, label = heapq.heappop(heap)
        if labels[r, c] != 0:
            continue
        li = label - 1
        labels[r, c] = label
        sums[li] += h[r, c]
        counts[li] += 1
        for dr_, dc_ in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            rr, cc = r + dr_, c + dc_
            if not (0 <= rr < nr and 0 <= cc < nc) or labels[rr, cc] == label:
                continue
            first = touched_by[rr, cc] != label
            touched_by[rr, cc] = label
            if labels[rr, cc] != 0:
                contacts[li] += first
                continue
            hq = h[rr, cc]
            ok = not np.isnan(hq) and hq <= ceiling[li]
            if ok:
                if rule == "apex":
                    ok = hq >= floor[li]
                else:
                    ok = hq > floor[li] and hq > th_cr * sums[li] / counts[li]
            if not ok:
                contacts[li] += first
                continue
            dr, dc = rr - rows[li], cc - cols[li]
            if dr * dr + dc * dc > max_r2:
                contacts[li] += first
                refused[li] += first
                continue
            heapq.heappush(heap, (-float(hq), counter, rr, cc, label))
            counter += 1
    return labels, _capped_fraction(refused, contacts)
