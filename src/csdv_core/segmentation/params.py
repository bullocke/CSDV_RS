"""csdv_core.segmentation.params — segmentation parameters as data.

Every knob that changes a crown lives here, in metres, so one parameter set
transfers between the 0.6 m NAIP-CHM and a 1 m airborne lidar CHM without
edits. Nothing is expressed in pixels, because a pixel-valued parameter
silently changes physical meaning with resolution and that is exactly the
failure this module exists to prevent.

:class:`SegmentationParams` hashes to a short stable key. Crown artefacts are
written under that key, so a run cannot pick up crowns produced by a different
parameter set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

__all__ = [
    "WindowFunction",
    "SegmentationParams",
    "WINDOW_FUNCTIONS",
    "DEFAULT_PARAMS",
    "LEGACY_PARAMS",
    "LEGACY_AS_SHIPPED_PARAMS",
]


@dataclass(frozen=True)
class WindowFunction:
    """Search window **diameter** in metres as a function of canopy height.

    ``ws(h) = clip(a + b*h + c*h**2, lo, hi)``.

    Diameter, not radius, and not a peak separation. This follows lidR's
    ``lmf(ws=...)``, where a pixel is a tree top when it is the maximum inside
    a circle of diameter ``ws`` centred on it. The implied minimum separation
    between two tops is therefore ``ws/2``.

    The distinction is not academic. The previous implementation passed ``ws``
    straight to ``skimage.peak_local_max(min_distance=...)``, which takes a
    minimum separation, so it spaced tree tops twice as far apart as the same
    equation does in lidR.
    """

    a: float
    b: float = 0.0
    c: float = 0.0
    lo: float = 3.0
    hi: float = 12.0
    name: str = ""

    def __call__(self, height_m: np.ndarray | float) -> np.ndarray:
        """Window diameter in metres for each height."""
        h = np.asarray(height_m, dtype="float64")
        ws = self.a + self.b * h + self.c * h * h
        return np.clip(ws, self.lo, self.hi)

    def describe(self) -> str:
        """One-line algebraic form, for figure labels and logs."""
        terms = [f"{self.a:g}"]
        if self.b:
            terms.append(f"{self.b:+g}h")
        if self.c:
            terms.append(f"{self.c:+g}h²")
        return f"clip({' '.join(terms)}, {self.lo:g}, {self.hi:g})"


#: Candidate window functions for the parameter sweep.
#:
#: ``legacy`` is what the pipeline shipped with. ``popescu_deciduous`` is the
#: crown-width equation from Popescu and Wynne (2004) for deciduous stands,
#: used directly as the window so the search circle matches the crown it is
#: meant to find. ``popescu_linear`` is a straight-line fit to that curve over
#: 2 to 30 m, kept because a linear rule is easier to defend and to port.
#: ``shallow`` is the shallow-slope starting point this work was asked to test.
WINDOW_FUNCTIONS: dict[str, WindowFunction] = {
    "legacy": WindowFunction(a=2.0, b=0.5, lo=3.0, hi=12.0, name="legacy"),
    "shallow": WindowFunction(a=3.0, b=0.10, lo=3.0, hi=7.0, name="shallow"),
    "popescu_deciduous": WindowFunction(
        a=2.51503, c=0.00901, lo=3.0, hi=12.0, name="popescu_deciduous"
    ),
    "popescu_linear": WindowFunction(
        a=0.60, b=0.33, lo=3.0, hi=12.0, name="popescu_linear"
    ),
    "fixed_5m": WindowFunction(a=5.0, lo=5.0, hi=5.0, name="fixed_5m"),
    "fixed_8m": WindowFunction(a=8.0, lo=8.0, hi=8.0, name="fixed_8m"),
}


@dataclass(frozen=True)
class SegmentationParams:
    """A complete, resolution-independent segmentation configuration.

    Attributes:
        min_height_m: Canopy floor. Pixels below it are not forest.
        smooth_radius_m: Mean-filter radius. Zero disables smoothing. Converted
            to an odd pixel kernel per raster, so the physical scale holds
            across resolutions. The old fixed 3x3 kernel was 1.8 m on a 0.6 m
            CHM and 3.0 m on a 1 m CHM, and it is what caps crown count.
        window: Search window diameter as a function of height.
        th_cr: Crown extent bound. A pixel stays in a crown only while it is
            above ``th_cr`` times the height at that crown's tree top, which is
            the rule that stops a crown at the edge of its own canopy. Zero
            reproduces the old unbounded watershed, where every canopy pixel is
            assigned to some crown and mean crown area is forced to
            ``10000 / density``. Mirrors ``th_cr`` in lidR's ``dalponte2016``.
        max_crown_radius_m: Hard ceiling on crown radius. A guard rail against
            one tall tree claiming a clearing, not the mechanism that shapes
            crowns. ``None`` disables it.
        min_crown_area_m2: Segments smaller than this are noise.
        min_separation_m: Floor on tree-top separation, independent of height.
            Stops adjacent pixels of one flat crown top becoming two trees.
    """

    min_height_m: float = 2.0
    smooth_radius_m: float = 0.6
    window: WindowFunction = field(default_factory=lambda: WINDOW_FUNCTIONS["fixed_5m"])
    th_cr: float = 0.70
    max_crown_radius_m: float | None = 12.0
    min_crown_area_m2: float = 1.0
    min_separation_m: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary, suitable for JSON and for a sidecar manifest."""
        out = asdict(self)
        out["window"] = {k: v for k, v in asdict(self.window).items() if k != "name"}
        out["window_name"] = self.window.name or self.window.describe()
        return out

    @property
    def key(self) -> str:
        """Short stable hash. Identical parameters always give the same key.

        Used to name crown artefacts so that re-running after a parameter
        change cannot silently reuse the previous run's crowns.
        """
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode(), digest_size=5).hexdigest()

    def replace(self, **changes: Any) -> SegmentationParams:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def describe(self) -> str:
        """Compact human-readable summary for logs and figure captions."""
        cap = (
            "none"
            if self.max_crown_radius_m is None
            else f"{self.max_crown_radius_m:g} m"
        )
        return (
            f"smooth {self.smooth_radius_m:g} m, "
            f"ws {self.window.describe()}, "
            f"th_cr {self.th_cr:g}, "
            f"max radius {cap}"
        )


#: The parameter set this pipeline uses across the eastern United States.
#:
#: Chosen in ``docs/guides/segmentation_optimization/``, which records how and
#: why. One thing there is worth repeating here, because it explains a choice
#: that looks wrong in isolation.
#:
#: The pre-registered decision rule, scored on the Indiana tuning tiles alone,
#: selected a 5 m window with no smoothing. That set reached 76 crowns per
#: hectare in Indiana and then produced 200 and 234 per hectare on the two
#: airborne lidar sites, because no smoothing on a sharper sensor finds three
#: times as many maxima. It was tuned to the smoothness of a model-inferred
#: canopy height model rather than to forest structure.
#:
#: This set keeps a 0.6 m smoothing radius instead. It reads 8 crowns per
#: hectare below the Indiana density floor, and in exchange it holds together
#: across three sites and two sensors at 67, 88 and 109 per hectare. For a
#: single parameter set meant to run everywhere, that trade is the point.
DEFAULT_PARAMS = SegmentationParams()

#: The legacy window equation under the corrected semantics. Useful for asking
#: how much of the old behaviour was the equation, as opposed to the way the
#: equation was applied. This is **not** what the pipeline used to produce.
LEGACY_PARAMS = SegmentationParams(
    smooth_radius_m=0.6,
    window=WINDOW_FUNCTIONS["legacy"],
    th_cr=0.0,
    max_crown_radius_m=None,
)

#: What the pipeline actually produced before this work, for before-and-after
#: comparisons.
#:
#: The old engine passed the window diameter to ``peak_local_max`` as a minimum
#: separation, so the separation it enforced was ``ws`` where lidR would
#: enforce ``ws/2``. Doubling the window reproduces that spacing under the
#: corrected code: ``clip(2 + 0.5h, 3, 12)`` doubled is
#: ``clip(4 + 1.0h, 6, 24)``. The old window was also a scene scalar rather
#: than a per-pixel function, which matters little here because it saturated at
#: its upper clip wherever mean canopy height passed 20 m, as it does at
#: Elkinsville.
#:
#: This is an approximation, not a replay. It gives about 14 crowns per hectare
#: at a mean diameter near 29 m, against the 10.3 per hectare and 33.2 m in the
#: archived crown files. The residual gap is the scene-scalar window, which
#: pinned the separation at exactly 12 m everywhere, while the doubled window
#: here still varies a little with height. Use the archived crowns when the
#: exact old output is what matters.
LEGACY_AS_SHIPPED_PARAMS = SegmentationParams(
    smooth_radius_m=0.6,
    window=WindowFunction(a=4.0, b=1.0, lo=6.0, hi=24.0, name="legacy_as_shipped"),
    th_cr=0.0,
    max_crown_radius_m=None,
)
