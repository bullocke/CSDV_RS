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
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "WindowFunction",
    "SegmentationParams",
    "WINDOW_FUNCTIONS",
    "DEFAULT_PARAMS",
    "OPTIMIZED_POST_TRIM_PARAMS",
    "LEGACY_PARAMS",
    "TALLO_BIOMES",
    "tallo_window",
    "tallo_params",
    "LEGACY_AS_SHIPPED_PARAMS",
]


@dataclass(frozen=True)
class WindowFunction:
    """Search window **diameter** in metres as a function of canopy height.

    Two forms:

    ``poly`` (default)  ``ws(h) = clip(k*(a + b*h + c*h**2), lo, hi)``
    ``power``           ``ws(h) = clip(k*a*h**b, lo, hi)``

    The power form is a crown-width allometry ``CW = a*h**b`` scaled by the
    window-to-crown proportion ``k``, as in the refAGB Tallo windows, where
    ``a`` and ``b`` are fitted per biome to the Tallo database and ``k`` is 0.5.

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
    form: str = "poly"
    k: float = 1.0

    def __post_init__(self) -> None:
        if self.form not in ("poly", "power"):
            raise ValueError(f"Unknown window form {self.form!r}")

    def __call__(self, height_m: np.ndarray | float) -> np.ndarray:
        """Window diameter in metres for each height."""
        h = np.asarray(height_m, dtype="float64")
        if self.form == "power":
            with np.errstate(invalid="ignore", divide="ignore"):
                ws = self.k * self.a * np.power(np.maximum(h, 0.0), self.b)
            ws = np.where(np.isfinite(ws), ws, self.lo)
        else:
            ws = self.k * (self.a + self.b * h + self.c * h * h)
        return np.clip(ws, self.lo, self.hi)

    def as_dict(self) -> dict[str, Any]:
        """Coefficients for hashing and sidecars. ``form`` and ``k`` appear only
        when they differ from the polynomial default, so the keys of every
        parameter set written before they existed are unchanged."""
        out: dict[str, Any] = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "lo": self.lo,
            "hi": self.hi,
        }
        if self.form != "poly":
            out["form"] = self.form
        if self.k != 1.0:
            out["k"] = self.k
        return out

    def describe(self) -> str:
        """One-line algebraic form, for figure labels and logs."""
        if self.form == "power":
            body = f"{self.k:g}*{self.a:.4g}*h^{self.b:.4g}"
            return f"clip({body}, {self.lo:g}, {self.hi:g})"
        terms = [f"{self.a:g}"]
        if self.b:
            terms.append(f"{self.b:+g}h")
        if self.c:
            terms.append(f"{self.c:+g}h²")
        body = " ".join(terms)
        if self.k != 1.0:
            body = f"{self.k:g}*({body})"
        return f"clip({body}, {self.lo:g}, {self.hi:g})"


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


#: Values of the newer fields that reproduce the engine as it was before they
#: existed. ``as_dict`` omits a field holding its neutral value.
_NEUTRAL: dict[str, Any] = {
    "bound_mode": "post_trim",
    "th_cr_ref": "apex",
    "th_seed": 0.0,
    "min_crown_depth_m": None,
    "max_crown_depth_m": None,
    "window_metric": "pixel",
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
        bound_mode: How ``th_cr`` bounds a crown. ``"flood"`` applies each
            crown's admission test while the crowns grow, so a pixel one crown
            rejects can still join a neighbour (:mod:`.flood`). ``"post_trim"``
            is the earlier engine: an unconstrained watershed, then each basin
            trimmed to its own floor. Kept to reproduce
            ``docs/guides/crown_segmentation/optimization/``.
        th_cr_ref: What ``th_cr`` is relative to under ``"flood"``. ``"apex"``
            is the tree-top height. ``"mean"`` is the crown's running mean
            height, which is lidR's ``dalponte2016`` rule and is provided for
            comparison only.
        th_seed: Floor relative to the tree-top height, as lidR's ``th_seed``.
            Under the apex rule it binds only when it exceeds ``th_cr``.
        min_crown_depth_m: Under ``"flood"`` with the apex rule, every crown
            may extend at least this far below its apex, whatever ``th_cr``
            says. A fixed fraction of height is a thin slice on a short tree.
        max_crown_depth_m: The opposite bound for tall trees, the absolute
            height difference of Dalponte and Coomes (2016). ``None`` disables.
        window_metric: ``"exact"`` tests tree tops against true distances
            between pixel centres, as lidR's ``lmf`` does. ``"pixel"`` rounds
            the window radius to whole pixels first, the earlier behaviour,
            which at 1 m made a 5 m window the same as a 3 m one.
    """

    min_height_m: float = 2.0
    smooth_radius_m: float = 0.6
    window: WindowFunction = field(default_factory=lambda: WINDOW_FUNCTIONS["fixed_5m"])
    th_cr: float = 0.70
    max_crown_radius_m: float | None = 12.0
    min_crown_area_m2: float = 1.0
    min_separation_m: float = 1.0
    bound_mode: str = "flood"
    th_cr_ref: str = "apex"
    th_seed: float = 0.0
    min_crown_depth_m: float | None = None
    max_crown_depth_m: float | None = None
    window_metric: str = "exact"

    def __post_init__(self) -> None:
        if self.window_metric not in ("exact", "pixel"):
            raise ValueError(f"Unknown window_metric {self.window_metric!r}")
        if self.bound_mode not in ("flood", "post_trim"):
            raise ValueError(f"Unknown bound_mode {self.bound_mode!r}")
        if self.th_cr_ref not in ("apex", "mean"):
            raise ValueError(f"Unknown th_cr_ref {self.th_cr_ref!r}")
        if self.bound_mode == "post_trim" and (
            self.th_cr_ref != "apex"
            or self.th_seed
            or self.min_crown_depth_m is not None
            or self.max_crown_depth_m is not None
        ):
            raise ValueError(
                "post_trim supports th_cr only; th_cr_ref, th_seed and the crown "
                "depth bounds need bound_mode='flood'"
            )

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary, suitable for JSON and for a sidecar manifest.

        Fields added after the optimization study are left out while they hold
        the value that reproduces the earlier engine, so every key recorded
        there still names the same crowns.
        """
        out = asdict(self)
        out["window"] = self.window.as_dict()
        out["window_name"] = self.window.name or self.window.describe()
        for name, neutral in _NEUTRAL.items():
            if out[name] == neutral:
                del out[name]
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
        text = (
            f"smooth {self.smooth_radius_m:g} m, "
            f"ws {self.window.describe()}, "
            f"th_cr {self.th_cr:g}"
        )
        if self.th_cr_ref != "apex":
            text += f" of {self.th_cr_ref}"
        if self.th_seed:
            text += f", th_seed {self.th_seed:g}"
        if self.min_crown_depth_m is not None:
            text += f", min depth {self.min_crown_depth_m:g} m"
        if self.max_crown_depth_m is not None:
            text += f", max depth {self.max_crown_depth_m:g} m"
        return text + f", max radius {cap}, {self.bound_mode}"


#: The parameter set chosen by the optimization study, under the earlier
#: post-trim bound.
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
OPTIMIZED_POST_TRIM_PARAMS = SegmentationParams(
    bound_mode="post_trim", window_metric="pixel"
)

#: The default: the study's set under the flood bound.
#:
#: ``th_cr`` and ``min_crown_depth_m`` were chosen on the calibration units of
#: 626 NEON field plots by a rule fixed beforehand
#: (``scripts/segmentation/compare_lidr/before_after.py``). Its primary branch,
#: the most canopy assigned without crowns growing larger relative to field
#: crowns than under the post-trim bound, had no solution, because any bound
#: that keeps more canopy also enlarges crowns. The fallback ranks sets by the
#: calibration R² of a line on Tallo crown AGB, which barely moved across the
#: grid (0.333 to 0.340), and its tie-break kept ``th_cr`` at 0.70 with no
#: minimum depth. Against the post-trim bound at the same values, the median
#: unit puts 86 rather than 77 percent of its canopy in a crown, and tree tops
#: are unchanged. Lower ``th_cr`` (refAGB uses 0.55) assigns more canopy still.
DEFAULT_PARAMS = SegmentationParams()

#: The legacy window equation under the corrected semantics. Useful for asking
#: how much of the old behaviour was the equation, as opposed to the way the
#: equation was applied. This is **not** what the pipeline used to produce.
LEGACY_PARAMS = SegmentationParams(
    bound_mode="post_trim",
    window_metric="pixel",
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
    bound_mode="post_trim",
    window_metric="pixel",
    smooth_radius_m=0.6,
    window=WindowFunction(a=4.0, b=1.0, lo=6.0, hi=24.0, name="legacy_as_shipped"),
    th_cr=0.0,
    max_crown_radius_m=None,
)


@lru_cache(maxsize=1)
def _tallo_window_table() -> pd.DataFrame:
    ref = resources.files("csdv_core.segmentation").joinpath(
        "coefs/tallo_window_coefs.csv"
    )
    with resources.as_file(ref) as path:
        return pd.read_csv(path, comment="#").set_index("biome")


#: RESOLVE biomes with a Tallo crown-width fit.
TALLO_BIOMES: tuple[str, ...] = tuple(_tallo_window_table().index)


def tallo_window(
    biome: str, *, k: float = 0.5, lo: float = 3.0, hi: float = 25.0
) -> WindowFunction:
    """The refAGB Tallo window for one RESOLVE biome.

    ``w(h) = clip(k * a * h**b, lo, hi)`` with ``a`` and ``b`` from a power-law
    crown-width fit to the Tallo database. At ``k = 0.5`` the 3 m floor binds
    below about 16 to 32 m depending on the biome, so for most trees the window
    is a fixed 3 m.
    """
    table = _tallo_window_table()
    if biome not in table.index:
        raise KeyError(f"No Tallo window for biome {biome!r}; known: {TALLO_BIOMES}")
    row = table.loc[biome]
    return WindowFunction(
        a=float(row["a"]),
        b=float(row["b"]),
        lo=lo,
        hi=hi,
        k=k,
        form="power",
        name=f"tallo:{biome}",
    )


def tallo_params(biome: str, **changes: Any) -> SegmentationParams:
    """The refAGB production segmentation parameters for one biome.

    refAGB (``config/site_configs/tallo_2``) runs lidR on 1 m lidar CHMs with
    ``hmin = th_tree = 1``, ``th_seed = 0.45``, ``th_cr = 0.55``,
    ``max_cr = 15`` pixels, ``min_area = 1`` and a 3x3 mean filter. In metres
    that is a 1 m canopy floor, a 1 m smoothing radius (a 3 m wide filter) and
    a 15 m crown radius cap, which this function returns so the set means the
    same thing at 0.6 m. Note that lidR applies ``max_cr`` as a square
    half-width, ``|dx| < max_cr``, not as a radius or diameter.

    ``th_cr`` here is relative to the apex, which differs from lidR's
    mean-relative ``th_cr``. Pass ``th_cr_ref="mean"`` for lidR's rule.
    """
    base = SegmentationParams(
        min_height_m=1.0,
        smooth_radius_m=1.0,
        window=tallo_window(biome),
        th_cr=0.55,
        th_seed=0.45,
        max_crown_radius_m=15.0,
        min_crown_area_m2=1.0,
    )
    return base.replace(**changes) if changes else base
