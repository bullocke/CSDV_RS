"""csdv_core.label.requests: build the requests sent to the vision models.

Inputs: a window stack and one candidate.
Outputs: a :class:`Request` with the prompt and the JPEG images.

Two requests exist. A verdict request shows every date with the candidate
outlined and asks for the state of the canopy on each date. A polygon
request shows the two dates around a confirmed loss with a cross inside it
and asks for the outline of the disturbed area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from PIL import ImageDraw
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from csdv_core.config._models import LabelViews
from csdv_core.label.lattice import GSD
from csdv_core.label.render import LETTERS, captions, jpeg_bytes, render_date
from csdv_core.label.stacks import WindowStack
from csdv_core.label.views import View, _inside, _snap, hybrid_view

#: Frozen prompt versions. A new wording gets a new file and a new version.
VERDICT_PROMPT = "verdict_v2"
POLYGON_PROMPT = "polygon_v1"

#: Location text given to the models: state and physiographic region. Site
#: labels such as "CA_CampFire" name the answer and are never shown.
LOCATIONS: dict[str, str] = {
    "q3208840": "western Alabama, Gulf Coastal Plain",
    "q3508207": "western North Carolina, southern Blue Ridge Mountains",
    "q3508321": "eastern Tennessee, southern Appalachian Mountains",
    "q3511151": "northern Arizona, Colorado Plateau ponderosa pine forest",
    "q3708104": "southern West Virginia, Appalachian Plateau",
    "q3709154": "southern Missouri, Ozark Highlands",
    "q3907420": "southern New Jersey, Atlantic Coastal Plain pine barrens",
    "q3908654": "south-central Indiana, Interior Low Plateau hills",
    "q3912112": "northern California, Sierra Nevada foothills",
    "q4010549": "north-central Colorado, Southern Rocky Mountains",
    "q4107127": "Rhode Island, southern New England",
    "q4407152": "northern New Hampshire, White Mountains",
    "q4407462": "northern New York, Adirondack Mountains",
    "q4412328": "western Oregon, Coast Range",
    "q4506921": "northern Maine, Acadian forest",
    "q4612245": "western Washington, Cascade foothills",
    "q4709438": "northern Minnesota, Laurentian mixed forest",
}

__all__ = [
    "LOCATIONS",
    "POLYGON_PROMPT",
    "VERDICT_PROMPT",
    "Request",
    "location_of",
    "polygon_request",
    "polygon_view",
    "prompt_text",
    "verdict_request",
]


@dataclass
class Request:
    """One request about one candidate."""

    cand_id: str
    kind: str  # verdict or polygon
    prompt: str
    images: list[bytes]
    dates: list[str]
    view: View
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> dict[str, Any]:
        return {"cand_id": self.cand_id, "kind": self.kind, **self.extra}


def prompt_text(name: str) -> str:
    """A packaged prompt template by name (``verdict_v2``)."""
    path = resources.files("csdv_core.label") / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def location_of(unit: str) -> str:
    """Location text of a unit.

    Raises:
        KeyError: If the unit has none. Add it to :data:`LOCATIONS`: a prompt
            without a location has not been tested.
    """
    return LOCATIONS[unit]


def verdict_request(
    stack: WindowStack,
    cand_id: str,
    geom: BaseGeometry,
    unit: str,
    cfg: LabelViews,
) -> Request:
    """Every date of the stack with the candidate outlined in yellow."""
    view = hybrid_view(geom, stack.bounds, cfg, key=cand_id)
    outline = geom.intersection(box(*stack.bounds))
    caps = captions(stack.dates)
    images = [
        jpeg_bytes(
            render_date(stack, i, view, caption=caps[i], outline=outline),
            cfg.jpeg_quality,
        )
        for i in range(len(stack.dates))
    ]
    prompt = prompt_text(VERDICT_PROMPT).format(
        n_images=len(images),
        edge_m=view.edge_m,
        location=location_of(unit),
        layout_note=(
            "Each image has its letter and acquisition date in its top-left corner."
        ),
        date_list="\n".join(f"{LETTERS[i]}: {d}" for i, d in enumerate(stack.dates)),
        band_note="The images are natural colour.",
    )
    return Request(
        cand_id,
        "verdict",
        prompt,
        images,
        list(stack.dates),
        view,
        {"prompt": VERDICT_PROMPT},
    )


def polygon_view(geom: BaseGeometry, stack: WindowStack, cfg: LabelViews) -> View:
    """A view wide enough to show an event larger than its candidate.

    Three times the candidate's longer side, at least the standard view and
    at most the stack, centred on the candidate.
    """
    seen = geom.intersection(box(*stack.bounds))
    minx, miny, maxx, maxy = (seen if not seen.is_empty else geom).bounds
    extent = max(maxx - minx, maxy - miny)
    stack_edge = stack.bounds[2] - stack.bounds[0]
    edge = _snap(min(max(3.0 * extent, cfg.detail_m), stack_edge))
    px = int(min(cfg.max_px, max(cfg.px, round(edge / GSD))))
    x0 = _snap((minx + maxx) / 2 - edge / 2)
    y0 = _snap((miny + maxy) / 2 - edge / 2)
    return View(_inside((x0, y0, x0 + edge, y0 + edge), stack.bounds), px)


def polygon_request(
    stack: WindowStack,
    cand_id: str,
    geom: BaseGeometry,
    unit: str,
    pre: int,
    post: int,
    cfg: LabelViews,
) -> Request:
    """The dates before and after a loss, with a cross inside the candidate."""
    view = polygon_view(geom, stack, cfg)
    seen = geom.intersection(box(*view.bounds))
    point = (seen if not seen.is_empty else geom).representative_point()
    x, y = view.to_px(point.x, point.y)

    def image(index: int, letter: str) -> bytes:
        img = render_date(stack, index, view, caption=f"{letter}  {stack.dates[index]}")
        draw = ImageDraw.Draw(img)
        for extra, colour in ((2, (0, 0, 0)), (0, (0, 255, 255))):
            draw.line([x - 9, y, x + 9, y], fill=colour, width=3 + extra)
            draw.line([x, y - 9, x, y + 9], fill=colour, width=3 + extra)
        return jpeg_bytes(img, cfg.jpeg_quality)

    prompt = prompt_text(POLYGON_PROMPT).format(
        edge_m=view.edge_m,
        location=location_of(unit),
        pre_date=stack.dates[pre],
        post_date=stack.dates[post],
        type_note="a forest disturbance",
    )
    return Request(
        cand_id,
        "polygon",
        prompt,
        [image(pre, "A"), image(post, "B")],
        [stack.dates[pre], stack.dates[post]],
        view,
        {"prompt": POLYGON_PROMPT, "pre": pre, "post": post},
    )
