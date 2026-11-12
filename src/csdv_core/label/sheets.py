"""csdv_core.label.sheets: review sheets of labelled windows.

Inputs: a window's imagery stack, its label rasters and its candidates.
Outputs: PNG images. An overview sheet shows every date of a window with the
label of each interval. A pair sheet shows one training sample as the model
will see it: the two dates, the label at 0.6 m and the target pooled to
output cells.

Colours follow one scheme everywhere. Loss is red. Where the change mask keeps
two severities, partial loss is amber, and the map of reasons always shows
which severity the models gave. Background is green (forest), light green (inside a candidate the
models judged) or blue (non-forest). Unlabelled ground is grey and hatched,
darker for precautions without a verdict. The hues pass a colour-blindness
check as a set, and the hatching and the outlines carry the same information
without colour.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from csdv_core.config._models import LabelPooling
from csdv_core.label.labels import JudgedCandidate, WindowLabels
from csdv_core.label.lattice import GSD, Bounds, pixel_slices
from csdv_core.label.pooling import pool_change
from csdv_core.label.render import BANDS, font, stretch_limits, to_rgb
from csdv_core.label.rules import IGNORE, REASON_GROUPS, reason_group
from csdv_core.label.stacks import WindowStack
from csdv_core.label.views import View

#: Fill of each reason group, in the order of ``REASON_GROUPS``.
GROUP_COLOURS: dict[str, tuple[int, int, int]] = {
    "loss_full": (227, 73, 72),
    "loss_partial": (237, 161, 0),
    "ignore_outline": (214, 213, 207),
    "ignore_verdict": (155, 154, 148),
    "ignore_guard": (95, 94, 89),
    "background_verified": (140, 200, 120),
    "background_nonforest": (42, 120, 214),
    "background_forest": (0, 131, 0),
}
GROUP_LABELS: dict[str, str] = {
    "loss_full": "loss, stand-replacing",
    "loss_partial": "loss, partial",
    "ignore_outline": "ignore: outline band or extension",
    "ignore_verdict": "ignore: models disagree or reject",
    "ignore_guard": "ignore: precaution, no verdict",
    "background_verified": "background, judged by models",
    "background_nonforest": "background, non-forest",
    "background_forest": "background, forest (not judged)",
}
#: Fill of the pooled target: background, partial, stand-replacing, ignore.
#: Background is a pale neutral, not a green: the target has one background
#: class for forest and non-forest alike, and the reason map beside it uses
#: green for forest only.
CHANGE_COLOURS = {
    0: (214, 210, 198),
    1: (237, 161, 0),
    2: (227, 73, 72),
    IGNORE: (155, 154, 148),
}
#: Outline of a candidate by its routed status.
STATUS_COLOURS: dict[str, tuple[int, int, int]] = {
    "accepted": (255, 255, 255),
    "date_disputed": (255, 120, 220),
    "disagree": (255, 120, 220),
    "failed": (255, 120, 220),
    "rejected": (0, 220, 235),
    "rejected_weak": (0, 220, 235),
    "unjudged": (190, 190, 190),
}
INK = (20, 20, 20)
MUTED = (90, 90, 90)
PAPER = (252, 252, 251)

__all__ = [
    "CHANGE_COLOURS",
    "GROUP_COLOURS",
    "GROUP_LABELS",
    "STATUS_COLOURS",
    "WindowView",
    "candidate_lines",
    "caption",
    "cell_image",
    "choose_pairs",
    "date_image",
    "draw_candidates",
    "header",
    "label_overlay",
    "legend",
    "overview_sheet",
    "pair_note",
    "pair_sheet",
    "reason_image",
    "row_of",
    "stack_rows",
]


def _hatch(shape: tuple[int, int], step: int = 8) -> np.ndarray:
    rows, cols = np.indices(shape)
    return (rows + cols) % step == 0


def label_overlay(
    rgb: np.ndarray, reason: np.ndarray, single_loss: bool = False
) -> np.ndarray:
    """An image with its label drawn over it.

    Loss is tinted red or amber and non-forest background blue. Unlabelled
    ground is veiled and hatched. Forest background is left clear, so the
    canopy stays visible.

    Args:
        rgb: ``(rows, cols, 3)`` uint8 image.
        reason: ``(rows, cols)`` reason codes on the same grid.
        single_loss: The label has one loss class, so all loss is tinted red
            whatever severity the models gave.
    """
    out = rgb.astype(np.float32)
    group = reason_group(reason)
    names = list(REASON_GROUPS)
    tints = {
        "loss_full": 0.5,
        "loss_partial": 0.5,
        "background_nonforest": 0.3,
        "ignore_outline": 0.3,
        "ignore_verdict": 0.4,
        "ignore_guard": 0.5,
    }
    for name, alpha in tints.items():
        mask = group == names.index(name)
        if mask.any():
            fill = "loss_full" if single_loss and name == "loss_partial" else name
            colour = np.array(GROUP_COLOURS[fill], dtype=np.float32)
            out[mask] = (1 - alpha) * out[mask] + alpha * colour
    ignored = np.isin(group, [names.index(n) for n in names if n.startswith("ignore")])
    out[ignored & _hatch(reason.shape)] = 245.0
    return out.astype(np.uint8)


def reason_image(reason: np.ndarray) -> np.ndarray:
    """Reason groups as flat colours, with unlabelled ground hatched."""
    group = reason_group(reason)
    table = np.zeros((256, 3), dtype=np.uint8)
    names = list(REASON_GROUPS)
    for index, name in enumerate(names):
        table[index] = GROUP_COLOURS[name]
    out = table[group]
    ignored = np.isin(group, [names.index(n) for n in names if n.startswith("ignore")])
    out[ignored & _hatch(reason.shape)] = (252, 252, 251)
    return out


def cell_image(pooled: np.ndarray, size: int, single_loss: bool = False) -> Image.Image:
    """A pooled change mask as flat colours, enlarged to ``size`` pixels.

    With ``single_loss`` the value 1 means loss of any severity and is drawn
    in the loss colour, not the partial one.
    """
    table = np.zeros((256, 3), dtype=np.uint8)
    for value, colour in CHANGE_COLOURS.items():
        table[value] = colour
    if single_loss:
        table[1] = CHANGE_COLOURS[2]
    img = Image.fromarray(table[pooled]).resize((size, size), Image.NEAREST)
    cells = pooled.shape[0]
    if size // cells >= 8:
        draw = ImageDraw.Draw(img)
        step = size / cells
        for i in range(1, cells):
            draw.line([(i * step, 0), (i * step, size)], fill=(252, 252, 251), width=1)
            draw.line([(0, i * step), (size, i * step)], fill=(252, 252, 251), width=1)
    return img


def legend(width: int, pooled: bool = False, single_loss: bool = False) -> Image.Image:
    """A legend strip: every fill with its meaning, text in ink."""
    f = font(15)
    labels = dict(GROUP_LABELS)
    if single_loss:
        labels["loss_full"] = "loss (models said stand-replacing)"
        labels["loss_partial"] = "loss (models said partial)"
    entries = [
        (GROUP_COLOURS[n], labels[n], n.startswith("ignore")) for n in REASON_GROUPS
    ]
    if pooled and single_loss:
        entries = [
            (CHANGE_COLOURS[2], "cell: loss", False),
            (CHANGE_COLOURS[0], "cell: background (no loss, forest or not)", False),
            (CHANGE_COLOURS[IGNORE], "cell: ignore", False),
        ]
    elif pooled:
        entries = [
            (CHANGE_COLOURS[2], "cell: stand-replacing", False),
            (CHANGE_COLOURS[1], "cell: partial", False),
            (CHANGE_COLOURS[0], "cell: background (no loss, forest or not)", False),
            (CHANGE_COLOURS[IGNORE], "cell: ignore", False),
        ]
    lines = [
        ("accepted", "outline: accepted"),
        ("rejected", "outline: rejected by both"),
        ("disagree", "outline: disagree or date disputed"),
        ("unjudged", "outline: not judged"),
    ]
    img = Image.new("RGB", (width, 56), PAPER)
    draw = ImageDraw.Draw(img)
    x, y = 8, 6
    for colour, text, hatched in entries:
        w = draw.textlength(text, font=f) + 34
        if x + w > width:
            x, y = 8, y + 24
        draw.rectangle([x, y + 1, x + 18, y + 17], fill=colour)
        if hatched:
            for k in range(0, 18, 6):
                draw.line([x + k, y + 17, x + k + 6, y + 11], fill=PAPER, width=1)
        draw.text((x + 24, y), text, fill=INK, font=f)
        x += int(w) + 14
    if not pooled:
        for status, text in lines:
            w = draw.textlength(text, font=f) + 34
            if x + w > width:
                x, y = 8, y + 24
            draw.line([x, y + 9, x + 18, y + 9], fill=(0, 0, 0), width=5)
            draw.line([x, y + 9, x + 18, y + 9], fill=STATUS_COLOURS[status], width=3)
            draw.text((x + 24, y), text, fill=INK, font=f)
            x += int(w) + 14
    return img


def draw_candidates(
    img: Image.Image, view: View, candidates: list[JudgedCandidate], k: int | None
) -> None:
    """Candidate outlines by status. With ``k``, only those live in interval ``k``."""
    draw = ImageDraw.Draw(img)
    for cand in candidates:
        if k is not None and not (
            k < len(cand.routed.outcomes) and cand.routed.outcomes[k]
        ):
            continue
        colour = STATUS_COLOURS.get(cand.routed.status, (190, 190, 190))
        width = 1 if cand.routed.status == "unjudged" else 2
        for ring in view.rings_px(cand.geometry):
            draw.line(ring, fill=(0, 0, 0), width=width + 2, joint="curve")
            draw.line(ring, fill=colour, width=width, joint="curve")
        if cand.event is not None and not cand.event.extension.is_empty:
            polygon = cand.event.core.union(cand.event.extension)
            for ring in view.rings_px(polygon):
                draw.line(ring, fill=(0, 0, 0), width=3)
                draw.line(ring, fill=(255, 235, 0), width=1)


def caption(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    f = font(max(13, img.width // 34))
    w = draw.textlength(text, font=f)
    draw.rectangle([0, 0, w + 10, f.size + 8], fill=(0, 0, 0))
    draw.text((5, 3), text, fill=(255, 255, 255), font=f)
    return img


def header(lines: list[str], width: int, size: int = 16) -> Image.Image:
    f = font(size)
    wrapped: list[str] = []
    per_line = max(40, int(width / (size * 0.56)))
    for line in lines:
        wrapped += textwrap.wrap(line, per_line, subsequent_indent="    ") or [""]
    img = Image.new("RGB", (width, (size + 6) * len(wrapped) + 12), PAPER)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(wrapped):
        draw.text((8, 6 + i * (size + 6)), line, fill=INK if i == 0 else MUTED, font=f)
    return img


def stack_rows(parts: list[Image.Image], gap: int = 4) -> Image.Image:
    width = max(p.width for p in parts)
    sheet = Image.new(
        "RGB", (width, sum(p.height for p in parts) + gap * (len(parts) - 1)), PAPER
    )
    y = 0
    for part in parts:
        sheet.paste(part, (0, y))
        y += part.height + gap
    return sheet


def row_of(panels: list[Image.Image], gap: int = 6) -> Image.Image:
    height = max(p.height for p in panels)
    sheet = Image.new(
        "RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), height), PAPER
    )
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    return sheet


def date_image(
    stack: WindowStack, index: int, bounds: Bounds, bands: str
) -> np.ndarray:
    block = stack.data[index]
    limits = stretch_limits(block, BANDS[bands])
    rows, cols = pixel_slices(bounds, stack.bounds)
    return to_rgb(block[:, rows, cols], limits, bands)


def candidate_lines(candidates: list[JudgedCandidate], limit: int = 14) -> list[str]:
    """One text line per judged candidate: status, type and what each model said."""
    lines = []
    shown = [c for c in candidates if c.routed.status != "unjudged"]
    for cand in shown[:limit]:
        r = cand.routed
        types = "/".join(str(t) for t in r.change_types)
        fits = "/".join(str(f) for f in r.outline_fit)
        extra = ""
        if cand.event is not None:
            extra = f", outline {cand.event.source} ({cand.event.quality})"
        lines.append(
            f"{cand.cand_id} [{cand.source}]: {r.status}, {' '.join(_short(r.outcomes))}"
            f", type {types}, fit {fits}, confidence {r.confidence}{extra}"
        )
    if len(shown) > limit:
        lines.append(f"... and {len(shown) - limit} more judged candidates")
    return lines


_SHORT = {
    "loss_full": "FULL",
    "loss_partial": "PART",
    "loss_disputed": "LOSS?",
    "disagree": "dis",
    "date_disputed": "date?",
    "intact": "ok",
    "open": "open",
    "rejected_flagged": "REJ",
    "rejected_weak": "rej?",
    "flagged_unconfirmed": "flag?",
    "loss_repeated": "again?",
    "stressed": "stress",
    "failed": "fail",
    "unjudged": "-",
    "under_min": "small",
    "": ".",
}


def _short(outcomes: list[str]) -> list[str]:
    return [_SHORT.get(o, o) for o in outcomes]


def choose_pairs(labels: WindowLabels, window, max_sheets: int = 4) -> list[tuple]:
    """Chip and interval pairs worth a pair sheet.

    The two with the most loss, the one with the most unlabelled ground
    among the rest (if over 20%), and one quiet pair. Each entry is
    ``(chip, interval index, loss pixels, ignore share)``.
    """
    scored = []
    for chip in window.chips():
        r, c = pixel_slices(chip.bounds, window.bounds)
        for k in range(len(labels.intervals)):
            change = labels.change[k][r, c]
            loss = int(((change == 1) | (change == 2)).sum())
            scored.append((chip, k, loss, float((change == IGNORE).mean())))
    by_loss = sorted((s for s in scored if s[2] > 0), key=lambda s: -s[2])[:2]
    rest = [s for s in scored if s not in by_loss]
    by_ignore = sorted((s for s in rest if s[3] > 0.2), key=lambda s: -s[3])[:1]
    rest = [s for s in rest if s not in by_ignore]
    quiet = sorted((s for s in rest if s[2] == 0), key=lambda s: s[3])[:1]
    return (by_loss + by_ignore + quiet)[:max_sheets]


def pair_note(
    stack: WindowStack, labels: WindowLabels, k: int, loss_px: int, ignore: float
) -> str:
    """One line of numbers for the header of a pair sheet."""
    note = (
        f"0.6 m label: {loss_px * GSD * GSD / 1e4:.2f} ha loss, {ignore:.0%} ignore. "
        f"Native resolution {stack.native_m[k]} m to {stack.native_m[k + 1]} m."
    )
    if labels.unstable[k]:
        note += "  The forest mask is unreliable in this interval."
    return note


@dataclass
class WindowView:
    """What a sheet needs about one window."""

    window_id: str
    title: str
    bounds: Bounds
    stack: WindowStack
    labels: WindowLabels
    candidates: list[JudgedCandidate]
    #: The change mask has one loss class (the policy maps no reason to 2).
    single_loss: bool = False


def overview_sheet(win: WindowView, panel_px: int = 384) -> Image.Image:
    """Every date of a window, and the label of every interval.

    Row 1 is natural colour, row 2 colour-infrared, row 3 the later date of
    each interval with its label and the candidates live in that interval.
    """
    stack, labels = win.stack, win.labels
    edge = max(win.bounds[2] - win.bounds[0], win.bounds[3] - win.bounds[1])
    scale = panel_px / (edge / GSD)
    size = (
        max(1, round((win.bounds[2] - win.bounds[0]) / GSD * scale)),
        max(1, round((win.bounds[3] - win.bounds[1]) / GSD * scale)),
    )
    view = View(
        (win.bounds[0], win.bounds[3] - edge, win.bounds[0] + edge, win.bounds[3]),
        panel_px,
    )
    rows = []
    for bands in ("rgb", "cir"):
        panels = []
        for i, date in enumerate(stack.dates):
            img = Image.fromarray(date_image(stack, i, win.bounds, bands))
            img = img.resize(size, Image.LANCZOS)
            res = stack.native_m[i]
            tag = f"{date}" + (f"  {res:g} m" if res else "")
            panels.append(caption(img, tag if bands == "rgb" else f"{date} infrared"))
        rows.append(row_of(panels))
    panels = []
    first = Image.fromarray(date_image(stack, 0, win.bounds, "rgb")).resize(
        size, Image.LANCZOS
    )
    draw_candidates(first, view, win.candidates, None)
    panels.append(caption(first, "all candidates"))
    for k in range(len(labels.intervals)):
        rgb = date_image(stack, k + 1, win.bounds, "rgb")
        img = Image.fromarray(
            label_overlay(rgb, labels.reason[k], win.single_loss)
        ).resize(size, Image.LANCZOS)
        draw_candidates(img, view, win.candidates, k)
        note = " (forest mask unreliable)" if labels.unstable[k] else ""
        panels.append(
            caption(img, f"label {stack.dates[k][:4]}-{stack.dates[k + 1][:4]}{note}")
        )
    rows.append(row_of(panels))
    body = stack_rows(rows)
    head = header([win.title, *candidate_lines(win.candidates)], body.width)
    return stack_rows([head, body, legend(body.width, single_loss=win.single_loss)])


def pair_sheet(
    win: WindowView,
    chip_bounds: Bounds,
    chip_id: str,
    k: int,
    pooling: LabelPooling,
    notes: list[str] | None = None,
) -> Image.Image:
    """One training sample at native resolution.

    Panels: the earlier date, the later date, the later date with the 0.6 m
    label, the reasons as flat colours, and the target pooled to the
    coarsest cell size, with the finer cell sizes beside it.
    """
    stack, labels = win.stack, win.labels
    rows, cols = pixel_slices(chip_bounds, win.bounds)
    reason = labels.reason[k][rows, cols]
    change = labels.change[k][rows, cols]
    px = reason.shape[0]
    view = View(chip_bounds, px)
    pre = Image.fromarray(date_image(stack, k, chip_bounds, "rgb"))
    post = Image.fromarray(date_image(stack, k + 1, chip_bounds, "rgb"))
    over = Image.fromarray(label_overlay(np.asarray(post), reason, win.single_loss))
    draw_candidates(over, view, win.candidates, k)
    flat = Image.fromarray(reason_image(reason))
    cells = sorted(pooling.cells_m)
    coarse = pool_change(change, round(cells[-1] / GSD), pooling)
    target = cell_image(coarse, px, win.single_loss)
    small = [
        caption(
            cell_image(
                pool_change(change, round(c / GSD), pooling),
                px // 2 - 2,
                win.single_loss,
            ),
            f"{c:g} m cells",
        )
        for c in cells[:-1]
    ]
    side = stack_rows(small) if small else None
    panels = [
        caption(pre, f"earlier  {stack.dates[k]}"),
        caption(post, f"later  {stack.dates[k + 1]}"),
        caption(over, "later date with the 0.6 m label"),
        caption(flat, "why each pixel has its label"),
        caption(target, f"target, {cells[-1]:g} m cells"),
    ]
    if side is not None:
        panels.append(side)
    body = row_of(panels)
    live = [
        c for c in win.candidates if k < len(c.routed.outcomes) and c.routed.outcomes[k]
    ]
    head = header(
        [
            f"{win.title}   chip {chip_id}   interval {labels.intervals[k].replace('_', ' to ')}",
            *(notes or []),
            *candidate_lines(live, limit=8),
        ],
        body.width,
    )
    return stack_rows(
        [
            head,
            body,
            legend(body.width, single_loss=win.single_loss),
            legend(body.width, pooled=True, single_loss=win.single_loss),
        ]
    )
