"""csdv_core.label.render: draw a stack as images for the vision models.

Everything is drawn with PIL on the pixel grid of a
:class:`~csdv_core.label.views.View`, with no figure margins, so a coordinate
a model returns maps straight back to the ground.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry.base import BaseGeometry

from csdv_core.label.lattice import pixel_slices
from csdv_core.label.stacks import WindowStack
from csdv_core.label.views import View

LETTERS = "ABCDEFGHIJKL"
BANDS = {"rgb": (0, 1, 2), "cir": (3, 0, 1)}
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
)

__all__ = [
    "BANDS",
    "LETTERS",
    "captions",
    "draw_outline",
    "font",
    "jpeg_bytes",
    "montage",
    "render_date",
    "stretch_limits",
    "to_rgb",
]


def font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stretch_limits(
    block: np.ndarray, bands: tuple[int, int, int]
) -> tuple[float, float]:
    """One shared 1 to 99 percentile range for the three bands of a date.

    Computed on the whole stack of that date, so every view of a date has the
    same colours.
    """
    sample = block[list(bands)][:, ::4, ::4].astype(np.float32)
    sample = sample[:, sample.any(axis=0)] if sample.any() else sample
    lo, hi = np.percentile(sample, [1.0, 99.0])
    return float(lo), float(max(hi, lo + 1.0))


def to_rgb(
    block: np.ndarray,
    limits: tuple[float, float],
    bands: str = "rgb",
    gamma: float = 1.3,
) -> np.ndarray:
    """A ``(4, rows, cols)`` block as an 8-bit ``(rows, cols, 3)`` image."""
    lo, hi = limits
    part = block[list(BANDS[bands])].astype(np.float32)
    part = np.clip((part - lo) / (hi - lo), 0.0, 1.0) ** (1.0 / gamma)
    return (part.transpose(1, 2, 0) * 255).astype(np.uint8)


def draw_outline(
    img: Image.Image,
    view: View,
    geom: BaseGeometry,
    colour: tuple[int, int, int] = (255, 255, 0),
    width: int = 2,
    halo: bool = False,
) -> None:
    """Draw every ring of ``geom`` on ``img``."""
    draw = ImageDraw.Draw(img)
    for ring in view.rings_px(geom):
        if halo:
            draw.line(ring, fill=(0, 0, 0), width=width + 2, joint="curve")
        draw.line(ring, fill=colour, width=width, joint="curve")


def render_date(
    stack: WindowStack,
    index: int,
    view: View,
    *,
    bands: str = "rgb",
    caption: str | None = None,
    outline: BaseGeometry | None = None,
) -> Image.Image:
    """One date of a stack as an image of ``view``."""
    block = stack.data[index]
    limits = stretch_limits(block, BANDS[bands])
    rows, cols = pixel_slices(view.bounds, stack.bounds)
    img = Image.fromarray(to_rgb(block[:, rows, cols], limits, bands))
    if img.size != (view.px, view.px):
        img = img.resize((view.px, view.px), Image.LANCZOS)
    if outline is not None:
        draw_outline(img, view, outline)
    if caption:
        draw = ImageDraw.Draw(img)
        f = font(max(14, view.px // 36))
        w = draw.textlength(caption, font=f)
        draw.rectangle([0, 0, w + 10, f.size + 8], fill=(0, 0, 0))
        draw.text((5, 3), caption, fill=(255, 255, 255), font=f)
    return img


def captions(dates: list[str]) -> list[str]:
    return [f"{LETTERS[i]}  {d}" for i, d in enumerate(dates)]


def montage(
    images: list[Image.Image], cols: int | None = None, gap: int = 6
) -> Image.Image:
    """Tile images left to right, top to bottom, on a white sheet."""
    cols = cols or min(len(images), 4)
    rows = -(-len(images) // cols)
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    sheet = Image.new(
        "RGB",
        (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap),
        (255, 255, 255),
    )
    for i, img in enumerate(images):
        sheet.paste(img, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return sheet


def jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
