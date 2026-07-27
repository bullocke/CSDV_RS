"""Tests for shared figure styling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from csdv_core.viz.style import (  # noqa: E402
    STAGE_COLORS,
    add_scale_bar,
    panel_label,
    save_fig,
    setup_style,
)


def _image_axis():
    """An axis showing an image, so the y axis runs top to bottom."""
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((200, 300)))
    return fig, ax


def test_scale_bar_sits_at_the_visual_bottom_of_an_image_axis():
    """imshow inverts the y axis. Treating the smaller limit as the bottom puts
    the bar in the top corner, over the image."""
    fig, ax = _image_axis()
    y_bottom, y_top = ax.get_ylim()
    assert y_bottom > y_top  # inverted, as imshow leaves it
    add_scale_bar(ax, 1.0, bar_m=50.0)
    bar = ax.patches[0]
    midpoint = (y_bottom + y_top) / 2
    assert bar.get_y() > midpoint  # nearer the bottom in data terms
    plt.close(fig)


def test_scale_bar_sits_at_the_bottom_of_an_ordinary_axis():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 200)
    add_scale_bar(ax, 1.0, bar_m=50.0)
    bar = ax.patches[0]
    assert bar.get_y() < 100
    assert bar.get_height() > 0
    plt.close(fig)


def test_scale_bar_length_reflects_pixel_size():
    fig, ax = _image_axis()
    add_scale_bar(ax, 0.6, bar_m=60.0)
    assert ax.patches[0].get_width() == pytest.approx(100.0)
    plt.close(fig)


def test_scale_bar_is_skipped_when_it_would_not_fit():
    fig, ax = _image_axis()
    add_scale_bar(ax, 1.0, bar_m=10_000.0)
    assert not ax.patches
    plt.close(fig)


def test_scale_bar_is_skipped_for_an_unknown_pixel_size():
    fig, ax = _image_axis()
    add_scale_bar(ax, 0.0, bar_m=50.0)
    assert not ax.patches
    plt.close(fig)


def test_scale_bar_does_not_change_the_axis_limits():
    fig, ax = _image_axis()
    before = (ax.get_xlim(), ax.get_ylim())
    add_scale_bar(ax, 1.0, bar_m=50.0)
    assert (ax.get_xlim(), ax.get_ylim()) == before
    plt.close(fig)


def test_panel_label_is_parenthesised_once():
    fig, ax = plt.subplots()
    panel_label(ax, "a")
    panel_label(ax, "(b)")
    assert [t.get_text() for t in ax.texts] == ["(a)", "(b)"]
    plt.close(fig)


def test_setup_style_is_idempotent():
    setup_style()
    first = plt.rcParams["font.size"]
    setup_style()
    assert plt.rcParams["font.size"] == first


def test_stage_colors_cover_every_stage_in_order():
    assert list(STAGE_COLORS) == ["ESI", "LSI", "ESE", "LSE", "UR", "MA_OW", "OG"]


def test_save_fig_creates_the_parent_directory(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_fig(fig, tmp_path / "nested" / "deeper" / "plot.png")
    assert out.exists()
    assert out.stat().st_size > 0
