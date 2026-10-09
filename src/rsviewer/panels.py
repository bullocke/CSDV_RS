"""rsviewer.panels: side-panel widgets.

Each panel is an ipywidgets container with a small Python API, so project code
can build its own panels from the same parts:

* :class:`LayerPanel`: grouped on/off checkboxes, opacity sliders, legends.
* :class:`SelectorPanel`: a set of dropdowns, toggles or sliders that calls one
  function with all current values whenever any of them changes.
* :class:`PlotPanel`: a Plotly ``FigureWidget`` with a fixed width so it fits
  the side panel, plus a status line.
* :class:`InfoPanel`: an HTML readout for clicked features and pixels.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable, Sequence
from typing import Any

import ipywidgets as w
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

#: Width of the side panel and everything in it, in pixels.
PANEL_WIDTH = 440


class LayerPanel(w.VBox):
    """Grouped layer toggles with opacity sliders and inline legends.

    Args:
        on_visible: Called as ``on_visible(name, visible)``.
        on_opacity: Called as ``on_opacity(name, opacity)``.
    """

    def __init__(
        self,
        on_visible: Callable[[str, bool], None],
        on_opacity: Callable[[str, float], None],
    ) -> None:
        super().__init__(layout=w.Layout(width=f"{PANEL_WIDTH}px"))
        self._on_visible = on_visible
        self._on_opacity = on_opacity
        self._groups: dict[str, w.VBox] = {}
        self._rows: dict[str, tuple[str, w.VBox]] = {}
        self._checks: dict[str, w.Checkbox] = {}
        self._acc = w.Accordion(children=[])
        self.children = [self._acc]

    def _group_box(self, group: str) -> w.VBox:
        if group not in self._groups:
            box = w.VBox()
            self._groups[group] = box
            self._acc.children = [*self._acc.children, box]
            self._acc.set_title(len(self._acc.children) - 1, group)
        return self._groups[group]

    def add(
        self,
        name: str,
        group: str,
        visible: bool,
        opacity: float,
        legend_html: str = "",
        description: str = "",
    ) -> None:
        """Add a row for a layer, or replace the row of a layer with this name."""
        check = w.Checkbox(
            value=visible,
            description=name,
            indent=False,
            layout=w.Layout(width="250px"),
            tooltip=description or name,
        )
        slider = w.FloatSlider(
            value=opacity,
            min=0,
            max=1,
            step=0.05,
            readout=False,
            layout=w.Layout(width="150px"),
        )
        legend = w.HTML(legend_html, layout=w.Layout(margin="0 0 4px 22px"))
        legend.layout.display = "" if (visible and legend_html) else "none"

        def _vis(change: dict) -> None:
            legend.layout.display = "" if (change["new"] and legend_html) else "none"
            self._on_visible(name, change["new"])

        check.observe(_vis, "value")
        slider.observe(lambda c: self._on_opacity(name, c["new"]), "value")
        row = w.VBox([w.HBox([check, slider]), legend])
        self._checks[name] = check
        box = self._group_box(group)
        if name in self._rows:
            old_group, old_row = self._rows[name]
            if old_group == group:
                box.children = tuple(row if c is old_row else c for c in box.children)
                self._rows[name] = (group, row)
                return
            self.remove(name)
        box.children = (*box.children, row)
        self._rows[name] = (group, row)

    def set_checked(self, name: str, visible: bool) -> None:
        """Tick or untick a layer's checkbox (which also shows or hides it)."""
        check = self._checks.get(name)
        if check is not None and check.value != visible:
            check.value = visible

    def remove(self, name: str) -> None:
        """Remove a layer's row."""
        self._checks.pop(name, None)
        if name not in self._rows:
            return
        group, row = self._rows.pop(name)
        box = self._groups[group]
        box.children = tuple(c for c in box.children if c is not row)

    def open_group(self, group: str) -> None:
        """Expand one group of the accordion."""
        if group in self._groups:
            self._acc.selected_index = list(self._groups).index(group)


class SelectorPanel(w.VBox):
    """Controls that call ``on_change(values)`` whenever any of them changes.

    Args:
        controls: Control name to widget, e.g. ``{"grain": w.Dropdown(...)}``.
            Any widget with a ``value`` trait works.
        on_change: Called with ``{name: value}`` for all controls.
        title: Optional heading.
    """

    def __init__(
        self,
        controls: dict[str, w.ValueWidget],
        on_change: Callable[[dict[str, Any]], None],
        title: str = "",
    ) -> None:
        self.controls = controls
        self._on_change = on_change
        self._hold = False
        for ctl in controls.values():
            ctl.observe(self._changed, "value")
        head = [w.HTML(f"<b>{html.escape(title)}</b>")] if title else []
        super().__init__(
            [*head, *controls.values()], layout=w.Layout(width=f"{PANEL_WIDTH}px")
        )

    @property
    def values(self) -> dict[str, Any]:
        return {k: c.value for k, c in self.controls.items()}

    def _changed(self, _change: dict) -> None:
        if not self._hold:
            self._on_change(self.values)

    def set(self, fire: bool = True, **values: Any) -> None:
        """Set several controls at once, calling ``on_change`` once at most."""
        self._hold = True
        try:
            for k, v in values.items():
                self.controls[k].value = v
        finally:
            self._hold = False
        if fire:
            self._on_change(self.values)

    def set_options(self, name: str, options: Sequence[Any], keep: bool = True) -> None:
        """Replace a dropdown's options without firing ``on_change``.

        Keeps the current value when it is still an option and ``keep`` is set.
        """
        ctl = self.controls[name]
        old = ctl.value
        self._hold = True
        try:
            ctl.options = list(options)
            values = [o[1] if isinstance(o, tuple) else o for o in options]
            if keep and old in values:
                ctl.value = old
        finally:
            self._hold = False


class PlotPanel(w.VBox):
    """A Plotly ``FigureWidget`` sized to the side panel.

    Args:
        height: Figure height in pixels.
        rows: Number of stacked subplots sharing the x axis.
        row_heights: Relative subplot heights.
        titles: Subplot titles.
    """

    def __init__(
        self,
        height: int = 420,
        rows: int = 1,
        row_heights: Sequence[float] | None = None,
        titles: Sequence[str] | None = None,
    ) -> None:
        from plotly.subplots import make_subplots

        base = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=list(row_heights) if row_heights else None,
            subplot_titles=list(titles) if titles else None,
        )
        self.figure = go.FigureWidget(base)
        self.figure.update_layout(
            width=PANEL_WIDTH - 10,
            height=height,
            margin=dict(l=50, r=10, t=30, b=30),
            showlegend=True,
            legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
            template="plotly_white",
        )
        self.status = w.HTML("")
        super().__init__(
            [self.status, self.figure], layout=w.Layout(width=f"{PANEL_WIDTH}px")
        )

    def clear(self) -> None:
        """Remove all traces and shapes."""
        with self.figure.batch_update():
            self.figure.data = ()
            self.figure.layout.shapes = ()


class InfoPanel(w.HTML):
    """HTML readout for clicked features or pixels."""

    def __init__(self) -> None:
        super().__init__(
            "<i>Click the map or a feature.</i>",
            layout=w.Layout(width=f"{PANEL_WIDTH}px"),
        )

    def show_table(
        self, title: str, rows: dict[str, Any] | list[tuple[str, Any]]
    ) -> None:
        """Show a two-column table."""
        items = rows.items() if isinstance(rows, dict) else rows
        body = "".join(
            f"<tr><td style='padding:0 8px 0 0;color:#555'>{html.escape(str(k))}</td>"
            f"<td>{html.escape(_fmt(v))}</td></tr>"
            for k, v in items
        )
        self.value = (
            f"<b>{html.escape(title)}</b><table style='font-size:12px'>{body}</table>"
        )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.4g}"
    return str(v)
