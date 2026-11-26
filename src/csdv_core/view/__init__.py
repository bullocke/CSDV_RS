"""csdv_core.view: a browser viewer for the pair-model disturbance maps.

:mod:`~csdv_core.view.catalog` finds the exported runs and sites,
:mod:`~csdv_core.view.derive` turns the per-interval probability and agent
rasters into date, type and probability layers, :mod:`~csdv_core.view.tiles`
renders them as map tiles, :mod:`~csdv_core.view.notes` keeps the comment
points and :mod:`~csdv_core.view.server` serves the page in ``page/``. The
``csdv view`` command is in :mod:`~csdv_core.view.cli`. See
docs/guides/map_viewer.md.
"""
