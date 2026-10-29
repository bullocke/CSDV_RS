"""csdv_core.grid — CONUS processing grid and one-command unit runs.

:mod:`~csdv_core.grid.cells` defines the grid (7.5 minute quads),
:mod:`~csdv_core.grid.units` the units and their records,
:mod:`~csdv_core.grid.submit` the Slurm chain of a unit,
:mod:`~csdv_core.grid.status` its progress and :mod:`~csdv_core.grid.export`
the small download set of a run. The ``csdv grid`` commands are in
:mod:`~csdv_core.grid.cli`.
"""
