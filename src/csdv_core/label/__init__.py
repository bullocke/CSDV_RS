"""csdv_core.label: training labels for a NAIP disturbance model.

Candidates from the disturbance runs are verified by two vision models, and
the agreed answers become labelled windows of multi-date NAIP.
:mod:`~csdv_core.label.inputs` reads a unit, :mod:`~csdv_core.label.lattice`
places chips and windows, :mod:`~csdv_core.label.naip` and
:mod:`~csdv_core.label.stacks` read the imagery,
:mod:`~csdv_core.label.candidates` builds the candidate set,
:mod:`~csdv_core.label.requests` and :mod:`~csdv_core.label.verdicts` ask the
models, :mod:`~csdv_core.label.routing` and :mod:`~csdv_core.label.outlines`
combine their answers, and :mod:`~csdv_core.label.rules` turns the result into
label rasters. The ``csdv label`` commands are in
:mod:`~csdv_core.label.cli`. See docs/guides/disturbance_labels.md.
"""
