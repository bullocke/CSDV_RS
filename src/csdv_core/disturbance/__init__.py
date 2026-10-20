"""csdv_core.disturbance: NAIP-CHM forest disturbance detection pipeline.

A parameterized, documented version of the best Disturbance Index method from
``analysis/NAIP_disturbance_detection/2_Disturbance_Index``: change in mean
CHM height on 19.2 m cells, two-pass 150 m local standardization, rule D6
(spatial support and persistence), a 1-acre minimum mapping unit and 0.6 m
outline refinement. The algorithms live in :mod:`csdv_core.change`. This
package reads inputs, runs the stages and writes the run folder.

Modules: ``params`` (parameter resolution), ``inputs`` (files and grid),
``windows`` (pixel reads on the grid), ``layers`` (cell layers and masks),
``standardize`` (z), ``decide`` (rules, onset, MMU), ``outlines`` (objects
and refinement), ``store`` (run folder I/O), ``report`` (summary),
``figures`` (PNGs), ``pipeline`` (stages) and ``cli`` (Click commands).
Submodules are not imported here, so importing the package stays cheap.
"""

from __future__ import annotations
