"""csdv_core.disturbance._pipeline_inputs: prepare layers and z for a run.

Private helpers of :mod:`csdv_core.disturbance.pipeline`. A fresh run
discovers the site files, builds the cell layers and standardizes the change
layer. A ``--from-run`` rerun reloads the saved layers and z instead, copies
them into the new folder, and discovers the source files (metadata only) when
refinement or a NAIP backdrop needs them. Rules D7 and D7s also get the
two-step z stacks, standardized from the layers (or reloaded when the source
run saved them).

Inputs: run parameters and folders. Outputs: :class:`PreparedStacks`, with
layers and z written to the new run folder.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.decide import BRACKET_RULES
from csdv_core.disturbance.inputs import CellGrid, SiteInputs, discover_site_inputs
from csdv_core.disturbance.layers import (
    CellLayers,
    bracket_changes,
    build_cell_layers,
    change_layer,
    population_mask,
)
from csdv_core.disturbance.params import check_reuse_compatible, params_to_dict
from csdv_core.disturbance.report import (
    InputFileInfo,
    input_inventory,
    read_summary_json,
)
from csdv_core.disturbance.standardize import (
    quantize_z,
    standardize_brackets,
    standardize_stack,
)
from csdv_core.disturbance.store import RunPaths, load_run, write_layers, write_z

if TYPE_CHECKING:
    from csdv_core.disturbance.pipeline import StageTimer

logger = logging.getLogger(__name__)

#: Input whose layers need the NAIP imagery.
NAIP_INPUT = "d_ndvi"


@dataclass
class PreparedStacks:
    """Grid, layers and z ready for the decision stage.

    Attributes:
        grid: Common grid.
        layers: Cell layers.
        z: float32 ``(T, rows, cols)`` quantized z.
        site_inputs: Discovered inputs, or None when a ``--from-run`` rerun
            needs no source file.
        inventory: Input inventory for the report.
        bracket_z: Quantized ``(back, forward)`` two-step z stacks for rules
            D7 and D7s, else None.
    """

    grid: CellGrid
    layers: CellLayers
    z: np.ndarray
    site_inputs: SiteInputs | None
    inventory: list[InputFileInfo]
    bracket_z: tuple[np.ndarray, np.ndarray] | None = None


def bracket_z_stacks(
    layers: CellLayers, params: DisturbanceConfig, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-step z stacks of rules D7 and D7s, computed from the cell layers.

    Args:
        layers: Cell layers.
        params: Run parameters.
        cell_m: Cell side (m).

    Returns:
        Quantized ``(back_z, forward_z)``, float32 ``(T, rows, cols)``.
    """
    back_change, forward_change = bracket_changes(layers, params.inputs.input)
    population = population_mask(layers.forest)
    return standardize_brackets(back_change, forward_change, population, params, cell_m)


def write_bracket_z(
    run_paths: RunPaths,
    params: DisturbanceConfig,
    bracket_z: tuple[np.ndarray, np.ndarray],
    grid: CellGrid,
    years: tuple[int, ...],
) -> None:
    """Write the two-step z stacks to ``z/`` (same encoding as the main z).

    Args:
        run_paths: Paths of the run.
        params: Run parameters (name the files).
        bracket_z: ``(back_z, forward_z)``.
        grid: Common grid.
        years: NAIP years.
    """
    back_z, forward_z = bracket_z
    back_path, forward_path = run_paths.bracket_z_paths(params)
    write_z(back_path, back_z, grid, years)
    write_z(forward_path, forward_z, grid, years)


def discover_with_saved_grid(
    site: str, params: DisturbanceConfig, grid: CellGrid, *, need_naip: bool
) -> SiteInputs:
    """Discover a site's files and attach the grid of a saved run.

    Only file metadata is read. The saved grid replaces the rebuilt one so
    windows line up with the saved layers. A rebuilt grid that differs logs a
    warning, since the source files then changed after the saved run.

    Args:
        site: Site name.
        params: Run parameters.
        grid: Grid of the saved run.
        need_naip: List NAIP files too.

    Returns:
        Site inputs carrying ``grid``.
    """
    site_inputs = discover_site_inputs(site, params, need_naip=need_naip)
    if site_inputs.grid != grid:
        logger.warning(
            "The grid rebuilt from the source files differs from the saved grid, "
            "the saved grid is used"
        )
    return dataclasses.replace(site_inputs, grid=grid)


def _saved_inventory(
    run_paths: RunPaths, site_inputs: SiteInputs | None
) -> list[InputFileInfo]:
    """Input inventory of a saved run, from its ``summary.json`` when present.

    Args:
        run_paths: Paths of the saved run.
        site_inputs: Rediscovered inputs, used when the summary is missing.

    Returns:
        The inventory (empty when neither source is available).
    """
    if run_paths.summary_json.is_file():
        saved_summary = read_summary_json(run_paths.summary_json)
        return saved_summary.inputs
    if site_inputs is not None:
        return input_inventory(site_inputs)
    logger.warning(
        "No summary.json in %s, the input inventory is empty", run_paths.root
    )
    return []


def prepare_fresh(
    site: str, params: DisturbanceConfig, run_paths: RunPaths, timer: StageTimer
) -> PreparedStacks:
    """Run the ``inputs``, ``layers`` and ``standardize`` stages.

    Args:
        site: Site name.
        params: Run parameters.
        run_paths: Paths of the new run.
        timer: Stage timer.

    Returns:
        Grid, layers, quantized z and inputs, with layers and z written.
    """
    include_ndvi = params.inputs.input == NAIP_INPUT
    need_backdrop = params.figures.enabled and params.figures.naip_backdrop
    need_naip = include_ndvi or need_backdrop
    with timer.stage("inputs"):
        site_inputs = discover_site_inputs(site, params, need_naip=need_naip)
        inventory = input_inventory(site_inputs)
        grid = site_inputs.grid
    with timer.stage("layers"):
        layers = build_cell_layers(site_inputs, params, include_ndvi=include_ndvi)
        write_layers(run_paths, layers, grid)
    with timer.stage("standardize"):
        change = change_layer(layers, params.inputs.input)
        population = population_mask(layers.forest)
        z_unrounded = standardize_stack(change, population, params, grid.cell_m)
        z = quantize_z(z_unrounded)
        z_path = run_paths.z_path(params)
        write_z(z_path, z, grid, layers.years)
        bracket_z = None
        if params.decide.rule in BRACKET_RULES:
            bracket_z = bracket_z_stacks(layers, params, grid.cell_m)
            write_bracket_z(run_paths, params, bracket_z, grid, layers.years)
    return PreparedStacks(
        grid=grid,
        layers=layers,
        z=z,
        site_inputs=site_inputs,
        inventory=inventory,
        bracket_z=bracket_z,
    )


def prepare_from_run(
    site: str,
    params: DisturbanceConfig,
    run_paths: RunPaths,
    from_run: Path,
    timer: StageTimer,
) -> PreparedStacks:
    """Reload layers and z of a saved run (the ``inputs`` stage of a rerun).

    The saved layers and z are copied into the new folder so it is complete
    on its own. The source files are discovered only when refinement or a
    NAIP backdrop needs them. Rules D7 and D7s reuse the saved two-step z
    stacks, or compute them from the saved layers when the source run has
    none (for example a D6 run).

    Args:
        site: Site name (must equal the saved run's site).
        params: Run parameters.
        run_paths: Paths of the new run.
        from_run: Saved run folder.
        timer: Stage timer.

    Returns:
        Grid, layers, quantized z and inputs of the saved run.

    Raises:
        ValueError: If the site or a locked parameter section differs.
    """
    with timer.stage("inputs"):
        saved = load_run(from_run)
        if saved.site != site:
            raise ValueError(f"--from-run holds site {saved.site!r}, not {site!r}")
        saved_values = params_to_dict(saved.params)
        requested_values = params_to_dict(params)
        check_reuse_compatible(saved_values, requested_values)
        need_backdrop = params.figures.enabled and params.figures.naip_backdrop
        site_inputs = None
        if params.refine.enabled or need_backdrop:
            site_inputs = discover_with_saved_grid(
                site, params, saved.grid, need_naip=need_backdrop
            )
        inventory = _saved_inventory(saved.run_paths, site_inputs)
        write_layers(run_paths, saved.layers, saved.grid)
        z_path = run_paths.z_path(params)
        write_z(z_path, saved.z, saved.grid, saved.years)
        logger.info("Reused layers and z of %s", from_run)
        bracket_z = None
        if params.decide.rule in BRACKET_RULES:
            bracket_z = saved.bracket_z
            if bracket_z is None:
                logger.info("Computing the two-step z stacks from the saved layers")
                bracket_z = bracket_z_stacks(saved.layers, params, saved.grid.cell_m)
            write_bracket_z(run_paths, params, bracket_z, saved.grid, saved.years)
    return PreparedStacks(
        grid=saved.grid,
        layers=saved.layers,
        z=saved.z,
        site_inputs=site_inputs,
        inventory=inventory,
        bracket_z=bracket_z,
    )


__all__ = [
    "NAIP_INPUT",
    "PreparedStacks",
    "bracket_z_stacks",
    "discover_with_saved_grid",
    "prepare_fresh",
    "prepare_from_run",
    "write_bracket_z",
]
