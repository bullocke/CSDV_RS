"""csdv_core.disturbance.pipeline: stage orchestration and timing.

:func:`run_detection` runs one detection end to end:

1. ``setup``: claim the output folder, attach the file log handler, write
   ``run_params.json``.
2. ``inputs``: discover years and build the common grid.
3. ``layers``: cell layers, written to ``layers/``.
4. ``standardize``: change layer, population, z stack (quantized), written to
   ``z/``. Rules D7 and D7s also write the two-step z stacks.
5. ``decide``: rule, MMU per interval (with the growth rule) and onset.
6. ``outlines``: objects (and refinement), then the detection rasters, the
   GeoPackage and the CSV.
7. ``report``: ``summary.json`` and ``report.md``.
8. ``figures``: PNGs (when ``figures.enabled``), then ``summary.json`` and
   ``report.md`` again with the stage times and figure links.

With ``from_run``, the ``inputs`` stage reloads the saved layers and z instead
and copies them into the new folder, and ``layers`` and ``standardize`` do not
run. Input discovery still runs (without reading any raster) when refinement
or a NAIP backdrop needs the source files. A failed run keeps its folder and
log, and the exception is written to the log. The input preparation lives in
``_pipeline_inputs`` and the figure stage and :func:`rebuild_figures` in
``_pipeline_figures``.

Inputs: validated parameters, site name, output folder. Outputs: a run folder
and its :class:`~csdv_core.disturbance.report.RunSummary`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance._pipeline_figures import (
    rebuild_figures,
    write_run_figures,
)
from csdv_core.disturbance._pipeline_inputs import prepare_fresh, prepare_from_run
from csdv_core.disturbance.decide import decide, mmu_cells, onset_index_to_year
from csdv_core.disturbance.layers import monitored_mask
from csdv_core.disturbance.outlines import detection_objects, refine_objects
from csdv_core.disturbance.report import (
    RunSummary,
    build_summary,
    refresh_summary,
    write_report_markdown,
    write_summary_json,
)
from csdv_core.disturbance.store import (
    RunPaths,
    claim_output_dir,
    write_detection,
    write_run_params,
)

logger = logging.getLogger(__name__)

#: Format of the run log file.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
#: Logger that the run log handler is attached to (the package root).
PACKAGE_LOGGER_NAME = "csdv_core"


@dataclass
class StageTimer:
    """Wall-clock time per pipeline stage.

    Attributes:
        seconds: Stage name to elapsed seconds, in the order stages finished.
            A stage that raises still records its time before the error
            propagates.
    """

    seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block as stage ``name`` and log start and end.

        Args:
            name: Stage name.

        Yields:
            Nothing. The block runs inside the timer.
        """
        logger.info("Stage %s started", name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds[name] = elapsed
            logger.info("Stage %s finished in %.1f s", name, elapsed)


def attach_log_handler(log_path: Path, level: int = logging.INFO) -> logging.Handler:
    """Add a file handler for ``csdv_core`` loggers writing to ``log_path``.

    The handler is attached to the ``csdv_core`` logger (not the root) with
    :data:`LOG_FORMAT`. When the package logger would drop records at
    ``level``, its level is lowered to ``level`` so the file receives them.
    The caller removes the handler with :func:`detach_log_handler`.

    Args:
        log_path: Log file (appended to).
        level: Lowest level written to the file.

    Returns:
        The attached handler.
    """
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.addHandler(handler)
    effective_level = package_logger.getEffectiveLevel()
    if effective_level > level:
        package_logger.setLevel(level)
    return handler


def detach_log_handler(handler: logging.Handler) -> None:
    """Remove and close a handler added by :func:`attach_log_handler`.

    Args:
        handler: Handler returned by :func:`attach_log_handler`.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.removeHandler(handler)
    handler.close()


def _run_stages(
    params: DisturbanceConfig,
    site: str,
    run_paths: RunPaths,
    from_run: Path | None,
    command_line: str | None,
    config_path: Path | None,
) -> RunSummary:
    """Run every stage after the output folder is claimed.

    Args:
        params: Run parameters.
        site: Site name.
        run_paths: Paths of the claimed run folder.
        from_run: Saved run to reuse, or None.
        command_line: Command line to record, or None.
        config_path: User YAML to record, or None.

    Returns:
        The final run summary.
    """
    timer = StageTimer()
    with timer.stage("setup"):
        write_run_params(
            run_paths,
            params,
            site=site,
            command_line=command_line,
            config_path=config_path,
            from_run=from_run,
        )
    if from_run is None:
        prepared = prepare_fresh(site, params, run_paths, timer)
    else:
        prepared = prepare_from_run(site, params, run_paths, from_run, timer)
    grid = prepared.grid
    years = prepared.layers.years
    write_run_params(
        run_paths,
        params,
        site=site,
        command_line=command_line,
        config_path=config_path,
        from_run=from_run,
        grid=grid,
    )
    monitored = monitored_mask(prepared.layers.forest)
    with timer.stage("decide"):
        decision = decide(
            prepared.z,
            monitored,
            params,
            grid.cell_m,
            years=years,
            bracket_z=prepared.bracket_z,
        )
        onset_year = onset_index_to_year(decision.onset_index_mmu, years, monitored)
        onset_year_nommu = onset_index_to_year(decision.onset_index, years, monitored)
    with timer.stage("outlines"):
        objects = detection_objects(
            decision.detected,
            decision.probability,
            prepared.z,
            grid,
            years,
            min_cells=mmu_cells(params.mmu.mmu_ha, grid.cell_ha),
        )
        refinement = None
        if params.refine.enabled:
            refinement = refine_objects(objects, prepared.site_inputs, params)
        write_detection(
            run_paths,
            onset_year=onset_year,
            onset_year_nommu=onset_year_nommu,
            detected=decision.detected,
            probability=decision.probability,
            grid=grid,
            years=years,
            objects=objects,
            refinement=refinement,
        )
    with timer.stage("report"):
        summary = build_summary(
            site=site,
            params=params,
            inputs=prepared.inventory,
            grid=grid,
            layers=prepared.layers,
            monitored=monitored,
            decision=decision,
            objects=objects,
            refinement=refinement,
            stage_seconds=timer.seconds,
            run_paths=run_paths,
            command_line=command_line,
            config_path=config_path,
            from_run=from_run,
        )
        write_summary_json(summary, run_paths.summary_json)
        write_report_markdown(summary, run_paths.report_md, figure_paths={})
    figure_paths: dict[str, Path] = {}
    if params.figures.enabled:
        refined = None if refinement is None else refinement.outlines
        with timer.stage("figures"):
            figure_paths = write_run_figures(
                params=params,
                run_paths=run_paths,
                grid=grid,
                layers=prepared.layers,
                z=prepared.z,
                probability=decision.probability,
                onset_year=onset_year,
                detected=decision.detected,
                fits=decision.fits,
                intervals=summary.intervals,
                objects=objects,
                refined=refined,
                site_inputs=prepared.site_inputs,
                naip_backdrop=params.figures.naip_backdrop,
            )
    summary = refresh_summary(summary, run_paths, timer.seconds)
    write_summary_json(summary, run_paths.summary_json)
    write_report_markdown(summary, run_paths.report_md, figure_paths=figure_paths)
    total_seconds = sum(timer.seconds.values())
    logger.info("Run finished in %.1f s: %s", total_seconds, run_paths.root)
    return summary


def run_detection(
    params: DisturbanceConfig,
    site: str,
    output_dir: Path,
    from_run: Path | None = None,
    command_line: str | None = None,
    config_path: Path | None = None,
    log_level: int = logging.INFO,
) -> RunSummary:
    """Run disturbance detection for one site into a new folder.

    The folder is claimed first, so an existing folder fails before anything
    is read or written. Every later failure is logged with its traceback to
    ``detect_disturbance.log`` and re-raised, and the folder is left in place.

    Args:
        params: Validated run parameters (see
            :func:`csdv_core.disturbance.params.resolve_params`).
        site: Site name (folder under ``$CSDV_DATA_ROOT/naip_chm/``).
        output_dir: New run folder (must not exist).
        from_run: Saved run folder whose layers and z are reused. Its input,
            grid, forest and standardize parameters must equal ``params``
            (the CLI copies them from the saved run).
            The source rasters must still be available when refinement or a
            NAIP backdrop is requested.
        command_line: Command line to record, or None.
        config_path: User YAML to record, or None.
        log_level: Lowest level written to the run log.

    Returns:
        The run summary (also written to ``summary.json``).

    Raises:
        FileExistsError: If ``output_dir`` exists.
        FileNotFoundError: If inputs or the ``from_run`` files are missing.
        ValueError: If inputs are inconsistent or ``from_run`` parameters
            differ (see :func:`csdv_core.disturbance.params.check_reuse_compatible`).
    """
    run_paths = claim_output_dir(output_dir)
    handler = attach_log_handler(run_paths.log, log_level)
    try:
        return _run_stages(params, site, run_paths, from_run, command_line, config_path)
    except Exception:
        logger.exception("Run failed, the folder %s is kept for inspection", output_dir)
        raise
    finally:
        detach_log_handler(handler)


__all__ = [
    "StageTimer",
    "attach_log_handler",
    "detach_log_handler",
    "rebuild_figures",
    "run_detection",
]
