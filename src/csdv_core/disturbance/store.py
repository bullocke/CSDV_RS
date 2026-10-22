"""csdv_core.disturbance.store: run folder layout, raster and vector I/O, reload.

A run folder holds::

    run_params.json  detect_disturbance.log  summary.json  report.md
    layers/      chm_height.tif chm_cover.tif valid_share.tif [ndvi.tif] forest.tif
    z/           z_<input>_<method>[_<window>m].tif
                 [z_..._back2.tif z_..._fwd2.tif]  (rules D7 and D7s)
    detection/   onset_year.tif onset_year_nommu.tif detected_by_interval.tif
                 probability.tif disturbances.gpkg objects.csv
    figures/     *.png

Year stacks are multi-band GeoTIFFs on the cell grid, one band per year,
band description = year string, DEFLATE, tiled. ``run_params.json`` holds
the full :class:`~csdv_core.disturbance.inputs.CellGrid` so a run can be
reloaded without the source CHMs. :func:`claim_output_dir` never reuses an
existing folder.

Inputs: arrays, grid, years, GeoDataFrames. Outputs: files under a run folder
and :class:`SavedRun` on reload.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.io

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.inputs import CellGrid, grid_from_dict, grid_to_dict
from csdv_core.disturbance.layers import LAYER_NAMES, CellLayers
from csdv_core.disturbance.outlines import RefinementResult, objects_table
from csdv_core.disturbance.params import params_to_dict
from csdv_core.disturbance.standardize import (
    LOCAL_METHODS,
    Z_NODATA,
    decode_z,
    encode_z,
)

logger = logging.getLogger(__name__)

#: GeoTIFF creation options shared by every raster of a run.
GTIFF_PROFILE: dict[str, Any] = {
    "driver": "GTiff",
    "compress": "DEFLATE",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}
#: GeoPackage layer of the cell objects.
OBJECTS_LAYER = "objects"
#: GeoPackage layer of the refined envelopes.
REFINED_OUTLINES_LAYER = "refined_outlines"
#: GeoPackage layer of the refined openings.
REFINED_OPENINGS_LAYER = "refined_openings"
#: GeoTIFF predictor for float rasters (floating-point prediction).
FLOAT_PREDICTOR = 3
#: GeoTIFF predictor for integer rasters (horizontal differencing).
INTEGER_PREDICTOR = 2
#: Optional layer file, present only for runs that computed NDVI.
OPTIONAL_LAYER_NAME = "ndvi"
#: File name suffix of the back two-step z stack (rules D7 and D7s).
BACK_Z_SUFFIX = "back2"
#: File name suffix of the forward two-step z stack (rules D7 and D7s).
FORWARD_Z_SUFFIX = "fwd2"


@dataclass(frozen=True)
class RunPaths:
    """Every path of one run folder.

    Attributes:
        root: The run folder.
        run_params: ``run_params.json`` (final parameters, command line,
            config path, source run, grid).
        log: ``detect_disturbance.log``.
        summary_json: ``summary.json``.
        report_md: ``report.md``.
        layers_dir: ``layers/``.
        z_dir: ``z/``.
        detection_dir: ``detection/``.
        figures_dir: ``figures/``.
        onset_year: ``detection/onset_year.tif`` (int16, first detection
            after the MMU).
        onset_year_nommu: ``detection/onset_year_nommu.tif`` (int16, first
            date the rule passes, before the MMU).
        detected_by_interval: ``detection/detected_by_interval.tif`` (uint8,
            one band per year, 1 = detected in the interval closing that
            year, after the MMU).
        probability: ``detection/probability.tif`` (float32, one band per year).
        disturbances_gpkg: ``detection/disturbances.gpkg``.
        objects_csv: ``detection/objects.csv``.
    """

    root: Path
    run_params: Path
    log: Path
    summary_json: Path
    report_md: Path
    layers_dir: Path
    z_dir: Path
    detection_dir: Path
    figures_dir: Path
    onset_year: Path
    onset_year_nommu: Path
    detected_by_interval: Path
    probability: Path
    disturbances_gpkg: Path
    objects_csv: Path

    @classmethod
    def for_root(cls, root: Path) -> RunPaths:
        """Build the paths of a run folder (nothing is created)."""
        layers_dir = root / "layers"
        z_dir = root / "z"
        detection_dir = root / "detection"
        figures_dir = root / "figures"
        return cls(
            root=root,
            run_params=root / "run_params.json",
            log=root / "detect_disturbance.log",
            summary_json=root / "summary.json",
            report_md=root / "report.md",
            layers_dir=layers_dir,
            z_dir=z_dir,
            detection_dir=detection_dir,
            figures_dir=figures_dir,
            onset_year=detection_dir / "onset_year.tif",
            onset_year_nommu=detection_dir / "onset_year_nommu.tif",
            detected_by_interval=detection_dir / "detected_by_interval.tif",
            probability=detection_dir / "probability.tif",
            disturbances_gpkg=detection_dir / "disturbances.gpkg",
            objects_csv=detection_dir / "objects.csv",
        )

    def layer_path(self, name: str) -> Path:
        """``layers/<name>.tif`` for a :class:`CellLayers` field name."""
        return self.layers_dir / f"{name}.tif"

    def z_path(self, params: DisturbanceConfig) -> Path:
        """``z/`` file of the run's input and method, see :func:`z_filename`."""
        filename = z_filename(
            params.inputs.input, params.standardize.method, params.standardize.window_m
        )
        return self.z_dir / filename

    def bracket_z_paths(self, params: DisturbanceConfig) -> tuple[Path, Path]:
        """Back and forward two-step z files of rules D7 and D7s.

        Args:
            params: Run parameters (input and method name the files).

        Returns:
            ``(back, forward)``: :meth:`z_path` with ``_back2`` and ``_fwd2``
            added before the suffix.
        """
        main_path = self.z_path(params)
        back_path = self.z_dir / f"{main_path.stem}_{BACK_Z_SUFFIX}.tif"
        forward_path = self.z_dir / f"{main_path.stem}_{FORWARD_Z_SUFFIX}.tif"
        return back_path, forward_path


@dataclass(frozen=True)
class YearStack:
    """A multi-band year raster read back into memory.

    Attributes:
        values: ``(T, rows, cols)`` array. float32 with NaN nodata for float
            files, bool for forest, int16 decoded to float32 for z.
        years: Years from the band descriptions, length ``T``.
    """

    values: np.ndarray
    years: tuple[int, ...]


@dataclass(frozen=True)
class SavedRun:
    """A run folder reloaded for ``--from-run`` or ``disturbance-figures``.

    Attributes:
        run_paths: Paths of the saved run.
        site: Site name from ``run_params.json``.
        params: Parameters of the saved run.
        grid: Common grid rebuilt from ``run_params.json``.
        layers: Cell layers from ``layers/``.
        z: float32 ``(T, rows, cols)`` decoded z (already quantized).
        years: NAIP years, length ``T``.
        bracket_z: Decoded ``(back, forward)`` two-step z stacks when both
            files exist (runs with rule D7 or D7s), else None.
    """

    run_paths: RunPaths
    site: str
    params: DisturbanceConfig
    grid: CellGrid
    layers: CellLayers
    z: np.ndarray
    years: tuple[int, ...]
    bracket_z: tuple[np.ndarray, np.ndarray] | None = None


def z_filename(input_name: str, method: str, window_m: float) -> str:
    """File name of a z stack.

    ``z_<input>_<method>.tif`` for scene methods and
    ``z_<input>_<method>_<int(window_m)>m.tif`` for ``local`` and
    ``local_twopass`` (e.g. ``z_d_chm_h_local_twopass_150m.tif``).

    Args:
        input_name: Detection input (``inputs.input``).
        method: Standardization method.
        window_m: Local window width (m).

    Returns:
        The file name.
    """
    if method in LOCAL_METHODS:
        window_label = int(window_m)
        return f"z_{input_name}_{method}_{window_label}m.tif"
    return f"z_{input_name}_{method}.tif"


def claim_output_dir(path: Path) -> RunPaths:
    """Create a new run folder and its subfolders.

    Args:
        path: Run folder to create (parents are created as needed).

    Returns:
        The run paths, with ``layers/``, ``z/``, ``detection/`` and
        ``figures/`` created.

    Raises:
        FileExistsError: If ``path`` already exists (file or folder, empty or
            not). Nothing is ever overwritten.
    """
    if path.exists():
        raise FileExistsError(f"Output folder already exists: {path}")
    path.mkdir(parents=True)
    run_paths = RunPaths.for_root(path)
    for folder in (
        run_paths.layers_dir,
        run_paths.z_dir,
        run_paths.detection_dir,
        run_paths.figures_dir,
    ):
        folder.mkdir()
    logger.info("Claimed output folder %s", path)
    return run_paths


def _band_years(dataset: rasterio.io.DatasetReader) -> tuple[int, ...]:
    """Parse the band descriptions of an open dataset as years.

    Args:
        dataset: Open raster written by this module.

    Returns:
        One year per band.

    Raises:
        ValueError: If a band description is missing or not an integer.
    """
    years: list[int] = []
    for band_index, description in enumerate(dataset.descriptions, start=1):
        stripped = "" if description is None else description.strip()
        if not stripped.isdigit():
            raise ValueError(
                f"Band {band_index} of {dataset.name} has no year description: "
                f"{description!r}"
            )
        years.append(int(stripped))
    return tuple(years)


def _write_raster(
    path: Path,
    data: np.ndarray,
    grid: CellGrid,
    *,
    nodata: float | None,
    band_names: Sequence[str],
) -> Path:
    """Write a ``(bands, rows, cols)`` array on the cell grid.

    The predictor is 3 (floating point) for float data and 2 (horizontal
    differencing) for integers.

    Args:
        path: Output GeoTIFF.
        data: Array already cast to the stored dtype.
        grid: Common grid.
        nodata: Nodata value, or None for no nodata tag.
        band_names: One band description per band.

    Returns:
        ``path``.
    """
    is_float = np.issubdtype(data.dtype, np.floating)
    predictor = FLOAT_PREDICTOR if is_float else INTEGER_PREDICTOR
    band_count, rows, cols = data.shape
    profile = dict(GTIFF_PROFILE)
    profile.update(
        width=cols,
        height=rows,
        count=band_count,
        dtype=data.dtype.name,
        crs=grid.crs,
        transform=grid.cell_transform,
        nodata=nodata,
        predictor=predictor,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)
        for band_index, band_name in enumerate(band_names, start=1):
            dataset.set_band_description(band_index, band_name)
    logger.debug("Wrote %s (%d bands, %s)", path, band_count, data.dtype.name)
    return path


def _check_stack_shape(stack: np.ndarray, grid: CellGrid, years: Sequence[int]) -> None:
    """Check that a year stack matches the grid and the number of years.

    Args:
        stack: ``(T, rows, cols)`` array.
        grid: Common grid.
        years: Length ``T``.

    Raises:
        ValueError: If the shape is not ``(len(years), cell_rows, cell_cols)``.
    """
    expected_shape = (len(years), grid.cell_rows, grid.cell_cols)
    if stack.shape != expected_shape:
        raise ValueError(
            f"Stack shape {stack.shape} does not match grid and years "
            f"{expected_shape}"
        )


def write_year_stack(
    path: Path, stack: np.ndarray, grid: CellGrid, years: Sequence[int]
) -> Path:
    """Write a ``(T, rows, cols)`` stack on the cell grid, one band per year.

    float stacks are written float32 with nodata NaN. bool stacks are written
    uint8 0/1 without nodata. Band descriptions are the year strings.

    Args:
        path: Output GeoTIFF.
        stack: float or bool ``(T, grid.cell_rows, grid.cell_cols)``.
        grid: Common grid (``cell_transform`` and CRS are used).
        years: Length ``T``.

    Returns:
        ``path``.

    Raises:
        ValueError: If the shape does not match the grid and years, or the
            dtype is neither float nor bool.
    """
    _check_stack_shape(stack, grid, years)
    band_names = [str(year) for year in years]
    if stack.dtype == bool:
        stored = stack.astype("uint8")
        return _write_raster(path, stored, grid, nodata=None, band_names=band_names)
    if np.issubdtype(stack.dtype, np.floating):
        stored = stack.astype("float32")
        return _write_raster(
            path, stored, grid, nodata=float("nan"), band_names=band_names
        )
    raise ValueError(f"Year stacks must be float or bool, got {stack.dtype}")


def read_year_stack(path: Path) -> YearStack:
    """Read a stack written by :func:`write_year_stack`.

    uint8 files come back as bool, float files as float32 with NaN nodata.

    Args:
        path: GeoTIFF written by :func:`write_year_stack`.

    Returns:
        The values and band years.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a band description is not a year.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as dataset:
        years = _band_years(dataset)
        stored = dataset.read()
        nodata = dataset.nodata
    if stored.dtype == np.uint8:
        values = stored.astype(bool)
        return YearStack(values=values, years=years)
    values = stored.astype("float32")
    if nodata is not None and not np.isnan(nodata):
        values[stored == nodata] = np.nan
    return YearStack(values=values, years=years)


def write_z(path: Path, z: np.ndarray, grid: CellGrid, years: Sequence[int]) -> Path:
    """Write z as int16 ``round(z * 100)``, nodata -32768, band description = year.

    Uses :func:`csdv_core.disturbance.standardize.encode_z`.

    Args:
        path: Output GeoTIFF.
        z: float32 ``(T, rows, cols)``, NaN = missing.
        grid: Common grid.
        years: Length ``T``.

    Returns:
        ``path``.
    """
    _check_stack_shape(z, grid, years)
    encoded = encode_z(z)
    band_names = [str(year) for year in years]
    return _write_raster(path, encoded, grid, nodata=Z_NODATA, band_names=band_names)


def read_z(path: Path) -> YearStack:
    """Read a z file back to float32 with NaN (``standardize.decode_z``).

    Args:
        path: GeoTIFF written by :func:`write_z`.

    Returns:
        The decoded z and band years.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"z raster not found: {path}")
    with rasterio.open(path) as dataset:
        years = _band_years(dataset)
        encoded = dataset.read()
    values = decode_z(encoded)
    return YearStack(values=values, years=years)


def write_layers(
    run_paths: RunPaths, layers: CellLayers, grid: CellGrid
) -> dict[str, Path]:
    """Write every year stack of ``layers`` into ``layers/``.

    Files: ``chm_height.tif``, ``chm_cover.tif``, ``valid_share.tif``,
    ``forest.tif`` and ``ndvi.tif`` when ``layers.ndvi`` is set.

    Args:
        run_paths: Paths of the run.
        layers: Cell layers.
        grid: Common grid.

    Returns:
        Mapping from layer name to the written path.
    """
    written: dict[str, Path] = {}
    for layer_name in LAYER_NAMES:
        stack = getattr(layers, layer_name)
        if stack is None:
            continue
        layer_path = run_paths.layer_path(layer_name)
        written[layer_name] = write_year_stack(layer_path, stack, grid, layers.years)
    logger.info("Wrote %d layer stacks to %s", len(written), run_paths.layers_dir)
    return written


def read_layers(run_paths: RunPaths) -> CellLayers:
    """Read ``layers/`` back into :class:`CellLayers`.

    Args:
        run_paths: Paths of the run.

    Returns:
        The layers, with ``ndvi`` None when ``ndvi.tif`` is absent.

    Raises:
        FileNotFoundError: If a layer file other than ``ndvi.tif`` is missing.
        ValueError: If the files disagree in years.
    """
    values_by_name: dict[str, np.ndarray | None] = {}
    years: tuple[int, ...] | None = None
    for layer_name in LAYER_NAMES:
        layer_path = run_paths.layer_path(layer_name)
        if layer_name == OPTIONAL_LAYER_NAME and not layer_path.is_file():
            values_by_name[layer_name] = None
            continue
        stack = read_year_stack(layer_path)
        if years is None:
            years = stack.years
        elif stack.years != years:
            raise ValueError(
                f"Layer {layer_name} has years {stack.years}, others have {years}"
            )
        values_by_name[layer_name] = stack.values
    return CellLayers(years=years, **values_by_name)


def write_detection(
    run_paths: RunPaths,
    *,
    onset_year: np.ndarray,
    onset_year_nommu: np.ndarray,
    detected: np.ndarray,
    probability: np.ndarray | None,
    grid: CellGrid,
    years: Sequence[int],
    objects: gpd.GeoDataFrame,
    refinement: RefinementResult | None,
) -> dict[str, Path]:
    """Write the detection rasters, the GeoPackage and ``objects.csv``.

    ``onset_year*.tif``: single band int16 on the cell grid, no nodata value
    (0 = no detection, -1 = never monitored). ``detected_by_interval.tif``:
    uint8 ``(T, rows, cols)`` 0/1, band description = year (band 1, the
    first year, is all 0). ``probability.tif``: float32
    ``(T, rows, cols)``, nodata NaN, band description = year (skipped when
    ``probability`` is None, rule D1). GeoPackage layers
    :data:`OBJECTS_LAYER` and, when ``refinement`` is given,
    :data:`REFINED_OUTLINES_LAYER` and :data:`REFINED_OPENINGS_LAYER`.
    ``objects.csv`` comes from
    :func:`csdv_core.disturbance.outlines.objects_table`.

    Args:
        run_paths: Paths of the run.
        onset_year: int16 ``(rows, cols)`` after the MMU.
        onset_year_nommu: int16 ``(rows, cols)`` before the MMU.
        detected: bool ``(T, rows, cols)`` detections per interval.
        probability: float32 ``(T, rows, cols)``, or None.
        grid: Common grid.
        years: NAIP years, length ``T``.
        objects: Objects GeoDataFrame.
        refinement: Refinement result, or None when refinement is off.

    Returns:
        Mapping from a short name (``onset_year``, ``onset_year_nommu``,
        ``detected_by_interval``, ``probability``, ``disturbances_gpkg``,
        ``objects_csv``) to each written path.
    """
    written: dict[str, Path] = {}
    onset_rasters = {
        "onset_year": (onset_year, run_paths.onset_year),
        "onset_year_nommu": (onset_year_nommu, run_paths.onset_year_nommu),
    }
    for short_name, (onset_array, onset_path) in onset_rasters.items():
        stored = onset_array.astype("int16")
        single_band = stored[np.newaxis, :, :]
        written[short_name] = _write_raster(
            onset_path, single_band, grid, nodata=None, band_names=[short_name]
        )
    written["detected_by_interval"] = write_year_stack(
        run_paths.detected_by_interval, detected.astype(bool), grid, years
    )
    if probability is not None:
        written["probability"] = write_year_stack(
            run_paths.probability, probability, grid, years
        )
    gpkg_path = run_paths.disturbances_gpkg
    vector_layers = {OBJECTS_LAYER: objects}
    if refinement is not None:
        vector_layers[REFINED_OUTLINES_LAYER] = refinement.outlines
        vector_layers[REFINED_OPENINGS_LAYER] = refinement.openings
    for layer_name, frame in vector_layers.items():
        frame.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    written["disturbances_gpkg"] = gpkg_path
    table = objects_table(objects)
    table.to_csv(run_paths.objects_csv, index=False)
    written["objects_csv"] = run_paths.objects_csv
    logger.info("Wrote detection outputs to %s", run_paths.detection_dir)
    return written


def read_detected_by_interval(run_paths: RunPaths, years: Sequence[int]) -> np.ndarray:
    """Detections per interval of a saved run.

    Runs written before ``detected_by_interval.tif`` existed kept one
    detection per cell, so their ``onset_year.tif`` gives the stack
    (``out[t] = onset_year == years[t]``).

    Args:
        run_paths: Paths of the run.
        years: NAIP years of the run, length ``T``.

    Returns:
        bool ``(T, rows, cols)``.

    Raises:
        FileNotFoundError: If neither raster exists.
        ValueError: If the stack's band years differ from ``years``.
    """
    if run_paths.detected_by_interval.is_file():
        stack = read_year_stack(run_paths.detected_by_interval)
        if list(stack.years) != list(years):
            raise ValueError(
                f"{run_paths.detected_by_interval} has years {stack.years}, "
                f"expected {tuple(years)}"
            )
        return stack.values
    if not run_paths.onset_year.is_file():
        raise FileNotFoundError(f"Raster not found: {run_paths.onset_year}")
    with rasterio.open(run_paths.onset_year) as dataset:
        onset_year = dataset.read(1)
    year_values = np.asarray(years).reshape(-1, 1, 1)
    return onset_year[np.newaxis, :, :] == year_values


def write_run_params(
    run_paths: RunPaths,
    params: DisturbanceConfig,
    *,
    site: str,
    command_line: str | None,
    config_path: Path | None,
    from_run: Path | None,
    grid: CellGrid | None = None,
) -> Path:
    """Write ``run_params.json``.

    Keys: ``site``, ``created`` (ISO 8601 local time), ``command_line``,
    ``config_path``, ``from_run`` (strings or null), ``params``
    (:func:`csdv_core.disturbance.params.params_to_dict`) and ``grid``
    (:func:`csdv_core.disturbance.inputs.grid_to_dict`, null until the grid is
    known). The pipeline writes it once after claiming the folder and again
    once the grid exists.

    Args:
        run_paths: Paths of the run.
        params: Run parameters.
        site: Site name.
        command_line: Command line, or None.
        config_path: User YAML, or None.
        from_run: Source run folder, or None.
        grid: Common grid, or None before it is built.

    Returns:
        ``run_paths.run_params``.
    """
    now = datetime.datetime.now()
    created = now.astimezone().isoformat(timespec="seconds")
    config_text = None if config_path is None else str(config_path)
    from_run_text = None if from_run is None else str(from_run)
    grid_values = None if grid is None else grid_to_dict(grid)
    content = {
        "site": site,
        "created": created,
        "command_line": command_line,
        "config_path": config_text,
        "from_run": from_run_text,
        "params": params_to_dict(params),
        "grid": grid_values,
    }
    text = json.dumps(content, indent=2)
    run_paths.run_params.write_text(text + "\n", encoding="utf-8")
    logger.debug("Wrote %s", run_paths.run_params)
    return run_paths.run_params


def read_run_params(run_dir: Path) -> dict[str, Any]:
    """Read ``run_params.json`` of a run folder as a dict.

    Args:
        run_dir: Run folder.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    run_paths = RunPaths.for_root(run_dir)
    if not run_paths.run_params.is_file():
        raise FileNotFoundError(f"run_params.json not found in {run_dir}")
    text = run_paths.run_params.read_text(encoding="utf-8")
    return json.loads(text)


def read_bracket_z(
    run_paths: RunPaths, params: DisturbanceConfig, years: Sequence[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the two-step z stacks of a run when both files exist.

    Args:
        run_paths: Paths of the run.
        params: Parameters of the run.
        years: Layer years the stacks must match.

    Returns:
        Decoded ``(back, forward)`` z stacks, or None when a file is missing.

    Raises:
        ValueError: If a stack's years differ from ``years``.
    """
    back_path, forward_path = run_paths.bracket_z_paths(params)
    if not (back_path.is_file() and forward_path.is_file()):
        return None
    back_stack = read_z(back_path)
    forward_stack = read_z(forward_path)
    expected_years = tuple(years)
    for stack, path in ((back_stack, back_path), (forward_stack, forward_path)):
        if stack.years != expected_years:
            raise ValueError(
                f"{path} years {stack.years} differ from layer years {expected_years}"
            )
    return back_stack.values, forward_stack.values


def load_run(run_dir: Path) -> SavedRun:
    """Reload a run folder's parameters, grid, layers and z.

    Args:
        run_dir: Run folder.

    Returns:
        The saved run, with z located by :meth:`RunPaths.z_path` and the
        two-step z stacks by :meth:`RunPaths.bracket_z_paths` (None when
        absent).

    Raises:
        FileNotFoundError: If ``run_dir``, ``run_params.json``, a layer file or
            the z file is missing, or the saved grid is null.
        ValueError: If the z and layer years differ.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    run_paths = RunPaths.for_root(run_dir)
    saved = read_run_params(run_dir)
    params = DisturbanceConfig.model_validate(saved["params"])
    if saved.get("grid") is None:
        raise FileNotFoundError(
            f"{run_paths.run_params} holds no grid, the run stopped before the "
            "grid was built"
        )
    grid = grid_from_dict(saved["grid"])
    layers = read_layers(run_paths)
    z_path = run_paths.z_path(params)
    z_stack = read_z(z_path)
    if z_stack.years != layers.years:
        raise ValueError(
            f"z years {z_stack.years} differ from layer years {layers.years}"
        )
    bracket_z = read_bracket_z(run_paths, params, layers.years)
    logger.info("Loaded saved run %s (%d years)", run_dir, len(layers.years))
    return SavedRun(
        run_paths=run_paths,
        site=saved["site"],
        params=params,
        grid=grid,
        layers=layers,
        z=z_stack.values,
        years=layers.years,
        bracket_z=bracket_z,
    )


__all__ = [
    "BACK_Z_SUFFIX",
    "FORWARD_Z_SUFFIX",
    "OBJECTS_LAYER",
    "REFINED_OPENINGS_LAYER",
    "REFINED_OUTLINES_LAYER",
    "RunPaths",
    "SavedRun",
    "YearStack",
    "claim_output_dir",
    "load_run",
    "read_bracket_z",
    "read_detected_by_interval",
    "read_layers",
    "read_run_params",
    "read_year_stack",
    "read_z",
    "write_detection",
    "write_layers",
    "write_run_params",
    "write_year_stack",
    "write_z",
    "z_filename",
]
