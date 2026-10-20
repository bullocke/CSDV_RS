# Disturbance detection: API, parameters and outputs

This is the reference for `csdv_core.disturbance`, the package behind `csdv detect-disturbance` and `csdv disturbance-figures`. It lists every public module, dataclass and function, every parameter with its default and command-line option, and the content and encoding of every output file. For what the method does and how to run it, read [guides/disturbance_detection.md](guides/disturbance_detection.md) first.

The package is a thin layer over the algorithm modules in `csdv_core.change` (`di.py` for standardization, `di_decide.py` for empirical nulls and spatial probabilities, `refine.py` for 0.6 m outlines). Those modules are shared with the research scripts in `analysis/NAIP_disturbance_detection/2_Disturbance_Index/` and are not documented here. Submodules are not imported by `csdv_core.disturbance/__init__.py`, so import them directly (`from csdv_core.disturbance.decide import decide`).

Array conventions used throughout: a year stack has shape `(T, rows, cols)` with index `t` matching `years[t]`, cell rasters have shape `(rows, cols)` on the cell grid, float arrays are float32 with NaN for missing values, and `params` is always a validated `csdv_core.config._models.DisturbanceConfig`. Date index 0 is the first image. It has no earlier image, so it is never monitored and change layers are NaN there. Interval `t` (for `t >= 1`) runs from `years[t - 1]` to `years[t]`, and an onset at index `t` is labelled with the later year, `years[t]`.

## Module map

| Module | Responsibility |
|---|---|
| `params` | Merge packaged YAML, user YAML and CLI options, then validate |
| `inputs` | Find each year's CHM (and NAIP) files, build the common grid |
| `windows` | Read pixel windows of a year's files on the common grid |
| `layers` | Per-year cell means (height, cover, valid share, NDVI), forest, monitored and population masks, change layers |
| `standardize` | Change layer to z stack (scene, scene mean, local, local two-pass), z encoding |
| `decide` | Empirical null per date, rules D1, D2, D5, D5s, D6, D6s, onset date, minimum mapping unit |
| `outlines` | Detected cells to objects, 0.6 m refinement, the flat objects table |
| `store` | Run folder layout, GeoTIFF and GeoPackage writing, reload for `--from-run` |
| `report` | Run summary, `summary.json`, `report.md` |
| `report_markdown` | Renders the sections of `report.md` from a `RunSummary` (used by `report.write_report_markdown`) |
| `figures` | PNG figures (Agg backend), public entry points |
| `_figure_maps`, `_figure_charts`, `_figure_helpers` | Private drawing code re-exported by `figures`: scene maps, charts (null fits, areas, refined examples), and layout, scale bar and shared constants |
| `pipeline` | Stage order, timing, log file, `run_detection`, `rebuild_figures` |
| `_pipeline_inputs`, `_pipeline_figures` | Private helpers of `pipeline`: preparing layers and z (fresh or from a saved run), and the figure stage and figure rebuild |
| `sampling` | Stratified inspection sample of a run: detected objects by interval, near misses below the MMU, never-flagged forest cells |
| `chips` | NAIP RGB and CHM chips of an inspection sample, contact sheet, `samples.csv` and `index.md` |
| `_chip_panels` | Private drawing code of `chips`: `RunRasters`/`load_run_rasters`, `cells_in_polygon`, `context_bounds`, the onset and probability context maps, and the percentile time series |
| `cli` | Click commands `detect-disturbance`, `disturbance-figures` and `disturbance-chips` |

## `params`

Parameters come from three layers merged in this order: the packaged `src/csdv_core/config/disturbance.yaml`, an optional user YAML (`--config`), and explicit CLI options. Later layers win key by key. The merged mapping is validated once by `DisturbanceConfig`, which rejects unknown keys. For a `--from-run` rerun, the locked sections are then replaced by those of the saved run, after a check that no explicit value contradicts them.

| Name | Signature | Purpose |
|---|---|---|
| `PACKAGED_CONFIG_NAME` | `"disturbance"` | Stem of the packaged YAML |
| `REUSE_LOCKED_SECTIONS` | `("inputs", "grid", "forest", "standardize")` | Sections that shaped the saved layers and z, taken from the saved run by `--from-run` |
| `deep_merge` | `(base: Mapping, override: Mapping) -> dict` | Recursive merge. Mappings merge, other values (lists included) replace. Inputs are not modified |
| `expand_dotted` | `(overrides: Mapping) -> dict` | Turns `{"decide.rule": "D5"}` into `{"decide": {"rule": "D5"}}` and drops None values (an unset option never overrides) |
| `read_user_config` | `(config_path: Path) -> dict` | Reads a partial YAML. Raises `FileNotFoundError`, or `ValueError` when it is not a mapping |
| `flatten_mapping` | `(mapping, prefix="") -> dict` | Nested mapping to dotted keys (`{"decide": {"rule": "D6"}}` becomes `{"decide.rule": "D6"}`), used by `check_reuse_compatible` and the report's parameter table |
| `check_reuse_compatible` | `(saved_values: Mapping, requested_values: Mapping) -> None` | Compares every locked key present in `requested_values` (full parameters or only the explicit overrides) with the saved value. Raises `ValueError` listing each difference |
| `resolve_params` | `(config_path: Path \| None, cli_overrides: Mapping, *, saved_params: Mapping \| None = None) -> DisturbanceConfig` | The full merge and validation. With `saved_params` (the `params` of a saved `run_params.json`), the explicit config and CLI values are checked with `check_reuse_compatible` and the locked sections are copied from the saved run. Raises `ValueError` on a contradiction and `pydantic.ValidationError` on an unknown key or invalid value |
| `params_to_dict` | `(params: DisturbanceConfig) -> dict` | JSON-ready nested dict that validates back to the same model |

## `inputs`

| Name | Value or signature | Purpose |
|---|---|---|
| `DATE_PATTERN` | `re.compile(r"_(\d{8})_")` | Acquisition date token in file names |
| `RASTER_GLOB` | `"*.tif"` | Files listed inside a year folder |
| `M2_PER_HA` | `10000.0` | Square metres per hectare, shared by the package |
| `parse_acquisition_date` | `(path: Path) -> date \| None` | First valid `_YYYYMMDD_` token in the file name. Works for `CHM_<site>_<date>_0p6m.tif` and `NAIP_<site>_<date>_0p6m_chm.tif` |
| `discover_years` | `(chm_root, *, years=None, naip_root=None, chm_scale=None, chm_nodata=None) -> list[YearInput]` | Lists integer-named year folders with at least one `.tif`, reads dtype, nodata, CRS and pixel size once per year. Raises `FileNotFoundError` for a missing root, year or NAIP folder, and `ValueError` for fewer than two years or mixed dtype or CRS within a year |
| `build_common_grid` | `(year_inputs, grain_m, *, reference_year=None) -> CellGrid` | The reference year (first when None) fixes CRS, pixel size and alignment. The extent is the intersection of all years, shrunk to reference pixel edges and trimmed at the right and bottom to whole cells. Raises `ValueError` on differing CRS or pixel size (naming the years), an unknown reference year, a grain below one pixel, or an empty intersection |
| `discover_site_inputs` | `(site, params, *, paths=None, need_naip=False) -> SiteInputs` | `discover_years` plus `build_common_grid` from the run parameters and project roots |
| `grid_to_dict` / `grid_from_dict` | `(grid) -> dict` / `(data) -> CellGrid` | JSON form of the grid: `crs` (WKT), `pixel_transform` (GDAL geotransform), `pixel_m`, `block_px`, `cell_rows`, `cell_cols` |

`YearInput` (frozen) describes one NAIP year.

| Field | Type | Meaning |
|---|---|---|
| `year` | `int` | Folder name |
| `chm_paths` | `tuple[Path, ...]` | CHM GeoTIFFs, sorted by name, mosaicked on read |
| `acquisition_dates` | `tuple[date \| None, ...]` | Parsed date per CHM file |
| `chm_dtype` | `str` | Stored data type, for example `"uint16"` or `"float32"` |
| `chm_scale` | `float` | Stored value to metres. Config value, else 0.01 for integer and 1.0 for float files |
| `chm_nodata` | `float \| None` | Stored nodata. Config value, else file metadata, else None with a warning |
| `crs` | `CRS` | CRS of the year's files |
| `pixel_m` | `float` | Pixel size (m) |
| `naip_paths` | `tuple[Path, ...]` | NAIP GeoTIFFs, filled only for the NDVI input or the RGB backdrop |

`CellGrid` (frozen) is the common 0.6 m pixel grid and the cell grid laid over it. Pixel `(r, c)` belongs to cell `(r // block_px, c // block_px)`.

| Field or property | Type | Meaning |
|---|---|---|
| `crs` | `CRS` | Shared CRS |
| `pixel_transform` | `Affine` | Pixel grid transform, north up |
| `pixel_m` | `float` | Pixel size (m) |
| `block_px` | `int` | Pixels per cell side, `round(grain_m / pixel_m)` (32 at 19.2 m) |
| `cell_rows`, `cell_cols` | `int` | Cell grid size |
| `cell_m` | property | `pixel_m * block_px` |
| `cell_ha` | property | Cell area (ha), 0.036864 at 19.2 m |
| `cell_transform` | property | Cell grid transform (same origin) |
| `cell_shape`, `pixel_shape` | property | `(rows, cols)` of each grid |
| `bounds` | property | `(left, bottom, right, top)` |
| `extent()` | method | `(left, right, bottom, top)` for `imshow` |

`SiteInputs` (frozen) holds `site`, `chm_root` (`<data_root>/naip_chm/<site>`), `naip_root` (`<data_root>/naip/<site>`), `years` (one `YearInput` per year, ascending), `grid` and `reference_year`. The property `year_list` gives the years as a tuple, and `year_input(year)` returns one year (`KeyError` if absent).

## `windows`

Every function mosaics one band of a year's files into a window of the grid's pixel array. A file on the grid's CRS, pixel size and alignment (the usual case) is read directly by window, an exact copy. Any other file goes through a nearest-neighbour `WarpedVRT`. The direct read is needed for exactness as well as speed: on real NAIP the warper returned 0 for a uint8 value of 255. Each file fills only the grid pixels whose centres it covers, and where files overlap the first file with a valid value wins. Results are float32 with NaN nodata, already multiplied by `scale`.

| Name | Signature | Purpose |
|---|---|---|
| `read_window` | `(paths, grid, row_off, n_rows, *, band=1, scale=1.0, nodata=None) -> ndarray` | Full-width strip of the pixel grid. Raises `ValueError` for a strip outside the grid or no paths |
| `read_bounds` | `(paths, grid, bounds, *, band=1, scale=1.0, nodata=None) -> (ndarray, Affine)` | Window covering `(left, bottom, right, top)` as `from_bounds(...).round_offsets().round_lengths()`: offsets floored to the pixel edge at or before the left and top bounds, width and height rounded to whole pixels (so the far edges can fall half a pixel inside or outside), then clipped to the grid |
| `read_chm_bounds` | `(site_inputs, year, bounds) -> (ndarray, Affine)` | `read_bounds` on one year's CHM files with its scale and nodata, giving heights in metres. Used by refinement and the refined-examples figure |

## `layers`

| Name | Value or signature | Purpose |
|---|---|---|
| `DEFAULT_STRIP_CELLS` | `16` | Strip height in cells when reading |
| `LAYER_NAMES` | `("chm_height", "chm_cover", "valid_share", "forest", "ndvi")` | Year-stack fields of `CellLayers`, one `layers/<name>.tif` each |
| `INPUT_STATE_LAYER` | `{"d_chm_h": "chm_height", "d_chm_cov": "chm_cover", "d_ndvi": "ndvi"}` | State layer that each change input differences |
| `aggregate_block` | `(chm, block_px, *, height_m, min_valid_share) -> BlockAggregate` | Mean height, cover (share of valid pixels with height strictly above `height_m`) and valid share per cell. Height and cover are NaN where the valid share is below `min_valid_share` |
| `aggregate_ndvi` | `(red, nir, block_px, *, min_valid_share) -> ndarray` | Cell-mean NDVI. A pixel is valid where both bands are finite and `red + nir > 0`. Pixel NDVI is `(nir - red) / max(nir + red, 1)` |
| `forest_mask` | `(chm_cover, *, min_cover) -> ndarray` | `chm_cover >= min_cover`, NaN counts as non-forest |
| `build_cell_layers` | `(site_inputs, params, *, include_ndvi=False, strip_cells=16) -> CellLayers` | Reads every year in strips and aggregates. The result does not depend on `strip_cells` |
| `monitored_mask` | `(forest) -> ndarray` | Cells watched for loss: `out[0]` all False, `out[t] = forest[t - 1]` |
| `population_mask` | `(forest) -> ndarray` | Standardization population. As `monitored_mask` except `out[0] = forest[0]` |
| `change_layer` | `(layers, input_name) -> ndarray` | `out[0]` NaN, `out[t] = state[t] - state[t - 1]`. Raises `ValueError` for `d_ndvi` without NDVI |

`BlockAggregate` (frozen) has float32 cell arrays `height`, `cover` and `valid_share` (never NaN). `CellLayers` (frozen) has `years`, the float32 stacks `chm_height`, `chm_cover`, `valid_share`, the bool stack `forest` and the optional stack `ndvi`, plus the property `cell_shape`. Unlike the research `01_layers.py`, CHM layers are not masked where NAIP has no data.

## `standardize`

| Name | Value or signature | Purpose |
|---|---|---|
| `Z_SCALE` | `100` | Stored value is `round(z * 100)` |
| `Z_NODATA` | `-32768` | int16 nodata of stored z |
| `Z_CLIP` | `32767` | Largest stored magnitude |
| `LOCAL_METHODS` | `("local", "local_twopass")` | Methods that use a window and carry `window_m` in the z file name |
| `sigmas_in_cells` | `(params, cell_m) -> list[float]` | Smoothing scales `decide.spatial_sigma_m / cell_m` |
| `window_cells` | `(window_m, cell_m) -> int` | `round(window_m / cell_m)`, plus one when even. `window_cells(150, 19.2) == 9` (172.8 m) |
| `standardize_date` | `(values, population, params, cell_m) -> ndarray` | One date. `scene` is median and 1.4826 MAD, `scenemean` mean and SD, `local` a forest-only moving window with `min_frac`, `trim_z` and `sd_floor`. `local_twopass` computes the local z, the spatial loss probability, then the local z again without cells whose probability exceeds `decide.p_boundary` |
| `standardize_stack` | `(change, population, params, cell_m) -> ndarray` | Every date. Dates without finite values stay NaN |
| `encode_z` | `(z) -> ndarray` | `int16(clip(round(z * 100), -32767, 32767))`, `Z_NODATA` where NaN |
| `decode_z` | `(encoded) -> ndarray` | `float32(encoded) / 100`, NaN where `Z_NODATA` |
| `quantize_z` | `(z) -> ndarray` | `decode_z(encode_z(z))`. Decisions run on quantized z, so a `--from-run` rerun sees the same values as the original run |

## `decide`

| Name | Value or signature | Purpose |
|---|---|---|
| `NO_ONSET` | `-1` | Onset index of a cell without detection |
| `ONSET_YEAR_NONE` | `0` | `onset_year` value of a monitored cell without detection |
| `ONSET_YEAR_UNMONITORED` | `-1` | `onset_year` value of a cell never monitored |
| `SKIPPED_METHOD` | `"skipped"` | `DateFit.method` of a skipped date |
| `EIGHT_CONNECTED` | 3 by 3 True | Structuring element for objects (also used by `outlines`) |
| `skipped_fit` | `(year, n_cells) -> DateFit` | Fit of a date without a null: NaN parameters, method `"skipped"` |
| `fit_date_models` | `(z, monitored, params, years) -> (list[LfdrModel \| None], list[DateFit])` | Efron central-matching null and lfdr curve per date, with `seed = t`. Dates with fewer than `decide.min_fit_cells` monitored finite cells are skipped |
| `rule_d1` | `(z, monitored, params) -> ndarray` | `monitored & (z <= -baseline_z)` |
| `rule_d2` | `(z, monitored, models, params) -> (flags, probability)` | `monitored & (P > p_boundary)` with `P = 1 - lfdr` |
| `spatial_loss_probability` | `(z, monitored, models, params, cell_m) -> ndarray` | D5 probability: the largest calibrated probability over the smoothing scales `spatial_sigma_m / cell_m` |
| `persistence_probability` | `(z, monitored, models, loss_probability, params, cell_m) -> ndarray` | D6 probability: `loss_probability[t] * (1 - P_gain[t + 1])`. The last date, and dates whose next image has fewer than `min_fit_cells` usable cells, keep the D5 value |
| `cell_lean_mask` | `(z, models, params, *, strict) -> ndarray` | Cells whose own probability exceeds the date's prior (plain rules) or `p_boundary` (strict rules) |
| `rule_spatial` | `(z, monitored, models, params, cell_m, *, persistence, strict) -> (flags, probability)` | D5, D5s, D6 and D6s |
| `first_onset` | `(flags) -> ndarray` | int8 index of the first flagged date, `NO_ONSET` where none |
| `apply_mmu` | `(onset_index, cell_ha, mmu_ha) -> ndarray` | Removes objects (8-connected cells sharing an onset) with fewer than `ceil(mmu_ha / cell_ha)` cells. At 19.2 m the 1-acre MMU is 11 cells (0.406 ha) |
| `decide` | `(z, monitored, params, cell_m, *, years) -> DecisionResult` | Fits the nulls (for every rule, so the report always lists them), runs `decide.rule`, onset and MMU |
| `onset_index_to_year` | `(onset_index, years, monitored) -> ndarray` | int16 raster: the onset year, 0 where monitored but not detected, -1 where never monitored |

`DateFit` (frozen) records one date's null.

| Field | Type | Meaning |
|---|---|---|
| `year` | `int` | Later image of the interval |
| `n_cells` | `int` | Monitored cells with finite z |
| `mu0`, `sigma0` | `float` | Null centre and SD (NaN when skipped) |
| `p0` | `float` | Share of cells following the null (NaN when skipped) |
| `method` | `str` | `"central"`, `"iqr"` (fallback) or `"skipped"` |
| `skipped` | `bool` | True when too few cells, always True at index 0. Nothing is detected on a skipped date, except under D1 |

`DecisionResult` (frozen) holds `rule`, `onset_index` (int8, before the MMU), `onset_index_mmu` (int8, after the MMU), `probability` (float32 year stack, NaN on skipped dates and on cells not monitored at that date, so index 0 is all NaN, None for D1) and `fits` (one `DateFit` per date).

## `outlines`

| Name | Value or signature | Purpose |
|---|---|---|
| `OBJECT_COLUMNS`, `REFINED_COLUMNS`, `OPENING_COLUMNS`, `CENTROID_COLUMNS` | tuples | Column orders (see the output specification below) |
| `interval_label` | `(year_before, year) -> str` | `"2016-2018"` |
| `detection_objects` | `(onset_index, probability, z, grid, years) -> GeoDataFrame` | Labels 8-connected cells per onset index, polygonizes them on the cell grid and dissolves per label. `object_id` runs from 1 by onset index, then label. Empty frame with the same columns when nothing is detected |
| `refine_object` | `(chm_before, chm_after, transform, geometry, params) -> RefinedOutline \| None` | Redraws one object from the 0.6 m CHM pair (port of `08_refine_best.refine_object`). Returns None when the forest ring is too small, statistics fail, or the envelope is empty |
| `refine_objects` | `(objects, site_inputs, params) -> RefinementResult` | Reads each object's CHM windows (bounds buffered by `refine.pad_m`) and refines it |
| `objects_table` | `(objects) -> DataFrame` | Objects without geometry, plus centroids in grid CRS and in EPSG:4326 |

`refine_object` works in six steps. It takes the CHM change with a 3 by 3 maximum on the later CHM (`refine.tolerance_px`), builds a reference ring of earlier forest (CHM above `refine.forest_height_m`) outside the object buffered by `refine.ring_gap_m`, standardizes the change against the ring, fits Efron's null on the ring, marks loss where the posterior exceeds `refine.p_loss` within `refine.near_m` of the object, and closes the resulting openings into an envelope that must touch the object and reach the MMU. Refinement always reads the CHM, whatever the detection input.

`RefinedOutline` (frozen) has `envelope` and `openings` (shapely geometries), `envelope_ha`, `openings_ha`, `opening_share` (opening pixels over envelope pixels), `n_openings` (8-connected components) and `n_ring_px`. `RefinementResult` has the GeoDataFrames `outlines` and `openings`, the counts `n_objects` and `n_refined`, and `skipped_object_ids`.

## `store`

| Name | Value or signature | Purpose |
|---|---|---|
| `GTIFF_PROFILE` | DEFLATE, tiled 256 by 256 | Creation options of every raster |
| `OBJECTS_LAYER`, `REFINED_OUTLINES_LAYER`, `REFINED_OPENINGS_LAYER` | `"objects"`, `"refined_outlines"`, `"refined_openings"` | GeoPackage layer names |
| `z_filename` | `(input_name, method, window_m) -> str` | `z_<input>_<method>.tif`, or `z_<input>_<method>_<int(window_m)>m.tif` for local methods |
| `claim_output_dir` | `(path) -> RunPaths` | Creates the run folder and its four subfolders. Raises `FileExistsError` if `path` exists in any form |
| `write_year_stack` / `read_year_stack` | `(path, stack, grid, years) -> Path` / `(path) -> YearStack` | Multi-band year rasters on the cell grid |
| `write_z` / `read_z` | `(path, z, grid, years) -> Path` / `(path) -> YearStack` | int16 z stacks through `encode_z` and `decode_z` |
| `write_layers` / `read_layers` | `(run_paths, layers, grid) -> dict` / `(run_paths) -> CellLayers` | The `layers/` folder (`ndvi.tif` is optional on read) |
| `write_detection` | `(run_paths, *, onset_year, onset_year_nommu, probability, grid, years, objects, refinement) -> dict` | Detection rasters, GeoPackage and `objects.csv` |
| `write_run_params` / `read_run_params` | `(run_paths, params, *, site, command_line, config_path, from_run, grid=None) -> Path` / `(run_dir) -> dict` | `run_params.json`, written after claiming the folder and again once the grid exists |
| `load_run` | `(run_dir) -> SavedRun` | Reloads parameters, grid, layers and z without the source CHMs |

`RunPaths` (frozen) holds every path of a run folder: `root`, `run_params`, `log`, `summary_json`, `report_md`, `layers_dir`, `z_dir`, `detection_dir`, `figures_dir`, `onset_year`, `onset_year_nommu`, `probability`, `disturbances_gpkg` and `objects_csv`. Build it with `RunPaths.for_root(root)`. The methods `layer_path(name)` and `z_path(params)` return file paths. `YearStack` (frozen) holds `values` and `years` (from the band descriptions). `SavedRun` (frozen) holds `run_paths`, `site`, `params`, `grid`, `layers`, `z` (decoded, already quantized) and `years`.

## `report`

| Name | Value or signature | Purpose |
|---|---|---|
| `SUMMARY_SCHEMA_VERSION` | `1` | Version of the `summary.json` layout |
| `REPORTED_PACKAGES` | numpy, scipy, rasterio, geopandas, shapely, pydantic, matplotlib | Versions recorded in every run |
| `DEFAULT_CAVEATS` | three sentences | No persistence check on the last interval, refinement always uses the CHM, no reference scoring |
| `software_info` | `(repo_root=None) -> dict[str, str]` | `csdv_core` version, `git_commit`, `git_dirty`, `python` and package versions. Git failures give `"unknown"` |
| `input_inventory` | `(site_inputs) -> list[InputFileInfo]` | Every CHM and NAIP file by year |
| `grid_summary` | `(grid, params) -> GridSummary` | The grid and the window and sigma sizes in cells |
| `build_summary` | `(*, site, params, inputs, grid, layers, monitored, decision, objects, refinement, stage_seconds, run_paths, command_line, config_path, from_run) -> RunSummary` | Assembles the summary. Areas are cell counts times `cell_ha` |
| `summary_to_dict` / `summary_from_dict` | | JSON form. Year keys become strings, NaN becomes null |
| `refresh_summary` | `(summary, run_paths, stage_seconds=None) -> RunSummary` | Copy with `outputs` listed again from the run folder and, when given, new stage times. The pipeline calls it after the figures |
| `write_summary_json` / `read_summary_json` | `(summary, path) -> Path` / `(path) -> RunSummary` | `summary.json`, indent 2, keys in field order (stage times in stage order) |
| `write_report_markdown` | `(summary, path, *, figure_paths=None) -> Path` | `report.md` |

`InputFileInfo` (frozen) has `year`, `kind` (`"chm"` or `"naip"`), `path`, `acquisition_date` (ISO string or None), `dtype`, `scale` and `nodata`. `GridSummary` (frozen) has `crs`, `bounds`, `pixel_m`, `block_px`, `cell_m`, `cell_rows`, `cell_cols`, `area_ha`, `window_cells` and `window_m_effective` (None for scene methods) and `sigma_cells`. `IntervalSummary` (frozen) has `year_before`, `year`, `label`, `monitored_ha`, `detected_ha`, `detected_ha_nommu`, `n_objects`, `share_of_monitored`, `fit` (a `DateFit`), `skipped`, `n_refined` and `refined_ha`. `RunSummary` is described with `summary.json` below.

## `figures`

All functions use the Agg backend, take explicit arrays and return the written path. `draw_*` functions draw into a supplied matplotlib `Axes` so panels can be reused in other figures.

| Name | Purpose |
|---|---|
| `FIGURE_FILES` | Figure name to file name (nine figures, listed below) |
| `read_naip_backdrop(naip_paths, grid, *, bands=(1, 2, 3), max_px=2000) -> NaipBackdrop` | Stretched RGB over the grid extent. Always bands 1, 2 and 3 (`inputs.naip_red_band` only serves NDVI) |
| `chm_height_backdrop(chm_height, grid) -> NaipBackdrop` | One year of cell CHM height as a grey image, 0 m to the 99th percentile (`CHM_BACKDROP_PERCENTILE`), taller canopy lighter |
| `onset_backdrop(inputs) -> NaipBackdrop \| None` | Backdrop of the onset maps: the NAIP backdrop when given, otherwise the grey CHM height of the first year when `chm_height` is set |
| `draw_forest`, `plot_forest_initial`, `plot_forest_by_year` | Forest mask at the first year and per year |
| `draw_onset`, `plot_disturbance_onset`, `plot_onset_by_interval` | Onset map (one colour per year) and one panel per interval, over the backdrop from `onset_backdrop` |
| `plot_z_maps` | Diverging z panel per interval, limits plus and minus 4 |
| `plot_probability_maps` | Probability panel per interval. The colour ramp changes from neutral to warm at `p_boundary` (a colour break, not a contour line), the colour bar marks the boundary, and NaN cells are grey |
| `draw_null_fit`, `plot_null_fits` | Monitored z histogram with the fitted null `p0 N(mu0, sigma0)` per interval. The dashed boundary is approximate: it marks where the cell-level posterior `1 - p0 f0 / f`, with `f` from the plotted histogram, reaches `p_boundary`, not the smoothed lfdr curve used for decisions |
| `draw_area_by_interval`, `plot_area_by_interval` | Detected area per interval, after and before the MMU |
| `plot_refined_examples` | CHM before and after for the largest refined objects, with the cell object and envelope |
| `make_figures(inputs, figures_dir, *, dpi=200) -> dict[str, Path]` | Writes every figure the inputs allow. A failing figure is logged and left out |

`NaipBackdrop` (frozen) holds `rgb` (uint8, decimated) and `extent`. `FigureInputs` collects what `make_figures` draws: `grid`, `years`, `forest`, `monitored`, `z`, `probability`, `onset_year`, `fits`, `intervals`, `objects`, `refined`, `p_boundary`, `bin_w`, and the optional `site_inputs`, `backdrop` and `chm_height` (the cell CHM height stack, which supplies the default grey onset-map backdrop).

## `pipeline`

| Name | Signature | Purpose |
|---|---|---|
| `StageTimer` | dataclass with `seconds: dict[str, float]` and context manager `stage(name)` | Wall time per stage, recorded even when a stage fails |
| `attach_log_handler` / `detach_log_handler` | `(log_path, level=INFO) -> Handler` / `(handler) -> None` | File log on the `csdv_core` logger (its level is lowered to `level` when needed) |
| `run_detection` | `(params, site, output_dir, from_run=None, command_line=None, config_path=None, log_level=INFO) -> RunSummary` | One run end to end into a new folder. Raises `FileExistsError`, `FileNotFoundError` or `ValueError` |
| `rebuild_figures` | `(run_dir, *, naip_backdrop=None) -> dict[str, Path]` | Rewrites `figures/` and `report.md` of an existing run in place. `naip_backdrop=None` follows the saved `figures.naip_backdrop`. `summary.json` is not changed |

`run_detection` runs the stages in this order. Setup claims the folder, attaches the log and writes `run_params.json`. Inputs discovers years and builds the grid. Layers writes `layers/`. Standardize builds the change layer and the z stack and writes `z/`. Decide applies the rule, onset and MMU. Outlines builds the objects and refined outlines and writes the detection rasters, the GeoPackage and the CSV. Report writes `summary.json` and `report.md`. Figures writes the PNGs, and then `summary.json` and `report.md` are written again with the full stage times and the figure links. With `from_run`, the inputs stage reloads the saved layers and z (checked with `params.check_reuse_compatible`, and the site must match) and copies them into the new folder, and the layers and standardize stages do not run. When refinement or the NAIP backdrop is on, the site's files are discovered again (metadata only) and the saved grid is attached, so the source CHMs, and NAIP for a backdrop, must still be available. The input inventory comes from the saved `summary.json`. A failed run keeps its folder, and the exception and traceback go to its log.

## `sampling`

| Name | Purpose |
|---|---|
| `SampleDesign(n_detected=24, n_below_mmu=4, n_no_detection=0, min_per_interval=2, seed=0)` | Sample sizes and seed |
| `STRATUM_DETECTED`, `STRATUM_BELOW_MMU`, `STRATUM_NO_DETECTION`, `STRATA` | Stratum names |
| `NEAR_MISS_MMU_SHARE` | Smallest near miss drawn, as a share of the MMU (0.5) |
| `SAMPLE_COLUMNS` | Columns of the sample table |
| `allocate_by_stratum(counts, n_total, min_per_stratum) -> dict` | Floor per stratum, then the remainder by the D'Hondt rule, never more than a stratum holds |
| `sample_objects(objects, n_samples, min_per_interval, rng) -> GeoDataFrame` | Objects drawn at random within intervals |
| `below_mmu_objects(saved) -> GeoDataFrame` | Objects flagged by the rule but removed by the MMU |
| `no_detection_cells(saved, n_samples, rng) -> GeoDataFrame` | Monitored forest cells never flagged, each with a random monitored interval |
| `draw_samples(saved, design) -> GeoDataFrame` | The full sample with `SAMPLE_COLUMNS` and polygons, ids `S01`, `S02`, ... |

## `chips`

| Name | Purpose |
|---|---|
| `ChipSettings(pad_m=40.0, min_chip_m=150.0, show_chm=True, context_factor=4.0, min_context_m=600.0, dpi=150)` | Chip size, context map size and look |
| `InspectionResult` | Paths written: `output_dir`, `samples`, `table_path`, `vector_path`, `chip_paths`, `contact_sheet`, `index_path` |
| `DEFAULT_FOLDER_NAME` | `"inspection"`, inside the run folder |
| `chip_bounds(geometry, settings) -> tuple` | Square bounds around a unit, at least `min_chip_m` wide |
| `read_rgb_chip(site_inputs, year, bounds) -> (rgb, extent)` | NAIP bands 1 to 3 on the run grid, stretched for display |
| `plot_sample_chips(sample, site_inputs, rasters, envelope, out_path, settings) -> Path` | RGB and CHM of every year for one unit, the onset and probability context maps, and the DI and CHM height series of the unit's cells |
| `plot_contact_sheet(samples, site_inputs, out_path, settings) -> Path` | Before and after RGB of every unit |
| `chip_filename(sample) -> str` | `<sample_id>_<stratum>_<interval>.png` |
| `write_index_markdown(samples, run_dir, design, out_path) -> Path` | `index.md` with the contact sheet and chip links |
| `make_inspection_chips(run_dir, output_dir=None, *, design=None, settings=None, sample_only=False) -> InspectionResult` | Loads the run, claims the output folder (raises `FileExistsError` if it exists), draws the sample, writes the table and the chips |

## `cli`

| Name | Purpose |
|---|---|
| `OPTION_TO_KEY` | Click option name to dotted config key |
| `parse_years(text) -> list[int] \| None` | Parses `"2014,2016,2018"` |
| `cli_overrides_from_options(options) -> dict` | Options to dotted overrides for `resolve_params` |
| `default_output_dir(site) -> Path` | `project_paths().disturbance_dir(site)`, that is `$CSDV_RESULTS_ROOT/disturbance/<site>` |
| `configure_console_logging(level_name) -> int` | `logging.basicConfig` at the chosen level for terminal output |
| `detect_disturbance` | The `csdv detect-disturbance` command |
| `disturbance_figures` | The `csdv disturbance-figures` command (`--run-dir`, `--naip-backdrop/--no-naip-backdrop`, default: the saved run's setting, `--log-level`) |
| `disturbance_chips` | The `csdv disturbance-chips` command (`--run-dir`, `--output-dir`, `--n-detected`, `--n-below-mmu`, `--n-no-detection`, `--min-per-interval`, `--seed`, `--pad-m`, `--min-chip-m`, `--chm/--no-chm`, `--sample-only`, `--log-level`) |

`csdv detect-disturbance` options that are not parameters: `--site` (required), `--output-dir`, `--config`, `--from-run` and `--log-level` (DEBUG, INFO, WARNING, ERROR). With `--from-run`, the command reads the saved `run_params.json` and passes its `params` to `resolve_params(..., saved_params=...)`, so a contradicting option fails before any folder is created. An existing output folder is reported as an error that names `--output-dir`. The command echoes the run folder on stdout.

## Parameters

Defaults reproduce the best tested case from `2_Disturbance_Index`. Every key may appear in a `--config` YAML. A key without a CLI option can only be changed there. Validation bounds come from the Pydantic model.

| Key | Default | CLI option | Meaning |
|---|---|---|---|
| `inputs.input` | `d_chm_h` | `--input` | Change layer: `d_chm_h` (mean CHM height), `d_chm_cov` (CHM cover above `forest.height_m`) or `d_ndvi` (NAIP NDVI) |
| `inputs.years` | null (all) | `--years 2014,2016,2018` | Years to use, at least two, no duplicates |
| `inputs.chm_scale` | null | `--chm-scale` | Stored CHM value to metres. Null gives 0.01 for integer files, 1.0 for float files |
| `inputs.chm_nodata` | null | `--chm-nodata` | Stored CHM nodata, in stored units. Null takes it from the file |
| `inputs.naip_red_band` | 1 | none | 1-based NAIP red band |
| `inputs.naip_nir_band` | 4 | none | 1-based NAIP near-infrared band |
| `grid.grain_m` | 19.2 | `--grain-m` | Cell size (m), rounded to whole source pixels |
| `grid.reference_year` | null (first) | `--reference-year` | Year that fixes CRS and pixel alignment |
| `grid.min_valid_share` | 0.5 | none | Least share of valid CHM pixels in a cell, else NaN |
| `forest.height_m` | 5.0 | none | CHM height (m) above which a pixel counts as canopy |
| `forest.min_cover` | 0.5 | none | Least canopy share for a forest cell |
| `standardize.method` | `local_twopass` | `--method` | `scene`, `scenemean`, `local` or `local_twopass` |
| `standardize.window_m` | 150.0 | `--window-m` | Window side (m), rounded to an odd number of cells (9 cells, 172.8 m at 19.2 m) |
| `standardize.min_frac` | 0.2 | none | Least forest share in a window before falling back to scene statistics |
| `standardize.trim_z` | 3.0 | none | Population cells beyond this robust scene z are left out of window statistics (null disables) |
| `standardize.sd_floor` | 0.25 | none | Least window SD as a share of the scene scale |
| `decide.rule` | `D6` | `--rule` | `D1`, `D2`, `D5`, `D5s`, `D6` or `D6s` |
| `decide.p_boundary` | 0.5 | `--p-boundary` | Bayes boundary on posterior probabilities (also the second-pass exclusion level) |
| `decide.baseline_z` | 3.0 | none | Fixed threshold of rule D1 (`z <= -3`) |
| `decide.spatial_sigma_m` | [10.0, 25.0] | none | Gaussian smoothing SDs (m) for spatial support |
| `decide.pct0` | 0.25 | none | Quantile range for Efron's central matching (quartiles) |
| `decide.bin_w` | 0.05 | none | Histogram bin width (z units) |
| `decide.smooth_bins` | 2.0 | none | Gaussian smoothing of the mixture histogram (bins) |
| `decide.min_prior` | 0.01 | none | Floor on the per-date prior probability of disturbance |
| `decide.min_fit_cells` | 1000 | none | Fewest monitored cells to fit a date (at least 100). Smaller dates are skipped and reported |
| `mmu.mmu_ha` | 0.405 | `--mmu-ha` | Minimum mapping unit (ha), 1 acre. 0 keeps every object |
| `refine.enabled` | true | `--refine/--no-refine` | Redraw objects from the 0.6 m CHM |
| `refine.pad_m` | 60.0 | none | Read window buffer around each object (m) |
| `refine.ring_gap_m` | 20.0 | none | Gap between object and reference ring (m) |
| `refine.near_m` | 10.0 | none | Loss pixels must lie within this distance of the object (m) |
| `refine.forest_height_m` | 5.0 | none | Earlier CHM height (m) that marks ring forest |
| `refine.min_ring_px` | 2000 | none | Fewest ring pixels, else the object is skipped |
| `refine.p_loss` | 0.5 | none | Pixel posterior above which a pixel is loss |
| `refine.canopy_m`, `open_m`, `drop_m` | 10.0, 5.0, 8.0 | none | `RefineParams` height thresholds. The lfdr loss test replaces them, so they only fill `RefineParams` |
| `refine.tolerance_px` | 1 | none | Half width of the maximum filter on the later CHM |
| `refine.open_radius_m` | 1.0 | none | Opening radius that removes crown slivers (m) |
| `refine.min_opening_m2` | 25.0 | none | Smaller openings are dropped (m²) |
| `refine.close_radius_m` | 8.0 | none | Closing radius that joins openings into an envelope (m) |
| `refine.fill_holes_m2` | 2500.0 | none | Holes up to this area are filled (m²) |
| `refine.smooth_m` | 3.0 | none | Buffer out and in that rounds envelope outlines (m) |
| `refine.simplify_m` | 0.6 | none | Douglas-Peucker tolerance (m) |
| `figures.enabled` | true | `--figures/--no-figures` | Write the PNGs |
| `figures.naip_backdrop` | false | `--naip-backdrop/--no-naip-backdrop` | Draw NAIP RGB under the maps (reads NAIP) |
| `figures.dpi` | 200 | none | Figure resolution (50 to 1200) |

The least refined envelope area is `mmu.mmu_ha`. With `--from-run`, the `inputs`, `grid`, `forest` and `standardize` sections are taken from the saved run, and an explicit option or config value in those sections must equal the saved one. `decide`, `mmu`, `refine` and `figures` may change, and they come from the packaged defaults, the rerun's `--config` and its options.

## Output specification

### Folder tree

```
<run>/                         default $CSDV_RESULTS_ROOT/disturbance/<site>/
├── run_params.json
├── detect_disturbance.log
├── summary.json
├── report.md
├── layers/
│   ├── chm_height.tif
│   ├── chm_cover.tif
│   ├── valid_share.tif
│   ├── ndvi.tif               only for --input d_ndvi
│   └── forest.tif
├── z/
│   └── z_<input>_<method>[_<window>m].tif
├── detection/
│   ├── onset_year.tif
│   ├── onset_year_nommu.tif
│   ├── probability.tif        not written for rule D1
│   ├── disturbances.gpkg
│   └── objects.csv
└── figures/
    ├── forest_initial.png
    ├── forest_by_year.png
    ├── disturbance_onset.png
    ├── onset_by_interval.png
    ├── z_maps.png
    ├── probability_maps.png   not written for rule D1
    ├── null_fits.png
    ├── area_by_interval.png
    └── refined_examples.png   only when refinement redrew at least one object
```

The default z file is `z/z_d_chm_h_local_twopass_150m.tif`. The window label is `int(window_m)`, the requested size, while `summary.json` and the report give the effective size in cells and metres.

### Rasters

All rasters share the cell grid (`CellGrid.cell_transform` and CRS), DEFLATE compression and 256 by 256 tiles. In multi-band files band `b` (1-based) is year `years[b - 1]`, and each band description holds the year as a string.

| File | Bands | dtype | nodata | Values |
|---|---|---|---|---|
| `layers/chm_height.tif` | one per year | float32 | NaN | Mean CHM height (m) over valid pixels |
| `layers/chm_cover.tif` | one per year | float32 | NaN | Share of valid pixels above `forest.height_m`, 0 to 1 |
| `layers/valid_share.tif` | one per year | float32 | NaN (never present) | Share of the cell's pixels with valid CHM, 0 to 1 |
| `layers/ndvi.tif` | one per year | float32 | NaN | Cell-mean NAIP NDVI |
| `layers/forest.tif` | one per year | uint8 | none | 1 forest, 0 not forest |
| `z/z_*.tif` | one per year | int16 | -32768 | `round(z * 100)`, clipped to plus or minus 32767. Divide by 100 for z. Band 1 is all nodata for change inputs |
| `detection/onset_year.tif` | 1 | int16 | none | Onset year after the MMU, 0 monitored without detection, -1 never monitored |
| `detection/onset_year_nommu.tif` | 1 | int16 | none | The same before the MMU |
| `detection/probability.tif` | one per year | float32 | NaN | Decision probability of the rule (D2 lfdr posterior, D5 and D5s spatial loss probability, D6 and D6s with persistence). NaN on skipped dates and on cells not monitored at that date, so band 1 is all NaN |

An onset year is the later image of its interval. A value of 2018 in a 2012 to 2022 biennial series means the loss happened between the 2016 and 2018 images. A cell is monitored at date `t` when it was forest at `t - 1`, so -1 marks cells that were never forest at any image before the last.

### GeoPackage `detection/disturbances.gpkg`

All layers are in the grid CRS. Areas are in hectares.

Layer `objects`, one row per detected object after the MMU:

| Column | Type | Meaning |
|---|---|---|
| `object_id` | int | 1-based, ordered by onset index, then label |
| `onset_year` | int | Later image year of the interval |
| `year_before` | int | Earlier image year |
| `interval` | str | `"<year_before>-<onset_year>"` |
| `onset_index` | int | Date index `t` in `years` |
| `n_cells` | int | Cells in the object |
| `area_ha` | float | `n_cells * cell_ha` |
| `mean_prob`, `max_prob` | float | Mean and maximum decision probability at the onset date (NaN for D1) |
| `mean_z` | float | Mean z at the onset date |
| `geometry` | (Multi)Polygon | Dissolved cell outline |

Layer `refined_outlines`, one row per object that refinement could redraw (absent with `--no-refine`):

| Column | Type | Meaning |
|---|---|---|
| `object_id`, `onset_year`, `year_before`, `interval` | | As in `objects` |
| `envelope_ha` | float | Envelope area |
| `openings_ha` | float | Opening area inside the envelope |
| `opening_share` | float | Opening pixels over envelope pixels |
| `n_openings` | int | 8-connected opening components |
| `n_ring_px` | int | Ring pixels used for the null |
| `geometry` | (Multi)Polygon | Smoothed and simplified envelope |

Layer `refined_openings`, one row per refined object: `object_id`, `onset_year`, `openings_ha` and the unsmoothed openings geometry.

### `detection/objects.csv`

One row per object, with every column of the `objects` layer except `geometry`, plus `centroid_x` and `centroid_y` (polygon centroid in grid CRS units) and `centroid_lon` and `centroid_lat` (the same point in EPSG:4326). It is meant for drawing interpretation samples in a spreadsheet or pandas without GIS software.

### `run_params.json`

| Key | Content |
|---|---|
| `site` | Site name |
| `created` | ISO 8601 local time |
| `command_line` | The command that started the run, or null |
| `config_path` | The `--config` YAML, or null |
| `from_run` | The source run of `--from-run`, or null |
| `params` | The final merged parameters, one key per section |
| `grid` | `grid_to_dict` of the common grid (null only if the run failed before the grid was built) |

The file is enough to rebuild the grid and reload the run without the source CHMs.

### `summary.json`

`summary.json` is `summary_to_dict(RunSummary)`, written with indent 2 and keys in field order. Year keys are strings and NaN is null.

| Key | Content |
|---|---|
| `schema_version` | 1 |
| `site`, `created`, `output_dir` | Run identity |
| `command_line`, `config_path`, `from_run` | As in `run_params.json` |
| `params` | Final parameters |
| `input_name`, `method`, `rule` | Shortcuts to the main choices |
| `years` | NAIP years |
| `inputs` | Inventory: one `InputFileInfo` per file (year, kind, path, acquisition date, dtype, scale, nodata) |
| `grid` | `GridSummary`: CRS, bounds, pixel and cell size, cell counts, area (ha), window size in cells and metres, smoothing scales in cells |
| `forest_ha_by_year` | Forest area per year (ha) |
| `intervals` | One `IntervalSummary` per interval: monitored, detected (after and before the MMU) and refined area, object counts, share of monitored forest, null fit, skipped flag |
| `skipped_years` | Years whose null fit was skipped (the first year excluded) |
| `n_objects`, `detected_ha` | Totals after the MMU |
| `n_refined`, `n_refine_skipped`, `refined_ha` | Refinement totals (0 with `--no-refine`) |
| `stage_seconds` | Wall time per stage (s) |
| `software` | `software_info()`: package version, git commit and dirty flag, Python and package versions |
| `caveats` | Sentences printed in the report |
| `outputs` | Short name to file path relative to the run folder |

### `report.md` and the log

`report.md` presents the same content for reading: run, parameters (with effective window and sigma sizes), input inventory, grid, forest area by year, detections by interval, null fits by date (skipped dates flagged), refinement, runtime by stage, software, caveats, figure links and output files. `detect_disturbance.log` holds every `csdv_core` log record of the run, including warnings such as a CHM year without a nodata value.
