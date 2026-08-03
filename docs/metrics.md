# Metrics

This is the reference for every metric the pipeline computes. The unit of analysis is the stand polygon, so each entry describes the value as it appears in `stand_metrics.parquet`, one row per stand per date. The fixed-window raster path still exists and each entry notes its variant, but the stand path is the default.

Related: [architecture.md](architecture.md) for how the packages fit together, [data_layout.md](data_layout.md) for where the tables land on disk.

## How a stand metric is computed

The same five steps run for every metric, so the entries below do not repeat them.

**1. Load the stands.** `io/stands.py::read_ais_stands` reads an AIS disturbance-polygon file geodatabase. One impact polygon is one stand. Each gets a `stand_id` of the form `{module}-U{UID1}-{UID2}-{UID3}`, with a letter suffix when one footprint splits into several polygons. The reader requires a projected CRS and repairs invalid geometry with `shapely.make_valid`.

**2. Read a window, not the scene.** `zonal/mask.py::read_stand_array` reads only the stand's bounding box out of the raster using `rasterio.windows.from_bounds`. It converts nodata to NaN and applies a read scale, which is `0.01` for a uint16 CHM in centimetres and `1.0` for float32 metres.

**3. Build the membership mask.** `zonal/mask.py::stand_window` calls `rasterio.features.geometry_mask(..., all_touched=False)`. A pixel belongs to the stand when its centre falls inside the polygon. A stand that contains no pixel centres raises rather than returning an empty result.

**4. Compute.** Each metric is a pure function of `(array, mask)` that returns one float. Nothing is rasterised wall to wall and no zonal-stats reducer runs. The polygon is the support, so there is no window size to choose.

**5. Assemble.** `zonal/compute.py::compute_stand_record` builds one `StandMetricRecord` per stand per date. `compute_module_metrics` returns the tidy frame, sorted by `stand_id` then `year`.

Two exceptions to step 4. Crown metrics read a vector layer segmented over the whole scene, not the CHM window. Satellite metrics reduce inside Earth Engine at 30 m and join back on `(stand_id, year)`.

### Missing values

No metric is ever silently blank. A metric that cannot be computed is NaN and the reason lands in the `unavailable` column, formatted `"metric: why; metric: why"`. Sample-size columns carry the `support_` prefix. Satellite support columns use `sat_` so the join cannot collide.

## Reading the entries

Each entry gives the definition, the source raster or vector, how the number comes out of the masked array, the unit, the parameters and where they are set, and which classification rules read it. "Feeds" names the stage envelopes in `config/stages.yaml` and the trajectory rules in `config/trajectories.yaml`. "Windowed variant" names the equivalent function in `csdv_core/metrics/` and any place the two disagree on purpose.

Thresholds live in two places. `config/metrics.yaml` and `config/satellite.yaml` hold the values the typed loader reads. Structural constants that define what a metric means, such as the 2 m canopy threshold, are module constants in Python. See [Where parameters live](#where-parameters-live).

## Metric summary

| Metric | Category | Source | Units | Feeds |
|---|---|---|---|---|
| `gap_fraction` | cover and gaps | CHM | fraction | all 7 stages, 14 trajectory rules |
| `crown_fraction` | cover and gaps | CHM | fraction | 7 LC trajectory rules |
| `gap_persistence` | cover and gaps | CHM, two dates | fraction | all 7 stages, LC7 |
| `shrub_fraction` | height | CHM | fraction | 6 stages, EF1, EF2 |
| `small_tree_fraction` | height | CHM | fraction | no rule |
| `mid_canopy_fraction` | height | CHM | fraction | no rule |
| `tall_canopy_fraction` | height | CHM | fraction | no rule |
| `height_mean` | height | CHM | m | no rule |
| `height_median` | height | CHM | m | no rule |
| `height_p90` | height | CHM | m | no rule |
| `height_max` | height | CHM | m | no rule |
| `height_cv` | height | CHM | dimensionless | no rule |
| `crown_cv` | crowns | crown polygons | dimensionless | all 7 stages, 7 trajectory rules |
| `crown_p90` | crowns | crown polygons | m | no rule |
| `crown_mean` | crowns | crown polygons | m | no rule |
| `crown_median` | crowns | crown polygons | m | no rule |
| `crown_std` | crowns | crown polygons | m | no rule |
| `crown_count` | crowns | crown polygons | count | no rule |
| `glcm_texture` | texture | NAIP band 4 (NIR) | bits | all 7 stages, 5 trajectory rules |
| `edge_density` | spatial pattern | CHM | 1/m | no rule |
| `linearity_index` | spatial pattern | CHM | 0 to 1 | LC7 |
| `row_directionality` | spatial pattern | CHM | 0 to 1 | FC2 |
| `d_gap_fraction` | change | derived | fraction | no rule |
| `d_crown_fraction` | change | derived | fraction | no rule |
| `d_crown_p90` | change | derived | m | no rule |
| `d_crown_count` | change | derived | count | no rule |
| `d_edge_density` | change | derived | 1/m | no rule |
| `ndvi` | satellite, per scene | Landsat C2 L2 | index | input to annual metrics |
| `nbr` | satellite, per scene | Landsat C2 L2 | index | registered, not extracted |
| `ndmi` | satellite, per scene | Landsat C2 L2 | index | registered, not extracted |
| `ndvi_mean` | satellite, annual | `ndvi` series | index | all 7 stages, LC6 |
| `ndvi_seasonal_amplitude` | satellite, annual | `ndvi` series | index | all 7 stages, LC1, LC3, LC5 |
| `ndvi_trend` | satellite, annual | `ndvi_mean` series | index per year | DS2, DS3a, EF3 |

---

## Canopy cover and gaps

Source module: `zonal/pixel.py`. All three metrics split the CHM at `CANOPY_HEIGHT_THRESHOLD_M = 2.0`. A pixel below 2 m is gap, a pixel at or above 2 m is canopy.

### gap_fraction

Share of valid in-stand pixels whose canopy height falls below the 2 m threshold.

**Source.** NAIP-derived CHM, single band, heights in metres.
**Calculation.** Take the in-stand pixels that are finite, test each against the threshold, return the mean of the boolean. Returns NaN when the stand holds no valid pixel.
**Units.** Fraction, 0 to 1.
**Parameters.** `height_threshold_m = 2.0`, from `CANOPY_HEIGHT_THRESHOLD_M` in `zonal/pixel.py`. The raster path reads `defaults.chm_gap_threshold_m` in `config/metrics.yaml`, which carries the same value.
**Feeds.** Every stage envelope, ESI through OG. Fourteen trajectory rules across all four groups.
**Support.** `support_n_valid_pixels`, `support_nodata_fraction`.
**Windowed variant.** `metrics/gap.py::gap_fraction`, `window_m = 25.0`. Same arithmetic per tile.

### crown_fraction

Share of valid in-stand pixels at or above 2 m. The complement of `gap_fraction`.

**Source.** NAIP-derived CHM.
**Calculation.** `1 - gap_fraction`, computed on the same pixel set so the two always sum to 1. NaN propagates.
**Units.** Fraction, 0 to 1.
**Parameters.** `height_threshold_m = 2.0`.
**Feeds.** No stage envelope. Seven land-conversion trajectory rules, LC1 through LC7.
**Support.** Same as `gap_fraction`.
**Windowed variant.** `metrics/gap.py::crown_fraction`, `window_m = 25.0`.

### gap_persistence

Share of pixels that read as gap at both of two dates.

**Source.** Two NAIP-derived CHMs on an identical grid.
**Calculation.** The denominator is in-stand pixels that are finite at both dates, so a hole in one date does not count against the other. The numerator is pixels below 2 m at both dates. The two rasters must share a transform. A mismatch raises rather than resampling, because a difference across dates already compounds the error of both inputs.
**Units.** Fraction, 0 to 1.
**Parameters.** `height_threshold_m = 2.0`.
**Feeds.** Every stage envelope. LC7.
**Support.** NaN at the first date of a stand's series, with the reason `"first date of the series, nothing to compare to"`.
**Windowed variant.** `metrics/gap.py::gap_persistence`, `window_m = 25.0`. It is the one registered metric that takes two arrays, so a generic caller has to special-case it.

---

## Height distribution

Source module: `zonal/pixel.py`. Four cover fractions split the CHM into height bands. Five summary statistics describe the canopy surface.

The bands are half-open, `[lo, hi)`. They do not partition the height range. Nothing falls in `[0, 0.5)`, and the shrub band straddles the 2 m canopy threshold, so `shrub_fraction` and `gap_fraction` overlap by design.

| Metric | Band |
|---|---|
| `shrub_fraction` | `[0.5, 2.0)` m |
| `small_tree_fraction` | `[2.0, 10.0)` m |
| `mid_canopy_fraction` | `[10.0, 20.0)` m |
| `tall_canopy_fraction` | `[20.0, 100.0)` m |

### shrub_fraction

Share of valid in-stand pixels between 0.5 m and 2 m. Reads as low woody cover and regenerating shrubs.

**Source.** NAIP-derived CHM.
**Calculation.** `height_band_fraction` tests each finite in-stand pixel against `lo <= h < hi` and returns the mean.
**Units.** Fraction, 0 to 1.
**Parameters.** `lo_m = 0.5`, `hi_m = 2.0`, from `SHRUB_BAND_M`. Mirrored in `config/metrics.yaml` for the raster path.
**Feeds.** Six stage envelopes, LSI through OG. ESI omits it on purpose, because shrub cover is not constrained at that stage. EF1 and EF2.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** `metrics/cover.py::shrub_fraction`, `window_m = 25.0`.

### small_tree_fraction

Share of valid in-stand pixels between 2 m and 10 m.

**Source.** NAIP-derived CHM.
**Calculation.** As above with `lo_m = 2.0`, `hi_m = 10.0`.
**Units.** Fraction, 0 to 1.
**Parameters.** `SMALL_TREE_BAND_M`.
**Feeds.** No stage envelope and no trajectory rule. Computed for description and figures.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** `metrics/cover.py::small_tree_fraction`, `window_m = 25.0`.

### mid_canopy_fraction

Share of valid in-stand pixels between 10 m and 20 m.

**Source.** NAIP-derived CHM.
**Calculation.** As above with `lo_m = 10.0`, `hi_m = 20.0`.
**Units.** Fraction, 0 to 1.
**Parameters.** `MID_CANOPY_BAND_M`.
**Feeds.** No rule.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** `metrics/cover.py::mid_canopy_fraction`, `window_m = 25.0`.

### tall_canopy_fraction

Share of valid in-stand pixels between 20 m and 100 m.

**Source.** NAIP-derived CHM.
**Calculation.** As above with `lo_m = 20.0`, `hi_m = 100.0`. The upper bound rejects CHM artefacts rather than describing a real ceiling.
**Units.** Fraction, 0 to 1.
**Parameters.** `TALL_CANOPY_BAND_M`.
**Feeds.** No rule.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** `metrics/cover.py::tall_canopy_fraction`, `window_m = 25.0`.

### height_mean, height_median, height_p90, height_max

Central tendency and upper tail of the canopy height surface.

**Source.** NAIP-derived CHM.
**Calculation.** `height_stats` first drops every pixel below 2 m, so the statistics describe the canopy instead of a mixture of canopy and ground. It then takes the mean, median, 90th percentile and maximum of what remains. All five statistics return NaN together below `MIN_CANOPY_PIXELS = 10` canopy pixels.
**Units.** Metres.
**Parameters.** `height_threshold_m = 2.0`, `min_pixels = 10`. Both are module constants in `zonal/pixel.py` with no YAML entry.
**Feeds.** No rule. These describe stand condition and support the worked examples.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** None. The registered windowed metrics have no height-statistic equivalent.

### height_cv

Coefficient of variation of canopy height, a measure of vertical roughness.

**Source.** NAIP-derived CHM.
**Calculation.** Standard deviation of canopy pixels over their mean. NaN when the mean is zero or negative, and NaN below 10 canopy pixels.
**Units.** Dimensionless.
**Parameters.** Same as the other height statistics.
**Feeds.** No rule.
**Support.** `support_n_valid_pixels`.
**Windowed variant.** None.

---

## Crown geometry

Source module: `zonal/crowns.py`. This is the only category whose input is a vector layer.

Crowns come from a segmentation run over the whole scene, not per stand. `segment_scene_crowns` tiles the CHM in blocks with overlap so a crown near a block edge is not cut in two, and it drops anything below 2 m. Each crown carries a measured diameter in the `crown_diam_m` column.

`crowns_in_stand` assigns a crown to a stand when the crown centroid falls inside the polygon. A crown therefore counts exactly once, and a crown overhanging the stand boundary belongs to whichever stand holds its centre.

Every statistic except `crown_count` returns NaN below `MIN_CROWNS = 3` crowns, with the reason recorded. `crown_count` is exempt because zero crowns is a real measurement, not a failure.

### crown_cv

Coefficient of variation of crown diameter. The primary structural diagnostic in the classifier.

**Source.** Crown polygons, `crown_diam_m` column.
**Calculation.** Standard deviation of the in-stand crown diameters over their mean. Low values mean crowns of one size, which is the signature of an even-aged closed canopy. High values mean a mixed size distribution, which is the signature of old growth or a mature stand with a broken canopy.
**Units.** Dimensionless.
**Parameters.** `min_crowns = 3`, from `MIN_CROWNS`. The raster path reads `defaults.min_crowns_per_window` in `config/metrics.yaml`, same value.
**Feeds.** Every stage envelope, and it carries the widest spread of any stage metric, 0.0 to 0.1 at ESI up to 0.6 to 1.5 at OG. Seven trajectory rules including DS1, FC1 and FC2.
**Support.** `support_n_crowns`.
**Windowed variant.** `metrics/crown.py::crown_cv`, `window_m = 50.0`, built from the shared `crown_stats` engine. Crowns are assigned to a tile by centroid there too.

### crown_p90

90th percentile crown diameter. The diagnostic for removal of the largest trees.

**Source.** Crown polygons, `crown_diam_m` column.
**Calculation.** 90th percentile of the in-stand crown diameters. A drop between dates with the count roughly held means the big crowns went, which separates high-grading from a general thinning.
**Units.** Metres.
**Parameters.** `min_crowns = 3`.
**Feeds.** No rule directly. Its change metric `d_crown_p90` carries the V5 code dCW90.
**Support.** `support_n_crowns`.
**Windowed variant.** `metrics/crown.py::crown_p90`, `window_m = 50.0`.

### crown_mean, crown_median, crown_std

Central tendency and spread of crown diameter.

**Source.** Crown polygons, `crown_diam_m` column.
**Calculation.** Mean, median and standard deviation of the in-stand crown diameters. `crown_std` is the unnormalised counterpart of `crown_cv`.
**Units.** Metres.
**Parameters.** `min_crowns = 3`.
**Feeds.** No rule.
**Support.** `support_n_crowns`.
**Windowed variant.** `metrics/crown.py::crown_mean` at `window_m = 50.0`. `crown_median` and `crown_std` reach the raster path only through `crown_stats(stat=...)` and are not registered.

### crown_count

Number of crowns whose centroid falls inside the stand.

**Source.** Crown polygons.
**Calculation.** A count, not a density. Compare it against `area_m2` to reason about stem density, or use `d_crown_count` to read removal between dates.
**Units.** Count.
**Parameters.** None. `min_crowns` does not apply.
**Feeds.** No rule.
**Support.** `support_n_crowns`, which holds the same value.
**Windowed variant.** `metrics/crown.py::crown_count`, `window_m = 50.0`. The windowed version initialises empty tiles to `0.0` rather than NaN, so an empty tile and a zero-crown tile look the same.

---

## Canopy texture

Source module: `zonal/texture.py`.

### glcm_texture

Shannon entropy of the grey-level co-occurrence matrix. High entropy means a visually varied canopy surface, low entropy means a uniform one.

**Source.** NAIP band 4, the near infrared band, read over the stand bounding box. This is the one structural metric that reads reflectance instead of the CHM.
**Calculation.** Three steps. `quantize_masked` stretches the in-stand values linearly to 16 grey levels using the in-stand minimum and maximum. `masked_glcm` then counts pixel pairs at the four angles, keeping a pair only when both members are in-stand. `glcm_entropy` returns `-sum(p * log2 p)` over the normalised matrix.
**Units.** Bits. The theoretical maximum is `2 * log2(levels)`, so 8.0 at 16 levels. Observed forest values sit between 4 and 8.
**Parameters.** `levels = 16`, `distances = (1,)`, `angles = (0, pi/4, pi/2, 3pi/4)`, `min_valid_pixels = 256`. All are module constants in `zonal/texture.py`. `config/metrics.yaml` sets `levels: 16` and `prop: entropy` for the raster path.
**Feeds.** Every stage envelope, and it is the metric that separates LSE, at 0.0 to 4.0 bits, from the rest. DS1, DS3c, FC1, FC3, FC4.
**Support.** `support_texture_n_valid`. NaN below 256 valid pixels, and NaN when the stand has no tonal variation. Both reasons are recorded.
**Caveat.** `config/trajectories.yaml` rule DS1 tests `glcm_texture <= 0.30`, which is a 0 to 1 scale. The metric emits bits. That predicate can never fire. See [Known gaps](#known-gaps).
**Windowed variant.** `metrics/texture.py::glcm_texture`, `window_m = 50.0`, via `skimage.feature.graycomatrix`. The two disagree on purpose. The windowed version sets invalid pixels to grey level 0, which puts a spurious spike at cell (0, 0) proportional to the share of the box outside the stand. For a polygon it reports the shape of the bounding box as much as the texture of the canopy. Use the stand version for stand work.

---

## Spatial pattern

Source module: `zonal/spatial.py`. These three read the arrangement of canopy and gap rather than how much of each there is. `stand_spatial_metrics` derives both masks from the CHM at the 2 m threshold and computes all three in one call.

All three need a rectangular grid, so they run over the stand bounding box with the mask applied. Two corrections keep the polygon shape out of the answer. `interior_edge_mask` erodes the stand mask by one pixel and keeps only edges strictly inside it, so the polygon outline is not counted as a canopy edge. Denominators use in-stand area, not bounding box area.

The normalisations in `linearity_index` and `row_directionality` are not calibrated. Read the magnitudes as relative and compare stands against each other.

### edge_density

Canopy edge length per unit stand area.

**Source.** NAIP-derived CHM, thresholded at 2 m into a canopy mask.
**Calculation.** Count interior edge pixels, multiply by pixel size to get an edge length in metres, divide by the in-stand area in square metres. Dividing by stand area rather than box area keeps a long thin stand comparable to a compact one.
**Units.** 1/m. Values scale with pixel size, so compare only across a common resolution.
**Parameters.** `height_threshold_m = 2.0`.
**Feeds.** No stage envelope and no trajectory rule. Its change metric `d_edge_density` carries the V5 code dEdge.
**Support.** `support_support_fraction`, the in-stand share of the bounding box. NaN when the stand holds no pixels.
**Windowed variant.** `metrics/spatial.py::edge_density`, `window_m = 50.0`. It divides by window area and counts the polygon outline as an edge, so it reads higher on any stand that does not fill its window.

### linearity_index

How much the gap pattern lines up along a single direction.

**Source.** NAIP-derived CHM, thresholded at 2 m into a gap mask.
**Calculation.** Run an edge detector over the interior gap mask, then take a Hough transform over `n_angles` orientations. The value is `clip(1 - mean/peak, 0, 1)` of the accumulator. A strong single peak against a low mean means the openings align, which is the signature of a utility corridor or another maintained strip.
**Units.** Dimensionless, 0 to 1. Uncalibrated.
**Parameters.** `n_angles = 90`. `stand_spatial_metrics` takes it as a keyword default. `config/metrics.yaml` carries the same value for the raster path.
**Feeds.** No stage envelope. LC7, utility corridor.
**Support.** `support_support_fraction`. NaN with the reason `"no gap edges inside the stand"` when the interior gap mask is empty.
**Windowed variant.** `metrics/spatial.py::linearity_index`, `window_m = 50.0`. It returns `0.0` rather than NaN on an empty window, so the windowed output is effectively never NaN.

### row_directionality

How regularly and directionally spaced the canopy is. Targets row-structured plantations.

**Source.** NAIP-derived CHM as a continuous surface, not a mask.
**Calculation.** Apply a Hanning window, take a 2-D FFT, bin the power spectrum into `n_bins` angular bins over `[0, pi)`, then return `clip(1 - mean/peak, 0, 1)`. Invalid pixels are filled with the in-stand mean before the transform.
**Units.** Dimensionless, 0 to 1. Uncalibrated.
**Parameters.** `n_bins = 36`, a keyword default on `stand_spatial_metrics`, with the same value in `config/metrics.yaml` for the raster path. `min_support = MIN_DIRECTIONALITY_SUPPORT = 0.5`.
**Feeds.** No stage envelope. FC2, row-structured plantation, at a threshold of 0.50.
**Support.** `support_support_fraction`. Reported only when the stand fills at least half its bounding box. Below that the transform measures the shape of the mask instead of the canopy, so the value is NaN with the reason recorded.
**Windowed variant.** `metrics/spatial.py::row_directionality`, `window_m = 50.0`. No support gate, because a full window has no mask to confound it.

---

## Change between NAIP dates

Source module: `zonal/deltas.py`.

`add_change_metrics` differences consecutive dates within a `stand_id`, ordered by year, and writes the result under a `d_` prefix. The first date of every stand is NaN. A metric that is absent from the frame still gets its column, filled with NaN, rather than being skipped.

Units are inherited from the source metric. The five change metrics correspond to the V5 inter-NAIP codes.

| Column | Source metric | Units | V5 code |
|---|---|---|---|
| `d_crown_fraction` | `crown_fraction` | fraction | dCF |
| `d_gap_fraction` | `gap_fraction` | fraction | dGF |
| `d_crown_p90` | `crown_p90` | m | dCW90 |
| `d_crown_count` | `crown_count` | count | dCD |
| `d_edge_density` | `edge_density` | 1/m | dEdge |

**Feeds.** No stage envelope and no trajectory rule reads a `d_` column today. Trajectory rules operate on the multi-date metric cube through reducers instead.

**Reading them.** A negative `d_crown_p90` with `d_crown_count` near zero points at removal of the largest trees. Both negative points at a general thinning or a harvest. `d_crown_count` is the pair to watch carefully, because a windowed `crown_count` of zero can mean an empty tile rather than a stand with no crowns.

**Windowed variant.** `metrics/deltas.py` computes the same five differences under a `delta_` prefix, through a registry separate from the metric registry. It enforces strict alignment on shape, transform, CRS and `window_m`, and raises rather than resampling.

---

## Satellite spectral

Source package: `satellite/`. This category differs from the others in three ways. The imagery is 30 m Landsat rather than 0.6 m NAIP, the reduction happens server side in Earth Engine, and the output is annual rather than per NAIP date.

### Data path

**Archive.** Landsat Collection 2 Level 2, Tier 1 only. Five sensors are registered in `satellite/sensors.py`: `L4`, `L5`, `L7`, `L8`, `L9`, covering 1982 to the present. Landsat 7 SLC-off scenes are kept. No Sentinel-2 sensor is registered, though adding one is a `SensorSpec` data change rather than new code.

**Masking.** `prepare_image` applies four filters in order. QA_PIXEL bits from `qa.mask_bits`, which defaults to `fill`, `dilated_cloud`, `cirrus`, `cloud`, `cloud_shadow`, `snow`. An optional cloud-confidence guard, off by default. Per-band QA_RADSAT saturation bits. A reflectance range check against `[0.0, 1.0]`. Surface reflectance then comes from the digital number as `dn * 2.75e-05 - 0.2`.

**Reduction.** `satellite/extract.py` calls `image.reduceRegions` at 30 m with one combined reducer: area-weighted `mean`, unweighted `count`, and `sum` over a constant band. Earth Engine offers no pixel-centre rule, so the mean is area weighted. This differs from the `zonal/` pixel-centre rule on purpose and the manifest records it as `pixel_rule: "ee_area_weighted"`.

**Tables.** Two products. The observation table holds one row per stand per scene, cached as `landsat_observations.parquet`. The annual table holds one row per stand per year, written to `satellite_annual.parquet`. `annual_table` emits a row for every stand and every year in range, including years with no surviving observation, so a gap is a row with a reason rather than a missing row.

**Quality control.** `filter_observations` runs six gates per stand before any metric, each counted separately into `sat_dropped_<cause>`: null index, index outside `[-1, 1]`, `n_pixels < 4`, `pixel_weight_sum < 2.0`, `coverage_fraction < 0.60`, `area_m2 < 3600`. The last gate excludes stands smaller than four Landsat pixels.

### Per-scene indices

All three are normalised differences computed on harmonised band names, so the same formula runs on every sensor. All are dimensionless with a valid range of -1 to 1. `apply_indices` drops the source reflectance bands, so only index bands reach the reduction.

| Index | Formula | Reads |
|---|---|---|
| `ndvi` | `(nir - red) / (nir + red)` | greenness and canopy cover |
| `nbr` | `(nir - swir2) / (nir + swir2)` | burn severity and canopy loss |
| `ndmi` | `(nir - swir1) / (nir + swir1)` | canopy moisture |

`config/satellite.yaml` sets `extraction.indices: [ndvi]`, so `nbr` and `ndmi` are registered but not extracted. Requesting more than one index does not currently work, because `extract.py` keeps only the first. See [Known gaps](#known-gaps).

The observation table stores the reduced index under the bare index name, plus `<index>_n_pixels`, `n_pixels`, `pixel_weight_sum`, `area_m2`, `expected_pixels` and `coverage_fraction`. `expected_pixels` is `area_m2 / 900`, one Landsat pixel being 900 m².

### ndvi_mean

Growing-season mean NDVI for one stand in one year. The level term.

**Source.** The `ndvi` observation series for that stand and year, after QC.
**Calculation.** Unweighted mean of every surviving observation with day of year between 152 and 258, which is 1 June to 15 September. NaN below `min_obs = 3` observations.
**Units.** Index, dimensionless.
**Parameters.** `doy_min = 152`, `doy_max = 258`, `min_obs = 3`, in `config/satellite.yaml` under `annual_metrics.ndvi_mean.params`.
**Feeds.** Every stage envelope. LC6, at a threshold of 0.40.
**Support.** `sat_ndvi_mean_n_obs`, `sat_ndvi_mean_median`, plus the shared `sat_median_coverage_fraction` and `sat_n_sensors`.
**Caveat.** NDVI saturates over closed canopy, so the bands for the five closed-canopy stages overlap almost completely. Adding it raises every closed-canopy stage score by the same amount without changing the ranking.

### ndvi_seasonal_amplitude

Peak-to-trough swing of the annual NDVI cycle. Separates deciduous forest from agriculture, pasture and evergreen cover.

**Source.** The `ndvi` observation series for that stand and year, full calendar year.
**Calculation.** `fit_single_harmonic` fits `y(t) = offset + b1*cos(2*pi*t) + b2*sin(2*pi*t)` by least squares, with `t = doy / days_in_year` and 366 days in a leap year. The value is `2 * hypot(b1, b2)`, so it is peak to trough rather than an amplitude about the mean.
**Units.** Index, peak to trough. Theoretical maximum 2.0.
**Parameters.** `doy_min = 1`, `doy_max = 366`, `min_obs = 6`, `min_doy_span = 150.0`, `max_condition = 30.0`, `max_amplitude = 1.5`, in `config/satellite.yaml`.
**Feeds.** Every stage envelope. LC1, LC3, LC5.
**Support.** `sat_amplitude_n_obs`, `sat_amplitude_doy_span`, `sat_amplitude_condition`, `sat_amplitude_rmse`, `sat_amplitude_r2`, `sat_amplitude_phase_doy`, `sat_amplitude_offset`.
**Guards.** Four guards fire in a fixed order: observation count, day-of-year span, design-matrix condition number, then implausible amplitude. `r2` is recorded but is deliberately not a guard, because a low `r2` is the expected outcome over closed-canopy forest and using it would void the stands the classifier most needs.

### ndvi_trend

Multi-year rate of change in growing-season NDVI. The only satellite metric that describes a rate.

**Source.** The yearly `ndvi_mean` series for that stand.
**Calculation.** Theil-Sen slope over a trailing window of years, labelled at the last year of the window. `scipy.stats.theilslopes` supplies the slope and a confidence interval. The metric calls `get_annual("ndvi_mean")` and reuses that metric's season parameters, so level and trend can never be computed over different seasons.
**Units.** Index per year.
**Parameters.** `window_years = 5`, `min_years = 4`, in `config/satellite.yaml`.
**Feeds.** No stage envelope, on purpose. A rate has no meaning in a single-date envelope, so it lives at the trajectory layer. DS2 at `<= -0.004`, DS3a and EF3 at `<= 0.002`.
**Support.** `sat_trend_n_years`, `sat_trend_window_years`, `sat_trend_slope_lo`, `sat_trend_slope_hi`. Use the interval to tell a real decline from a flat series with noise.

### Metric names are pinned

`ndvi_mean`, `ndvi_seasonal_amplitude` and `ndvi_trend` are referenced by name in `config/stages.yaml` and by seven rules in `config/trajectories.yaml`: DS2, DS3a, EF3, LC1, LC3, LC5 and LC6. Renaming one silently unhooks those rules rather than raising. A unit test in `tests/unit/test_config.py` asserts the YAML names and the registered names match.

---

## Where parameters live

Two sources. The typed loader reads `config/*.yaml`. Structural constants that define what a metric means stay in Python, because changing them changes the metric rather than tuning it.

### YAML, `config/metrics.yaml`

| Key | Value | Read by |
|---|---|---|
| `defaults.window_sizes_m` | `[25, 50, 100, 200]` | raster path only |
| `defaults.chm_gap_threshold_m` | `2.0` | raster gap and cover metrics |
| `defaults.min_crowns_per_window` | `3` | raster crown metrics |
| `metrics.glcm_texture.params.levels` | `16` | raster texture |
| `metrics.glcm_texture.params.prop` | `entropy` | raster texture |
| `metrics.linearity_index.params.n_angles` | `90` | raster path. The stand path uses a matching Python default |
| `metrics.row_directionality.params.n_bins` | `36` | raster path. The stand path uses a matching Python default |
| `metrics.<band>_fraction.params.lo_m` / `hi_m` | see height table | raster cover metrics |
| `metrics.<name>.params.window_m` | `25.0` or `50.0` | see below |

### YAML, `config/satellite.yaml`

| Block | Key parameters |
|---|---|
| `earth_engine` | `project`, `high_volume`, `deadline_ms`, `max_workers` |
| `extraction` | `sensors`, `indices`, `start_year 1985`, `end_year 2025`, `scale_m 30.0`, `tile_scale 2`, `chunk year`, `max_scene_cloud_cover 80.0`, `aoi_buffer_m 1000.0`, `simplify_tolerance_m 1.0` |
| `qa` | `mask_bits`, `reflectance_min/max`, `index_valid_range`, `min_pixels 4`, `min_effective_pixels 2.0`, `min_coverage_fraction 0.60`, `min_area_m2 3600.0` |
| `annual_defaults` | `doy_min 1`, `doy_max 366`, `min_obs 6` |
| `annual_metrics.<name>` | `index`, `units`, `params` merged over `annual_defaults` |

### Python module constants

These have no YAML entry.

| Constant | Value | Module |
|---|---|---|
| `CANOPY_HEIGHT_THRESHOLD_M` | `2.0` | `zonal/pixel.py` |
| `SHRUB_BAND_M` | `(0.5, 2.0)` | `zonal/pixel.py` |
| `SMALL_TREE_BAND_M` | `(2.0, 10.0)` | `zonal/pixel.py` |
| `MID_CANOPY_BAND_M` | `(10.0, 20.0)` | `zonal/pixel.py` |
| `TALL_CANOPY_BAND_M` | `(20.0, 100.0)` | `zonal/pixel.py` |
| `MIN_CANOPY_PIXELS` | `10` | `zonal/pixel.py` |
| `MIN_CROWNS` | `3` | `zonal/crowns.py` |
| `DIAMETER_COLUMN` | `crown_diam_m` | `zonal/crowns.py` |
| `DEFAULT_LEVELS` | `16` | `zonal/texture.py` |
| `DEFAULT_DISTANCES` | `(1,)` | `zonal/texture.py` |
| `DEFAULT_ANGLES` | `(0, pi/4, pi/2, 3pi/4)` | `zonal/texture.py` |
| `MIN_VALID_PIXELS` | `256` | `zonal/texture.py` |
| `MIN_DIRECTIONALITY_SUPPORT` | `0.5` | `zonal/spatial.py` |
| `NAIP_NIR_BAND` | `4` | `zonal/compute.py` |
| `CHANGE_METRICS`, `DELTA_PREFIX` | five names, `d_` | `zonal/deltas.py` |

### A note on window_m

`window_m` belongs to the fixed-window raster path and has no meaning for a stand.

The stand path never reads it. `grep -rn "window_m" src/csdv_core/zonal/` returns one hit, and that hit is prose in a docstring. The polygon is the support, so there is nothing to size.

The per-metric `window_m` in `config/metrics.yaml` is also dead at runtime. `metrics/orchestrator.py` overwrites it with the CLI `--window-m` on every call:

```python
params = dict(spec.defaults)
params.setdefault("window_m", window_m)
params["window_m"] = float(window_m)
```

The YAML values still record the intended scale per metric, 25 m for pixel and cover metrics and 50 m for crown, texture and pattern metrics, and they document that intent even though the orchestrator ignores them.

The closest analogues in the stand path are `bbox_fill_fraction` and `support_support_fraction`, which report how much of a bounding box the polygon fills. They describe how trustworthy a bounding-box computation is. They are not a scale parameter.

---

## Known gaps

Each of these is real in the code today. Check them before acting on a rule or a column.

- **Understory Reinitiation can never be assigned.** The UR envelope in `config/stages.yaml` differs from ESE only in `ndvi_mean`, 0.83 to 0.92 against 0.82 to 0.92. `stages/stand.py` breaks ties toward the earlier stage in `stage_order`, and ESE precedes UR. The file states this is not a transcription error. The source workbook describes UR by direction of change, not by level, so it is not separable on a single date.
- **DS1 tests `glcm_texture` on the wrong scale.** `config/trajectories.yaml` uses `glcm_texture <= 0.30`, a 0 to 1 scale. The metric emits Shannon entropy in bits, roughly 0 to 8, which is the scale the stage envelopes use. That predicate can never be satisfied.
- **The raster orchestrator wires 4 of 15 registered metrics.** `PASS1_METRICS` in `metrics/orchestrator.py` covers `gap_fraction`, `crown_fraction`, `crown_cv` and `glcm_texture`. Everything else raises `NotImplementedError`. Of the seven metrics the stage envelopes need, the raster path can supply three. The full set is reachable only through `zonal/` plus `csdv satellite join`.
- **Two rule metrics have no implementation.** `wetness_persistence` (LC2) and `impervious_fraction` (LC4) appear in `config/trajectories.yaml` and exist nowhere in `src/`.
- **Stratification falls through to `type_00`.** Every continuous threshold in `config/site_types.yaml` is `null`, so no stand reaches a real site type. The `type_00` envelopes in `config/stages.yaml` are described by their own author as unreviewed guesses.
- **`type_01` populates only LSE.** Under that site type every other stage has an empty envelope, so LSE is the only stage that can score and everything else returns a no-envelope reason.
- **Only the first index survives extraction.** `satellite/extract.py` normalises and selects columns for `indices[0]` alone, and derives the coverage band from it. Passing `--indices ndvi,nbr` silently drops `nbr`.
- **Five stand metrics have no windowed counterpart.** `height_mean`, `height_median`, `height_p90`, `height_max`, `height_cv`, plus `crown_median` and `crown_std`, exist only in the stand path.
- **Windowed and stand values disagree for three metrics.** `edge_density`, `linearity_index` and `glcm_texture` are reimplemented in `zonal/` with different denominators and masking. The two are not interchangeable for the same ground area. Use the stand version for stand work.
