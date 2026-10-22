"""Smoke tests for csdv_core.disturbance.figures on small synthetic runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from csdv_core.disturbance import figures, windows
from csdv_core.disturbance.decide import DateFit
from csdv_core.disturbance.inputs import CellGrid, SiteInputs, YearInput
from csdv_core.disturbance.report import IntervalSummary

#: Synthetic NAIP years.
YEARS: tuple[int, ...] = (2012, 2014, 2016, 2018)
#: Cell grid shape of the synthetic scene.
CELL_SHAPE: tuple[int, int] = (40, 60)
#: Pixels per cell side (19.2 m cells of 0.6 m pixels).
BLOCK_PX = 32
#: Synthetic CRS.
TEST_CRS = CRS.from_epsg(26916)
#: Top-left corner of the synthetic grid.
ORIGIN_XY: tuple[float, float] = (500_000.0, 3_700_000.0)


def make_grid() -> CellGrid:
    """Return a 40 x 60 cell grid of 19.2 m cells."""
    pixel_transform = Affine(0.6, 0.0, ORIGIN_XY[0], 0.0, -0.6, ORIGIN_XY[1])
    return CellGrid(
        crs=TEST_CRS,
        pixel_transform=pixel_transform,
        pixel_m=0.6,
        block_px=BLOCK_PX,
        cell_rows=CELL_SHAPE[0],
        cell_cols=CELL_SHAPE[1],
    )


def make_site_inputs(grid: CellGrid) -> SiteInputs:
    """Return site inputs with one fake CHM path per year."""
    year_inputs = tuple(
        YearInput(
            year=year,
            chm_paths=(Path(f"/nonexistent/chm_{year}.tif"),),
            acquisition_dates=(None,),
            chm_dtype="float32",
            chm_scale=1.0,
            chm_nodata=None,
            crs=TEST_CRS,
            pixel_m=0.6,
        )
        for year in YEARS
    )
    return SiteInputs(
        site="Synthetic",
        chm_root=Path("/nonexistent"),
        naip_root=Path("/nonexistent"),
        years=year_inputs,
        grid=grid,
        reference_year=YEARS[0],
    )


def make_figure_inputs(
    *, with_probability: bool = True, with_refined: bool = True
) -> figures.FigureInputs:
    """Build a synthetic run with one clearcut per interval."""
    grid = make_grid()
    rng = np.random.default_rng(seed=3)
    n_dates = len(YEARS)
    rows, cols = CELL_SHAPE
    forest_first = np.ones(CELL_SHAPE, dtype=bool)
    forest_first[:, :8] = False
    forest = np.repeat(forest_first[np.newaxis], n_dates, axis=0)
    onset_year = np.where(forest_first, 0, -1).astype(np.int16)
    z = rng.normal(0.0, 1.0, size=(n_dates, rows, cols)).astype(np.float32)
    probability = np.zeros((n_dates, rows, cols), dtype=np.float32)
    patch_slices = [
        (slice(5, 12), slice(15, 24)),
        (slice(20, 30), slice(30, 38)),
        (slice(28, 35), slice(45, 56)),
    ]
    for date_index, (row_slice, col_slice) in enumerate(patch_slices, start=1):
        forest[date_index:, row_slice, col_slice] = False
        onset_year[row_slice, col_slice] = YEARS[date_index]
        z[date_index, row_slice, col_slice] -= 6.0
        probability[date_index, row_slice, col_slice] = 0.9
    z[0] = np.nan
    monitored = np.zeros_like(forest)
    monitored[1:] = forest[:-1]
    z[~monitored] = np.nan
    fits = [DateFit(YEARS[0], 0, np.nan, np.nan, np.nan, "skipped", True)]
    fits += [
        DateFit(year, 2000, 0.05, 1.0, 0.95, "central", False) for year in YEARS[1:]
    ]
    intervals = []
    object_records = []
    for date_index in range(1, n_dates):
        row_slice, col_slice = patch_slices[date_index - 1]
        n_cells = (row_slice.stop - row_slice.start) * (
            col_slice.stop - col_slice.start
        )
        area_ha = n_cells * grid.cell_ha
        label = f"{YEARS[date_index - 1]}-{YEARS[date_index]}"
        intervals.append(
            IntervalSummary(
                year_before=YEARS[date_index - 1],
                year=YEARS[date_index],
                label=label,
                monitored_ha=500.0,
                detected_ha=area_ha,
                detected_ha_nommu=area_ha * 1.3,
                n_objects=1,
                share_of_monitored=area_ha / 500.0,
                fit=fits[date_index],
                skipped=False,
                n_refined=1,
                refined_ha=area_ha * 0.8,
            )
        )
        left = ORIGIN_XY[0] + col_slice.start * grid.cell_m
        right = ORIGIN_XY[0] + col_slice.stop * grid.cell_m
        top = ORIGIN_XY[1] - row_slice.start * grid.cell_m
        bottom = ORIGIN_XY[1] - row_slice.stop * grid.cell_m
        object_records.append(
            {
                "object_id": date_index,
                "onset_year": YEARS[date_index],
                "year_before": YEARS[date_index - 1],
                "interval": label,
                "area_ha": area_ha,
                "geometry": box(left, bottom, right, top),
            }
        )
    objects = gpd.GeoDataFrame(object_records, crs=TEST_CRS)
    refined = None
    if with_refined:
        refined_geometry = objects.geometry.buffer(-8.0)
        refined = gpd.GeoDataFrame(
            {
                "object_id": objects["object_id"],
                "envelope_ha": refined_geometry.area / 10_000.0,
            },
            geometry=refined_geometry,
            crs=TEST_CRS,
        )
    return figures.FigureInputs(
        grid=grid,
        years=YEARS,
        forest=forest,
        monitored=monitored,
        z=z,
        probability=probability if with_probability else None,
        onset_year=onset_year,
        fits=fits,
        intervals=intervals,
        objects=objects,
        refined=refined,
        p_boundary=0.5,
        bin_w=0.05,
        site_inputs=make_site_inputs(grid),
    )


def fake_read_bounds(
    paths: Sequence[Path],
    grid: CellGrid,
    bounds: tuple[float, float, float, float],
    *,
    band: int = 1,
    scale: float = 1.0,
    nodata: float | None = None,
) -> tuple[np.ndarray, Affine]:
    """Return a synthetic CHM chip covering ``bounds`` at the grid pixel size."""
    left, bottom, right, top = bounds
    n_cols = int(round((right - left) / grid.pixel_m))
    n_rows = int(round((top - bottom) / grid.pixel_m))
    rng = np.random.default_rng(seed=len(str(paths[0])))
    chm = rng.uniform(10.0, 28.0, size=(n_rows, n_cols)).astype(np.float32)
    if "2016" in str(paths[0]):
        chm[n_rows // 3 : 2 * n_rows // 3, n_cols // 3 : 2 * n_cols // 3] = 0.5
    transform = Affine(grid.pixel_m, 0.0, left, 0.0, -grid.pixel_m, top)
    return chm, transform


@pytest.fixture()
def figure_inputs(monkeypatch: pytest.MonkeyPatch) -> figures.FigureInputs:
    """Synthetic figure inputs with CHM reads patched."""
    monkeypatch.setattr(windows, "read_bounds", fake_read_bounds)
    return make_figure_inputs()


def assert_png(path: Path) -> None:
    """Check that ``path`` is a non-empty PNG."""
    assert path.exists()
    header = path.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 1000


def test_each_plot_writes_png(
    figure_inputs: figures.FigureInputs, tmp_path: Path
) -> None:
    grid = figure_inputs.grid
    years = figure_inputs.years
    written = [
        figures.plot_forest_initial(
            figure_inputs.forest[0], grid, years[0], tmp_path / "a.png", dpi=60
        ),
        figures.plot_forest_by_year(
            figure_inputs.forest, grid, years, tmp_path / "b.png", dpi=60
        ),
        figures.plot_disturbance_onset(
            figure_inputs.onset_year, years, grid, tmp_path / "c.png", dpi=60
        ),
        figures.plot_onset_by_interval(
            figure_inputs.onset_year, years, grid, tmp_path / "d.png", dpi=60
        ),
        figures.plot_onset_by_interval(
            figure_inputs.onset_year,
            years,
            grid,
            tmp_path / "d2.png",
            detected=np.stack([figure_inputs.onset_year == year for year in years]),
            dpi=60,
        ),
        figures.plot_z_maps(figure_inputs.z, years, grid, tmp_path / "e.png", dpi=60),
        figures.plot_probability_maps(
            figure_inputs.probability,
            years,
            grid,
            tmp_path / "f.png",
            p_boundary=0.5,
            dpi=60,
        ),
        figures.plot_null_fits(
            figure_inputs.z,
            figure_inputs.monitored,
            figure_inputs.fits,
            tmp_path / "g.png",
            bin_w=0.05,
            p_boundary=0.5,
            dpi=60,
        ),
        figures.plot_area_by_interval(
            figure_inputs.intervals, tmp_path / "h.png", dpi=60
        ),
        figures.plot_refined_examples(
            figure_inputs.objects,
            figure_inputs.refined,
            figure_inputs.site_inputs,
            tmp_path / "i.png",
            n_examples=2,
            dpi=60,
        ),
    ]
    for path in written:
        assert_png(path)


def test_onset_with_backdrop(
    figure_inputs: figures.FigureInputs, tmp_path: Path
) -> None:
    grid = figure_inputs.grid
    rgb = np.full((50, 75, 3), 120, dtype=np.uint8)
    backdrop = figures.NaipBackdrop(rgb=rgb, extent=grid.extent())
    out_path = figures.plot_disturbance_onset(
        figure_inputs.onset_year,
        figure_inputs.years,
        grid,
        tmp_path / "onset.png",
        backdrop=backdrop,
        dpi=60,
    )
    assert_png(out_path)


def test_make_figures_writes_all(
    figure_inputs: figures.FigureInputs, tmp_path: Path
) -> None:
    written = figures.make_figures(figure_inputs, tmp_path, dpi=60)
    assert set(written) == set(figures.FIGURE_FILES)
    for path in written.values():
        assert_png(path)


def test_make_figures_skips_probability_and_refined(tmp_path: Path) -> None:
    figure_inputs = make_figure_inputs(with_probability=False, with_refined=False)
    written = figures.make_figures(figure_inputs, tmp_path, dpi=60)
    assert "probability_maps" not in written
    assert "refined_examples" not in written
    assert "disturbance_onset" in written
    assert not (tmp_path / figures.FIGURE_FILES["probability_maps"]).exists()


def test_make_figures_skips_refined_without_site_inputs(
    figure_inputs: figures.FigureInputs, tmp_path: Path
) -> None:
    figure_inputs.site_inputs = None
    written = figures.make_figures(figure_inputs, tmp_path, dpi=60)
    assert "refined_examples" not in written
    assert "probability_maps" in written


def test_skipped_fit_draws_histogram_only(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    skipped_fit = DateFit(2014, 10, np.nan, np.nan, np.nan, "skipped", True)
    z_values = np.linspace(-2.0, 2.0, 10)
    figures.draw_null_fit(ax, z_values, skipped_fit, bin_w=0.5)
    assert "skipped" in ax.get_title()
    plt.close(figure)
