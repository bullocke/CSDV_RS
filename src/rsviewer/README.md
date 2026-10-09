# rsviewer

`rsviewer` is a small interactive map viewer for remote sensing projects. It shows project rasters and vectors on top of a satellite basemap, lets you switch layers on and off or swap them from dropdowns, and passes map clicks to Python so that a panel can plot the values under the cursor. It runs in the VS Code Interactive Window from a plain `.py` file with `# %%` cells, or in any Jupyter front end, so there is no need to leave VS Code (or WSL) to look at results in QGIS or Google Earth.

The package is a thin layer over three libraries. [leafmap](https://leafmap.org) provides the map (on its ipyleaflet backend), [localtileserver](https://github.com/banesullivan/localtileserver) serves local GeoTIFFs as map tiles, and Plotly draws the plots. What `rsviewer` adds is a registry of named layers that can be replaced in place, a side panel with grouped layer toggles, legends and an info readout, a YAML catalogue format for a project's static layers, and helpers for writing Cloud Optimized GeoTIFFs and turning clicks into pixel coordinates.

`rsviewer` lives in the CSDV repository for now (`src/rsviewer/`), but it never imports CSDV code. A test (`tests/unit/test_rsviewer_isolation.py`) enforces this so the package can move to its own repository unchanged.

## Installation

The CSDV environment already has everything. For another project, the conda-forge packages are:

```bash
micromamba install -c conda-forge leafmap localtileserver ipyleaflet ipywidgets \
    plotly anywidget xyzservices rio-cogeo rasterio geopandas pyproj
```

`geemap` and `earthengine-api` are optional and are only needed for Earth Engine layers. Plotly `FigureWidget` needs `anywidget` from Plotly 6 onwards.

## Running in VS Code

Write the viewer as a `.py` file split into cells with `# %%` lines. "Run Cell" (or Shift+Enter) sends a cell to the Interactive Window, which starts a Jupyter kernel for the selected interpreter. The cell that ends with `viewer.show()` displays the map. Widgets stay live for as long as the kernel runs. Closing the Interactive Window ends the kernel, so the map has to be rebuilt by running the cells again.

## Quickstart

```python
# %%
from rsviewer import RasterLayer, RasterStyle, VectorLayer, VectorStyle, Viewer
from rsviewer.cog import file_to_cog

naip = file_to_cog("naip_2018.tif", ".cache/naip_2018_cog.tif")  # once, ~20 s

v = Viewer(title="My site")
v.add(RasterLayer("NAIP 2018", path=naip, style=RasterStyle(bands=(1, 2, 3))))
v.add(
    RasterLayer(
        "Canopy height",
        path="chm_2018.tif",
        style=RasterStyle(colormap="greens", vmin=0, vmax=35, units="m"),
        visible=False,
    )
)
v.add(VectorLayer("Plots", path="plots.gpkg", style=VectorStyle(color="#ffff00")))
v.on_click(lambda lon, lat: print(lon, lat))
v.zoom_to("NAIP 2018")

# %%
v.show()
```

## Concepts

### Layers and styles

Layers are dataclasses in `rsviewer.layers` that describe what to show. They hold no map state.

| Class | Use |
|---|---|
| `RasterLayer` | A GeoTIFF served by localtileserver. Give `path`, or `source`, a function that returns a path and is called when the layer is first added. A `source` function can build a cached file on demand. |
| `VectorLayer` | Any file geopandas can read, reprojected to WGS84 and drawn as GeoJSON. Clicking a feature shows its `popup` attributes in the Info tab. |
| `TileLayerSpec` | Any XYZ tile URL, for example an Earth Engine map. |

`RasterStyle` sets the bands (one band colour-mapped, three as RGB), the rio-tiler colormap name (lower case, such as `viridis`, `rdbu`, `greens`), the stretch (`vmin`, `vmax` in raw pixel units), a `scale` and `units` used by the legend and the pixel readout, and optionally `categories`. A categorical style maps uint8 pixel values to a `Category(label, color)`. The tile server applies it as an exact lookup table, with no stretching. The tile server cannot make one value transparent, so values that should not be drawn must be the file's nodata value. `VectorStyle` sets outline colour, width and fill, and can colour features by an attribute (`by` plus `categories`). Several values may share one category. The legend then lists the category once.

### Tile serving and COGs

localtileserver starts one HTTP tile server per Python process, on `127.0.0.1` with a free port. VS Code forwards that port to the Interactive Window, which works under WSL2 without any setup. The server reads only the parts of a file that a tile needs. A large raster without internal overviews makes every zoomed-out tile read full-resolution pixels. For a 0.6 m NAIP quarter quad that means about 2 s per tile, against 0.15 to 0.3 s from a Cloud Optimized GeoTIFF. Convert such files once with `rsviewer.cog.file_to_cog(src, dst)`. Small rasters (a few thousand pixels on a side) can be served as they are.

`rsviewer.cog.array_to_cog(arr, dst, transform=..., crs=..., nodata=...)` writes in-memory results, for example arrays kept in `.npz` files, as georeferenced COGs. `rsviewer.cog.is_stale(dst, *sources)` reports whether a cached file is missing or older than its sources. With `RasterLayer(source=...)` this gives layers that are computed on first view and reused afterwards.

### The viewer

`Viewer` keeps an ordered registry of layers by name.

| Method | Effect |
|---|---|
| `add(spec)` | Adds a layer. Rasters go below all vectors. A missing file is logged and skipped, so one absent output does not stop a catalogue from loading. |
| `replace(name, spec)` | Swaps a layer in place. The draw order and the user's visibility and opacity settings are kept. |
| `remove(name)` | Removes a layer. |
| `set_visible(name, on)`, `set_opacity(name, value)` | Changes a layer and keeps the layer panel in step. |
| `zoom_to(name)` | Fits the map to a layer. |
| `split(left, right)`, `unsplit()` | Shows two layers on either side of a draggable divider. |
| `add_panel(title, widget)` | Adds a tab to the side panel. |
| `on_click(fn)` | Calls `fn(lon, lat)` on every map click. |

`viewer.map` is the underlying leafmap map, so any leafmap or ipyleaflet call still works on it.

By default (`inspect=True`) a click also reads every visible raster at that point and lists the values, or the category labels, in the Info tab. The labels serve as a second encoding for colour-coded classes. Warnings and errors from click handlers and layer loading appear under the side panel, because output printed inside widget callbacks is otherwise easy to miss.

### Panels

`rsviewer.panels` has the building blocks used by the side panel. `SelectorPanel` wraps a set of widgets (dropdowns, checkboxes, toggle buttons) and calls one function with all their current values whenever any of them changes. Its `set()` updates several controls with a single call, and `set_options()` replaces a dropdown's options without firing. This is enough to build dependent dropdowns, where the options of one control depend on another. `PlotPanel` is a Plotly `FigureWidget` with stacked subplots, sized to the side panel. It is redrawn in place, which is much faster than re-rendering a figure. `InfoPanel` shows a two-column table.

### Catalogue files

Static layers can be listed in YAML and loaded with `rsviewer.from_catalog(path)`, which returns a `Viewer` with the layers added.

```yaml
title: Site A
zoom: 14
basemap: Esri World Imagery        # a key of rsviewer.BASEMAPS
vars:
  cache: ${MY_CACHE:-../.cache}/viewer   # ${ENV} or ${ENV:-default}
layers:
  - type: raster
    name: NAIP {year}
    path: "{cache}/naip_{year}.tif"
    foreach: {year: [2016, 2018, 2020]}  # one layer per value
    visible: {year: [2018]}              # or true / false
    group: NAIP
    style: {bands: [1, 2, 3]}
  - type: raster
    name: Land cover
    path: landcover.tif
    style:
      categories: {1: [Forest, "#008300"], 2: [Water, "#2a78d6"]}
  - type: vector
    name: Plots
    path: ../data/plots.gpkg
    layer: plots
    style: {by: status, categories: {done: [Done, "#1baf7a"]}}
    popup: [plot_id, status]
  - type: tiles
    name: USGS topo
    url: https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}
```

Strings may refer to variables in `vars` (which can build on earlier ones) and to `foreach` loop variables with `{name}`. Environment variables use `${NAME}`. Relative paths resolve against the folder of the YAML file. Layers that change with user choices are better built in Python and added with `viewer.add` or `viewer.replace`.

### Earth Engine

Earth Engine serves its own tiles, so `rsviewer.ee.ee_layer(image, vis, name)` returns a `TileLayerSpec` for any `ee.Image` without the local tile server. Call `rsviewer.ee.init(project)` first. Authenticate once per machine with `earthengine authenticate`. The tile URLs expire after a few hours, so long sessions may need the layer to be added again.

## Building a project viewer

A project viewer usually has four parts. In the CSDV disturbance-index viewer (`analysis/NAIP_disturbance_detection/2_Disturbance_Index/`) they are:

1. **A catalogue** (`viewer.yaml`) for static layers: imagery per year, canopy height, reference polygons.
2. **A layer module** (`_viewer_layers.py`) that turns project outputs into COGs on demand and returns layer specs, such as the onset, outcome, z and probability maps of a chosen variant.
3. **A panel module** (`_viewer_panels.py`) with the project's selectors and click handlers. One class keeps the map in step with the variant dropdowns through `viewer.replace`. Another registers `viewer.on_click` and redraws a `PlotPanel` with the time series of the clicked cell.
4. **An app script** (`viewer.py`) with `# %%` cells that builds the viewer, adds the panels and ends with `viewer.show()`.

A cache script that pre-builds the slow COGs (`11_viewer_cache.py` in CSDV) keeps the first view fast.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Basemap shows, local layers are blank | The Interactive Window cannot reach the tile server. Check that VS Code lists the forwarded port under "Ports". In a remote setup, localtileserver's `client_host` and `client_prefix` settings or `jupyter-server-proxy` can route tiles through the Jupyter server. |
| Tiles are slow when zoomed out | The raster has no overviews. Convert it with `file_to_cog`. |
| "Connecting to kernel" hangs for a minute or two | VS Code's Jupyter extension checks the interpreter from the VS Code install folder. On WSL that folder is on the Windows drive, and a slow read there delays the check. Wait, or reload the window. This is unrelated to `rsviewer`. |
| A layer is missing from the panel | Its file did not exist when it was added. The message line under the panel names the path. |
| A colormap name is rejected | Use rio-tiler names, which are lower case (`rdbu`, not `RdBu`). |
| Memory grows in long sessions | Each distinct raster keeps a tile client. Restart the kernel. Set `GDAL_CACHEMAX` (e.g. 512 MB) on machines with little RAM. |

## Moving to its own repository

1. Create a new repository and copy `src/rsviewer/` (including this README) and `tests/unit/test_rsviewer_*.py` into it.
2. Add a `pyproject.toml` that names `rsviewer` as the package, with the dependencies listed under Installation, and move the shared test fixture `synthetic_chm` (from `tests/conftest.py`) with the tests.
3. Install it in each project environment with `pip install -e <path>`.
4. Delete `src/rsviewer/` and the `rsviewer` tests from CSDV, and drop the `viewer` extra from CSDV's `pyproject.toml` in favour of a dependency on the new package.

Project code does not change, because the import name stays `rsviewer`.
