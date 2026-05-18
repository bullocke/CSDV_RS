# Random notes related to the code needed, using it for this project, and specific information about authentication, data access, and processing.

## Google Earth Engine Python API (`earthengine-api`)

The `earthengine-api` is a Python client library for interacting with Google Earth Engine (GEE). It allows you to access and manipulate geospatial data stored in GEE, run analyses, and export results. It is already installed in the environment and authenticated. To use it, you need to import the library and initialize it with a specific project, which is `dyce-biomass` unless otherwise specified:

```python
import ee
ee.Initialize(project='dyce-biomass')
```

Be cautious with the mixing server-side and client-side operations, such as using `.getInfo()` to print an Earth Engine object from the server to the client. Using this for large files will block execution, and often cause errors or hang the kernel. 

## wxee

_documentation:_ `https://wxee.readthedocs.io/en/latest/`

From the `wxee` documentation: "wxee was built to make processing gridded, mesoscale time series data quick and easy by integrating the data catalog and processing power of Google Earth Engine with the flexibility of xarray, with no complicated setup required. To accomplish this, wxee implements convenient methods for data processing, aggregation, downloading, and ingestion."

`wxee` is installed in the default micromamba environment of this project.

Here is an example from the documentation on downloading an image:

```python
import ee, wxee
ee.Initialize(
    project='dyce-biomass',
    opt_url='https://earthengine-highvolume.googleapis.com')

img = ee.Image("MODIS/006/MOD09GA/2021_06_01").select(["sur_refl_b01", "sur_refl_b04" ,"sur_refl_b03"])
# The file name to save
description = "modis_img"
# The coordinate reference system to use (NAD83 Albers CONUS)
crs = "EPSG:5070"
# Spatial resolution in CRS units (meters)
scale = 500
# The region to download the image within.
region = ee.Geometry.Polygon(
    [[[-120.0576549852371, 46.185091976777684],
      [-120.0576549852371, 45.577074504710005],
      [-118.91782344226834, 45.577074504710005],
      [-118.91782344226834, 46.185091976777684]]]
)
file = img.wx.to_tif(out_dir='data', description=description, region=region, scale=scale, crs=crs)
```

This is an example on downloading a full collection as GeoTIFFs:

```python
import ee, wxee
wxee.Initialize(project='dyce-biomass')
collection = (ee.ImageCollection("MODIS/006/MOD09GA")
    .filterDate("2021-06-01", "2021-06-05")
    .select(["sur_refl_b01", "sur_refl_b04" ,"sur_refl_b03"])
)
files = collection.wx.to_tif(
    out_dir="data",
    prefix="wx_",
    region=region,
    scale=scale,
    crs=crs
)
```