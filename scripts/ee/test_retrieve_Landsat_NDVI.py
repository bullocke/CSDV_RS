import ee
import xarray as xr

# Initialize Earth Engine 
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# Define ROI
polygon_roi = ee.Geometry.Polygon([
    [[-122.15, 37.40], [-122.15, 37.50], [-122.00, 37.50], [-122.00, 37.40]]
])

start_date = '1990-01-01'
end_date = '2026-01-01'

# 1. Unified prep function for Landsat 4, 5, and 7
def prep_l457(image):
    # Cloud and Shadow Mask
    qa = image.select('QA_PIXEL')
    cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
    mask = cloud_shadow_mask.And(cloud_mask)
    
    # Apply Collection 2 scale and offset
    optical_scaled = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    
    # Overwrite unscaled bands, mask, rename to generic names, and keep time
    return image.addBands(optical_scaled, None, True) \
                .updateMask(mask) \
                .select(['SR_B3', 'SR_B4'], ['Red', 'NIR']) \
                .copyProperties(image, ['system:time_start'])

# 2. Unified prep function for Landsat 8 and 9
def prep_l89(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
    mask = cloud_shadow_mask.And(cloud_mask)
    
    optical_scaled = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    
    return image.addBands(optical_scaled, None, True) \
                .updateMask(mask) \
                .select(['SR_B4', 'SR_B5'], ['Red', 'NIR']) \
                .copyProperties(image, ['system:time_start'])

# 3. Standardized NDVI Function (now extremely lightweight)
def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    return image.addBands(ndvi).select('NDVI')

# 4. Filter dates and bounds FIRST, apply prep, then merge
# (Note: Included L8 here just in case you want continuous coverage)
l4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(polygon_roi).filterDate(start_date, end_date).map(prep_l457)
l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(polygon_roi).filterDate(start_date, end_date).map(prep_l457)
l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(polygon_roi).filterDate(start_date, end_date).map(prep_l457)
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(polygon_roi).filterDate(start_date, end_date).map(prep_l89)
l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(polygon_roi).filterDate(start_date, end_date).map(prep_l89)

merged_collection = (l4.merge(l5).merge(l7).merge(l8).merge(l9)
                     .map(calculate_ndvi)
                     .sort('system:time_start'))

# 5. Compile the data into a local Xarray Dataset
ds = xr.open_dataset(
    merged_collection,
    engine='ee',
    projection=ee.Projection('EPSG:4326'),
    scale=30,
    geometry=polygon_roi
)

print(ds)