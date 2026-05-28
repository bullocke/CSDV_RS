import rasterio

def check_tiff_compression(filepath):
    with rasterio.open(filepath) as src:
        profile = src.profile
        print(f"File: {filepath}")
        print(f"Driver: {profile.get('driver')}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"Data Type: {profile.get('dtype')}")
        
        # The critical check
        compression = profile.get('compress', 'None')
        print(f"Compression Tag: {compression}")

check_tiff_compression("naip_2024_pa_test.tif")