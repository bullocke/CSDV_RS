import requests

def download_naip_test_image():
    # The ArcGIS REST exportImage endpoint
    url = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"

    # Your target coordinates in PA
    center_lon = -76.68928
    center_lat = 41.04895
    
    # Create a small bounding box (approx. 1km square depending on latitude)
    offset = 0.005 
    bbox = f"{center_lon - offset},{center_lat - offset},{center_lon + offset},{center_lat + offset}"

    # Define the API parameters
    params = {
        "bbox": bbox,
        "bboxSR": "4326",         # Specifies the bounding box is in WGS84 Lat/Lon
        "size": "1000,1000",      # Dimensions of the output image in pixels
        "imageSR": "3857",        # Output projection (Web Mercator, native to this service)
        "format": "tiff",         # Requesting a TIFF instead of a web-optimized JPEG
        "pixelType": "U8",        # Unsigned 8-bit integers (retains raw pixel values)
        "compression": "None",    # Crucial: Disables lossy server-side compression
        "f": "image"              # Tells the server to return the binary image file directly
    }

    print("Requesting uncompressed NAIP subset...")
    
    # Send the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # The API will sometimes return JSON if there is an error, even with a 200 status.
        # We check the headers to ensure we are receiving a TIFF.
        if 'image/tiff' in response.headers.get('Content-Type', ''):
            output_filename = "naip_2024_pa_test.tif"
            with open(output_filename, "wb") as file:
                file.write(response.content)
            print(f"Success! Uncompressed image saved as {output_filename}")
        else:
            print("Error: The server did not return a TIFF. It returned:")
            print(response.text)
    else:
        print(f"HTTP Error {response.status_code}: {response.reason}")

if __name__ == "__main__":
    download_naip_test_image()