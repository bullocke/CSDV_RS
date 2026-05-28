import requests

def test_2024_naip_availability():
    # ---------------------------------------------------------
    # STEP 1: Query the service for 2024 data
    # ---------------------------------------------------------
    query_url = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/query"
    
    # Filter for 2024 data in Pennsylvania. 
    # Change to "Year_ts = 2024" to search the entire country.
    where_clause = "Year_ts > 2021" 
    
    query_params = {
        "where": "Year_ts > 2021",
        "returnGeometry": "true",
        "outSR": "4326",
        "outFields": "Name,ST,Year_ts,QQDATE",
        "orderByFields": "Year_ts DESC", # Forces the database to bump the newest data to the top
        "resultRecordCount": "1",        # Returns the very first 2024 image it finds
        "f": "json"
    }
    
    print(f"Searching database for: {where_clause}...")
    query_response = requests.get(query_url, params=query_params)
    
    if query_response.status_code != 200:
        print(f"HTTP Error querying service: {query_response.status_code}")
        return
        
    data = query_response.json()
    features = data.get("features", [])
    # print(data)
    if not features:
        print("Result: No imagery found.")
        print("The 2024 uncompressed data for this query is not yet published to the PRIME service.")
        return
        
    feature = features[0]
    attributes = feature.get("attributes", {})
    state = attributes.get('ST', 'Unknown')
    name = attributes.get('Name', 'Unknown')
    
    print(f"Success: Found 2024 image '{name}' in {state}.")
    print(feature)
    # # ---------------------------------------------------------
    # # STEP 2: Extract geometry and download uncompressed subset
    # # ---------------------------------------------------------
    # # ArcGIS polygons are returned as a list of coordinate rings: [[[lon, lat], [lon, lat]...]]
    # rings = feature.get("geometry", {}).get("rings", [])
    
    # if not rings:
    #     print("Error: No geometry returned with the feature.")
    #     return
        
    # # Grab the first vertex of the polygon to use as our center point
    # lon = rings[0][0][0]
    # lat = rings[0][0][1]
    
    # export_url = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"
    
    # # Create a small bounding box around the coordinate
    # offset = 0.005
    # bbox = f"{lon - offset},{lat - offset},{lon + offset},{lat + offset}"
    
    # export_params = {
    #     "bbox": bbox,
    #     "bboxSR": "4326",
    #     "size": "1000,1000",
    #     "imageSR": "3857",
    #     "format": "tiff",
    #     "pixelType": "U8",
    #     "compression": "None", # Requests raw digital numbers
    #     "f": "image"
    # }
    
    # print(f"Downloading uncompressed 1000x1000 subset at {lon:.5f}, {lat:.5f}...")
    # export_response = requests.get(export_url, params=export_params)
    
    # if export_response.status_code == 200 and 'image/tiff' in export_response.headers.get('Content-Type', ''):
    #     output_filename = f"naip_2024_{state}_uncompressed_test.tif"
    #     with open(output_filename, "wb") as f:
    #         f.write(export_response.content)
    #     print(f"Download Complete! Saved as {output_filename}")
    # else:
    #     print("Error: Failed to download the TIFF subset.")

if __name__ == "__main__":
    test_2024_naip_availability()