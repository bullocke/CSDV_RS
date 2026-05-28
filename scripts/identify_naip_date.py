import requests
import datetime
import json

def identify_naip_date():
    url = "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/identify"

    lon = -76.68928
    lat = 41.04895
    
    # Define the geometry as a JSON object containing the spatial reference
    geometry_obj = {
        "x": lon,
        "y": lat,
        "spatialReference": {"wkid": 4326}
    }
    
    # Define the API parameters
    params = {
        "geometry": json.dumps(geometry_obj),
        "geometryType": "esriGeometryPoint",
        "returnGeometry": "false",
        "f": "json"
    }

    print(f"Querying the NAIP Image Service at {lon}, {lat}...\n")
    
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        
        if "catalogItems" in data and "features" in data["catalogItems"] and len(data["catalogItems"]["features"]) > 0:
            attributes = data["catalogItems"]["features"][0]["attributes"]
            
            name = attributes.get("Name", "Unknown")
            year = attributes.get("Year_ts", "Unknown")
            qqdate_ms = attributes.get("QQDATE", None)
            
            if qqdate_ms:
                flight_date = datetime.datetime.fromtimestamp(qqdate_ms / 1000.0).strftime('%Y-%m-%d')
            else:
                flight_date = "Unknown"

            print("--- Image Source Attributes ---")
            print(f"Source File Name: {name}")
            print(f"Flight Year:      {year}")
            print(f"Exact Date:       {flight_date}")
            
        else:
            print("No imagery found at this location.")
    else:
        print(f"HTTP Error {response.status_code}: {response.reason}")

if __name__ == "__main__":
    identify_naip_date()