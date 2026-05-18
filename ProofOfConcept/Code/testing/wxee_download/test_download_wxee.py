
import ee
import wxee

ee.Initialize(
    project='dyce-biomass',
    opt_url='https://earthengine-highvolume.googleapis.com')

# Subset in SCBI
region = ee.Geometry.Polygon(
    [[[-78.15303514106559, 38.89905430300793],
      [-78.15303514106559, 38.88783134643515],
      [-78.13483903510856, 38.88783134643515],
      [-78.13483903510856, 38.89905430300793]]]);

# Output directory to save the file
outdir = '/home/bullocke/vscode_projects/csdv/ProofOfConcept/Code/testing/wxee_download'

# NEON CHM: ee.ImageCollection('projects/neon-prod-earthengine/assets/CHM/001')
scale = 1

img = ee.Image('projects/neon-prod-earthengine/assets/CHM/001/2023_SCBI_6')

# NAD83/ Conus Albers Equal Area
crs="EPSG:5070"

# The file name to save
description='NEON_CHM_SCBI_Subset_2023'

# Download the image as a GeoTIFF file using wxee
file = img.wx.to_tif(out_dir=outdir, description=description, region=region, scale=scale, crs=crs)
file