"""csdv_core.national — feasibility tooling for a CONUS-wide crown and AGB map.

The national NAIP-CHM (Morford et al., 2026) covers CONUS in about 213,000
quarter-quad tiles. This package builds the tile inventory, takes a cheap
canopy census of every tile from its internal overviews, and fits the cost
model that turns benchmark timings into national compute, transfer and storage
estimates. It does not run the national segmentation itself.
"""
