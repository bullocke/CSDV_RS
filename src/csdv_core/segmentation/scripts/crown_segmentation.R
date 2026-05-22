# 02_crown_segmentation.R
#
# Individual tree crown segmentation from the NEON ALS canopy height model
# at SCBI using lidR. Produces:
#   (1) A GeoPackage of crown polygons with per-crown diameter estimates
#   (2) A raster of crown width CV computed per 50m analysis window
#
# Both outputs land in Results/summary_document/intermediate/.
#
# Algorithm
# ---------
#   1. Load the NEON CHM (1m, EPSG:5070).
#   2. Smooth with a 3x3 mean kernel to reduce pitting artifacts.
#   3. Detect local maxima (tree tops) using the lmf() variable-window function.
#      Window size scales with canopy height following a simple linear function
#      fitted to eastern hardwood literature (ws = 2 + 0.5 * height).
#   4. Segment crowns with the Dalponte–Coomes algorithm (dalponte2016).
#   5. Vectorize crown raster to polygons; compute area and estimated diameter.
#   6. Build a 50m grid; for each cell compute CV of crown diameters from all
#      crowns whose centroid falls in that cell.
#   7. Write outputs.
#
# Usage
# -----
#   Rscript 02_crown_segmentation.R
#   (Run from any directory; paths are resolved relative to this script's location.)

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(sf)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve script location from commandArgs (works with Rscript)
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 1 && nchar(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
} else {
  script_dir <- getwd()
}

project_root <- normalizePath(file.path(script_dir, "..", "..", ".."))

# Allow optional positional args: <chm_path> <out_gpkg> <out_tif>
# This lets the zoom pipeline call this script with different paths.
trailing_args <- commandArgs(trailingOnly = TRUE)

if (length(trailing_args) >= 3) {
  neon_chm_path <- trailing_args[1]
  out_crowns    <- trailing_args[2]
  out_cv_raster <- trailing_args[3]
  out_dir       <- dirname(out_crowns)
} else {
  neon_chm_path <- file.path(
    project_root, "ProofOfConcept", "Data", "NEON", "CHM",
    "NEON_CHM_SCBI_Subset_2023.tif"
  )
  out_dir <- file.path(
    project_root, "ProofOfConcept", "Results", "summary_document", "intermediate"
  )
  out_crowns    <- file.path(out_dir, "crown_polygons_SCBI.gpkg")
  out_cv_raster <- file.path(out_dir, "crown_cv_50m_SCBI.tif")
}

# Optional 4th positional arg: scale factor applied to raw CHM values.
# Default 1.0 (NEON CHM is already in meters).
# Pass 0.01 for NAIP-CHM stored as UInt16 (height_m * 100).
if (length(trailing_args) >= 4) {
  scale_factor <- as.numeric(trailing_args[4])
} else {
  scale_factor <- 1.0
}

cat("Project root :", project_root, "\n")
cat("CHM path     :", neon_chm_path, "\n")
cat("Scale factor :", scale_factor, "\n")
cat("Output dir   :", out_dir, "\n")

if (!file.exists(neon_chm_path)) {
  stop("NEON CHM not found: ", neon_chm_path,
       "\nRun 01_download_data.py and the setup step first.")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. Load and inspect CHM
# ---------------------------------------------------------------------------
cat("\n[1/6] Loading CHM ...\n")
chm <- terra::rast(neon_chm_path)
if (scale_factor != 1.0) {
  cat("  Applying scale factor:", scale_factor, "\n")
  chm <- chm * scale_factor
}
cat("  CRS  :", terra::crs(chm, describe = TRUE)$authority,
    terra::crs(chm, describe = TRUE)$code, "\n")
cat("  Res  :", paste(terra::res(chm), collapse = " x "), "m\n")
cat("  Dims :", paste(dim(chm)[1:2], collapse = " rows x "), "cols\n")
cat("  Range:", round(terra::minmax(chm)[1], 1), "–",
    round(terra::minmax(chm)[2], 1), "m\n")

# ---------------------------------------------------------------------------
# 2. Smooth CHM (3x3 mean) to reduce pitting
# ---------------------------------------------------------------------------
cat("\n[2/6] Smoothing CHM (3x3 mean kernel) ...\n")
chm_smooth <- terra::focal(chm, w = matrix(1, 3, 3), fun = "mean", na.rm = TRUE)

# Mask pixels below 2m (non-forest / ground returns treated as gaps)
chm_smooth[chm_smooth < 2] <- NA

# ---------------------------------------------------------------------------
# 3. Detect tree tops (local maxima)
# ---------------------------------------------------------------------------
cat("\n[3/6] Detecting tree tops with variable-window lmf ...\n")

# Window size function: ws = 2 + 0.5 * h, bounded [3, 12] m
# Tuned for eastern hardwood stands (moderate crown size variability)
ws_fun <- function(h) {
  pmax(3, pmin(12, 2 + 0.5 * h))
}

ttops <- lidR::locate_trees(chm_smooth, lidR::lmf(ws = ws_fun))
cat("  Tree tops detected:", nrow(ttops), "\n")

# ---------------------------------------------------------------------------
# 4. Crown segmentation (Dalponte–Coomes watershed)
# ---------------------------------------------------------------------------
cat("\n[4/6] Segmenting crowns (Dalponte 2016 algorithm) ...\n")
# dalponte2016() returns a closure; invoke it directly on the CHM raster
algo <- lidR::dalponte2016(chm_smooth, ttops)
crown_raster <- algo()
cat("  Segmentation complete.\n")

# ---------------------------------------------------------------------------
# 5. Vectorize to polygons; compute per-crown diameter
# ---------------------------------------------------------------------------
cat("\n[5/6] Vectorizing crown segments to polygons ...\n")

# Convert to terra SpatVector polygons, dissolve by segment ID
crown_polys_terra <- terra::as.polygons(crown_raster, dissolve = TRUE)

# Convert to sf for easier attribute handling
crown_sf <- sf::st_as_sf(crown_polys_terra)
names(crown_sf)[1] <- "segment_id"

# Remove NA segment (background / unsegmented pixels)
crown_sf <- crown_sf[!is.na(crown_sf$segment_id), ]

# Compute area and estimated crown diameter (assuming circular crown)
crown_sf$area_m2 <- as.numeric(sf::st_area(crown_sf))
crown_sf$crown_diam_m <- 2 * sqrt(crown_sf$area_m2 / pi)

# Basic QC: remove very small segments (< 1 m² = likely noise)
crown_sf <- crown_sf[crown_sf$area_m2 >= 1, ]

cat("  Crown polygons retained:", nrow(crown_sf), "\n")
cat("  Crown diam range (m):", round(range(crown_sf$crown_diam_m), 1), "\n")

sf::st_write(crown_sf, out_crowns, layer = "crowns", delete_dsn = TRUE, quiet = TRUE)
cat("  Saved:", out_crowns, "\n")

# ---------------------------------------------------------------------------
# 6. Per-50m-window crown width CV raster
# ---------------------------------------------------------------------------
cat("\n[6/6] Computing crown width CV on 50m grid ...\n")

# Build 50m grid aligned to CHM extent
ext <- terra::ext(chm)
grid_50m <- terra::rast(
  xmin = ext$xmin, xmax = ext$xmax,
  ymin = ext$ymin, ymax = ext$ymax,
  resolution = 50,
  crs = terra::crs(chm)
)

# Crown centroids
centroids <- sf::st_centroid(crown_sf)
centroid_coords <- sf::st_coordinates(centroids)

# Rasterize CV: for each 50m cell, compute CV of crown diameters
cv_mat <- matrix(NA_real_, nrow = nrow(grid_50m), ncol = ncol(grid_50m))

# Map each centroid to a grid cell
cell_ids <- terra::cellFromXY(grid_50m, centroid_coords)
diams <- crown_sf$crown_diam_m

# Aggregate by cell
valid_mask <- !is.na(cell_ids)
cell_ids_v <- cell_ids[valid_mask]
diams_v <- diams[valid_mask]

cell_cv <- tapply(diams_v, cell_ids_v, function(x) {
  if (length(x) < 3) return(NA_real_)   # need >= 3 crowns per cell for CV
  sd(x) / mean(x)
})

cv_values <- rep(NA_real_, terra::ncell(grid_50m))
cv_values[as.integer(names(cell_cv))] <- as.numeric(cell_cv)
terra::values(grid_50m) <- cv_values

terra::writeRaster(grid_50m, out_cv_raster, overwrite = TRUE,
                   datatype = "FLT4S", NAflag = -9999)
cat("  Saved:", out_cv_raster, "\n")

non_na_cells <- sum(!is.na(cv_values))
cat("  Cells with data:", non_na_cells, "of", terra::ncell(grid_50m), "\n")
cat("  CV range (non-NA):", round(range(cv_values, na.rm = TRUE), 3), "\n")

cat("\n=== Crown segmentation complete ===\n")
