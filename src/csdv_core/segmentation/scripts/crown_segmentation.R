# crown_segmentation.R
#
# Individual tree crown segmentation from a canopy height model using lidR.
# This is the independent reference implementation. Production segmentation
# runs in Python (csdv_core.segmentation.chm_watershed); this script exists so
# the Python engine can be checked against a published algorithm.
#
# Every parameter is named and every length is in METRES. The pixel-valued
# arguments lidR takes are converted here from terra::res(), because a
# pixel-valued parameter changes physical meaning with resolution. lidR's
# dalponte2016 default of max_cr = 10 pixels, for instance, is a 10 m crown
# radius on a 1 m CHM and a 6 m radius on a 0.6 m CHM. Comparing two
# implementations across that difference measures the units, not the
# algorithms.
#
# Algorithm
# ---------
#   1. Load the CHM, optionally rescaling raw values to metres.
#   2. Mean-smooth with a kernel whose radius is given in metres.
#   3. Mask below the canopy floor.
#   4. Locate tree tops with lmf(), variable window, ws given as a DIAMETER.
#      With --seeds, read tree tops from a CSV (columns x, y, z) instead, so
#      dalponte2016 can grow from exactly the seeds another engine found.
#   5. Segment with dalponte2016(). Note that C_dalponte2016 applies max_cr as
#      a square half-width, |dx| < max_cr and |dy| < max_cr, so the reach is
#      max_cr - 1 pixels along an axis and more along a diagonal.
#   6. Vectorize, compute area and area-equivalent diameter, drop noise.
#   7. Optionally write a crown-diameter CV raster on a grid.
#
# Each step prints a "TIMING <step> <seconds>" line, and the script ends with
# "PEAK_RSS_KB <kB>" read from /proc/self/status. lidr_bridge parses both, so
# the national cost benchmark can time lidR stage by stage.
#
# Usage
# -----
#   Rscript crown_segmentation.R --chm=<in.tif> --out-crowns=<out.gpkg> \
#       [--ws-a=0.6] [--ws-b=0.33] [--ws-c=0.0] [--ws-lo=3] [--ws-hi=12] \
#       [--ws-form=poly|power] [--ws-k=1] [--threads=1] [--seeds=<tops.csv>] \
#       [--smooth-radius-m=0.6] [--min-height-m=2] \
#       [--th-seed=0.45] [--th-cr=0.55] [--max-crown-radius-m=8] \
#       [--min-crown-area-m2=1] [--scale-factor=1] \
#       [--out-cv-raster=<out.tif>] [--cv-grid-m=50] [--cv-min-crowns=3]

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(sf)
})

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parse_args <- function(argv) {
  out <- list()
  for (a in argv) {
    if (!grepl("^--", a)) {
      stop("Arguments must be --name=value form, got: ", a)
    }
    kv <- sub("^--", "", a)
    key <- sub("=.*$", "", kv)
    val <- sub("^[^=]*=", "", kv)
    out[[gsub("-", "_", key)]] <- val
  }
  out
}

arg_num <- function(args, name, default) {
  if (is.null(args[[name]])) return(default)
  as.numeric(args[[name]])
}

arg_chr <- function(args, name, default = NULL) {
  if (is.null(args[[name]])) return(default)
  args[[name]]
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

.t_last <- proc.time()[["elapsed"]]
lap <- function(step) {
  now <- proc.time()[["elapsed"]]
  cat(sprintf("TIMING %s %.4f\n", step, now - .t_last))
  .t_last <<- now
}

peak_rss_kb <- function() {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  if (length(line) == 0) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", line))
}

chm_path   <- arg_chr(args, "chm")
out_crowns <- arg_chr(args, "out_crowns")
if (is.null(chm_path) || is.null(out_crowns)) {
  stop("--chm and --out-crowns are required")
}

ws_a  <- arg_num(args, "ws_a", 0.60)
ws_b  <- arg_num(args, "ws_b", 0.33)
ws_c  <- arg_num(args, "ws_c", 0.00)
ws_lo <- arg_num(args, "ws_lo", 3.0)
ws_hi <- arg_num(args, "ws_hi", 12.0)
ws_form <- arg_chr(args, "ws_form", "poly")
ws_k    <- arg_num(args, "ws_k", 1.0)
threads <- arg_num(args, "threads", 1)
seeds_path <- arg_chr(args, "seeds", NULL)
lidR::set_lidr_threads(as.integer(threads))

smooth_radius_m    <- arg_num(args, "smooth_radius_m", 0.6)
min_height_m       <- arg_num(args, "min_height_m", 2.0)
th_seed            <- arg_num(args, "th_seed", 0.45)
th_cr              <- arg_num(args, "th_cr", 0.55)
max_crown_radius_m <- arg_num(args, "max_crown_radius_m", 8.0)
min_crown_area_m2  <- arg_num(args, "min_crown_area_m2", 1.0)
scale_factor       <- arg_num(args, "scale_factor", 1.0)

out_cv_raster <- arg_chr(args, "out_cv_raster", NULL)
cv_grid_m     <- arg_num(args, "cv_grid_m", 50)
cv_min_crowns <- arg_num(args, "cv_min_crowns", 3)

if (!file.exists(chm_path)) stop("CHM not found: ", chm_path)
dir.create(dirname(out_crowns), recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
chm <- terra::rast(chm_path)
if (scale_factor != 1.0) chm <- chm * scale_factor

pixel_size_m <- mean(terra::res(chm))
cat("CHM   :", chm_path, "\n")
cat("Pixel :", pixel_size_m, "m\n")
lap("load")

# ---------------------------------------------------------------------------
# 2. Smooth. Radius in metres -> odd kernel in pixels, matching
#    csdv_core.segmentation.chm_watershed.smooth_kernel_px exactly.
# ---------------------------------------------------------------------------
kernel_px <- 2 * max(round(smooth_radius_m / pixel_size_m), 0) + 1
cat("Smooth:", smooth_radius_m, "m ->", kernel_px, "x", kernel_px, "px\n")
if (kernel_px > 1) {
  chm_smooth <- terra::focal(
    chm, w = matrix(1, kernel_px, kernel_px), fun = "mean", na.rm = TRUE
  )
} else {
  chm_smooth <- chm
}
lap("smooth")

# ---------------------------------------------------------------------------
# 3. Canopy floor
# ---------------------------------------------------------------------------
chm_smooth[chm_smooth < min_height_m] <- NA
lap("mask")

# ---------------------------------------------------------------------------
# 4. Tree tops. ws is the search window DIAMETER in metres, which is what
#    lmf() expects. The implied minimum separation between tops is ws/2.
# ---------------------------------------------------------------------------
ws_fun <- if (ws_form == "power") {
  function(h) {
    w <- ws_k * ws_a * pmax(h, 0)^ws_b
    w[!is.finite(w)] <- ws_lo
    pmax(ws_lo, pmin(ws_hi, w))
  }
} else {
  function(h) pmax(ws_lo, pmin(ws_hi, ws_k * (ws_a + ws_b * h + ws_c * h * h)))
}
if (is.null(seeds_path)) {
  ttops <- lidR::locate_trees(
    chm_smooth, lidR::lmf(ws = ws_fun, hmin = min_height_m, shape = "circular")
  )
} else {
  seeds <- utils::read.csv(seeds_path)
  ttops <- sf::st_as_sf(
    data.frame(treeID = seq_len(nrow(seeds)), Z = seeds$z, x = seeds$x, y = seeds$y),
    coords = c("x", "y"), crs = terra::crs(chm)
  )
}
cat("Tops  :", nrow(ttops), "\n")
lap("seeds")
if (nrow(ttops) == 0) stop("No tree tops detected")

# ---------------------------------------------------------------------------
# 5. Segment. max_cr is in PIXELS in lidR, so convert from metres here.
# ---------------------------------------------------------------------------
max_cr_px <- max(1, round(max_crown_radius_m / pixel_size_m))
cat("max_cr:", max_crown_radius_m, "m ->", max_cr_px, "px\n")
algo <- lidR::dalponte2016(
  chm_smooth, ttops,
  th_tree = min_height_m, th_seed = th_seed, th_cr = th_cr, max_cr = max_cr_px
)
crown_raster <- algo()
lap("grow")

# ---------------------------------------------------------------------------
# 6. Vectorize
# ---------------------------------------------------------------------------
crown_sf <- sf::st_as_sf(terra::as.polygons(crown_raster, dissolve = TRUE))
names(crown_sf)[1] <- "segment_id"
crown_sf <- crown_sf[!is.na(crown_sf$segment_id), ]
crown_sf$area_m2 <- as.numeric(sf::st_area(crown_sf))
crown_sf$crown_diam_m <- 2 * sqrt(crown_sf$area_m2 / pi)
crown_sf <- crown_sf[crown_sf$area_m2 >= min_crown_area_m2, ]

# Carry the tree-top height and position, so the Python and R outputs can be
# compared on the same terms. apex height is taken at the top, never as the
# maximum over the segment, because a segment maximum grows with segment size.
tt <- sf::st_as_sf(ttops)
tt_coords <- sf::st_coordinates(tt)
height_col <- if ("Z" %in% names(tt)) {
  "Z"
} else {
  names(tt)[sapply(tt, is.numeric)][1]
}
tt_id <- if ("treeID" %in% names(tt)) tt$treeID else seq_len(nrow(tt))
match_idx <- match(crown_sf$segment_id, tt_id)
crown_sf$apex_h_m <- as.numeric(tt[[height_col]])[match_idx]
crown_sf$seed_x <- tt_coords[match_idx, 1]
crown_sf$seed_y <- tt_coords[match_idx, 2]

lap("vectorize")
cat("Crowns:", nrow(crown_sf), "\n")
cat("Diam  :", round(range(crown_sf$crown_diam_m), 2), "m\n")
cat("Mean  :", round(mean(crown_sf$crown_diam_m), 2), "m\n")

sf::st_write(crown_sf, out_crowns, layer = "crowns",
             delete_dsn = TRUE, quiet = TRUE)
cat("Saved :", out_crowns, "\n")
lap("write")

# ---------------------------------------------------------------------------
# 7. Optional crown-diameter CV raster
# ---------------------------------------------------------------------------
if (!is.null(out_cv_raster)) {
  dir.create(dirname(out_cv_raster), recursive = TRUE, showWarnings = FALSE)
  ext <- terra::ext(chm)
  grid <- terra::rast(
    xmin = ext$xmin, xmax = ext$xmax, ymin = ext$ymin, ymax = ext$ymax,
    resolution = cv_grid_m, crs = terra::crs(chm)
  )
  centroid_coords <- sf::st_coordinates(sf::st_centroid(crown_sf))
  cell_ids <- terra::cellFromXY(grid, centroid_coords)
  ok <- !is.na(cell_ids)
  cell_cv <- tapply(crown_sf$crown_diam_m[ok], cell_ids[ok], function(x) {
    if (length(x) < cv_min_crowns) return(NA_real_)
    sd(x) / mean(x)
  })
  cv_values <- rep(NA_real_, terra::ncell(grid))
  cv_values[as.integer(names(cell_cv))] <- as.numeric(cell_cv)
  terra::values(grid) <- cv_values
  terra::writeRaster(grid, out_cv_raster, overwrite = TRUE,
                     datatype = "FLT4S", NAflag = -9999)
  cat("Saved :", out_cv_raster, "\n")
}

cat(sprintf("PEAK_RSS_KB %.0f\n", peak_rss_kb()))
cat("=== done ===\n")
