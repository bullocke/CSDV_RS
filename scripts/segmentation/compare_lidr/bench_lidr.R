# bench_lidr.R
#
# Times lidR tree detection (locate_trees + lmf) and crown growth
# (dalponte2016) on a list of CHM rasters, with packages loaded once, and
# writes tree tops and crown label rasters for comparison with the Python
# engine. Inputs are expected to be smoothed and masked already, so both
# engines see identical pixels; no smoothing happens here.
#
# Usage
#   Rscript bench_lidr.R --manifest=<jobs.csv> --out=<timings.csv>
#
# Manifest columns (one row per job)
#   job_id       label carried to the output
#   chm          float32 GeoTIFF in metres, NaN or NA outside the canopy
#   min_height   canopy floor; lmf hmin and dalponte2016 th_tree
#   ws_form      "poly" or "power"
#   ws_a, ws_b, ws_c, ws_k, ws_lo, ws_hi
#                window diameter in metres: poly clip(k*(a+b*h+c*h^2), lo, hi),
#                power clip(k*a*h^b, lo, hi)
#   th_seed, th_cr, max_cr_px
#                dalponte2016 arguments; max_cr in pixels as lidR expects
#   seeds        optional CSV of tree tops (x, y, z); empty runs lmf
#   threads      set_lidr_threads()
#   reps         timed repetitions of each step
#   grow         1 to run dalponte2016, 0 for detection only
#   out_tops     optional CSV for the tree tops lmf found
#   out_labels   optional GeoTIFF for the crown labels (tree top order)
#   profile      1 to run one extra growth under Rprof and report the time
#                spent inside C_dalponte2016
#
# Output columns: job_id, n_px, n_valid, n_tops, t_lmf (median), t_lmf_all,
# t_grow (median), t_grow_all, t_cpp (Rprof), vmhwm_mb (process high-water
# mark after the job; run jobs in ascending size to read it per job).

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(sf)
})

args <- commandArgs(trailingOnly = TRUE)
kv <- function(name) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) stop("missing --", name)
  sub(paste0("^--", name, "="), "", hit[1])
}
jobs <- utils::read.csv(kv("manifest"), stringsAsFactors = FALSE)
out_path <- kv("out")

vmhwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(gsub("[^0-9]", "", line)) / 1024
}

elapsed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  force(expr)
  proc.time()[["elapsed"]] - t0
}

window_fun <- function(j) {
  if (j$ws_form == "power") {
    function(h) {
      w <- j$ws_k * j$ws_a * pmax(h, 0)^j$ws_b
      w[!is.finite(w)] <- j$ws_lo
      pmax(j$ws_lo, pmin(j$ws_hi, w))
    }
  } else {
    function(h) pmax(j$ws_lo, pmin(j$ws_hi, j$ws_k * (j$ws_a + j$ws_b * h + j$ws_c * h * h)))
  }
}

run_job <- function(j) {
  set_lidr_threads(as.integer(j$threads))
  chm <- terra::rast(j$chm)
  chm[chm < j$min_height] <- NA
  n_valid <- as.numeric(terra::global(!is.na(chm), "sum")[1, 1])
  ws <- window_fun(j)
  reps <- max(1L, as.integer(j$reps))

  t_lmf <- NA_real_
  t_lmf_all <- ""
  if (is.na(j$seeds) || j$seeds == "") {
    times <- numeric(reps)
    for (r in seq_len(reps)) {
      times[r] <- elapsed(ttops <- locate_trees(chm, lmf(ws = ws, hmin = j$min_height, shape = "circular")))
    }
    t_lmf <- stats::median(times)
    t_lmf_all <- paste(sprintf("%.4f", times), collapse = ";")
    if (!is.na(j$out_tops) && j$out_tops != "") {
      xy <- sf::st_coordinates(ttops)
      utils::write.csv(data.frame(treeID = ttops$treeID, x = xy[, 1], y = xy[, 2], z = ttops$Z),
                       j$out_tops, row.names = FALSE)
    }
  } else {
    s <- utils::read.csv(j$seeds)
    ttops <- sf::st_as_sf(
      data.frame(treeID = seq_len(nrow(s)), Z = s$z, x = s$x, y = s$y),
      coords = c("x", "y"), crs = terra::crs(chm)
    )
  }

  t_grow <- NA_real_
  t_grow_all <- ""
  t_cpp <- NA_real_
  if (as.integer(j$grow) == 1 && nrow(ttops) > 0) {
    algo <- dalponte2016(chm, ttops, th_tree = j$min_height, th_seed = j$th_seed,
                         th_cr = j$th_cr, max_cr = j$max_cr_px)
    times <- numeric(reps)
    for (r in seq_len(reps)) times[r] <- elapsed(crowns <- algo())
    t_grow <- stats::median(times)
    t_grow_all <- paste(sprintf("%.4f", times), collapse = ";")
    if (!is.na(j$profile) && as.integer(j$profile) == 1) {
      prof <- tempfile(fileext = ".out")
      utils::Rprof(prof, interval = 0.005)
      crowns <- algo()
      utils::Rprof(NULL)
      tot <- utils::summaryRprof(prof)$by.total
      key <- grep("C_dalponte2016", rownames(tot), value = TRUE)
      t_cpp <- if (length(key)) tot[key[1], "total.time"] else NA_real_
      unlink(prof)
    }
    if (!is.na(j$out_labels) && j$out_labels != "") {
      crowns[is.na(crowns)] <- 0
      terra::writeRaster(crowns, j$out_labels, overwrite = TRUE, datatype = "INT4S")
    }
  }

  data.frame(
    job_id = j$job_id, n_px = terra::ncell(chm), n_valid = n_valid,
    n_tops = nrow(ttops), threads = j$threads,
    t_lmf = t_lmf, t_lmf_all = t_lmf_all, t_grow = t_grow, t_grow_all = t_grow_all,
    t_cpp = t_cpp, vmhwm_mb = vmhwm_mb()
  )
}

rows <- list()
for (i in seq_len(nrow(jobs))) {
  j <- jobs[i, ]
  rows[[i]] <- tryCatch(
    run_job(j),
    error = function(e) {
      cat(sprintf("%s: ERROR %s\n", j$job_id, conditionMessage(e)))
      data.frame(job_id = j$job_id, n_px = NA, n_valid = NA, n_tops = NA, threads = j$threads,
                 t_lmf = NA, t_lmf_all = "", t_grow = NA, t_grow_all = "", t_cpp = NA,
                 vmhwm_mb = vmhwm_mb(), error = conditionMessage(e))
    }
  )
  if (is.null(rows[[i]]$error)) rows[[i]]$error <- ""
  if (i %% 50 == 0 || i == nrow(jobs)) {
    utils::write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
  }
}
