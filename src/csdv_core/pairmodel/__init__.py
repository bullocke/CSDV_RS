"""csdv_core.pairmodel: a Siamese pair model for canopy loss between two NAIP dates.

Inputs: the labelled chips of a label run (:mod:`csdv_core.label.chips`) for
training, and the NAIP mosaics of a grid unit for mapping.
Outputs: a checkpoint with its training record, a loss-probability raster of
a unit with one band per pair of consecutive dates, and tables and panels
that set the map beside the disturbance run and the held-out labels.

The model reads four-band NAIP only. The canopy height model and the
disturbance probability propose and constrain the labels and are never
inputs. See ``docs/guides/pair_model.md``.

GDAL has to load before PyTorch. The pip wheels of torch bring in the
system's libstdc++, which is older than the one conda's GDAL was built
against, and whichever loads first is the one the process keeps. With torch
first, ``import rasterio`` fails with "CXXABI_1.3.15 not found". Every module
here that imports torch therefore imports rasterio above it.
"""

from __future__ import annotations

__all__: list[str] = []
