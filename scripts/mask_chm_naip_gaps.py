"""Mask NAIP gaps in an existing NAIP-CHM without re-running inference.

CHMs predicted before ``csdv chm-inference`` masked NAIP gaps hold model
heights, usually near 0 m, wherever the NAIP mosaic had no imagery. NorthMaine
2023 is one: USDA never finished Maine in 2023, so 9 of the site's 15 quarter
quads are black fill. This script sets those CHM pixels, plus a band of
``--edge-m`` around each gap, to the CHM's nodata value, in place. It reads
``naip/<site>/<year>/*.tif`` and ``naip_chm/<site>/<year>/*.tif`` under
``$CSDV_DATA_ROOT``.

The unmasked CHM is kept beside the masked one as ``<name>.tif.unmasked``,
which no ``*.tif`` glob picks up. Delete it once the masked CHM checks out.

Example::

    python scripts/mask_chm_naip_gaps.py --site NorthMaine --year 2023
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from csdv_core.chm_inference.naip_gaps import mask_naip_gaps
from csdv_core.config import load_metrics
from csdv_core.io.paths import project_paths

logger = logging.getLogger(__name__)


def single_tif(folder: Path, what: str) -> Path:
    """Return the only ``*.tif`` in ``folder``.

    Raises:
        SystemExit: If the folder holds no GeoTIFF or more than one.
    """
    paths = sorted(folder.glob("*.tif"))
    if len(paths) != 1:
        raise SystemExit(f"Expected one {what} GeoTIFF in {folder}, found {len(paths)}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Default: $CSDV_DATA_ROOT.",
    )
    parser.add_argument(
        "--edge-m",
        type=float,
        default=None,
        help="Band (m) also masked around each gap. Default: metrics.yaml "
        "chm_inference.gap_edge_m.",
    )
    parser.add_argument(
        "--no-keep-original",
        action="store_true",
        help="Replace the CHM without keeping the unmasked copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be used and exit.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )

    data_root = args.data_root or project_paths().data_root
    edge_m = args.edge_m
    if edge_m is None:
        edge_m = load_metrics().chm_inference.gap_edge_m
    naip = single_tif(data_root / "naip" / args.site / str(args.year), "NAIP")
    chm = single_tif(data_root / "naip_chm" / args.site / str(args.year), "CHM")
    logger.info("NAIP: %s", naip)
    logger.info("CHM:  %s", chm)
    logger.info("Band around gaps: %g m", edge_m)
    if args.dry_run:
        return

    try:
        result = mask_naip_gaps(
            naip, chm, edge_m=edge_m, keep_original=not args.no_keep_original
        )
    except FileExistsError as exc:
        raise SystemExit(f"{exc}. Nothing done.") from exc
    if result.original_path is not None:
        logger.info("Unmasked CHM kept as %s", result.original_path)


if __name__ == "__main__":
    sys.exit(main())
