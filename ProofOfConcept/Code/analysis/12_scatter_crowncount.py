"""
12_scatter_crowncount.py — Per-window crown count scatter plots: NEON ALS
CHM (x) versus NAIP deep-learning CHM (y), at three window sizes.

For each site, writes one (1 × 3) figure showing 25 m, 50 m, and 100 m
windows side by side. Style mirrors ``10_scatter_validation.py`` (semi-
transparent points, 1:1 reference line, OLS fit, stats annotation box).

The crown-count rasters consumed here are written by
``09b_regenerate_crownCV_figures.py`` into ``intermediate/mediumzoom/{SITE}/
v2/crown_count_{neon,naip}_{SITE}_{w}m_v2.tif``. Both NEON and NAIP rasters
are computed against the same NEON CHM reference grid, so per-pixel pairs
are directly comparable.

Usage
-----
    python 12_scatter_crowncount.py --site SCBI
    python 12_scatter_crowncount.py --site HARV
    python 12_scatter_crowncount.py --site all
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from poc_lib import figures as fig_utils, get_site, read_band

# Reuse the scatter helpers from 10_scatter_validation.py
_SV_PATH = _HERE.parent / "10_scatter_validation.py"
_spec = importlib.util.spec_from_file_location("scatter_validation", _SV_PATH)
_sv = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["scatter_validation"] = _sv
_spec.loader.exec_module(_sv)  # type: ignore[union-attr]

_make_scatter_panel = _sv._make_scatter_panel
_regression_stats = _sv._regression_stats

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_FIGURES_OUT = _POC / "Results" / "summary_document" / "figures" / "metrics" / "crownCV"
_INTERMEDIATE = _POC / "Results" / "summary_document" / "intermediate" / "mediumzoom"

WINDOW_SIZES = [25.0, 50.0, 100.0]
POINT_COLOR = "#1f78b4"
REGRESSION_COLOR = "#e31a1c"


def _load_count_raster(path: Path) -> np.ndarray:
    arr, _, _ = read_band(path)
    return arr.astype(np.float32)


def make_scatter_for_site(site_code: str) -> Path:
    cfg = get_site(site_code)
    v2_dir = _INTERMEDIATE / site_code / "v2"

    fig, axes = plt.subplots(1, len(WINDOW_SIZES), figsize=(13.2, 4.4))

    for ax, w in zip(axes, WINDOW_SIZES):
        wi = int(w)
        neon_p = v2_dir / f"crown_count_neon_{site_code}_{wi}m_v2.tif"
        naip_p = v2_dir / f"crown_count_naip_{site_code}_{wi}m_v2.tif"
        if not neon_p.exists() or not naip_p.exists():
            logger.warning("Missing crown-count raster for %s %d m", site_code, wi)
            ax.text(0.5, 0.5, "Missing raster", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            continue

        x = _load_count_raster(neon_p).ravel()
        y = _load_count_raster(naip_p).ravel()
        stats = _regression_stats(x, y)
        _make_scatter_panel(
            ax,
            x,
            y,
            window_m=w,
            stats=stats,
            point_color=POINT_COLOR,
            regression_color=REGRESSION_COLOR,
            xlabel="NEON crown count",
            ylabel="NAIP crown count",
        )

    fig.suptitle(
        f"{cfg.label} \u2014 Crown count per window: NEON vs NAIP CHM",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = _FIGURES_OUT / f"fig_scatter_v3_crowncount_NEON_vs_NAIP_{site_code}.png"
    fig_utils.save_fig(fig, out)
    return out


@click.command()
@click.option(
    "--site",
    type=click.Choice(["SCBI", "HARV", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
def main(site: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fig_utils.setup_style()
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    sites = ["SCBI", "HARV"] if site.lower() == "all" else [site.upper()]
    for s in sites:
        make_scatter_for_site(s)
    logger.info("Done.")


if __name__ == "__main__":
    main()
