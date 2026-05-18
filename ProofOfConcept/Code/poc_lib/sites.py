"""
poc_lib/sites.py — Site configuration for CSDV proof-of-concept analyses.

Defines the SiteConfig dataclass and the SITES registry. Both SCBI and HARV
are pre-configured. To add a new site, append an entry to SITES.

Data paths follow the convention established in 08_compare_chm_sources.py.
NEON CHM filenames are matched via glob fallback because wxee appends a
timestamp suffix (e.g. .time.19700101T000000) when downloading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project-root resolution: this file is at ProofOfConcept/Code/poc_lib/sites.py
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_DATA = _PROJECT_ROOT / "ProofOfConcept" / "Data"


@dataclass
class SiteConfig:
    """Configuration for one NEON field site.

    Parameters
    ----------
    site_code : str
        Four-letter NEON site code (e.g. "SCBI").
    neon_chm_path : Path
        Expected path to the NEON ALS CHM GeoTIFF. If missing, resolve_neon_chm()
        falls back to a glob match with wxee timestamp suffix.
    naip_rgb_dir : Path
        Directory containing the NAIP RGBN mosaic GeoTIFF(s).
    naip_rgb_prefix : str
        Filename prefix for NAIP RGBN (used by find_latest_tif).
    naip_chm_dir : Path
        Directory containing the NAIP CHM (Morford et al. 2025) GeoTIFF(s).
    naip_chm_prefix : str
        Filename prefix for NAIP CHM.
    deepzoom_size_m : float
        Side length (m) of the deep-zoom square window. Default 400 m.
    mediumzoom_size_m : float
        Side length (m) of the medium-zoom square window. Default 800 m.
    label : str
        Human-readable site label for figure titles and captions.
    """

    site_code: str
    neon_chm_path: Path
    naip_rgb_dir: Path
    naip_rgb_prefix: str
    naip_chm_dir: Path
    naip_chm_prefix: str
    deepzoom_size_m: float = 400.0
    mediumzoom_size_m: float = 800.0
    label: str = ""

    def resolve_neon_chm(self) -> Path:
        """Return the NEON CHM path, falling back to a glob match with wxee suffix."""
        if self.neon_chm_path.exists():
            return self.neon_chm_path
        matches = sorted(
            self.neon_chm_path.parent.glob(f"{self.neon_chm_path.stem}*.tif"),
            key=lambda p: p.stat().st_mtime,
        )
        if matches:
            return matches[-1]
        raise FileNotFoundError(
            f"NEON CHM not found: {self.neon_chm_path}\n"
            "Run the matching download script first."
        )


SITES: dict[str, SiteConfig] = {
    "SCBI": SiteConfig(
        site_code="SCBI",
        neon_chm_path=_DATA / "NEON" / "CHM" / "NEON_CHM_SCBI_Subset_2023.tif",
        naip_rgb_dir=_DATA / "NAIP" / "Imagery",
        naip_rgb_prefix="NAIP_SCBI",
        naip_chm_dir=_DATA / "NAIP" / "National_CHM",
        naip_chm_prefix="NAIPCHM_SCBI",
        deepzoom_size_m=400.0,
        label="SCBI — Smithsonian Conservation Biology Institute, VA",
    ),
    "HARV": SiteConfig(
        site_code="HARV",
        neon_chm_path=_DATA / "NEON" / "CHM" / "NEON_CHM_HARV_Subset_2023.tif",
        naip_rgb_dir=_DATA / "NAIP" / "Imagery",
        naip_rgb_prefix="NAIP_HARV",
        naip_chm_dir=_DATA / "NAIP" / "National_CHM",
        naip_chm_prefix="NAIPCHM_HARV",
        deepzoom_size_m=400.0,
        label="HARV — Harvard Forest, MA",
    ),
}


def get_site(code: str) -> SiteConfig:
    """Return the SiteConfig for *code*, raising a clear error if unknown."""
    code_upper = code.upper()
    if code_upper not in SITES:
        raise KeyError(
            f"Unknown site '{code}'. Available sites: {', '.join(sorted(SITES))}. "
            "Add a new entry to poc_lib/sites.py to register additional sites."
        )
    return SITES[code_upper]
