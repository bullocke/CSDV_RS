"""Shared site configuration for NAIP-CHM AOI inference.

Site centers are NEON tower coordinates (WGS84). The PoC analysis AOIs in
``ProofOfConcept/Code/download_gee.py`` are smaller (~1-2 km) and serve a
different purpose (matched NEON+NAIP comparison). Here we use a larger AOI
(default 5 km square) so each prediction includes spatial context for
temporal-consistency analysis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteCenter:
    """Tower coordinates for a PoC site."""

    code: str
    lat: float
    lon: float
    label: str


SITES: dict[str, SiteCenter] = {
    "SCBI": SiteCenter("SCBI", 38.8929, -78.1454, "Smithsonian Conservation Biology Institute, VA"),
    "HARV": SiteCenter("HARV", 42.5369, -72.1727, "Harvard Forest, MA"),
    "TALL": SiteCenter("TALL", 32.9505, -87.3933, "Talladega National Forest, AL"),
    "MLBS": SiteCenter("MLBS", 37.3783, -80.5247, "Mountain Lake Biological Station, VA"),
}


def get_site(code: str) -> SiteCenter:
    """Look up a site by 4-letter code (case insensitive)."""
    key = code.upper()
    if key not in SITES:
        raise KeyError(f"Unknown site '{code}'. Valid: {sorted(SITES)}")
    return SITES[key]
