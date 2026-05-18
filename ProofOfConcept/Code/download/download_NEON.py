"""
NEON Data Downloader

This module provides functionality to download NEON vegetation structure data
from the NEON API.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class NEONDataDownloader:
    """Downloads NEON vegetation structure data for specified sites and date ranges."""
    
    NEON_API_BASE = "https://data.neonscience.org/api/v0"
    VST_PRODUCT_CODE = "# DP1.10017.001" # Hemispherical imagery
    

    def __init__(self, session: Optional[requests.Session] = None):
        """
        Initialize the downloader.
        
        Args:
            session: Optional requests Session for connection pooling.
        """
        self.session = session or requests.Session()

    def list_site_months(self, site_code: str) -> Tuple[List[str], List[str]]:
        """
        Retrieve available months and data URLs for a site.
        
        Args:
            site_code: NEON site code (e.g., 'ABBY', 'WREF').
            
        Returns:
            Tuple of (months, urls) where months are YYYY-MM strings.
            
        Raises:
            ValueError: If site not found in product data.
            requests.HTTPError: If API request fails.
        """
        url = f"{self.NEON_API_BASE}/products/{self.VST_PRODUCT_CODE}"
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        
        site_data = None
        for site_info in data["data"].get("siteCodes", []):
            if site_info.get("siteCode") == site_code:
                site_data = site_info
                break
                
        if not site_data:
            raise ValueError(f"Site {site_code} not found in product data.")
            
        months = site_data.get("availableMonths", []) or []
        urls = site_data.get("availableDataUrls", []) or []
        return months, urls

    def _filter_by_date_range(
        self, 
        months: List[str], 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> List[str]:
        """Filter months by date range."""
        out = []
        for m in months:
            if start_date and m < start_date:
                continue
            if end_date and m > end_date:
                continue
            out.append(m)
        return out

    def _filter_provisional(
        self, 
        months: List[str], 
        available_urls: List[str]
    ) -> List[str]:
        """
        Filter to non-provisional months when possible.
        
        Args:
            months: List of YYYY-MM month strings.
            available_urls: List of data URLs.
            
        Returns:
            Filtered list of months (non-provisional if available).
        """
        non_prov = set()
        for url in available_urls:
            if "PROVISIONAL" not in url.upper():
                parts = url.rstrip("/").split("/")
                if parts:
                    month = parts[-1]
                    if month in months:
                        non_prov.add(month)
                        
        if non_prov:
            filtered = sorted([m for m in months if m in non_prov])
            logger.info(f"Filtered to {len(filtered)} non-provisional months")
            return filtered
            
        logger.warning("Could not determine provisional status; using all months")
        return sorted(months)

    def download_site_data(
        self,
        site_code: str,
        output_dir: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exclude_provisional: bool = True
    ) -> List[Path]:
        """
        Download all available data for a NEON site.
        
        Args:
            site_code: NEON site code (e.g., 'ABBY').
            output_dir: Base directory for downloads.
            start_date: Optional start date filter (YYYY-MM).
            end_date: Optional end date filter (YYYY-MM).
            exclude_provisional: Whether to exclude provisional data.
            
        Returns:
            List of Path objects for downloaded month directories.
        """
        logger.info(f"Preparing to download NEON VST data for site {site_code}")
        months, urls = self.list_site_months(site_code)
        logger.info(f"Found {len(months)} total months for {site_code}")

        if start_date or end_date:
            months = self._filter_by_date_range(months, start_date, end_date)
            logger.info(f"After date filtering: {len(months)} months")

        if exclude_provisional:
            months = self._filter_provisional(months, urls)

        if not months:
            logger.warning(f"No months to download for {site_code}")
            return []

        downloaded_dirs: List[Path] = []
        for i, month in enumerate(months, 1):
            try:
                logger.info(f"[{i}/{len(months)}] {site_code} {month}")
                d = self._download_month(site_code, month, output_dir)
                downloaded_dirs.append(d)
            except Exception as e:
                logger.error(f"Failed month {month}: {e}")
                
        return downloaded_dirs

    def _download_month(
        self, 
        site_code: str, 
        month: str, 
        output_dir: Path
    ) -> Path:
        """
        Download data for a specific site-month.
        
        Args:
            site_code: NEON site code.
            month: Month string (YYYY-MM).
            output_dir: Base output directory.
            
        Returns:
            Path to the downloaded month directory.
        """
        url = f"{self.NEON_API_BASE}/data/{self.VST_PRODUCT_CODE}/{site_code}/{month}"
        r = self.session.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        files = data["data"].get("files", [])

        month_dir = output_dir / site_code / month
        month_dir.mkdir(parents=True, exist_ok=True)

        # Look for zip file (prefer basic package)
        zip_file = None
        for f in files:
            if f.get("name", "").endswith(".zip"):
                if "basic" in f["name"].lower():
                    zip_file = f
                    break
                if zip_file is None:
                    zip_file = f

        if zip_file:
            logger.info(
                f"Downloading zip: {zip_file['name']} "
                f"({zip_file.get('size', '?')} bytes)"
            )
            zr = self.session.get(zip_file["url"], timeout=600)
            zr.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
                zf.extractall(month_dir)
            logger.info(f"Extracted to {month_dir}")
        else:
            # Fallback to individual CSV files
            csvs = [f for f in files if f.get("name", "").endswith(".csv")]
            if not csvs:
                raise ValueError(f"No CSV or ZIP files for {site_code} {month}")
            logger.info(f"Downloading {len(csvs)} CSV files")
            for f in csvs:
                fr = self.session.get(f["url"], timeout=120)
                fr.raise_for_status()
                (month_dir / f["name"]).write_bytes(fr.content)
                
        return month_dir