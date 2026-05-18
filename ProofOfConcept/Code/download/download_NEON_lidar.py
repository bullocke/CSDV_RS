#!/usr/bin/env python3
"""
NEON Lidar Data Downloader and Organizer

This script downloads missing NEON discrete return lidar data (DP1.30003.001)
and creates symbolic links to existing data to consolidate everything in one location.

Requirements:
    pip install requests pandas tqdm
"""

import os
import sys
import json
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
# Override these via environment variables or config before calling functions
EXISTING_BASE_PATH = os.environ.get("REFAGB_LIDAR_PATH", "")
NEW_BASE_PATH = os.environ.get("REFAGB_LIDAR_PATH", "")
NEON_API_BASE = "https://data.neonscience.org/api/v0"
PRODUCT_CODE = "DP1.30003.001"  # Discrete return lidar point cloud

# All NEON terrestrial sites (as of 2024)
NEON_TERRESTRIAL_SITES = [
    "ABBY", "BARR", "BART", "BLAN", "BONA", "CLBJ", "CPER", "DCFS", "DEJU",
    "DELA", "DSNY", "GRSM", "GUAN", "HARV", "HEAL", "JERC", "JORN", "KONA",
    "KONZ", "LAJA", "LENO", "MLBS", "MOAB", "NIWO", "NOGP", "OAES", "ONAQ",
    "ORNL", "OSBS", "PUUM", "RMNP", "SCBI", "SERC", "SJER", "SOAP", "SRER",
    "STEI", "STER", "TALL", "TEAK", "TOOL", "TREE", "UKFS", "UNDE", "WOOD",
    "WREF", "YELL"
]

def create_directory_structure(base_path):
    """Create the base directory structure if it doesn't exist."""
    Path(base_path).mkdir(parents=True, exist_ok=True)
    print(f"Created/verified base directory: {base_path}")

def check_existing_sites():
    """Check which sites already have lidar data in the existing location."""
    existing_sites = {}
    
    print("\nChecking existing sites...")
    for site in NEON_TERRESTRIAL_SITES:
        site_path = Path(EXISTING_BASE_PATH) / site
        lidar_path = site_path / "lidar"
        
        if lidar_path.exists():
            # Check for date folders with classified point cloud files
            date_folders = []
            for date_folder in lidar_path.iterdir():
                if date_folder.is_dir():
                    # Check if it contains classified point cloud files
                    laz_files = list(date_folder.glob("*classified_point_cloud_colorized.laz"))
                    if laz_files:
                        date_folders.append(date_folder.name)
            
            if date_folders:
                existing_sites[site] = {
                    'path': str(lidar_path),
                    'dates': sorted(date_folders)
                }
                print(f"  {site}: Found {len(date_folders)} acquisition dates")
    
    return existing_sites

def create_symbolic_links(existing_sites):
    """Create symbolic links to existing data."""
    print("\nCreating symbolic links to existing data...")
    
    for site, info in existing_sites.items():
        target_path = Path(info['path'])
        link_path = Path(NEW_BASE_PATH) / site / "lidar"
        
        # Create parent directory
        link_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symbolic link
        if link_path.exists():
            if link_path.is_symlink():
                print(f"  {site}: Symbolic link already exists")
            else:
                print(f"  {site}: Path exists but is not a symbolic link, skipping")
        else:
            try:
                link_path.symlink_to(target_path)
                print(f"  {site}: Created symbolic link")
            except Exception as e:
                print(f"  {site}: Error creating symbolic link: {e}")

def get_available_sites_from_api():
    """Get list of sites with available lidar data from NEON API."""
    print("\nQuerying NEON API for available lidar data...")
    
    url = f"{NEON_API_BASE}/products/{PRODUCT_CODE}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        available_sites = []
        if 'data' in data and 'siteCodes' in data['data']:
            for site_info in data['data']['siteCodes']:
                site_code = site_info.get('siteCode')
                if site_code in NEON_TERRESTRIAL_SITES:
                    available_months = site_info.get('availableMonths', [])
                    if available_months:
                        available_sites.append({
                            'site': site_code,
                            'months': available_months
                        })
        
        print(f"Found {len(available_sites)} sites with available lidar data")
        return available_sites
        
    except Exception as e:
        print(f"Error querying NEON API: {e}")
        return []

def get_download_urls(site, year_month):
    """Get download URLs for a specific site and year-month."""
    url = f"{NEON_API_BASE}/data/{PRODUCT_CODE}/{site}/{year_month}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        download_urls = []
        if 'data' in data and 'files' in data['data']:
            for file_info in data['data']['files']:
                file_url = file_info.get('url')
                file_name = file_info.get('name')
                
                # Only download classified point cloud files
                if file_name and 'classified_point_cloud_colorized.laz' in file_name:
                    download_urls.append({
                        'url': file_url,
                        'name': file_name,
                        'size': file_info.get('size', 0)
                    })
        
        return download_urls
        
    except Exception as e:
        print(f"Error getting download URLs for {site} {year_month}: {e}")
        return []

def download_file(url, destination, max_retries=3):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {url}: {e}")
                return False

def download_site_data(site, year_months, existing_sites):
    """Download all data for a specific site."""
    site_downloaded = 0
    site_skipped = 0
    
    for year_month in year_months:
        # Check if we already have this data
        if site in existing_sites:
            if year_month in existing_sites[site]['dates']:
                site_skipped += 1
                continue
        
        # Create directory structure
        site_path = Path(NEW_BASE_PATH) / site / "lidar" / year_month
        try:
            site_path.mkdir(parents=True, exist_ok=True)
        except:
            continue
        # Get download URLs
        download_urls = get_download_urls(site, year_month)
        
        if download_urls:
            print(f"\n  Downloading {site} {year_month}: {len(download_urls)} files")
            
            for file_info in tqdm(download_urls, desc=f"    {year_month}", leave=False):
                destination = site_path / file_info['name']
                
                if destination.exists():
                    # Check if file size matches
                    if destination.stat().st_size == file_info['size']:
                        continue
                
                if download_file(file_info['url'], destination):
                    site_downloaded += 1
                    
        else:
            print(f"  No classified point cloud files found for {site} {year_month}")
    
    return site_downloaded, site_skipped

def main():
    """Main function to orchestrate the download and organization process."""
    print("NEON Lidar Data Downloader and Organizer")
    print("=" * 50)
    
    # Create base directory
    create_directory_structure(NEW_BASE_PATH)
    
    # Check existing sites
    existing_sites = check_existing_sites()
    print(f"\nFound {len(existing_sites)} sites with existing lidar data")
    
    # Create symbolic links to existing data
    #create_symbolic_links(existing_sites)
    
    # Get available sites from API
    available_sites = get_available_sites_from_api()
    
    # Determine which sites need to be downloaded
    sites_to_download = []
    for site_info in available_sites:
        site = site_info['site']
        
        if site in existing_sites:
            # Check if we need additional months
            existing_months = set(existing_sites[site]['dates'])
            available_months = set(site_info['months'])
            missing_months = available_months - existing_months
            
            if missing_months:
                sites_to_download.append({
                    'site': site,
                    'months': sorted(list(missing_months))
                })
        else:
            # Site doesn't exist at all
            sites_to_download.append({
                'site': site,
                'months': sorted(site_info['months'])
            })
    
    print(f"\nNeed to download data for {len(sites_to_download)} sites")
    
    if not sites_to_download:
        print("All available data already exists!")
        return
    
    # Show download plan
    print("\nDownload plan:")
    total_months = 0
    for site_info in sites_to_download:
        print(f"  {site_info['site']}: {len(site_info['months'])} months")
        total_months += len(site_info['months'])
    print(f"\nTotal: {total_months} site-months to download")
    
    # Confirm download
#    response = input("\nProceed with download? (y/n): ")
#    if response.lower() != 'y':
#        print("Download cancelled")
#        return
    
    # Download data
    print("\nStarting downloads...")
    total_downloaded = 0
    total_skipped = 0
    
    for i, site_info in enumerate(sites_to_download, 1):
        site = site_info['site']
        months = site_info['months']
        
        print(f"\n[{i}/{len(sites_to_download)}] Downloading {site}...")
        downloaded, skipped = download_site_data(site, months, existing_sites)
        
        total_downloaded += downloaded
        total_skipped += skipped
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary:")
    print(f"  Files downloaded: {total_downloaded}")
    print(f"  Files skipped (already exist): {total_skipped}")
    print(f"  Symbolic links created: {len(existing_sites)}")
    print(f"\nAll data is now available in: {NEW_BASE_PATH}")
    
    # Create a summary file
    summary_file = Path(NEW_BASE_PATH) / "download_summary.json"
    summary_data = {
        'download_date': datetime.now().isoformat(),
        'existing_sites_linked': list(existing_sites.keys()),
        'sites_downloaded': [s['site'] for s in sites_to_download],
        'total_files_downloaded': total_downloaded,
        'base_path': NEW_BASE_PATH
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()