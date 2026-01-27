#!/usr/bin/env python
"""Download VBN metadata files to local cache.

This script initializes the AllenSDK cache and downloads the metadata CSV files
(~160MB total) without downloading any session NWB files.

Usage:
    python scripts/download_metadata.py [--cache-dir PATH]

Example:
    python scripts/download_metadata.py
    python scripts/download_metadata.py --cache-dir ~/data/vbn_cache
"""

import argparse
from pathlib import Path

from vbn.cache import get_cache, get_sessions_table, get_probes_table, get_units_table
from vbn.config import get_cache_dir
from vbn.utils import setup_logging, print_diagnostic, format_size


def main():
    parser = argparse.ArgumentParser(
        description="Download VBN metadata files to local cache"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: ~/data/vbn_cache)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine cache directory
    cache_dir = args.cache_dir or get_cache_dir()
    logger.info(f"Cache directory: {cache_dir}")
    
    # Initialize cache (this downloads metadata)
    logger.info("Downloading metadata...")
    cache = get_cache(cache_dir)
    
    # Load tables to verify
    logger.info("Loading metadata tables...")
    sessions = get_sessions_table(cache, filter_by_validity=False)
    probes = get_probes_table(cache)
    units = get_units_table(cache)
    
    # Calculate metadata size
    metadata_dir = cache_dir / "visual-behavior-neuropixels" / "project_metadata"
    if metadata_dir.exists():
        total_size = sum(f.stat().st_size for f in metadata_dir.glob("*.csv"))
    else:
        total_size = 0
    
    # Print summary
    print_diagnostic("VBN Metadata Summary", {
        "Cache directory": cache_dir,
        "Metadata size": format_size(total_size),
        "Total sessions": len(sessions),
        "Valid sessions": len(sessions[sessions["abnormal_histology"].isna() & sessions["abnormal_activity"].isna()]) if "abnormal_histology" in sessions.columns else len(sessions),
        "Total probes": len(probes),
        "Total units": len(units),
    })
    
    # Show sample session IDs
    sample_ids = list(sessions.index[:5])
    logger.info(f"Sample session IDs: {sample_ids}")
    
    logger.info("Metadata download complete!")


if __name__ == "__main__":
    main()
