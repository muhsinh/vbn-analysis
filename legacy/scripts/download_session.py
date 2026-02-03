#!/usr/bin/env python
"""Download a specific VBN session NWB file.

This script downloads the NWB file for a specific session and validates
that it can be loaded correctly.

Usage:
    python scripts/download_session.py SESSION_ID [--cache-dir PATH]

Example:
    python scripts/download_session.py 1055240613
    python scripts/download_session.py 1055240613 --cache-dir ~/data/vbn_cache
"""

import argparse
from pathlib import Path

from vbn.cache import get_cache, get_sessions_table, get_session_nwb_path
from vbn.config import get_cache_dir
from vbn.io import load_session, summarize_session
from vbn.utils import setup_logging, print_diagnostic, format_size, check_disk_space


def main():
    parser = argparse.ArgumentParser(
        description="Download a specific VBN session NWB file"
    )
    parser.add_argument(
        "session_id",
        type=int,
        help="Ecephys session ID to download"
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
    
    # Check disk space (need ~4GB for a session)
    required_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
    if not check_disk_space(cache_dir, required_bytes):
        logger.warning(f"Low disk space. Need ~4GB free in {cache_dir}")
    
    # Initialize cache
    cache = get_cache(cache_dir)
    
    # Validate session exists
    sessions = get_sessions_table(cache, filter_by_validity=False)
    
    if args.session_id not in sessions.index:
        logger.error(f"Session {args.session_id} not found in dataset")
        sample_ids = list(sessions.index[:10])
        logger.info(f"Valid session IDs include: {sample_ids}")
        logger.info(f"Total sessions: {len(sessions)}")
        return 1
    
    # Check if already downloaded
    nwb_path = get_session_nwb_path(cache_dir, args.session_id)
    if nwb_path:
        logger.info(f"Session already downloaded: {nwb_path}")
        nwb_size = nwb_path.stat().st_size
        logger.info(f"File size: {format_size(nwb_size)}")
    else:
        logger.info(f"Downloading session {args.session_id}...")
    
    # Load session (downloads if needed)
    session = load_session(cache, args.session_id)
    
    # Get NWB path after download
    nwb_path = get_session_nwb_path(cache_dir, args.session_id)
    nwb_size = nwb_path.stat().st_size if nwb_path else 0
    
    # Summarize session
    summary = summarize_session(session)
    
    # Get session metadata from table
    session_row = sessions.loc[args.session_id]
    
    print_diagnostic(f"Session {args.session_id}", {
        "NWB file": nwb_path.name if nwb_path else "Unknown",
        "File size": format_size(nwb_size),
        "Mouse ID": session_row.get("mouse_id"),
        "Genotype": session_row.get("genotype"),
        "Sex": session_row.get("sex"),
        "Age (days)": session_row.get("age_in_days"),
        "Session type": session_row.get("session_type"),
        "Unit count": summary.get("unit_count"),
        "Probe count": summary.get("probe_count"),
        "Has eye tracking": summary.get("has_eye_tracking"),
        "Eye tracking samples": summary.get("eye_tracking_samples"),
        "Eye tracking duration (min)": round(summary.get("eye_tracking_duration_min", 0), 1),
        "Stimulus presentations": summary.get("stimulus_presentations"),
        "Trials": summary.get("trials"),
    })
    
    logger.info("Session download complete!")
    return 0


if __name__ == "__main__":
    exit(main())
