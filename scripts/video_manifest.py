#!/usr/bin/env python
"""Scan cache for video files and generate manifest.

This script searches the cache directory for video files associated with
a session and reports what's available.

Usage:
    python scripts/video_manifest.py SESSION_ID [--cache-dir PATH] [--output PATH]

Example:
    python scripts/video_manifest.py 1055240613
    python scripts/video_manifest.py 1055240613 --output manifest.json
"""

import argparse
import json
from pathlib import Path

from vbn.cache import get_cache, get_session_nwb_path
from vbn.config import get_cache_dir, get_session_output_dir
from vbn.io import load_session, get_eye_tracking
from vbn.video import discover_videos, generate_video_manifest
from vbn.utils import setup_logging, print_diagnostic


def main():
    parser = argparse.ArgumentParser(
        description="Scan cache for video files and generate manifest"
    )
    parser.add_argument(
        "session_id",
        type=int,
        help="Ecephys session ID to scan for"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: ~/data/vbn_cache)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for manifest JSON"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine cache directory
    cache_dir = args.cache_dir or get_cache_dir()
    logger.info(f"Searching {cache_dir} for session {args.session_id}...")
    
    # Search for video files
    videos = discover_videos(cache_dir, args.session_id)
    
    # Generate manifest
    output_path = args.output
    if output_path is None:
        output_dir = get_session_output_dir(args.session_id)
        output_path = output_dir / "video_manifest.json"
    
    manifest = generate_video_manifest(cache_dir, args.session_id, output_path)
    
    # Print results
    print_diagnostic("Video Search Results", {
        "Search path": cache_dir,
        "Session ID": args.session_id,
        "Files searched": manifest["files_searched"],
        "Video files found": manifest["video_files_found"],
    })
    
    if videos:
        logger.info("Video files found:")
        for v in videos:
            logger.info(f"  - {v['path'].name} ({v['camera_type_guess']}, {v['size_mb']:.1f} MB)")
    else:
        logger.warning("No standalone video files found for this session")
    
    # Check NWB for behavioral data
    logger.info("Checking NWB file for behavioral data...")
    
    try:
        cache = get_cache(cache_dir)
        session = load_session(cache, args.session_id)
        
        eye_df = get_eye_tracking(session)
        
        if eye_df is not None:
            duration_min = (eye_df.index.max() - eye_df.index.min()) / 60
            logger.info(f"NWB contains eye_tracking data ({len(eye_df):,} samples, {duration_min:.1f} min)")
        else:
            logger.info("NWB does NOT contain eye tracking data")
            
    except Exception as e:
        logger.warning(f"Could not check NWB file: {e}")
    
    # Print recommendations
    if manifest["recommendations"]:
        print("\nRecommendations:")
        for rec in manifest["recommendations"]:
            print(f"  {rec}")
    
    if not videos:
        print("\n" + "=" * 60)
        print("The VBN public dataset includes processed eye tracking")
        print("(pupil/eye position time series) but not raw behavior video files.")
        print("Use `preview_video.py --source eye-tracking` to visualize the available data.")
        print("=" * 60)
    
    logger.info(f"Manifest saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
