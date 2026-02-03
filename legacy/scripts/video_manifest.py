#!/usr/bin/env python
"""Scan cache for video files and generate manifest.

This script searches the cache directory for video files associated with
a session and reports what's available.

Usage:
    python scripts/video_manifest.py SESSION_ID [--cache-dir PATH] [--output PATH]
           [--video-dir PATH ...] [--camera body|eye|face|any] [--stage symlink|copy|none]

Example:
    python scripts/video_manifest.py 1055240613
    python scripts/video_manifest.py 1055240613 --output manifest.json
"""

import argparse
import json
from pathlib import Path

from vbn.cache import get_cache, get_session_nwb_path
from vbn.config import (
    get_cache_dir,
    get_outputs_dir,
    get_session_output_dir,
    get_video_dirs,
    get_video_preferred_camera,
    get_video_stage,
)
from vbn.io import load_session, get_eye_tracking
from vbn.video import (
    discover_videos,
    generate_video_manifest,
    resolve_video_path,
    stage_video,
    validate_video_open,
)
from vbn.utils import setup_logging, print_diagnostic


def main():
    default_camera = get_video_preferred_camera()
    default_stage = get_video_stage()

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
    parser.add_argument(
        "--video-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional video search root (repeatable)"
    )
    parser.add_argument(
        "--camera",
        choices=["body", "eye", "face", "any"],
        default=default_camera,
        help=f"Which camera to use when selecting videos (default: {default_camera})"
    )
    parser.add_argument(
        "--stage",
        choices=["symlink", "copy", "none"],
        default=default_stage,
        help=f"Stage selected video into outputs (default: {default_stage})"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine cache directory
    cache_dir = args.cache_dir or get_cache_dir()
    outputs_dir = get_outputs_dir()
    video_dirs = [*args.video_dir, *get_video_dirs()]
    logger.info(f"Searching {cache_dir} for session {args.session_id}...")
    
    # Search for video files
    videos = discover_videos(
        cache_dir=cache_dir,
        session_id=args.session_id,
        search_dirs=video_dirs,
        include_outputs_dir=True,
        outputs_dir=outputs_dir,
    )
    
    # Generate manifest
    output_path = args.output
    if output_path is None:
        output_dir = get_session_output_dir(args.session_id)
        output_path = output_dir / "video_manifest.json"
    
    manifest = generate_video_manifest(
        cache_dir=cache_dir,
        session_id=args.session_id,
        output_path=output_path,
        search_dirs=video_dirs,
        include_outputs_dir=True,
        outputs_dir=outputs_dir,
    )

    selected_video = resolve_video_path(
        session_id=args.session_id,
        camera=args.camera,
        video_dirs=video_dirs,
        cache_dir=cache_dir,
        outputs_dir=outputs_dir,
        include_outputs_dir=True,
    )

    staged_video = None
    validation = None
    if selected_video is not None:
        validation = validate_video_open(selected_video)
        if validation.get("ok", False) and args.stage != "none":
            staged_video = stage_video(
                selected_video,
                session_id=args.session_id,
                method=args.stage,
                outputs_dir=outputs_dir,
                selected_camera=args.camera,
                validation=validation,
            )
    
    # Print results
    print_diagnostic("Video Search Results", {
        "Search roots": manifest.get("search_roots", []),
        "Session ID": args.session_id,
        "Video files found": manifest["video_files_found"],
        "Camera": args.camera,
        "Selected video": selected_video or "(none)",
        "Staged video": staged_video or ("(none)" if selected_video else "(n/a)"),
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
