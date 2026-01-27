#!/usr/bin/env python
"""Generate video preview from session data.

This script creates a preview video either from raw video files (if available)
or from eye tracking data (default fallback).

Usage:
    python scripts/preview_video.py SESSION_ID [--source video|eye-tracking]
           [--start SEC] [--duration SEC] [--output PATH] [--cache-dir PATH]

Example:
    python scripts/preview_video.py 1055240613
    python scripts/preview_video.py 1055240613 --duration 15 --output preview.mp4
    python scripts/preview_video.py 1055240613 --source eye-tracking
"""

import argparse
from pathlib import Path

from vbn.cache import get_cache
from vbn.config import get_cache_dir, get_session_output_dir
from vbn.io import load_session, get_eye_tracking
from vbn.video import discover_videos, preview_video_file, preview_eye_tracking_as_video
from vbn.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate video preview from session data"
    )
    parser.add_argument(
        "session_id",
        type=int,
        help="Ecephys session ID"
    )
    parser.add_argument(
        "--source",
        choices=["video", "eye-tracking", "auto"],
        default="auto",
        help="Data source for preview (default: auto)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0,
        help="Start time in seconds (default: 0)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15,
        help="Duration in seconds (default: 15)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for preview video"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: ~/data/vbn_cache)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output FPS for eye tracking preview (default: 30)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine cache directory
    cache_dir = args.cache_dir or get_cache_dir()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = get_session_output_dir(args.session_id)
        output_path = output_dir / "preview_eye_tracking.mp4"
    
    # Check for video files
    videos = discover_videos(cache_dir, args.session_id)
    
    use_video = False
    video_path = None
    
    if args.source == "video":
        if not videos:
            logger.error("No video files found. Use --source eye-tracking instead.")
            return 1
        use_video = True
        video_path = videos[0]["path"]
        
    elif args.source == "auto":
        if videos:
            logger.info(f"Found video file: {videos[0]['path']}")
            use_video = True
            video_path = videos[0]["path"]
        else:
            logger.info("No raw video files found, using eye tracking data")
            use_video = False
    
    # Generate preview
    if use_video:
        logger.info(f"Rendering video preview ({args.duration} sec)...")
        preview_video_file(
            video_path,
            start_sec=args.start,
            duration_sec=args.duration,
            output_path=output_path
        )
    else:
        # Use eye tracking data
        logger.info("Loading session for eye tracking data...")
        cache = get_cache(cache_dir)
        session = load_session(cache, args.session_id)
        
        eye_df = get_eye_tracking(session)
        
        if eye_df is None or len(eye_df) == 0:
            logger.error("No eye tracking data available for this session")
            return 1
        
        logger.info(f"Rendering eye tracking preview ({args.duration} sec)...")
        preview_eye_tracking_as_video(
            eye_df,
            output_path=output_path,
            fps=args.fps,
            duration_sec=args.duration,
            start_sec=args.start
        )
    
    logger.info(f"Preview saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
