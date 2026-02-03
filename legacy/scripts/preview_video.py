#!/usr/bin/env python
"""Generate video preview from session data.

This script creates a preview video either from raw video files (if available)
or from eye tracking data (default fallback).

Usage:
    python scripts/preview_video.py SESSION_ID [--source video|eye-tracking]
           [--start SEC] [--duration SEC] [--output PATH] [--cache-dir PATH]
           [--video-path PATH] [--video-dir PATH ...] [--camera body|eye|face|any]
           [--stage symlink|copy|none]

Example:
    python scripts/preview_video.py 1055240613
    python scripts/preview_video.py 1055240613 --duration 15 --output preview.mp4
    python scripts/preview_video.py 1055240613 --source eye-tracking
"""

import argparse
from pathlib import Path

from vbn.cache import get_cache
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
    preview_video_file,
    preview_eye_tracking_as_video,
    resolve_video_path,
    stage_video,
    validate_video_open,
)
from vbn.utils import setup_logging, print_diagnostic


def main():
    default_camera = get_video_preferred_camera()
    default_stage = get_video_stage()

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
        "--video-path",
        type=Path,
        help="Explicit video path override"
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

    outputs_dir = get_outputs_dir()
    video_dirs = [*args.video_dir, *get_video_dirs()]

    use_video = False
    video_path: Path | None = None
    
    if args.source == "video":
        use_video = True
        video_path = args.video_path or resolve_video_path(
            session_id=args.session_id,
            camera=args.camera,
            video_dirs=video_dirs,
            cache_dir=cache_dir,
            outputs_dir=outputs_dir,
        )
        if video_path is None:
            logger.error(
                f"No video found for session {args.session_id} (camera={args.camera}). "
                "Set VBN_VIDEO_DIRS or pass --video-path/--video-dir. "
                "Try --camera any if camera tags are missing."
            )
            return 1
        
    elif args.source == "auto":
        video_path = args.video_path or resolve_video_path(
            session_id=args.session_id,
            camera=args.camera,
            video_dirs=video_dirs,
            cache_dir=cache_dir,
            outputs_dir=outputs_dir,
        )
        if video_path:
            logger.info(f"Found video file: {video_path}")
            use_video = True
        else:
            logger.info("No raw video files found, using eye tracking data")
            use_video = False
    else:
        # eye-tracking
        use_video = False

    # Determine output path (if not provided)
    if args.output:
        output_path = args.output
    else:
        output_dir = get_session_output_dir(args.session_id)
        output_path = output_dir / ("preview_video.mp4" if use_video else "preview_eye_tracking.mp4")
    
    # Generate preview
    if use_video:
        assert video_path is not None
        validation = validate_video_open(video_path)
        if not validation.get("ok", False):
            logger.error(f"Video found but could not be opened: {validation.get('error')}")
            logger.error(f"Path: {video_path}")
            return 1

        if args.stage != "none":
            video_path = stage_video(
                video_path,
                session_id=args.session_id,
                method=args.stage,
                outputs_dir=outputs_dir,
                selected_camera=args.camera,
                validation=validation,
            )

        print_diagnostic("Video Preview Configuration", {
            "Session ID": args.session_id,
            "Camera": args.camera,
            "Cache dir": cache_dir,
            "Video dirs": [str(p) for p in video_dirs] if video_dirs else "(none)",
            "Selected video": video_path,
            "Stage": args.stage,
            "Output": output_path,
        })

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
