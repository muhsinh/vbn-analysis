#!/usr/bin/env python
"""Export frames and timestamps for labeling.

This script extracts frames from video files or renders them from eye tracking
data, saving PNG images and a timestamps.csv file.

Usage:
    python scripts/export_frames.py SESSION_ID [--source video|eye-tracking]
           [--n-frames N] [--every-n N] [--output-dir PATH]
           [--video-path PATH] [--video-dir PATH ...] [--camera body|eye|face|any]
           [--stage symlink|copy|none]

Example:
    python scripts/export_frames.py 1055240613
    python scripts/export_frames.py 1055240613 --n-frames 100
    python scripts/export_frames.py 1055240613 --source video --every-n 30
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
from vbn.video import resolve_video_path, stage_video, validate_video_open
from vbn.frames import (
    extract_frames_from_video,
    extract_frames_from_eye_tracking,
    export_timestamps_csv,
)
from vbn.utils import setup_logging, print_diagnostic


def main():
    default_camera = get_video_preferred_camera()
    default_stage = get_video_stage()

    parser = argparse.ArgumentParser(
        description="Export frames and timestamps for labeling"
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
        help="Data source for frames (default: auto)"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=100,
        help="Number of frames to extract (default: 100)"
    )
    parser.add_argument(
        "--every-n",
        type=int,
        help="Extract every Nth frame (overrides --n-frames)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for frames"
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
        "--start-sec",
        type=float,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        help="End time in seconds"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine directories
    cache_dir = args.cache_dir or get_cache_dir()
    outputs_dir = get_outputs_dir()
    video_dirs = [*args.video_dir, *get_video_dirs()]
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_session_output_dir(args.session_id) / "frames"
    
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
    
    # Extract frames
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

        logger.info(f"Extracting frames from video...")
        
        if args.every_n:
            frames_df = extract_frames_from_video(
                video_path,
                output_dir,
                every_n=args.every_n,
                max_frames=args.n_frames
            )
        else:
            frames_df = extract_frames_from_video(
                video_path,
                output_dir,
                max_frames=args.n_frames
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
        
        logger.info(f"Rendering {args.n_frames} frames from eye tracking data...")
        frames_df = extract_frames_from_eye_tracking(
            eye_df,
            output_dir,
            n_frames=args.n_frames,
            start_sec=args.start_sec,
            end_sec=args.end_sec
        )
    
    # Save timestamps
    timestamps_path = output_dir / "timestamps.csv"
    export_timestamps_csv(frames_df, timestamps_path)
    
    # Print summary
    print_diagnostic("Frame Export Summary", {
        "Session ID": args.session_id,
        "Output directory": output_dir,
        "Frames exported": len(frames_df),
        "Source": "video" if use_video else "eye-tracking",
        "Timestamps file": timestamps_path.name,
        "Video dirs": [str(p) for p in video_dirs] if video_dirs else "(none)",
        "Camera": args.camera,
        "Stage": args.stage if use_video else "(n/a)",
    })
    
    # Show columns in timestamps
    logger.info(f"timestamps.csv columns: {list(frames_df.columns)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
