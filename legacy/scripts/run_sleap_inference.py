#!/usr/bin/env python
"""Run SLEAP inference and convert to standard format.

This script runs SLEAP pose estimation on a video and converts the
output to the standardized pose format.

Usage:
    python scripts/run_sleap_inference.py VIDEO_PATH MODEL_PATH
           [--output PATH] [--batch-size N] [--session-id ID]
           [--timestamps-csv PATH]
           [--video-path PATH] [--video-dir PATH ...] [--camera body|eye|face|any]
           [--stage symlink|copy|none]

    # Alternate (resolve video from session_id + VBN_VIDEO_DIRS):
    python scripts/run_sleap_inference.py MODEL_PATH --session-id ID

Example:
    python scripts/run_sleap_inference.py video.mp4 models/sleap/model --session-id 1055240613
"""

import argparse
from pathlib import Path

import pandas as pd

from vbn.pose import run_sleap_inference, convert_sleap_to_standard, save_pose_outputs
from vbn.config import (
    get_outputs_dir,
    get_video_dirs,
    get_video_preferred_camera,
    get_video_stage,
)
from vbn.utils import setup_logging, print_diagnostic
from vbn.video import resolve_video_path, stage_video, validate_video_open


def main():
    default_camera = get_video_preferred_camera()
    default_stage = get_video_stage()

    parser = argparse.ArgumentParser(
        description="Run SLEAP inference and convert to standard format"
    )
    parser.add_argument(
        "video_or_model_path",
        type=Path,
        help="Either VIDEO_PATH (if providing both) or MODEL_PATH (if resolving video)"
    )
    parser.add_argument(
        "maybe_model_path",
        type=Path,
        nargs="?",
        help="MODEL_PATH (required if first arg is VIDEO_PATH)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for pose data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        required=True,
        help="Session ID to include in output"
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
        "--timestamps-csv",
        type=Path,
        help="CSV with frame timestamps mapping"
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Video FPS (used if timestamps not provided)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()

    # Backward compatible positional parsing:
    # - If two positionals are provided: VIDEO_PATH MODEL_PATH
    # - If one positional is provided: MODEL_PATH (video resolved via session-id)
    if args.maybe_model_path is None:
        model_path = args.video_or_model_path
        positional_video_path: Path | None = None
    else:
        positional_video_path = args.video_or_model_path
        model_path = args.maybe_model_path

    video_dirs = [*args.video_dir, *get_video_dirs()]

    video_path = args.video_path or positional_video_path
    if video_path is None:
        video_path = resolve_video_path(
            session_id=args.session_id,
            camera=args.camera,
            video_dirs=video_dirs,
            cache_dir=None,
            outputs_dir=get_outputs_dir(),
        )
        if video_path is None:
            logger.error(
                f"No video found for session {args.session_id} (camera={args.camera}). "
                "Set VBN_VIDEO_DIRS or pass --video-path/--video-dir. "
                "Try --camera any if camera tags are missing."
            )
            return 1

    validation = validate_video_open(video_path)
    if not validation.get("ok", False):
        logger.error(f"Selected video could not be opened: {validation.get('error')}")
        logger.error(f"Path: {video_path}")
        return 1

    if args.stage != "none":
        video_path = stage_video(
            video_path,
            session_id=args.session_id,
            method=args.stage,
            outputs_dir=get_outputs_dir(),
            selected_camera=args.camera,
            validation=validation,
        )

    # Determine output paths
    output_dir = get_outputs_dir() / str(args.session_id) / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slp_output = output_dir / "sleap_raw.h5"
    
    if args.output:
        pose_output = args.output
    else:
        pose_output = output_dir / "pose_sleap.csv"
    
    gpu = args.gpu if args.gpu >= 0 else None
    
    print_diagnostic("SLEAP Inference Configuration", {
        "Video": video_path,
        "Model": model_path,
        "Session ID": args.session_id,
        "Batch size": args.batch_size,
        "GPU": gpu if gpu is not None else "CPU",
        "Output": pose_output,
        "Video dirs": [str(p) for p in video_dirs] if video_dirs else "(none)",
        "Camera": args.camera,
        "Stage": args.stage,
    })
    
    # Run inference
    logger.info(f"Running SLEAP inference on {video_path}")
    run_sleap_inference(
        video_path,
        model_path,
        slp_output,
        batch_size=args.batch_size,
        gpu=gpu
    )
    
    # Load timestamps if provided
    timestamps = None
    if args.timestamps_csv:
        logger.info(f"Loading timestamps from {args.timestamps_csv}")
        ts_df = pd.read_csv(args.timestamps_csv)
        timestamps = ts_df["timestamp_sec"]
    
    # Convert to standard format
    logger.info("Converting to standard pose format")
    pose_df = convert_sleap_to_standard(
        slp_output,
        args.session_id,
        timestamps=timestamps,
        fps=args.fps
    )
    
    # Save
    save_pose_outputs(pose_df, pose_output)
    
    print_diagnostic("Inference Results", {
        "Total detections": len(pose_df),
        "Unique frames": pose_df["frame_idx"].nunique(),
        "Nodes detected": pose_df["node"].nunique(),
        "Output file": pose_output,
    })
    
    logger.info(f"Pose data saved to: {pose_output}")
    return 0


if __name__ == "__main__":
    exit(main())
