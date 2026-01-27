#!/usr/bin/env python
"""Run DeepLabCut inference and convert to standard format.

This script runs DLC pose estimation on a video and converts the
output to the standardized pose format.

Usage:
    python scripts/run_dlc_inference.py VIDEO_PATH CONFIG_PATH
           [--output-dir PATH] [--session-id ID] [--timestamps-csv PATH]

Example:
    python scripts/run_dlc_inference.py video.mp4 dlc_project/config.yaml --session-id 1055240613
"""

import argparse
from pathlib import Path

import pandas as pd

from vbn.pose import run_dlc_inference, convert_dlc_to_standard, save_pose_outputs
from vbn.config import get_outputs_dir
from vbn.utils import setup_logging, print_diagnostic


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepLabCut inference and convert to standard format"
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to input video file"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to DLC project config.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        required=True,
        help="Session ID to include in output"
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
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_outputs_dir() / str(args.session_id) / "pose"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gpu = args.gpu if args.gpu >= 0 else None
    
    print_diagnostic("DLC Inference Configuration", {
        "Video": args.video_path,
        "Config": args.config_path,
        "Session ID": args.session_id,
        "GPU": gpu if gpu is not None else "CPU",
        "Output directory": output_dir,
    })
    
    # Run inference
    logger.info(f"Running DLC inference on {args.video_path}")
    dlc_output = run_dlc_inference(
        args.video_path,
        args.config_path,
        output_dir,
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
    pose_df = convert_dlc_to_standard(
        dlc_output,
        args.session_id,
        timestamps=timestamps,
        fps=args.fps
    )
    
    # Save
    pose_output = output_dir / "pose_dlc.csv"
    save_pose_outputs(pose_df, pose_output)
    
    print_diagnostic("Inference Results", {
        "Total detections": len(pose_df),
        "Unique frames": pose_df["frame_idx"].nunique(),
        "Bodyparts detected": pose_df["node"].nunique(),
        "DLC output": dlc_output,
        "Standard output": pose_output,
    })
    
    logger.info(f"Pose data saved to: {pose_output}")
    return 0


if __name__ == "__main__":
    exit(main())
