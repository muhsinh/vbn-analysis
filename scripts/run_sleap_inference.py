#!/usr/bin/env python
"""Run SLEAP inference and convert to standard format.

This script runs SLEAP pose estimation on a video and converts the
output to the standardized pose format.

Usage:
    python scripts/run_sleap_inference.py VIDEO_PATH MODEL_PATH
           [--output PATH] [--batch-size N] [--session-id ID]
           [--timestamps-csv PATH]

Example:
    python scripts/run_sleap_inference.py video.mp4 models/sleap/model --session-id 1055240613
"""

import argparse
from pathlib import Path

import pandas as pd

from vbn.pose import run_sleap_inference, convert_sleap_to_standard, save_pose_outputs
from vbn.config import get_outputs_dir
from vbn.utils import setup_logging, print_diagnostic


def main():
    parser = argparse.ArgumentParser(
        description="Run SLEAP inference and convert to standard format"
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to input video file"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to trained SLEAP model"
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
        "Video": args.video_path,
        "Model": args.model_path,
        "Session ID": args.session_id,
        "Batch size": args.batch_size,
        "GPU": gpu if gpu is not None else "CPU",
        "Output": pose_output,
    })
    
    # Run inference
    logger.info(f"Running SLEAP inference on {args.video_path}")
    run_sleap_inference(
        args.video_path,
        args.model_path,
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
