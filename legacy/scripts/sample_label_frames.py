#!/usr/bin/env python
"""Sample frames for labeling from exported frame set.

This script selects a subset of frames from previously exported frames
using various sampling strategies, then copies them to a labeling directory.

Usage:
    python scripts/sample_label_frames.py SESSION_ID [--n-samples N]
           [--strategy uniform|random|behavior-change] [--output-dir PATH]

Example:
    python scripts/sample_label_frames.py 1055240613
    python scripts/sample_label_frames.py 1055240613 --n-samples 50 --strategy behavior-change
"""

import argparse
from pathlib import Path

import pandas as pd

from vbn.config import get_session_output_dir
from vbn.frames import sample_frames_for_labeling, copy_frames_for_labeling
from vbn.utils import setup_logging, print_diagnostic


def main():
    parser = argparse.ArgumentParser(
        description="Sample frames for labeling from exported frame set"
    )
    parser.add_argument(
        "session_id",
        type=int,
        help="Ecephys session ID"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of frames to sample (default: 50)"
    )
    parser.add_argument(
        "--strategy",
        choices=["uniform", "random", "behavior-change"],
        default="behavior-change",
        help="Sampling strategy (default: behavior-change)"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory containing exported frames (default: outputs/SESSION/frames)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for labeled frames (default: outputs/SESSION/labeling)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine directories
    session_dir = get_session_output_dir(args.session_id)
    
    if args.frames_dir:
        frames_dir = args.frames_dir
    else:
        frames_dir = session_dir / "frames"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = session_dir / "labeling"
    
    # Load timestamps
    timestamps_path = frames_dir / "timestamps.csv"
    
    if not timestamps_path.exists():
        logger.error(f"Timestamps file not found: {timestamps_path}")
        logger.error("Run export_frames.py first to generate frames")
        return 1
    
    logger.info(f"Loading timestamps from {timestamps_path}")
    timestamps_df = pd.read_csv(timestamps_path)
    
    logger.info(f"Found {len(timestamps_df)} frames to sample from")
    
    # Sample frames
    logger.info(f"Sampling {args.n_samples} frames (strategy: {args.strategy})")
    selected_df = sample_frames_for_labeling(
        timestamps_df,
        n_samples=args.n_samples,
        strategy=args.strategy
    )
    
    # Copy frames to labeling directory
    logger.info(f"Copying selected frames to {output_dir}")
    selected_csv = output_dir / "selected_frames.csv"
    
    final_df = copy_frames_for_labeling(
        selected_df,
        source_dir=frames_dir,
        output_dir=output_dir,
        selected_frames_csv=selected_csv
    )
    
    # Print summary
    print_diagnostic("Frame Sampling Summary", {
        "Session ID": args.session_id,
        "Source frames": len(timestamps_df),
        "Sampled frames": len(final_df),
        "Sampling strategy": args.strategy,
        "Output directory": output_dir,
        "Selected frames CSV": selected_csv.name,
    })
    
    # Show frame range
    if len(final_df) > 0:
        min_time = final_df["timestamp_sec"].min()
        max_time = final_df["timestamp_sec"].max()
        logger.info(f"Time range: {min_time:.2f}s - {max_time:.2f}s")
    
    logger.info(f"Frames ready for labeling in: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
