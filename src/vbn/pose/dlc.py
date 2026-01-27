"""DeepLabCut pose estimation inference and conversion for VBN analysis."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..utils import setup_logging, ensure_dir
from .schema import POSE_SCHEMA_COLUMNS, validate_pose_schema


def run_dlc_inference(
    video_path: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    save_as_csv: bool = True,
    gpu: int | None = 0,
    use_shelve: bool = False
) -> Path:
    """Run DeepLabCut inference on a video.
    
    Uses deeplabcut.analyze_videos() API for inference.
    
    Args:
        video_path: Path to input video file
        config_path: Path to DLC project config.yaml
        output_dir: Directory for output files
        save_as_csv: Also save results as CSV
        gpu: GPU device ID (None for CPU)
        use_shelve: Use shelve for storing data (DLC option)
        
    Returns:
        Path to output H5 file
        
    Raises:
        FileNotFoundError: If video or config not found
        RuntimeError: If inference fails
    """
    logger = setup_logging()
    
    video_path = Path(video_path)
    config_path = Path(config_path)
    output_dir = ensure_dir(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Set GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    try:
        import deeplabcut
    except ImportError:
        raise ImportError(
            "deeplabcut is required for DLC inference. "
            "Install with: pip install deeplabcut"
        )
    
    logger.info(f"Running DLC inference on {video_path}")
    
    try:
        deeplabcut.analyze_videos(
            str(config_path),
            [str(video_path)],
            destfolder=str(output_dir),
            save_as_csv=save_as_csv,
            dynamic=(False, 0.5, 10),  # Default dynamic cropping
        )
    except Exception as e:
        raise RuntimeError(f"DLC inference failed: {e}") from e
    
    # Find output file
    # DLC names outputs like: video_name + DLC_model_name + shuffle + .h5
    output_files = list(output_dir.glob(f"{video_path.stem}*.h5"))
    
    if not output_files:
        raise RuntimeError(f"No output file found in {output_dir}")
    
    # Return the most recent file
    output_path = max(output_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"DLC inference complete: {output_path}")
    return output_path


def convert_dlc_to_standard(
    dlc_output: Path | str,
    session_id: int,
    timestamps: pd.Series | np.ndarray | None = None,
    fps: float | None = None,
    scorer: str | None = None
) -> pd.DataFrame:
    """Convert DeepLabCut output to standard pose DataFrame.
    
    Args:
        dlc_output: Path to DLC output H5 file
        session_id: VBN session ID to include in output
        timestamps: Optional timestamps for each frame
        fps: Frames per second (used if timestamps not provided)
        scorer: Specific scorer to extract (None = use first)
        
    Returns:
        DataFrame with standard pose schema
    """
    logger = setup_logging()
    dlc_output = Path(dlc_output)
    
    if not dlc_output.exists():
        raise FileNotFoundError(f"DLC output not found: {dlc_output}")
    
    # Read DLC H5 file
    df_dlc = pd.read_hdf(dlc_output)
    
    # DLC uses MultiIndex columns: (scorer, bodypart, x/y/likelihood)
    if isinstance(df_dlc.columns, pd.MultiIndex):
        # Get scorer name
        scorers = df_dlc.columns.get_level_values(0).unique()
        if scorer is None:
            scorer = scorers[0]
        elif scorer not in scorers:
            raise ValueError(f"Scorer '{scorer}' not found. Available: {list(scorers)}")
        
        # Get bodyparts for this scorer
        bodyparts = df_dlc[scorer].columns.get_level_values(0).unique()
    else:
        raise ValueError("Unexpected DLC output format - expected MultiIndex columns")
    
    records = []
    
    for frame_idx in range(len(df_dlc)):
        # Calculate timestamp
        if timestamps is not None and frame_idx < len(timestamps):
            timestamp_sec = float(timestamps[frame_idx])
        elif fps:
            timestamp_sec = frame_idx / fps
        else:
            timestamp_sec = float(frame_idx)
        
        for bodypart in bodyparts:
            try:
                x = df_dlc.loc[frame_idx, (scorer, bodypart, "x")]
                y = df_dlc.loc[frame_idx, (scorer, bodypart, "y")]
                likelihood = df_dlc.loc[frame_idx, (scorer, bodypart, "likelihood")]
            except KeyError:
                continue
            
            # Skip NaN values
            if pd.isna(x) or pd.isna(y):
                continue
            
            records.append({
                "session_id": session_id,
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "node": bodypart,
                "x": float(x),
                "y": float(y),
                "score": float(likelihood) if not pd.isna(likelihood) else 0.0,
            })
    
    df = pd.DataFrame(records, columns=POSE_SCHEMA_COLUMNS)
    
    if len(df) > 0:
        validate_pose_schema(df)
        logger.info(f"Converted {len(df)} pose detections from DLC output")
    else:
        logger.warning("No pose detections found in DLC output")
    
    return df


def get_dlc_bodyparts(config_path: Path | str) -> list[str]:
    """Extract bodypart names from DLC project config.
    
    Args:
        config_path: Path to DLC config.yaml
        
    Returns:
        List of bodypart names
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    bodyparts = config.get("bodyparts", [])
    
    if isinstance(bodyparts, str):
        # Sometimes stored as multiline string
        bodyparts = [bp.strip() for bp in bodyparts.split("\n") if bp.strip()]
    
    return bodyparts


def get_dlc_scorer_name(config_path: Path | str) -> str:
    """Get the scorer name from DLC config.
    
    Args:
        config_path: Path to DLC config.yaml
        
    Returns:
        Scorer name string
    """
    config_path = Path(config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config.get("scorer", "unknown")


def filter_dlc_by_likelihood(
    df: pd.DataFrame,
    min_likelihood: float = 0.5
) -> pd.DataFrame:
    """Filter pose DataFrame by likelihood/score threshold.
    
    Args:
        df: Pose DataFrame with standard schema
        min_likelihood: Minimum score to keep
        
    Returns:
        Filtered DataFrame
    """
    return df[df["score"] >= min_likelihood].reset_index(drop=True)
