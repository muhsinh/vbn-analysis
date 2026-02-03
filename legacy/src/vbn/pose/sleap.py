"""SLEAP pose estimation inference and conversion for VBN analysis."""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import setup_logging, ensure_dir
from .schema import POSE_SCHEMA_COLUMNS, validate_pose_schema


def run_sleap_inference(
    video_path: Path | str,
    model_path: Path | str,
    output_path: Path | str,
    batch_size: int = 4,
    gpu: int | None = 0
) -> Path:
    """Run SLEAP inference via CLI subprocess.
    
    Calls: sleap-track VIDEO -m MODEL -o OUTPUT --batch_size N
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained SLEAP model directory
        output_path: Path for output .slp or .h5 file
        batch_size: Batch size for inference
        gpu: GPU device ID (None for CPU)
        
    Returns:
        Path to output file
        
    Raises:
        FileNotFoundError: If video or model not found
        RuntimeError: If inference fails
    """
    logger = setup_logging()
    
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    ensure_dir(output_path.parent)
    
    # Build command
    cmd = [
        "sleap-track",
        str(video_path),
        "-m", str(model_path),
        "-o", str(output_path),
        "--batch_size", str(batch_size),
    ]
    
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    
    logger.info(f"Running SLEAP inference: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("SLEAP inference completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"SLEAP inference failed: {e.stderr}")
        raise RuntimeError(f"SLEAP inference failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "sleap-track command not found. "
            "Install SLEAP with: pip install sleap"
        )
    
    return output_path


def convert_sleap_to_standard(
    sleap_output: Path | str,
    session_id: int,
    timestamps: pd.Series | np.ndarray | None = None,
    fps: float | None = None
) -> pd.DataFrame:
    """Convert SLEAP output to standard pose DataFrame.
    
    Args:
        sleap_output: Path to SLEAP output file (.slp or .h5)
        session_id: VBN session ID to include in output
        timestamps: Optional timestamps for each frame (index -> timestamp_sec)
        fps: Frames per second (used if timestamps not provided)
        
    Returns:
        DataFrame with standard pose schema
    """
    logger = setup_logging()
    sleap_output = Path(sleap_output)
    
    if not sleap_output.exists():
        raise FileNotFoundError(f"SLEAP output not found: {sleap_output}")
    
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to read SLEAP outputs")
    
    records = []
    
    # Try to read as analysis H5 file
    if sleap_output.suffix in [".h5", ".hdf5"]:
        with h5py.File(sleap_output, "r") as f:
            # SLEAP analysis format
            if "tracks" in f:
                tracks = f["tracks"][:]  # shape: (frames, tracks, nodes, 2)
                node_names = [n.decode() for n in f["node_names"][:]]
                
                # Get confidence scores if available
                if "point_scores" in f:
                    scores = f["point_scores"][:]
                else:
                    scores = np.ones(tracks.shape[:-1])
                
                n_frames = tracks.shape[0]
                
                for frame_idx in range(n_frames):
                    # Calculate timestamp
                    if timestamps is not None:
                        if frame_idx < len(timestamps):
                            timestamp_sec = float(timestamps[frame_idx])
                        else:
                            timestamp_sec = frame_idx / fps if fps else float(frame_idx)
                    elif fps:
                        timestamp_sec = frame_idx / fps
                    else:
                        timestamp_sec = float(frame_idx)
                    
                    # Iterate over tracks (instances)
                    for track_idx in range(tracks.shape[1]):
                        for node_idx, node_name in enumerate(node_names):
                            x, y = tracks[frame_idx, track_idx, node_idx]
                            score = scores[frame_idx, track_idx, node_idx] if scores.ndim > 2 else scores[frame_idx, track_idx]
                            
                            # Skip NaN values
                            if np.isnan(x) or np.isnan(y):
                                continue
                            
                            records.append({
                                "session_id": session_id,
                                "frame_idx": frame_idx,
                                "timestamp_sec": timestamp_sec,
                                "node": node_name,
                                "x": float(x),
                                "y": float(y),
                                "score": float(score) if not np.isnan(score) else 0.0,
                            })
            else:
                logger.warning("Unexpected SLEAP H5 format, attempting generic read")
                raise ValueError("Unrecognized SLEAP H5 format")
    
    # Try to read as .slp file
    elif sleap_output.suffix == ".slp":
        try:
            import sleap
            labels = sleap.load_file(str(sleap_output))
            
            for frame_idx, lf in enumerate(labels):
                # Calculate timestamp
                if timestamps is not None and frame_idx < len(timestamps):
                    timestamp_sec = float(timestamps[frame_idx])
                elif fps:
                    timestamp_sec = frame_idx / fps
                else:
                    timestamp_sec = float(frame_idx)
                
                for instance in lf.instances:
                    for node in instance.nodes:
                        point = instance[node]
                        if point.x is not None and point.y is not None:
                            records.append({
                                "session_id": session_id,
                                "frame_idx": frame_idx,
                                "timestamp_sec": timestamp_sec,
                                "node": node.name,
                                "x": float(point.x),
                                "y": float(point.y),
                                "score": float(point.score) if point.score is not None else 1.0,
                            })
        except ImportError:
            raise ImportError(
                "sleap package is required to read .slp files. "
                "Install with: pip install sleap"
            )
    
    else:
        raise ValueError(f"Unsupported SLEAP output format: {sleap_output.suffix}")
    
    df = pd.DataFrame(records, columns=POSE_SCHEMA_COLUMNS)
    
    if len(df) > 0:
        validate_pose_schema(df)
        logger.info(f"Converted {len(df)} pose detections from SLEAP output")
    else:
        logger.warning("No pose detections found in SLEAP output")
    
    return df


def get_sleap_node_names(model_path: Path | str) -> list[str]:
    """Extract node/keypoint names from trained SLEAP model.
    
    Args:
        model_path: Path to SLEAP model directory
        
    Returns:
        List of node names
    """
    model_path = Path(model_path)
    
    # Look for training_config.json or skeleton info
    config_files = list(model_path.glob("*.json"))
    
    try:
        import json
        
        for config_file in config_files:
            with open(config_file) as f:
                config = json.load(f)
            
            # Check for skeleton info
            if "model" in config and "skeletons" in config["model"]:
                skeleton = config["model"]["skeletons"][0]
                return [node["name"] for node in skeleton["nodes"]]
            
            if "data" in config and "skeletons" in config["data"]:
                skeleton = config["data"]["skeletons"][0]
                return [node["name"] for node in skeleton["nodes"]]
    
    except Exception as e:
        setup_logging().warning(f"Could not read node names from model config: {e}")
    
    # Fallback: try to load the model
    try:
        import sleap
        model = sleap.load_model(str(model_path))
        return [node.name for node in model.skeleton.nodes]
    except Exception:
        pass
    
    return []
