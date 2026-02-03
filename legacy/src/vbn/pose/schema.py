"""Standardized pose output schema for VBN analysis.

Defines a common format for pose estimation outputs from any backend
(SLEAP, DeepLabCut, etc.) to enable consistent downstream analysis.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np


@dataclass
class PoseOutput:
    """Standardized pose output for a single keypoint detection.
    
    Attributes:
        session_id: VBN session identifier
        frame_idx: Frame index in source video
        timestamp_sec: Timestamp in seconds (aligned to session clock)
        node: Keypoint/bodypart name (e.g., "nose", "left_ear", "tail_base")
        x: X coordinate in pixels
        y: Y coordinate in pixels
        score: Detection confidence score (0-1)
    """
    session_id: int
    frame_idx: int
    timestamp_sec: float
    node: str
    x: float
    y: float
    score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Standard column names for pose DataFrame
POSE_SCHEMA_COLUMNS = [
    "session_id",
    "frame_idx", 
    "timestamp_sec",
    "node",
    "x",
    "y",
    "score"
]

# Expected dtypes
POSE_SCHEMA_DTYPES = {
    "session_id": "int64",
    "frame_idx": "int64",
    "timestamp_sec": "float64",
    "node": "object",  # string
    "x": "float64",
    "y": "float64",
    "score": "float64",
}


def validate_pose_schema(df: pd.DataFrame, raise_on_error: bool = True) -> bool:
    """Validate that a DataFrame conforms to the pose schema.
    
    Args:
        df: DataFrame to validate
        raise_on_error: If True, raise ValueError on validation failure
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If validation fails and raise_on_error is True
    """
    errors = []
    
    # Check required columns
    missing_cols = set(POSE_SCHEMA_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check dtypes for present columns
    for col, expected_dtype in POSE_SCHEMA_DTYPES.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            # Allow some flexibility in numeric types
            if expected_dtype == "int64" and actual_dtype not in ["int64", "int32", "int"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected integer")
            elif expected_dtype == "float64" and actual_dtype not in ["float64", "float32", "float"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected float")
    
    # Check value ranges
    if "score" in df.columns:
        if df["score"].min() < 0 or df["score"].max() > 1:
            errors.append("Score values must be between 0 and 1")
    
    if "frame_idx" in df.columns:
        if df["frame_idx"].min() < 0:
            errors.append("Frame indices must be non-negative")
    
    if errors:
        if raise_on_error:
            raise ValueError("Pose schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        return False
    
    return True


def load_pose_outputs(path: Path | str) -> pd.DataFrame:
    """Load pose outputs from CSV or H5 file.
    
    Args:
        path: Path to pose output file (.csv or .h5)
        
    Returns:
        DataFrame with standard pose schema
        
    Raises:
        ValueError: If file format is unsupported or schema validation fails
    """
    path = Path(path)
    
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".h5", ".hdf5"]:
        df = pd.read_hdf(path, key="pose")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    validate_pose_schema(df)
    return df


def save_pose_outputs(
    df: pd.DataFrame,
    path: Path | str,
    validate: bool = True
) -> None:
    """Save pose outputs to CSV or H5 file.
    
    Args:
        df: DataFrame with pose data
        path: Output path (.csv or .h5)
        validate: If True, validate schema before saving
        
    Raises:
        ValueError: If validation fails
    """
    path = Path(path)
    
    if validate:
        validate_pose_schema(df)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in [".h5", ".hdf5"]:
        df.to_hdf(path, key="pose", mode="w")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def create_empty_pose_df() -> pd.DataFrame:
    """Create an empty DataFrame with the standard pose schema.
    
    Returns:
        Empty DataFrame with correct columns and dtypes
    """
    return pd.DataFrame({
        col: pd.Series(dtype=dtype)
        for col, dtype in POSE_SCHEMA_DTYPES.items()
    })


def merge_pose_outputs(
    dfs: list[pd.DataFrame],
    validate: bool = True
) -> pd.DataFrame:
    """Merge multiple pose output DataFrames.
    
    Args:
        dfs: List of pose DataFrames to merge
        validate: If True, validate merged result
        
    Returns:
        Combined DataFrame sorted by session_id, frame_idx, node
    """
    if not dfs:
        return create_empty_pose_df()
    
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["session_id", "frame_idx", "node"]).reset_index(drop=True)
    
    if validate:
        validate_pose_schema(merged)
    
    return merged


def compute_pose_velocities(
    df: pd.DataFrame,
    fps: float | None = None
) -> pd.DataFrame:
    """Compute velocities for each node from pose data.
    
    Args:
        df: Pose DataFrame with standard schema
        fps: Frames per second (for velocity units). If None, uses frame differences.
        
    Returns:
        DataFrame with additional velocity columns (vx, vy, speed)
    """
    result = df.copy()
    
    # Group by session and node
    result = result.sort_values(["session_id", "node", "frame_idx"])
    
    # Compute differences
    result["dx"] = result.groupby(["session_id", "node"])["x"].diff()
    result["dy"] = result.groupby(["session_id", "node"])["y"].diff()
    
    if fps is not None:
        result["vx"] = result["dx"] * fps
        result["vy"] = result["dy"] * fps
    else:
        result["vx"] = result["dx"]
        result["vy"] = result["dy"]
    
    result["speed"] = np.sqrt(result["vx"]**2 + result["vy"]**2)
    
    return result
