"""Frame extraction and timestamp export for VBN analysis."""

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import setup_logging, ensure_dir


def extract_frames_from_video(
    video_path: Path | str,
    output_dir: Path | str,
    frame_indices: list[int] | None = None,
    every_n: int | None = None,
    max_frames: int | None = None
) -> pd.DataFrame:
    """Extract frames from video file as PNG images.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_indices: Specific frame indices to extract
        every_n: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        
    Returns:
        DataFrame with frame_idx, timestamp_sec, filename columns
    """
    logger = setup_logging()
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "opencv-python is required to extract frames from raw videos. "
            "Install with: pip install opencv-python"
        ) from e
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {total_frames} frames @ {fps:.1f} fps")
    
    # Determine which frames to extract
    if frame_indices is not None:
        frames_to_extract = frame_indices
    elif every_n is not None:
        frames_to_extract = list(range(0, total_frames, every_n))
    else:
        # Default: extract 100 frames evenly spaced
        n_frames = min(100 if max_frames is None else max_frames, total_frames)
        frames_to_extract = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
    
    if max_frames is not None:
        frames_to_extract = frames_to_extract[:max_frames]
    
    logger.info(f"Extracting {len(frames_to_extract)} frames...")
    
    results = []
    
    for idx, frame_idx in enumerate(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame {frame_idx}")
            continue
        
        # Save frame
        filename = f"frame_{frame_idx:08d}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        # Calculate timestamp
        timestamp_sec = frame_idx / fps if fps > 0 else 0
        
        results.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "filename": filename,
        })
    
    cap.release()
    
    df = pd.DataFrame(results)
    logger.info(f"Extracted {len(df)} frames to {output_dir}")
    
    return df


def extract_frames_from_eye_tracking(
    eye_df: pd.DataFrame,
    output_dir: Path | str,
    n_frames: int = 100,
    render_size: tuple[int, int] = (256, 256),
    start_sec: float | None = None,
    end_sec: float | None = None
) -> pd.DataFrame:
    """Render eye tracking data as individual frame images.
    
    Creates scatter plot images showing pupil position for each frame.
    
    Args:
        eye_df: Eye tracking DataFrame
        output_dir: Directory to save rendered frames
        n_frames: Number of frames to render
        render_size: Image size (width, height) in pixels
        start_sec: Start time in seconds (optional)
        end_sec: End time in seconds (optional)
        
    Returns:
        DataFrame with frame_idx, timestamp_sec, filename, and eye tracking columns
    """
    logger = setup_logging()
    output_dir = ensure_dir(output_dir)
    
    if len(eye_df) == 0:
        raise ValueError("Eye tracking DataFrame is empty")
    
    # Get time range
    timestamps = eye_df.index.values
    
    if start_sec is not None:
        start_idx = np.searchsorted(timestamps, start_sec)
    else:
        start_idx = 0
    
    if end_sec is not None:
        end_idx = np.searchsorted(timestamps, end_sec)
    else:
        end_idx = len(timestamps)
    
    # Select evenly spaced frames
    indices = np.linspace(start_idx, end_idx - 1, n_frames, dtype=int)
    
    logger.info(f"Rendering {n_frames} eye tracking frames...")
    
    # Get data columns
    pupil_x_col = "pupil_center_x" if "pupil_center_x" in eye_df.columns else None
    pupil_y_col = "pupil_center_y" if "pupil_center_y" in eye_df.columns else None
    pupil_area_col = "pupil_area" if "pupil_area" in eye_df.columns else None
    
    if pupil_x_col is None or pupil_y_col is None:
        raise ValueError("Eye tracking data must have pupil_center_x and pupil_center_y columns")
    
    # Calculate axis limits from full data
    x_data = eye_df[pupil_x_col].dropna()
    y_data = eye_df[pupil_y_col].dropna()
    
    x_min, x_max = x_data.min() - 10, x_data.max() + 10
    y_min, y_max = y_data.min() - 10, y_data.max() + 10
    
    results = []
    dpi = 100
    figsize = (render_size[0] / dpi, render_size[1] / dpi)
    
    for i, idx in enumerate(indices):
        row = eye_df.iloc[idx]
        timestamp = timestamps[idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot trail (last 50 points)
        trail_start = max(0, idx - 50)
        trail = eye_df.iloc[trail_start:idx + 1]
        
        ax.scatter(
            trail[pupil_x_col], trail[pupil_y_col],
            c=np.linspace(0.2, 0.8, len(trail)),
            cmap="Blues", s=10, alpha=0.5
        )
        
        # Plot current point
        pupil_x = row[pupil_x_col]
        pupil_y = row[pupil_y_col]
        
        if not (np.isnan(pupil_x) or np.isnan(pupil_y)):
            size = row[pupil_area_col] / 10 if pupil_area_col and not np.isnan(row[pupil_area_col]) else 100
            ax.scatter([pupil_x], [pupil_y], c="red", s=size, marker="o")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"t={timestamp:.2f}s")
        
        # Save frame
        filename = f"frame_{idx:08d}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        
        # Collect result
        result = {
            "frame_idx": idx,
            "timestamp_sec": float(timestamp),
            "filename": filename,
        }
        
        # Add eye tracking values
        if pupil_x_col:
            result["pupil_x"] = float(pupil_x) if not np.isnan(pupil_x) else None
        if pupil_y_col:
            result["pupil_y"] = float(pupil_y) if not np.isnan(pupil_y) else None
        if pupil_area_col and pupil_area_col in row:
            val = row[pupil_area_col]
            result["pupil_area"] = float(val) if not np.isnan(val) else None
        if "likely_blink" in row:
            result["likely_blink"] = bool(row["likely_blink"])
        
        results.append(result)
    
    df = pd.DataFrame(results)
    logger.info(f"Rendered {len(df)} frames to {output_dir}")
    
    return df


def sample_frames_for_labeling(
    timestamps_df: pd.DataFrame,
    n_samples: int = 50,
    strategy: Literal["uniform", "random", "behavior-change"] = "uniform"
) -> pd.DataFrame:
    """Select frames for labeling using specified sampling strategy.
    
    Args:
        timestamps_df: DataFrame with frame information (from extract_frames_*)
        n_samples: Number of frames to select
        strategy: Sampling strategy
            - "uniform": Evenly spaced throughout
            - "random": Random selection
            - "behavior-change": Select frames with high behavioral variance
            
    Returns:
        DataFrame with selected frame subset
    """
    logger = setup_logging()
    
    if len(timestamps_df) <= n_samples:
        logger.info(f"Requested {n_samples} samples but only {len(timestamps_df)} available")
        return timestamps_df.copy()
    
    n_samples = min(n_samples, len(timestamps_df))
    
    if strategy == "uniform":
        indices = np.linspace(0, len(timestamps_df) - 1, n_samples, dtype=int)
        selected = timestamps_df.iloc[indices]
        
    elif strategy == "random":
        selected = timestamps_df.sample(n=n_samples, random_state=42)
        selected = selected.sort_values("frame_idx")
        
    elif strategy == "behavior-change":
        # Select frames with high variance in behavioral measures
        # Use pupil area if available, otherwise fall back to uniform
        
        if "pupil_area" in timestamps_df.columns:
            # Calculate local variance
            pupil_area = timestamps_df["pupil_area"].fillna(timestamps_df["pupil_area"].median())
            
            # Rolling variance
            window = max(5, len(timestamps_df) // 50)
            variance = pupil_area.rolling(window=window, center=True).var().fillna(0)
            
            # Select frames with highest variance + some uniform samples
            n_high_var = n_samples // 2
            n_uniform = n_samples - n_high_var
            
            # High variance indices
            high_var_indices = variance.nlargest(n_high_var * 2).index.tolist()
            
            # Uniform indices
            uniform_indices = np.linspace(0, len(timestamps_df) - 1, n_uniform, dtype=int).tolist()
            
            # Combine and deduplicate
            all_indices = list(set(high_var_indices[:n_high_var] + uniform_indices))
            all_indices = sorted(all_indices)[:n_samples]
            
            selected = timestamps_df.iloc[all_indices]
        else:
            logger.warning("No pupil_area column found, falling back to uniform sampling")
            indices = np.linspace(0, len(timestamps_df) - 1, n_samples, dtype=int)
            selected = timestamps_df.iloc[indices]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    logger.info(f"Selected {len(selected)} frames using '{strategy}' strategy")
    return selected.reset_index(drop=True)


def export_timestamps_csv(
    frames_df: pd.DataFrame,
    output_path: Path | str
) -> None:
    """Write timestamps CSV file.
    
    Args:
        frames_df: DataFrame with frame information
        output_path: Path to save CSV
    """
    logger = setup_logging()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    frames_df.to_csv(output_path, index=False)
    logger.info(f"Timestamps saved to: {output_path}")


def copy_frames_for_labeling(
    timestamps_df: pd.DataFrame,
    source_dir: Path | str,
    output_dir: Path | str,
    selected_frames_csv: Path | str | None = None
) -> pd.DataFrame:
    """Copy selected frames to a labeling directory.
    
    Args:
        timestamps_df: DataFrame with selected frames (must have 'filename' column)
        source_dir: Directory containing source frames
        output_dir: Directory to copy frames to
        selected_frames_csv: Optional path to save selected frames CSV
        
    Returns:
        DataFrame with updated paths
    """
    logger = setup_logging()
    source_dir = Path(source_dir)
    output_dir = ensure_dir(output_dir)
    
    results = []
    
    for _, row in timestamps_df.iterrows():
        src_path = source_dir / row["filename"]
        dst_path = output_dir / row["filename"]
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            results.append(row.to_dict())
        else:
            logger.warning(f"Source file not found: {src_path}")
    
    df = pd.DataFrame(results)
    
    if selected_frames_csv:
        export_timestamps_csv(df, selected_frames_csv)
    
    logger.info(f"Copied {len(df)} frames to {output_dir}")
    return df
