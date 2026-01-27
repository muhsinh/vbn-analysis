"""Video discovery and preview functionality for VBN analysis."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from .config import get_cache_dir, get_outputs_dir
from .utils import setup_logging, format_size, ensure_dir


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".h5", ".nwb"}


def discover_videos(
    cache_dir: Path | str | None = None,
    session_id: int | None = None
) -> list[dict[str, Any]]:
    """Recursively search cache_dir for video files.
    
    Args:
        cache_dir: Directory to search. Defaults to get_cache_dir()
        session_id: If provided, only find files containing this session ID
        
    Returns:
        List of dicts with: path, extension, size_mb, camera_type_guess
    """
    logger = setup_logging()
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []
    
    videos = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # Actual video formats
    
    logger.info(f"Searching {cache_dir} for video files...")
    
    for ext in video_extensions:
        for video_path in cache_dir.rglob(f"*{ext}"):
            # Filter by session ID if specified
            if session_id is not None:
                if str(session_id) not in video_path.name:
                    continue
            
            # Guess camera type from filename
            name_lower = video_path.name.lower()
            if "body" in name_lower or "side" in name_lower:
                camera_type = "body"
            elif "eye" in name_lower:
                camera_type = "eye"
            elif "face" in name_lower or "front" in name_lower:
                camera_type = "face"
            else:
                camera_type = "unknown"
            
            videos.append({
                "path": video_path,
                "extension": ext,
                "size_mb": video_path.stat().st_size / (1024 * 1024),
                "camera_type_guess": camera_type,
            })
    
    logger.info(f"Found {len(videos)} video files")
    return videos


def generate_video_manifest(
    cache_dir: Path | str | None = None,
    session_id: int | None = None,
    output_path: Path | str | None = None
) -> dict[str, Any]:
    """Generate manifest of video files in cache.
    
    Args:
        cache_dir: Directory to search
        session_id: Filter by session ID
        output_path: If provided, save manifest as JSON
        
    Returns:
        Manifest dict with search_path, files_found, recommendations
    """
    logger = setup_logging()
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
    
    videos = discover_videos(cache_dir, session_id)
    
    manifest = {
        "search_path": str(cache_dir),
        "session_id": session_id,
        "files_searched": sum(1 for _ in cache_dir.rglob("*") if _.is_file()),
        "video_files_found": len(videos),
        "files": [
            {
                "path": str(v["path"]),
                "extension": v["extension"],
                "size_mb": round(v["size_mb"], 2),
                "camera_type": v["camera_type_guess"],
            }
            for v in videos
        ],
        "recommendations": [],
    }
    
    # Add recommendations
    if len(videos) == 0:
        manifest["recommendations"].append(
            "No standalone video files found. "
            "The VBN public dataset may not include raw behavior video files. "
            "Use eye tracking data instead: preview_video.py --source eye-tracking"
        )
    else:
        body_videos = [v for v in videos if v["camera_type_guess"] == "body"]
        if body_videos:
            manifest["recommendations"].append(
                f"Body camera video found: {body_videos[0]['path']}"
            )
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to: {output_path}")
    
    return manifest


def preview_video_file(
    video_path: Path | str,
    start_sec: float = 0,
    duration_sec: float = 15,
    output_path: Path | str | None = None,
    display: bool = True
) -> None:
    """Play or save preview clip from video file.
    
    Args:
        video_path: Path to video file
        start_sec: Start time in seconds
        duration_sec: Duration of preview in seconds
        output_path: If provided, save clip to this path instead of displaying
        display: If True and no output_path, display video in window
    """
    logger = setup_logging()
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    
    start_frame = int(start_sec * fps)
    end_frame = min(int((start_sec + duration_sec) * fps), total_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Output to file
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        logger.info(f"Preview saved to: {output_path}")
    
    # Display in window
    elif display:
        logger.info("Press 'q' to quit preview")
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Video Preview", frame)
            
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()
    
    cap.release()


def preview_eye_tracking_as_video(
    eye_tracking_df: pd.DataFrame,
    output_path: Path | str,
    fps: int = 30,
    duration_sec: float = 15,
    start_sec: float = 0,
    figsize: tuple[int, int] = (8, 6)
) -> None:
    """Render eye tracking data as animated video.
    
    Creates an animation showing pupil position and size over time.
    
    Args:
        eye_tracking_df: DataFrame with eye tracking data
        output_path: Path to save output video
        fps: Frames per second for output video
        duration_sec: Duration of preview in seconds
        start_sec: Start time in seconds
        figsize: Figure size in inches
    """
    logger = setup_logging()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    # Get time range
    timestamps = eye_tracking_df.index.values
    if len(timestamps) == 0:
        raise ValueError("Eye tracking DataFrame is empty")
    
    # Find frame indices for time range
    start_idx = np.searchsorted(timestamps, start_sec)
    end_time = start_sec + duration_sec
    end_idx = min(np.searchsorted(timestamps, end_time), len(timestamps))
    
    if end_idx <= start_idx:
        raise ValueError(f"No data in time range {start_sec} - {end_time}")
    
    # Subsample to target fps
    source_fps = len(timestamps) / (timestamps[-1] - timestamps[0])
    step = max(1, int(source_fps / fps))
    
    indices = range(start_idx, end_idx, step)
    n_frames = len(list(indices))
    
    logger.info(f"Rendering {n_frames} frames at {fps} fps...")
    
    # Extract data columns
    df = eye_tracking_df.iloc[start_idx:end_idx:step].copy()
    
    # Determine which columns exist
    pupil_x_col = "pupil_center_x" if "pupil_center_x" in df.columns else None
    pupil_y_col = "pupil_center_y" if "pupil_center_y" in df.columns else None
    pupil_area_col = "pupil_area" if "pupil_area" in df.columns else None
    
    if pupil_x_col is None or pupil_y_col is None:
        raise ValueError("Eye tracking data must have pupil_center_x and pupil_center_y columns")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Eye Tracking Preview")
    
    # Initialize plots
    ax_scatter = axes[0, 0]
    ax_scatter.set_xlabel("Pupil X")
    ax_scatter.set_ylabel("Pupil Y")
    ax_scatter.set_title("Pupil Position")
    
    # Set axis limits based on data range
    x_data = df[pupil_x_col].dropna()
    y_data = df[pupil_y_col].dropna()
    
    if len(x_data) > 0 and len(y_data) > 0:
        ax_scatter.set_xlim(x_data.min() - 10, x_data.max() + 10)
        ax_scatter.set_ylim(y_data.min() - 10, y_data.max() + 10)
    
    scatter = ax_scatter.scatter([], [], s=50, c="blue", alpha=0.7)
    current_point = ax_scatter.scatter([], [], s=200, c="red", marker="x")
    
    # Time series plots
    ax_x = axes[0, 1]
    ax_x.set_ylabel("Pupil X")
    ax_x.set_xlabel("Time (s)")
    line_x, = ax_x.plot([], [], "b-", linewidth=0.5)
    point_x = ax_x.axvline(x=0, color="red", linewidth=2)
    
    ax_y = axes[1, 0]
    ax_y.set_ylabel("Pupil Y")
    ax_y.set_xlabel("Time (s)")
    line_y, = ax_y.plot([], [], "b-", linewidth=0.5)
    point_y = ax_y.axvline(x=0, color="red", linewidth=2)
    
    ax_area = axes[1, 1]
    ax_area.set_ylabel("Pupil Area")
    ax_area.set_xlabel("Time (s)")
    line_area, = ax_area.plot([], [], "g-", linewidth=0.5)
    point_area = ax_area.axvline(x=0, color="red", linewidth=2)
    
    # Pre-compute all time series data
    times = df.index.values - df.index.values[0]
    x_vals = df[pupil_x_col].values
    y_vals = df[pupil_y_col].values
    area_vals = df[pupil_area_col].values if pupil_area_col else np.zeros(len(df))
    
    # Set time series limits
    ax_x.set_xlim(0, times[-1])
    ax_x.set_ylim(np.nanmin(x_vals) - 5, np.nanmax(x_vals) + 5)
    ax_y.set_xlim(0, times[-1])
    ax_y.set_ylim(np.nanmin(y_vals) - 5, np.nanmax(y_vals) + 5)
    ax_area.set_xlim(0, times[-1])
    ax_area.set_ylim(0, np.nanmax(area_vals) * 1.1 if np.any(~np.isnan(area_vals)) else 1)
    
    plt.tight_layout()
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        current_point.set_offsets(np.empty((0, 2)))
        line_x.set_data([], [])
        line_y.set_data([], [])
        line_area.set_data([], [])
        return scatter, current_point, line_x, line_y, line_area
    
    def animate(frame_idx):
        # Update scatter with trail
        trail_len = min(frame_idx + 1, 100)
        start = max(0, frame_idx - trail_len + 1)
        trail_x = x_vals[start:frame_idx + 1]
        trail_y = y_vals[start:frame_idx + 1]
        
        # Filter out NaN values
        mask = ~(np.isnan(trail_x) | np.isnan(trail_y))
        if np.any(mask):
            scatter.set_offsets(np.column_stack([trail_x[mask], trail_y[mask]]))
        
        # Current point
        if not np.isnan(x_vals[frame_idx]) and not np.isnan(y_vals[frame_idx]):
            current_point.set_offsets([[x_vals[frame_idx], y_vals[frame_idx]]])
        
        # Time series
        line_x.set_data(times[:frame_idx + 1], x_vals[:frame_idx + 1])
        line_y.set_data(times[:frame_idx + 1], y_vals[:frame_idx + 1])
        line_area.set_data(times[:frame_idx + 1], area_vals[:frame_idx + 1])
        
        # Update vertical lines
        current_time = times[frame_idx]
        point_x.set_xdata([current_time, current_time])
        point_y.set_xdata([current_time, current_time])
        point_area.set_xdata([current_time, current_time])
        
        return scatter, current_point, line_x, line_y, line_area, point_x, point_y, point_area
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(df), interval=1000 / fps, blit=True
    )
    
    # Save to file
    writer = FFMpegWriter(fps=fps, metadata={"title": "Eye Tracking Preview"})
    anim.save(str(output_path), writer=writer)
    
    plt.close(fig)
    logger.info(f"Eye tracking preview saved to: {output_path}")
