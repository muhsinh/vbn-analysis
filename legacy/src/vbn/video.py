"""Video discovery and preview functionality for VBN analysis."""

from __future__ import annotations

import json
import os
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import get_cache_dir, get_outputs_dir, get_session_output_dir
from .utils import setup_logging, format_size, ensure_dir


VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

CAMERA_CHOICES = ("body", "eye", "face", "any")
STAGE_CHOICES = ("symlink", "copy", "none")

_EXT_PREFERENCE = {
    ".mp4": 0,
    ".mov": 1,
    ".avi": 2,
    ".mkv": 3,
}

_CAMERA_KEYWORDS = {
    "body": ("body", "side", "behavior"),
    "eye": ("eye", "pupil"),
    "face": ("face", "front"),
}


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _guess_camera_type(path: Path) -> str:
    s = "/".join(path.parts).lower()
    for camera, keywords in _CAMERA_KEYWORDS.items():
        if any(k in s for k in keywords):
            return camera
    return "unknown"


def _iter_video_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            suffix = Path(name).suffix.lower()
            if suffix in extensions:
                yield Path(dirpath) / name


def discover_videos(
    cache_dir: Path | str | None = None,
    session_id: int | None = None,
    search_dirs: Sequence[Path | str] | None = None,
    include_outputs_dir: bool = False,
    outputs_dir: Path | str | None = None,
    camera: str | None = None,
    extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Recursively search cache_dir for video files.
    
    Args:
        cache_dir: Directory to search. Defaults to get_cache_dir()
        session_id: If provided, only find files containing this session ID (in full path)
        search_dirs: Additional roots to search (searched before cache_dir)
        include_outputs_dir: If True, also search outputs_dir (or get_outputs_dir())
        outputs_dir: Outputs root to search when include_outputs_dir=True
        camera: If provided, filter to this camera guess ("body"|"eye"|"face"|"any")
        extensions: File extensions to consider (default: common video extensions)
        
    Returns:
        List of dicts with: path, extension, size_mb, camera_type_guess
    """
    logger = setup_logging()

    if extensions is None:
        extensions = set(VIDEO_FILE_EXTENSIONS)
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    roots: list[Path] = []
    if search_dirs:
        roots.extend(Path(p).expanduser() for p in search_dirs)

    if cache_dir is None:
        roots.append(get_cache_dir())
    else:
        roots.append(Path(cache_dir).expanduser())

    if include_outputs_dir:
        if outputs_dir is None:
            roots.append(get_outputs_dir())
        else:
            roots.append(Path(outputs_dir).expanduser())

    roots = _dedupe_paths([p.resolve() if p.exists() else p for p in roots])

    videos = []

    if camera is not None:
        camera = camera.strip().lower()
        if camera not in CAMERA_CHOICES:
            raise ValueError(f"camera must be one of {CAMERA_CHOICES}, got: {camera}")

    for root in roots:
        if not root.exists():
            logger.warning(f"Video search root does not exist: {root}")
            continue
        logger.info(f"Searching {root} for video files...")

        for video_path in _iter_video_files(root, extensions):
            # Filter by session ID if specified (match full path, not just basename)
            if session_id is not None and str(session_id) not in str(video_path):
                continue

            camera_type = _guess_camera_type(video_path)
            if camera and camera != "any" and camera_type != camera:
                continue

            ext = video_path.suffix.lower()
            try:
                size_mb = video_path.stat().st_size / (1024 * 1024)
            except FileNotFoundError:
                continue

            videos.append(
                {
                    "path": video_path,
                    "extension": ext,
                    "size_mb": size_mb,
                    "camera_type_guess": camera_type,
                }
            )
    
    logger.info(f"Found {len(videos)} video files")
    return videos


def generate_video_manifest(
    cache_dir: Path | str | None = None,
    session_id: int | None = None,
    output_path: Path | str | None = None,
    search_dirs: Sequence[Path | str] | None = None,
    include_outputs_dir: bool = False,
    outputs_dir: Path | str | None = None,
    camera: str | None = None,
) -> dict[str, Any]:
    """Generate manifest of video files in cache.
    
    Args:
        cache_dir: Directory to search
        session_id: Filter by session ID
        output_path: If provided, save manifest as JSON
        search_dirs: Additional roots to search
        include_outputs_dir: Also search outputs root
        outputs_dir: Outputs root override
        camera: Optional camera filter ("body"|"eye"|"face"|"any")
        
    Returns:
        Manifest dict with search_path, files_found, recommendations
    """
    logger = setup_logging()
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    if outputs_dir is None:
        outputs_dir = get_outputs_dir()
    else:
        outputs_dir = Path(outputs_dir)

    roots: list[Path] = []
    if search_dirs:
        roots.extend(Path(p) for p in search_dirs)
    roots.append(Path(cache_dir))
    if include_outputs_dir:
        roots.append(Path(outputs_dir))

    videos = discover_videos(
        cache_dir=cache_dir,
        session_id=session_id,
        search_dirs=search_dirs,
        include_outputs_dir=include_outputs_dir,
        outputs_dir=outputs_dir,
        camera=camera,
    )
    
    manifest = {
        "search_path": str(cache_dir),
        "search_roots": [str(p) for p in _dedupe_paths([p.expanduser() for p in roots])],
        "session_id": session_id,
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


def resolve_video_path(
    session_id: int,
    camera: str = "body",
    video_dirs: Sequence[Path | str] | None = None,
    cache_dir: Path | str | None = None,
    outputs_dir: Path | str | None = None,
    include_outputs_dir: bool = True,
) -> Path | None:
    """Resolve the 'best' video path for a session with deterministic selection."""
    camera = camera.strip().lower()
    if camera not in CAMERA_CHOICES:
        raise ValueError(f"camera must be one of {CAMERA_CHOICES}, got: {camera}")

    candidates = discover_videos(
        cache_dir=cache_dir,
        session_id=session_id,
        search_dirs=video_dirs,
        include_outputs_dir=include_outputs_dir,
        outputs_dir=outputs_dir,
        camera=None if camera == "any" else camera,
    )

    if not candidates:
        return None

    # If "any", allow all candidates; otherwise candidates are already filtered.
    def score(item: dict[str, Any]) -> tuple[int, int, float, str]:
        p: Path = item["path"]
        ext = p.suffix.lower()
        ext_rank = _EXT_PREFERENCE.get(ext, 99)
        size_mb = float(item.get("size_mb") or 0.0)
        cam_guess = str(item.get("camera_type_guess") or "unknown")
        camera_match = 1 if (camera != "any" and cam_guess == camera) else 0
        # Higher camera_match and size_mb are better, ext_rank lower is better.
        return (-camera_match, ext_rank, -size_mb, str(p))

    best = min(candidates, key=score)
    return Path(best["path"])


def validate_video_open(video_path: Path | str) -> dict[str, Any]:
    """Validate that OpenCV can open a video and read its first frame.

    Returns a dict containing 'ok' plus basic metadata when available.
    This function does not raise for unreadable videos; it reports errors in the dict.
    """
    p = Path(video_path)
    result: dict[str, Any] = {
        "path": str(p),
        "ok": False,
        "error": None,
        "fps": None,
        "frame_count": None,
        "width": None,
        "height": None,
        "can_read_first_frame": False,
    }

    if not p.exists():
        result["error"] = f"File not found: {p}"
        return result

    try:
        import cv2  # type: ignore
    except ImportError:
        result["error"] = "opencv-python is not installed (cv2 import failed)."
        return result

    try:
        cap = cv2.VideoCapture(str(p))
    except Exception as e:
        result["error"] = f"VideoCapture init failed: {e}"
        return result

    try:
        if not cap.isOpened():
            result["error"] = "OpenCV could not open the video."
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, _frame = cap.read()

        result.update(
            {
                "ok": True,
                "fps": float(fps) if fps else None,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "can_read_first_frame": bool(ok),
            }
        )

        if not ok:
            result["ok"] = False
            result["error"] = "Opened but failed to read the first frame."

        return result
    finally:
        cap.release()


def stage_video(
    video_path: Path | str,
    session_id: int,
    method: str = "symlink",
    *,
    outputs_dir: Path | str | None = None,
    selected_camera: str | None = None,
    validation: dict[str, Any] | None = None,
) -> Path:
    """Stage a video into outputs/<session_id>/videos and write videos.json.

    Returns the staged path (or the original path if method="none").
    """
    logger = setup_logging()

    method = method.strip().lower()
    if method not in STAGE_CHOICES:
        raise ValueError(f"method must be one of {STAGE_CHOICES}, got: {method}")

    src = Path(video_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")

    if outputs_dir is not None:
        session_root = Path(outputs_dir).expanduser() / str(session_id)
        ensure_dir(session_root)
    else:
        session_root = get_session_output_dir(session_id)

    videos_dir = ensure_dir(session_root / "videos")
    src_resolved = src.resolve()

    try:
        if src_resolved.is_relative_to(videos_dir.resolve()):
            staged = src_resolved
        else:
            staged = videos_dir / src.name
    except Exception:
        staged = videos_dir / src.name

    if method == "none":
        _write_videos_index(
            videos_dir,
            session_id=session_id,
            selected_camera=selected_camera,
            original_path=str(src_resolved),
            staged_path=str(src_resolved),
            method=method,
            validation=validation,
        )
        return src_resolved

    staged = _ensure_unique_destination(staged, src_resolved)

    if method == "symlink":
        try:
            if staged.exists() or staged.is_symlink():
                # If it already points to the same target, keep it.
                if staged.is_symlink() and staged.resolve() == src_resolved:
                    pass
                else:
                    raise FileExistsError(f"Destination already exists: {staged}")
            staged.symlink_to(src_resolved)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create symlink at {staged}: {e}. "
                "Use --stage copy or --stage none."
            ) from e
    elif method == "copy":
        if staged.exists():
            raise FileExistsError(f"Destination already exists: {staged}")
        shutil.copy2(src_resolved, staged)

    logger.info(f"Staged video: {staged} -> {src_resolved} ({method})")

    _write_videos_index(
        videos_dir,
        session_id=session_id,
        selected_camera=selected_camera,
        original_path=str(src_resolved),
        staged_path=str(staged),
        method=method,
        validation=validation,
    )

    return staged


def _ensure_unique_destination(dest: Path, src_resolved: Path) -> Path:
    if not dest.exists() and not dest.is_symlink():
        return dest

    # If existing symlink points to the same source, reuse.
    if dest.is_symlink():
        try:
            if dest.resolve() == src_resolved:
                return dest
        except Exception:
            pass

    h = hashlib.sha1(str(src_resolved).encode("utf-8")).hexdigest()[:8]
    return dest.with_name(f"{dest.stem}__{h}{dest.suffix}")


def _write_videos_index(
    videos_dir: Path,
    *,
    session_id: int,
    selected_camera: str | None,
    original_path: str,
    staged_path: str,
    method: str,
    validation: dict[str, Any] | None,
) -> None:
    index_path = videos_dir / "videos.json"
    payload: dict[str, Any] = {
        "session_id": session_id,
        "selected_camera": selected_camera,
        "original_path": original_path,
        "staged_path": staged_path,
        "stage_method": method,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if validation is not None:
        payload["validation"] = validation

    ensure_dir(index_path.parent)
    with open(index_path, "w") as f:
        json.dump(payload, f, indent=2)


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

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "opencv-python is required to preview raw videos. "
            "Install with: pip install opencv-python"
        ) from e
    
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

    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.animation import FuncAnimation, FFMpegWriter  # type: ignore
    
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
