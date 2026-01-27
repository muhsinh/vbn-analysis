"""VBN - Local-first analysis tools for Allen Visual Behavior Neuropixels dataset."""

__version__ = "0.1.0"

from .config import load_config, get_cache_dir, get_outputs_dir
from .cache import get_cache, get_sessions_table, session_exists_locally
from .io import load_session, get_eye_tracking, get_running_speed, get_stimulus_presentations, get_trials
from .video import discover_videos, generate_video_manifest, preview_video_file, preview_eye_tracking_as_video
from .frames import (
    extract_frames_from_video,
    extract_frames_from_eye_tracking,
    sample_frames_for_labeling,
    export_timestamps_csv,
)

__all__ = [
    # Config
    "load_config",
    "get_cache_dir",
    "get_outputs_dir",
    # Cache
    "get_cache",
    "get_sessions_table",
    "session_exists_locally",
    # IO
    "load_session",
    "get_eye_tracking",
    "get_running_speed",
    "get_stimulus_presentations",
    "get_trials",
    # Video
    "discover_videos",
    "generate_video_manifest",
    "preview_video_file",
    "preview_eye_tracking_as_video",
    # Frames
    "extract_frames_from_video",
    "extract_frames_from_eye_tracking",
    "sample_frames_for_labeling",
    "export_timestamps_csv",
]
