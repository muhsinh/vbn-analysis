"""Configuration loading and path management for VBN analysis."""

from pathlib import Path
import os
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

VideoStage = Literal["symlink", "copy", "none"]
CameraType = Literal["body", "eye", "face", "any"]


def _get_package_root() -> Path:
    """Return the package root directory (where pyproject.toml lives)."""
    return Path(__file__).parent.parent.parent


def _expand_path(value: str | Path) -> Path:
    """Expand env vars + user home for a path-like value."""
    s = os.path.expandvars(str(value))
    return Path(s).expanduser()


def _parse_path_list(value: str) -> list[Path]:
    """Parse an OS-pathsep-separated list of paths into Path objects."""
    parts = [p.strip() for p in value.split(os.pathsep) if p.strip()]
    paths: list[Path] = []
    for p in parts:
        p = os.path.expandvars(p)
        try:
            paths.append(Path(p).expanduser().resolve())
        except Exception:
            paths.append(Path(p).expanduser())
    return paths


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load .env then configs/default.yaml, return merged dict.
    
    Args:
        config_path: Optional path to config YAML. Defaults to configs/default.yaml
        
    Returns:
        Configuration dictionary with all settings
    """
    # Load .env file if it exists
    env_file = _get_package_root() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Determine config file path
    if config_path is None:
        config_path = _get_package_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    # Load YAML config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Override paths from environment variables
    if "paths" not in config:
        config["paths"] = {}
    
    if os.environ.get("VBN_CACHE_DIR"):
        config["paths"]["cache_dir"] = os.environ["VBN_CACHE_DIR"]
    
    if os.environ.get("VBN_OUTPUTS_DIR"):
        config["paths"]["outputs_dir"] = os.environ["VBN_OUTPUTS_DIR"]

    if os.environ.get("VBN_VIDEO_DIRS"):
        config["paths"]["video_dirs"] = [
            str(p) for p in _parse_path_list(os.environ["VBN_VIDEO_DIRS"])
        ]

    if os.environ.get("VBN_VIDEO_CAMERA"):
        config.setdefault("video", {})
        config["video"]["preferred_camera"] = os.environ["VBN_VIDEO_CAMERA"]

    if os.environ.get("VBN_VIDEO_STAGE"):
        config.setdefault("video", {})
        config["video"]["stage"] = os.environ["VBN_VIDEO_STAGE"]
    
    return config


def get_cache_dir() -> Path:
    """Return cache directory path.
    
    Priority:
    1. $VBN_CACHE_DIR environment variable
    2. paths.cache_dir from config
    3. Default: ~/data/vbn_cache
    
    Returns:
        Path to cache directory (created if doesn't exist)
    """
    # Check environment variable first
    env_path = os.environ.get("VBN_CACHE_DIR")
    if env_path:
        cache_dir = _expand_path(env_path)
    else:
        # Load config and check
        config = load_config()
        config_path = config.get("paths", {}).get("cache_dir")
        if config_path:
            cache_dir = _expand_path(config_path)
        else:
            # Default
            cache_dir = Path.home() / "data" / "vbn_cache"
    
    # Ensure directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_outputs_dir() -> Path:
    """Return outputs directory path.
    
    Priority:
    1. $VBN_OUTPUTS_DIR environment variable
    2. paths.outputs_dir from config
    3. Default: ~/data/vbn_outputs
    
    Returns:
        Path to outputs directory (created if doesn't exist)
    """
    # Check environment variable first
    env_path = os.environ.get("VBN_OUTPUTS_DIR")
    if env_path:
        outputs_dir = _expand_path(env_path)
    else:
        # Load config and check
        config = load_config()
        config_path = config.get("paths", {}).get("outputs_dir")
        if config_path:
            outputs_dir = _expand_path(config_path)
        else:
            # Default
            outputs_dir = Path.home() / "data" / "vbn_outputs"
    
    # Ensure directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def get_session_output_dir(session_id: int) -> Path:
    """Return output directory for a specific session.
    
    Args:
        session_id: The ecephys session ID
        
    Returns:
        Path to session-specific output directory
    """
    output_dir = get_outputs_dir() / str(session_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_video_dirs() -> list[Path]:
    """Return configured video search roots.

    Priority:
    1. $VBN_VIDEO_DIRS (OS path separator-separated list)
    2. paths.video_dirs from config
    3. Default: []
    """
    env_value = os.environ.get("VBN_VIDEO_DIRS")
    if env_value:
        return _parse_path_list(env_value)

    config = load_config()
    raw = config.get("paths", {}).get("video_dirs", []) or []

    if isinstance(raw, str):
        return _parse_path_list(os.path.expandvars(raw))

    paths: list[Path] = []
    for item in raw:
        if not item:
            continue
        try:
            paths.append(_expand_path(item).resolve())
        except Exception:
            paths.append(_expand_path(item))
    return paths


def get_video_stage(default: VideoStage = "symlink") -> VideoStage:
    """Return default video staging method."""
    env_value = os.environ.get("VBN_VIDEO_STAGE")
    if env_value:
        v = env_value.strip().lower()
        if v in ("symlink", "copy", "none"):
            return v  # type: ignore[return-value]

    config = load_config()
    v = str(config.get("video", {}).get("stage", default)).strip().lower()
    if v in ("symlink", "copy", "none"):
        return v  # type: ignore[return-value]
    return default


def get_video_preferred_camera(default: CameraType = "body") -> CameraType:
    """Return default camera preference for video selection."""
    env_value = os.environ.get("VBN_VIDEO_CAMERA")
    if env_value:
        v = env_value.strip().lower()
        if v in ("body", "eye", "face", "any"):
            return v  # type: ignore[return-value]

    config = load_config()
    v = str(config.get("video", {}).get("preferred_camera", default)).strip().lower()
    if v in ("body", "eye", "face", "any"):
        return v  # type: ignore[return-value]
    return default
