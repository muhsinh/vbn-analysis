"""Configuration loading and path management for VBN analysis."""

from pathlib import Path
import os
from typing import Any

import yaml
from dotenv import load_dotenv


def _get_package_root() -> Path:
    """Return the package root directory (where pyproject.toml lives)."""
    return Path(__file__).parent.parent.parent


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
        cache_dir = Path(env_path).expanduser()
    else:
        # Load config and check
        config = load_config()
        config_path = config.get("paths", {}).get("cache_dir")
        if config_path:
            cache_dir = Path(config_path).expanduser()
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
        outputs_dir = Path(env_path).expanduser()
    else:
        # Load config and check
        config = load_config()
        config_path = config.get("paths", {}).get("outputs_dir")
        if config_path:
            outputs_dir = Path(config_path).expanduser()
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
