"""Shared utilities for VBN analysis."""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). 
               Defaults to VBN_LOG_LEVEL env var or INFO.
               
    Returns:
        Configured logger instance
    """
    if level is None:
        level = os.environ.get("VBN_LOG_LEVEL", "INFO")
    
    # Create logger
    logger = logging.getLogger("vbn")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_size(bytes_size: int) -> str:
    """Convert bytes to human-readable file size.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Human-readable string (e.g., "2.8 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_disk_space(path: Path) -> dict[str, int]:
    """Get disk space information for a path.
    
    Args:
        path: Path to check disk space for
        
    Returns:
        Dict with total, used, and free bytes
    """
    total, used, free = shutil.disk_usage(path)
    return {"total": total, "used": used, "free": free}


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        required_bytes: Bytes needed
        
    Returns:
        True if enough space available
    """
    space = get_disk_space(path)
    return space["free"] >= required_bytes


def print_diagnostic(title: str, items: dict[str, Any]) -> None:
    """Pretty-print diagnostic information to console.
    
    Args:
        title: Section title
        items: Dictionary of key-value pairs to display
    """
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)
    
    max_key_len = max(len(str(k)) for k in items.keys()) if items else 0
    
    for key, value in items.items():
        # Format value based on type
        if isinstance(value, bool):
            value_str = "Yes" if value else "No"
        elif isinstance(value, int) and key.lower().endswith(("bytes", "size")):
            value_str = format_size(value)
        elif isinstance(value, Path):
            value_str = str(value)
        elif isinstance(value, list):
            if len(value) <= 3:
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = f"{len(value)} items"
        else:
            value_str = str(value)
        
        print(f"  {key:<{max_key_len}} : {value_str}")
    
    print()


def validate_session_id(session_id: int, valid_ids: list[int]) -> None:
    """Validate that a session ID exists in the dataset.
    
    Args:
        session_id: Session ID to validate
        valid_ids: List of valid session IDs
        
    Raises:
        ValueError: If session ID is not valid
    """
    if session_id not in valid_ids:
        sample_ids = valid_ids[:10]
        raise ValueError(
            f"Session {session_id} not found in dataset.\n"
            f"Valid session IDs include: {sample_ids}\n"
            f"Total sessions: {len(valid_ids)}"
        )
