"""AllenSDK cache wrapper for Visual Behavior Neuropixels data."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import get_cache_dir
from .utils import setup_logging

if TYPE_CHECKING:
    from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )


def get_cache(cache_dir: Path | str | None = None) -> "VisualBehaviorNeuropixelsProjectCache":
    """Initialize or return AllenSDK cache at specified directory.

    Args:
        cache_dir: Path to cache directory. Defaults to get_cache_dir()

    Returns:
        VisualBehaviorNeuropixelsProjectCache instance

    Raises:
        ImportError: If allensdk is not installed
    """
    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError as e:
        raise ImportError(
            "allensdk is required. Install with: pip install allensdk"
        ) from e

    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging()
    logger.info(f"Initializing cache at: {cache_dir}")

    manifests_dir = cache_dir / "visual-behavior-neuropixels" / "manifests"

    # If manifests exist, we can use the strict local-cache path.
    if manifests_dir.exists():
        logger.info("Found manifests folder; using from_local_cache()")
        return VisualBehaviorNeuropixelsProjectCache.from_local_cache(
            cache_dir=str(cache_dir),
            use_static_cache=True,
        )

    logger.info("Manifests folder missing; bootstrapping with from_s3_cache()")
    return VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=str(cache_dir)
    )

def get_sessions_table(
    cache: "VisualBehaviorNeuropixelsProjectCache",
    filter_by_validity: bool = False
) -> pd.DataFrame:
    """Return ecephys_sessions table from cache.

    Works across AllenSDK versions where the method signature differs.
    """
    import inspect

    fn = cache.get_ecephys_session_table
    sig = inspect.signature(fn)

    # Newer AllenSDK versions support this kwarg
    if "filter_by_validity" in sig.parameters:
        return fn(filter_by_validity=filter_by_validity)

    # Older AllenSDK versions: no kwarg, so fetch then filter ourselves if requested
    df = fn()

    if filter_by_validity:
        # Try common validity column names across releases
        for col in ("is_valid", "valid", "ecephys_session_is_valid", "session_is_valid"):
            if col in df.columns:
                return df[df[col].astype(bool)]

        # If we can't find a clear validity flag, fail loudly but usefully
        raise KeyError(
            "filter_by_validity=True was requested, but no known validity column "
            f"was found in ecephys session table columns: {list(df.columns)}"
        )

    return df

def get_probes_table(cache: "VisualBehaviorNeuropixelsProjectCache") -> pd.DataFrame:
    """Return probes table from cache.
    
    Args:
        cache: Initialized cache instance
        
    Returns:
        DataFrame with probe metadata
    """
    return cache.get_probe_table()


def get_units_table(cache: "VisualBehaviorNeuropixelsProjectCache") -> pd.DataFrame:
    """Return units table from cache.
    
    Args:
        cache: Initialized cache instance
        
    Returns:
        DataFrame with unit metadata and quality metrics
    """
    return cache.get_unit_table()


def get_channels_table(cache: "VisualBehaviorNeuropixelsProjectCache") -> pd.DataFrame:
    """Return channels table from cache.
    
    Args:
        cache: Initialized cache instance
        
    Returns:
        DataFrame with channel metadata
    """
    return cache.get_channel_table()


def session_exists_locally(cache_dir: Path | str, session_id: int) -> bool:
    """Check if NWB file for session exists in cache.
    
    Args:
        cache_dir: Path to cache directory
        session_id: The ecephys session ID to check
        
    Returns:
        True if NWB file exists locally
    """
    cache_dir = Path(cache_dir)
    
    # Check for NWB file in expected location
    # AllenSDK stores files as: visual-behavior-neuropixels/ecephys_sessions/ecephys_session_{id}.nwb
    nwb_patterns = [
        cache_dir / "visual-behavior-neuropixels" / "ecephys_sessions" / f"ecephys_session_{session_id}.nwb",
        cache_dir / f"ecephys_session_{session_id}.nwb",
    ]
    
    for pattern in nwb_patterns:
        if pattern.exists():
            return True
    
    # Also check recursively for the file
    for nwb_file in cache_dir.rglob(f"*{session_id}*.nwb"):
        return True
    
    return False


def get_session_nwb_path(cache_dir: Path | str, session_id: int) -> Path | None:
    """Get path to NWB file for a session if it exists locally.
    
    Args:
        cache_dir: Path to cache directory
        session_id: The ecephys session ID
        
    Returns:
        Path to NWB file or None if not found
    """
    cache_dir = Path(cache_dir)
    
    # Check expected locations
    expected_path = (
        cache_dir / "visual-behavior-neuropixels" / "ecephys_sessions" 
        / f"ecephys_session_{session_id}.nwb"
    )
    
    if expected_path.exists():
        return expected_path
    
    # Search recursively
    for nwb_file in cache_dir.rglob(f"*{session_id}*.nwb"):
        return nwb_file
    
    return None


def list_local_sessions(cache_dir: Path | str | None = None) -> list[int]:
    """List all session IDs that have been downloaded locally.
    
    Args:
        cache_dir: Path to cache directory. Defaults to get_cache_dir()
        
    Returns:
        List of session IDs with local NWB files
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
    
    session_ids = []
    
    for nwb_file in cache_dir.rglob("*.nwb"):
        # Extract session ID from filename
        name = nwb_file.stem
        if "ecephys_session_" in name:
            try:
                session_id = int(name.replace("ecephys_session_", ""))
                session_ids.append(session_id)
            except ValueError:
                pass
    
    return sorted(set(session_ids))
