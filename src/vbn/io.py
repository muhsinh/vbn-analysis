"""Session loading and behavioral data access for VBN analysis."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .utils import setup_logging

if TYPE_CHECKING:
    from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )


def load_session(
    cache: "VisualBehaviorNeuropixelsProjectCache",
    session_id: int
) -> Any:
    """Load an ecephys session, downloading if needed.
    
    Args:
        cache: Initialized VBN cache instance
        session_id: The ecephys session ID to load
        
    Returns:
        BehaviorEcephysSession object
        
    Raises:
        ValueError: If session ID is not in the dataset
    """
    logger = setup_logging()
    
    # Validate session exists
    sessions_table = cache.get_ecephys_session_table(filter_by_validity=False)
    
    if session_id not in sessions_table.index:
        valid_ids = list(sessions_table.index[:10])
        raise ValueError(
            f"Session {session_id} not found in dataset.\n"
            f"Sample valid IDs: {valid_ids}\n"
            f"Total sessions: {len(sessions_table)}"
        )
    
    logger.info(f"Loading session {session_id}...")
    
    # This will download if not cached
    session = cache.get_ecephys_session(ecephys_session_id=session_id)
    
    logger.info(f"Session {session_id} loaded successfully")
    
    return session


def get_session_metadata(session: Any) -> dict[str, Any]:
    """Extract key metadata from a session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        Dictionary with session metadata
    """
    metadata = {}
    
    # Basic info
    if hasattr(session, "ecephys_session_id"):
        metadata["session_id"] = session.ecephys_session_id
    
    if hasattr(session, "metadata"):
        meta = session.metadata
        metadata.update({
            "mouse_id": meta.get("mouse_id"),
            "genotype": meta.get("genotype"),
            "sex": meta.get("sex"),
            "age_in_days": meta.get("age_in_days"),
            "session_type": meta.get("session_type"),
        })
    
    # Counts
    if hasattr(session, "units"):
        metadata["unit_count"] = len(session.units)
    
    if hasattr(session, "probes"):
        metadata["probe_count"] = len(session.probes)
    
    return metadata


def get_eye_tracking(session: Any) -> pd.DataFrame | None:
    """Get eye tracking data from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with eye tracking data or None if unavailable.
        Columns include: timestamps, pupil_area, pupil_center_x, pupil_center_y,
                        eye_area, eye_center_x, eye_center_y,
                        cr_area, cr_center_x, cr_center_y, likely_blink
    """
    logger = setup_logging()
    
    try:
        eye_tracking = session.eye_tracking
        
        if eye_tracking is None or len(eye_tracking) == 0:
            logger.warning("Session has no eye tracking data")
            return None
        
        logger.info(f"Eye tracking data: {len(eye_tracking)} samples")
        return eye_tracking
        
    except AttributeError:
        logger.warning("Session does not have eye_tracking attribute")
        return None
    except Exception as e:
        logger.warning(f"Could not load eye tracking: {e}")
        return None


def get_running_speed(session: Any) -> pd.DataFrame:
    """Get running speed time series from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with timestamps and speed columns
    """
    logger = setup_logging()
    
    try:
        running_speed = session.running_speed
        logger.info(f"Running speed data: {len(running_speed)} samples")
        return running_speed
    except AttributeError:
        logger.warning("Session does not have running_speed attribute")
        return pd.DataFrame(columns=["timestamps", "speed"])
    except Exception as e:
        logger.warning(f"Could not load running speed: {e}")
        return pd.DataFrame(columns=["timestamps", "speed"])


def get_stimulus_presentations(session: Any) -> pd.DataFrame:
    """Get stimulus presentation table from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with stimulus presentation times and parameters
    """
    logger = setup_logging()
    
    try:
        stim = session.stimulus_presentations
        logger.info(f"Stimulus presentations: {len(stim)} events")
        return stim
    except AttributeError:
        logger.warning("Session does not have stimulus_presentations attribute")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not load stimulus presentations: {e}")
        return pd.DataFrame()


def get_trials(session: Any) -> pd.DataFrame:
    """Get behavioral trials table from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with trial information
    """
    logger = setup_logging()
    
    try:
        trials = session.trials
        logger.info(f"Trials: {len(trials)}")
        return trials
    except AttributeError:
        logger.warning("Session does not have trials attribute")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not load trials: {e}")
        return pd.DataFrame()


def get_licks(session: Any) -> pd.DataFrame:
    """Get lick times from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with lick timestamps
    """
    logger = setup_logging()
    
    try:
        licks = session.licks
        logger.info(f"Licks: {len(licks)}")
        return licks
    except AttributeError:
        logger.warning("Session does not have licks attribute")
        return pd.DataFrame(columns=["timestamps"])
    except Exception as e:
        logger.warning(f"Could not load licks: {e}")
        return pd.DataFrame(columns=["timestamps"])


def get_rewards(session: Any) -> pd.DataFrame:
    """Get reward times from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with reward information
    """
    logger = setup_logging()
    
    try:
        rewards = session.rewards
        logger.info(f"Rewards: {len(rewards)}")
        return rewards
    except AttributeError:
        logger.warning("Session does not have rewards attribute")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not load rewards: {e}")
        return pd.DataFrame()


def get_units(session: Any) -> pd.DataFrame:
    """Get units table from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with unit information and quality metrics
    """
    return session.units


def get_probes(session: Any) -> pd.DataFrame:
    """Get probes table from session.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with probe information
    """
    return session.probes


def summarize_session(session: Any) -> dict[str, Any]:
    """Generate a summary of session contents.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        Dictionary with summary statistics
    """
    summary = get_session_metadata(session)
    
    # Eye tracking
    eye_df = get_eye_tracking(session)
    if eye_df is not None:
        summary["has_eye_tracking"] = True
        summary["eye_tracking_samples"] = len(eye_df)
        summary["eye_tracking_duration_min"] = (
            eye_df.index.max() - eye_df.index.min()
        ) / 60 if len(eye_df) > 1 else 0
    else:
        summary["has_eye_tracking"] = False
    
    # Running
    running_df = get_running_speed(session)
    summary["running_speed_samples"] = len(running_df)
    
    # Stimuli
    stim_df = get_stimulus_presentations(session)
    summary["stimulus_presentations"] = len(stim_df)
    
    # Trials
    trials_df = get_trials(session)
    summary["trials"] = len(trials_df)
    
    return summary
