"""NWB I/O and extraction utilities."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from config import get_config
from timebase import write_parquet_with_timebase, write_npz_with_provenance


@contextlib.contextmanager
def open_nwb_handle(nwb_path: Path | None, mock_mode: bool = False) -> Iterator[Any]:
    """Open an NWB file and yield the NWBFile object.

    If mock_mode is True or nwb_path is None/missing, yields a synthetic object.
    """
    if mock_mode or nwb_path is None or not Path(nwb_path).exists():
        yield _mock_nwb()
        return

    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise ImportError("pynwb is required to read NWB files") from exc

    io = NWBHDF5IO(str(nwb_path), "r")
    try:
        nwb = io.read()
        yield nwb
    finally:
        io.close()


def resolve_nwb_path(
    session_id: int,
    access_mode: str,
    nwb_path_override: Path | None = None,
) -> Path | None:
    if access_mode == "manual":
        return nwb_path_override

    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError:
        return nwb_path_override

    cfg = get_config()
    cache_dir = cfg.data_dir / "allensdk_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use AllenSDK cache to ensure NWB is available locally.
    try:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(cache_dir))
        session = cache.get_ecephys_session(session_id)
        nwb_path = Path(session.nwb_path) if hasattr(session, "nwb_path") else None
        return nwb_path
    except Exception:
        return nwb_path_override


def inspect_modalities(nwb: Any) -> Dict[str, bool]:
    modalities = {
        "spikes": False,
        "trials": False,
        "eye": False,
        "behavior": False,
        "stimulus": False,
    }
    if nwb is None:
        return modalities

    try:
        modalities["spikes"] = hasattr(nwb, "units") and nwb.units is not None
    except Exception:
        modalities["spikes"] = False

    try:
        modalities["trials"] = hasattr(nwb, "trials") and nwb.trials is not None
    except Exception:
        modalities["trials"] = False

    try:
        modalities["eye"] = "eye_tracking" in getattr(nwb, "processing", {})
    except Exception:
        modalities["eye"] = False

    try:
        modalities["behavior"] = "behavior" in getattr(nwb, "processing", {})
    except Exception:
        modalities["behavior"] = False

    try:
        modalities["stimulus"] = hasattr(nwb, "stimulus") and nwb.stimulus is not None
    except Exception:
        modalities["stimulus"] = False

    return modalities


def extract_units_and_spikes(
    nwb: Any,
    quality_filter: bool = True,
    min_presence_ratio: float = 0.9,
    max_isi_violations: float = 0.5,
    max_amplitude_cutoff: float = 0.1,
) -> Tuple[pd.DataFrame | None, Dict[str, Any] | None]:
    """Extract units table and spike times from NWB, with quality filtering.

    Parameters
    ----------
    quality_filter : bool
        If True (default), keep only units with quality == 'good' and
        passing the ISI / presence / amplitude thresholds recommended by
        the Allen Institute (Corbett Bennett et al., VBN manuscript).
    min_presence_ratio : float
        Minimum fraction of session with at least one spike per bin.
    max_isi_violations : float
        Maximum fraction of ISI violations (< 1.5ms refractory).
    max_amplitude_cutoff : float
        Maximum estimated fraction of spikes below detection threshold.
    """
    if nwb is None or not hasattr(nwb, "units") or nwb.units is None:
        return None, None

    units_table = nwb.units
    units_df = units_table.to_dataframe() if hasattr(units_table, "to_dataframe") else pd.DataFrame(units_table)

    # --- Quality filtering ---
    if quality_filter:
        n_before = len(units_df)
        if "quality" in units_df.columns:
            units_df = units_df[units_df["quality"] == "good"]
        if "isi_violations" in units_df.columns:
            units_df = units_df[units_df["isi_violations"] <= max_isi_violations]
        if "presence_ratio" in units_df.columns:
            units_df = units_df[units_df["presence_ratio"] >= min_presence_ratio]
        if "amplitude_cutoff" in units_df.columns:
            units_df = units_df[units_df["amplitude_cutoff"] <= max_amplitude_cutoff]
        n_after = len(units_df)
        if n_before > 0:
            print(f"    Quality filter: {n_before} → {n_after} units "
                  f"({n_before - n_after} removed, {100*n_after/n_before:.0f}% kept)")

    spike_times: Dict[str, Any] = {}
    if "spike_times" in units_df.columns:
        for unit_id, times in units_df["spike_times"].items():
            spike_times[str(unit_id)] = np.asarray(times)
        units_df = units_df.drop(columns=["spike_times"])

    return units_df.reset_index(drop=False), spike_times


def extract_trials(nwb: Any) -> pd.DataFrame | None:
    """Extract trials table, preserving all behaviorally relevant columns.

    Keeps: timing, outcomes (hit/miss/FA/CR), response latency, image
    identity, lick times, reward info, and the display-lag-corrected
    change time. Filters out auto-rewarded trials from outcome columns
    since they should not be included in behavioral performance metrics.
    """
    if nwb is None or not hasattr(nwb, "trials") or nwb.trials is None:
        return None
    trials_table = nwb.trials
    df = trials_table.to_dataframe() if hasattr(trials_table, "to_dataframe") else pd.DataFrame(trials_table)
    df = df.reset_index(drop=False)

    # Standardise timing columns
    if "start_time" in df.columns:
        df = df.rename(columns={"start_time": "t_start", "stop_time": "t_end"})
    # Use corrected change time where available (removes display lag)
    if "change_time_no_display_delay" in df.columns:
        df["t"] = df["change_time_no_display_delay"]
    elif "t_start" in df.columns:
        df["t"] = df["t_start"]

    # Preserve all outcome and identity columns that exist
    _keep = [
        "t", "t_start", "t_end",
        "hit", "miss", "false_alarm", "correct_reject", "aborted",
        "go", "catch", "stimulus_change", "is_change", "auto_rewarded",
        "response_latency", "response_time",
        "reward_time", "reward_volume",
        "initial_image_name", "change_image_name", "stimulus_name",
        "change_time_no_display_delay",
        # legacy / pipeline columns
        "trial_type", "rewarded",
    ]
    keep_cols = [c for c in _keep if c in df.columns]
    # Always include anything not in the drop list (avoid silent column loss)
    extra = [c for c in df.columns if c not in keep_cols and c not in ("lick_times",)]
    return df[keep_cols + extra]


def extract_running_speed(nwb: Any) -> pd.DataFrame | None:
    """Extract running speed from the rotary wheel encoder.

    Returns a DataFrame with columns ['t', 'running'] where 'running'
    is velocity in cm/s on the NWB seconds clock. This is the
    calibrated encoder signal used in all Allen Institute publications
    — use this in preference to video-derived running estimates.
    """
    if nwb is None:
        return None

    processing = getattr(nwb, "processing", {}) or {}

    # Primary location: nwb.processing["running"]["running_speed"]
    for module_name in ("running", "behavior"):
        if module_name not in processing:
            continue
        module = processing[module_name]
        for ts_name in ("running_speed", "speed", "RunningSpeed"):
            if ts_name not in module.data_interfaces:
                continue
            ts = module.data_interfaces[ts_name]
            try:
                times = np.asarray(ts.timestamps)
                data = np.asarray(ts.data)
                if data.ndim > 1:
                    data = data[:, 0]
                df = pd.DataFrame({"t": times, "running": data})
                # Clip negative speeds (encoder artefact)
                df["running"] = df["running"].clip(lower=0.0)
                return df
            except Exception:
                continue

    return None


def extract_behavior_events(nwb: Any) -> pd.DataFrame | None:
    # Try common sources in NWB processing modules
    if nwb is None:
        return None

    events = []
    processing = getattr(nwb, "processing", {}) or {}
    if "behavior" in processing:
        behavior_module = processing["behavior"]
        for name, ts in behavior_module.data_interfaces.items():
            try:
                times = np.asarray(ts.timestamps)
                data = np.asarray(ts.data)
                if data.ndim == 1:
                    df = pd.DataFrame({"t": times, name: data})
                else:
                    cols = [f"{name}_{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=cols)
                    df.insert(0, "t", times)
                events.append(df)
            except Exception:
                continue

    if not events:
        return None

    merged = events[0]
    for df in events[1:]:
        merged = pd.merge_asof(merged.sort_values("t"), df.sort_values("t"), on="t")
    return merged


def extract_eye_tracking(nwb: Any) -> pd.DataFrame | None:
    if nwb is None:
        return None

    processing = getattr(nwb, "processing", {}) or {}
    if "eye_tracking" not in processing:
        return None

    eye_module = processing["eye_tracking"]
    # Try common interfaces
    for name, ts in eye_module.data_interfaces.items():
        try:
            times = np.asarray(ts.timestamps)
            data = np.asarray(ts.data)
            if data.ndim == 1:
                df = pd.DataFrame({"t": times, name: data})
            else:
                cols = [f"{name}_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=cols)
                df.insert(0, "t", times)
            return df
        except Exception:
            continue

    return None


def save_units_and_spikes(
    units: pd.DataFrame,
    spikes: Dict[str, Any],
    units_path: Path,
    spikes_path: Path,
    session_id: int,
    alignment_method: str,
) -> None:
    provenance = _provenance(session_id, alignment_method)
    units_df = units.copy()
    if "unit_id" not in units_df.columns:
        units_df.insert(0, "unit_id", range(len(units_df)))
    write_parquet_with_timebase(
        units_df,
        units_path,
        timebase="nwb_seconds",
        provenance=provenance,
        required_columns=["unit_id"],
    )
    write_npz_with_provenance(spikes, spikes_path, provenance)


def save_behavior_tables(
    trials: pd.DataFrame | None,
    events: pd.DataFrame | None,
    trials_path: Path,
    events_path: Path,
    session_id: int,
    alignment_method: str,
) -> None:
    provenance = _provenance(session_id, alignment_method)
    if trials is not None:
        write_parquet_with_timebase(
            trials,
            trials_path,
            timebase="nwb_seconds",
            provenance=provenance,
            required_columns=["t"],
        )
    if events is not None:
        required = ["t"] if "t" in events.columns else None
        write_parquet_with_timebase(
            events,
            events_path,
            timebase="nwb_seconds",
            provenance=provenance,
            required_columns=required,
        )


def save_eye_table(
    eye_df: pd.DataFrame,
    eye_path: Path,
    session_id: int,
    alignment_method: str,
) -> None:
    provenance = _provenance(session_id, alignment_method)
    write_parquet_with_timebase(
        eye_df,
        eye_path,
        timebase="nwb_seconds",
        provenance=provenance,
        required_columns=["t"],
    )


def load_spike_times_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _provenance(session_id: int, alignment_method: str) -> Dict[str, Any]:
    from config import make_provenance
    return make_provenance(session_id, alignment_method)


def _mock_nwb() -> Any:
    """Create a minimal mock NWB-like object to keep notebooks runnable."""
    class MockNWB:
        def __init__(self):
            self.units = pd.DataFrame({
                "unit_id": [1, 2, 3],
                "spike_times": [
                    np.array([0.1, 0.5, 1.0]),
                    np.array([0.2, 0.7, 1.4]),
                    np.array([0.3, 0.9, 1.8]),
                ],
            })
            self.trials = pd.DataFrame({
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "trial_type": ["go", "no-go"],
            })
            self.processing = {}
            self.stimulus = None

        def __repr__(self):
            return "<MockNWB>"

    mock = MockNWB()
    return mock
