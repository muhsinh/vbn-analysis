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


def extract_units_and_spikes(nwb: Any) -> Tuple[pd.DataFrame | None, Dict[str, Any] | None]:
    if nwb is None or not hasattr(nwb, "units") or nwb.units is None:
        return None, None

    units_table = nwb.units
    if hasattr(units_table, "to_dataframe"):
        units_df = units_table.to_dataframe()
    else:
        units_df = pd.DataFrame(units_table)
    spike_times = {}
    if "spike_times" in units_df.columns:
        for unit_id, times in units_df["spike_times"].items():
            spike_times[str(unit_id)] = np.asarray(times)
        units_df = units_df.drop(columns=["spike_times"])
    return units_df.reset_index(drop=False), spike_times


def extract_trials(nwb: Any) -> pd.DataFrame | None:
    if nwb is None or not hasattr(nwb, "trials") or nwb.trials is None:
        return None
    trials_table = nwb.trials
    if hasattr(trials_table, "to_dataframe"):
        df = trials_table.to_dataframe()
    else:
        df = pd.DataFrame(trials_table)
    df = df.reset_index(drop=False)
    if "start_time" in df.columns:
        df = df.rename(columns={"start_time": "t_start", "stop_time": "t_end"})
    if "t_start" in df.columns and "t" not in df.columns:
        df["t"] = df["t_start"]
    return df


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
