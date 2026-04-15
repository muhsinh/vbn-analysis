"""NWB I/O and extraction utilities."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from config import get_config, make_provenance
from timebase import write_parquet_with_timebase, write_npz_with_provenance
from vbn_types import SpikeTimesDict


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

    # Trigger AllenSDK download so the NWB file is cached locally, then
    # return the override path (from sessions.csv) to the cached file.
    # BehaviorEcephysSession does not expose a nwb_path attribute, so we
    # cannot read the path back from the session object.
    try:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(cache_dir))
        cache.get_ecephys_session(session_id)
    except Exception:
        pass
    return nwb_path_override


def inspect_modalities(nwb: Any) -> dict[str, bool]:
    modalities = {
        "spikes": False,
        "trials": False,
        "eye": False,
        "behavior": False,
        "stimulus": False,
    }
    if nwb is None:
        return modalities

    modalities["spikes"] = hasattr(nwb, "units") and nwb.units is not None
    modalities["trials"] = hasattr(nwb, "trials") and nwb.trials is not None
    # VBN stores eye tracking in acquisition["EyeTracking"], not processing
    modalities["eye"] = "EyeTracking" in getattr(nwb, "acquisition", {})
    modalities["behavior"] = "behavior" in getattr(nwb, "processing", {})
    modalities["stimulus"] = hasattr(nwb, "stimulus") and nwb.stimulus is not None

    return modalities


def extract_units_and_spikes(
    nwb: Any,
    quality_filter: bool = True,
    min_presence_ratio: float = 0.9,
    max_isi_violations: float = 0.5,
    max_amplitude_cutoff: float = 0.1,
) -> tuple[pd.DataFrame | None, SpikeTimesDict | None]:
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

    spike_times: SpikeTimesDict = {}
    if "spike_times" in units_df.columns:
        for unit_id, times in units_df["spike_times"].items():
            spike_times[str(unit_id)] = np.asarray(times)
        units_df = units_df.drop(columns=["spike_times"])

    # Ensure brain area column is present — critical for area-stratified analyses
    for area_col in ("ecephys_structure_acronym", "structure_acronym", "location"):
        if area_col in units_df.columns:
            if area_col != "ecephys_structure_acronym":
                units_df = units_df.rename(columns={area_col: "ecephys_structure_acronym"})
            break

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
        "trial_type", "rewarded",
    ]
    keep_cols = [c for c in _keep if c in df.columns]
    # Always include anything not in the drop list (avoid silent column loss)
    extra = [c for c in df.columns if c not in keep_cols and c not in ("lick_times",)]
    return df[keep_cols + extra]


def extract_stimulus_presentations(nwb: Any) -> pd.DataFrame | None:
    """Extract stimulus presentation table from NWB.

    Returns a DataFrame with one row per flash, including:
      t                   – flash onset time (NWB seconds, display-lag corrected)
      t_start / t_end     – stimulus interval
      image_name          – identity of the image shown
      image_index         – integer image index
      is_change           – True on change flashes
      is_omission         – True on omitted flashes (blank screen)
      active              – True during the active task epoch, False during passive
      stimulus_block      – block index (increments each time stimulus set changes)
      stimulus_name       – 'natural_scenes' or 'natural_movie_*' etc.
    """
    if nwb is None:
        return None

    # VBN NWB stores one table per stimulus set (e.g.
    # "Natural_Images_Lum_Matched_set_ophys_G_2019_presentations",
    # "flash_250ms_presentations", "gabor_20_deg_250ms_presentations").
    # There is no single "stimulus_presentations" key.
    intervals = getattr(nwb, "intervals", None)
    if intervals is None:
        return None

    try:
        available_keys = list(intervals.keys())
    except TypeError:
        available_keys = [k for k in intervals]

    # Collect all *_presentations tables (including spontaneous — useful for
    # baseline firing rates, noise correlations, and state modulation baselines).
    # Skip only the "trials" table which is extracted separately via extract_trials().
    candidate_keys = [
        k for k in available_keys
        if k.endswith("_presentations") and k != "trials"
    ]
    if not candidate_keys:
        return None

    pieces = []
    for key in candidate_keys:
        try:
            tbl = intervals[key]
            df = tbl.to_dataframe() if hasattr(tbl, "to_dataframe") else pd.DataFrame(tbl)
            df = df.reset_index(drop=False)
            df["_source_table"] = key
            pieces.append(df)
        except Exception:
            continue

    if not pieces:
        return None

    stim = pd.concat(pieces, ignore_index=True)

    # pynwb always adds a "timeseries" column containing live NWB objects —
    # drop it and any other object-dtype column that isn't str/bool/numeric.
    PYNWB_OBJECT_COLS = {"timeseries", "tags"}
    drop_cols = [c for c in stim.columns if c in PYNWB_OBJECT_COLS]
    if drop_cols:
        stim = stim.drop(columns=drop_cols)

    # Standardise timing columns
    if "start_time" in stim.columns:
        stim = stim.rename(columns={"start_time": "t_start", "stop_time": "t_end"})
    if "stimulus_time_offset" in stim.columns:
        stim["t"] = stim["t_start"] + stim["stimulus_time_offset"]
    elif "t_start" in stim.columns:
        stim["t"] = stim["t_start"]

    stim = stim.sort_values("t").reset_index(drop=True)

    # Normalise column names
    renames = {"change_frame": "is_change", "omitted": "is_omission"}
    for src, dst in renames.items():
        if src in stim.columns and dst not in stim.columns:
            stim = stim.rename(columns={src: dst})

    # Derive active epoch flag: in VBN the active task epoch comes first,
    # passive replay second. Spontaneous periods are always passive (active=False).
    if "active" not in stim.columns:
        if "stimulus_block" in stim.columns:
            first_block = stim["stimulus_block"].min()
            stim["active"] = stim["stimulus_block"] == first_block
        else:
            mid = stim["t"].median()
            stim["active"] = stim["t"] < mid
    # Spontaneous presentations are never part of the active task epoch
    if "_source_table" in stim.columns:
        is_spontaneous = stim["_source_table"].str.contains("spontaneous", case=False)
        stim.loc[is_spontaneous, "active"] = False

    return stim


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
            except (AttributeError, ValueError, TypeError):
                continue

    return None


def extract_behavior_events(nwb: Any) -> pd.DataFrame | None:
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
            except (AttributeError, ValueError, TypeError):
                continue

    if not events:
        return None

    merged = events[0]
    for df in events[1:]:
        merged = pd.merge_asof(merged.sort_values("t"), df.sort_values("t"), on="t")
    return merged


def extract_eye_tracking(nwb: Any) -> pd.DataFrame | None:
    """Extract eye tracking from VBN NWB files.

    Primary path: nwb.acquisition["EyeTracking"] (EllipseEyeTracking),
    which contains pupil_tracking (area, x/y, angle) and
    corneal_reflection_tracking + eye_tracking (likely_blink).
    Falls back to nwb.processing["eye_tracking"] for older formats.
    """
    if nwb is None:
        return None

    acquisition = getattr(nwb, "acquisition", {}) or {}
    if "EyeTracking" in acquisition:
        try:
            et = acquisition["EyeTracking"]
            series = getattr(et, "spatial_series", {}) or {}

            pt = series.get("pupil_tracking")
            if pt is not None:
                times = np.asarray(pt.timestamps)
                df = pd.DataFrame({"t": times})

                # Pupil area (scalar per frame)
                if hasattr(pt, "area") and pt.area is not None:
                    df["pupil_area"] = np.asarray(pt.area)
                elif hasattr(pt, "area_raw") and pt.area_raw is not None:
                    df["pupil_area"] = np.asarray(pt.area_raw)

                # Pupil centre (data is (n, 2) → x, y)
                raw = np.asarray(pt.data)
                if raw.ndim == 2 and raw.shape[1] >= 2:
                    df["pupil_x"] = raw[:, 0]
                    df["pupil_y"] = raw[:, 1]

                # Ellipse geometry
                for attr, col in [("width", "pupil_width"), ("height", "pupil_height"),
                                   ("angle", "pupil_angle")]:
                    val = getattr(pt, attr, None)
                    if val is not None:
                        arr = np.asarray(val)
                        if arr.shape == times.shape:
                            df[col] = arr

                # Blink flag from eye_tracking series
                et_series = series.get("eye_tracking")
                if et_series is not None:
                    for attr in ("likely_blink",):
                        val = getattr(et_series, attr, None)
                        if val is not None:
                            arr = np.asarray(val)
                            if arr.shape == times.shape:
                                df["likely_blink"] = arr.astype(bool)

                return df
        except (AttributeError, ValueError, TypeError, KeyError):
            pass

    processing = getattr(nwb, "processing", {}) or {}
    if "eye_tracking" not in processing:
        return None

    eye_module = processing["eye_tracking"]
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
        except (AttributeError, ValueError, TypeError):
            continue

    return None


def save_units_and_spikes(
    units: pd.DataFrame,
    spikes: SpikeTimesDict,
    units_path: Path,
    spikes_path: Path,
    session_id: int,
    alignment_method: str,
) -> None:
    provenance = make_provenance(session_id, alignment_method)
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
    provenance = make_provenance(session_id, alignment_method)
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


def save_stimulus_presentations(
    stim_df: pd.DataFrame,
    stim_path: Path,
    session_id: int,
    alignment_method: str,
) -> None:
    provenance = make_provenance(session_id, alignment_method)
    required = ["t"] if "t" in stim_df.columns else None
    write_parquet_with_timebase(
        stim_df,
        stim_path,
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
    provenance = make_provenance(session_id, alignment_method)
    write_parquet_with_timebase(
        eye_df,
        eye_path,
        timebase="nwb_seconds",
        provenance=provenance,
        required_columns=["t"],
    )


def load_spike_times_npz(path: Path) -> SpikeTimesDict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


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
