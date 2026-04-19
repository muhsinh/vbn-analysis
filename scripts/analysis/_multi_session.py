"""Multi-session extraction + loading for cross-session replication.

Given any VBN session_id, either:
  (a) load pre-extracted parquets from outputs/cross_session/<id>/, or
  (b) extract from the cached NWB on disk and cache to parquet.

Produces the same schema as the session-1055240613 extraction so the
analysis scripts (A2/B2/D2/H) work unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import make_provenance  # noqa: E402
import io_nwb  # noqa: E402

CACHE_DIR = Path("/Users/muh/projects/vbn-analysis/data/allensdk_cache")
CROSS_DIR = ROOT / "outputs" / "cross_session"
BIN_SIZE = 0.025


def _nwb_path(session_id: int) -> Path:
    """Return the NWB path in the AllenSDK cache for a given session."""
    return (
        CACHE_DIR
        / "visual-behavior-neuropixels-0.5.0"
        / "behavior_ecephys_sessions"
        / str(session_id)
        / f"ecephys_session_{session_id}.nwb"
    )


def _session_dir(session_id: int) -> Path:
    d = CROSS_DIR / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_session(session_id: int, force: bool = False) -> Path:
    """Extract per-session artifacts from the NWB. Returns the per-session output dir.

    Schema written:
      units.parquet               units table with ecephys_structure_acronym joined
      spike_times.npz             {unit_id: np.ndarray}
      trials.parquet              standardized trials table
      stimuli.parquet             stimulus_presentations with block + is_change + active
      running.parquet             (t, running_cm_s)
      eye_features.parquet        (t, pupil, pupil_z, pupil_vel, pupil_x, pupil_y, blink)
      metadata.json
    """
    out = _session_dir(session_id)
    flag = out / ".extracted"
    if flag.exists() and not force:
        return out

    from pynwb import NWBHDF5IO

    nwb_path = _nwb_path(session_id)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB not found: {nwb_path}")
    print(f"  [{session_id}] extracting from {nwb_path.name} ({nwb_path.stat().st_size/1e9:.1f} GB)")

    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()

        # --- Units + spikes ---
        units_df, spikes = io_nwb.extract_units_and_spikes(nwb)
        if units_df is None:
            raise RuntimeError(f"No units extracted for {session_id}")
        # Drop object columns containing multi-dimensional arrays (pyarrow can't serialize)
        bad_cols = []
        for c in units_df.columns:
            if units_df[c].dtype == object:
                sample = units_df[c].dropna().head(1)
                if len(sample):
                    v = sample.iloc[0]
                    if hasattr(v, "ndim") and v.ndim > 1:
                        bad_cols.append(c)
        if bad_cols:
            units_df = units_df.drop(columns=bad_cols)
        units_df.to_parquet(out / "units.parquet", index=False)
        np.savez_compressed(
            out / "spike_times.npz",
            **{str(k): np.asarray(v) for k, v in spikes.items()},
        )
        print(f"    units: {len(units_df)}  spikes: {len(spikes)} units")

        # --- Trials ---
        trials = io_nwb.extract_trials(nwb)
        if trials is not None:
            trials.to_parquet(out / "trials.parquet", index=False)
            print(f"    trials: {len(trials)}")

        # --- Stimulus presentations ---
        stim = io_nwb.extract_stimulus_presentations(nwb)
        if stim is not None:
            stim.to_parquet(out / "stimuli.parquet", index=False)
            print(f"    stimuli: {len(stim)}")

        # --- Running speed ---
        running = None
        if "running" in nwb.processing:
            running_mod = nwb.processing["running"]
            if "speed" in running_mod.data_interfaces:
                ts = running_mod.data_interfaces["speed"]
                running = pd.DataFrame({
                    "t": ts.timestamps[:].astype(float),
                    "running": ts.data[:].astype(float),
                })
                running.to_parquet(out / "running.parquet", index=False)
                print(f"    running: {len(running)} samples")

        # --- Eye tracking: Allen VBN stores EllipseEyeTracking ---
        # pupil_tracking: .data (N,2) xy-center, .area (N,), .angle (N,), .timestamps (N,)
        # likely_blink: TimeSeries (N,) of bool
        eye = None
        if "EyeTracking" in nwb.acquisition:
            et = nwb.acquisition["EyeTracking"]
            pt = getattr(et, "pupil_tracking", None)
            lb = getattr(et, "likely_blink", None)
            if pt is not None and pt.timestamps is not None:
                t = pt.timestamps[:].astype(float)
                center = pt.data[:]  # (N, 2) xy
                area = pt.area[:] if hasattr(pt, "area") and pt.area is not None else np.full(len(t), np.nan)
                blink = lb.data[:].astype(bool) if lb is not None else np.zeros(len(t), dtype=bool)
                px = center[:, 0].astype(float) if center.ndim == 2 else np.full(len(t), np.nan)
                py = center[:, 1].astype(float) if center.ndim == 2 else np.full(len(t), np.nan)
                area = area.astype(float)
                eye = pd.DataFrame({
                    "t": t,
                    "pupil": area,
                    "pupil_x": px,
                    "pupil_y": py,
                    "blink": blink,
                })
                valid = np.isfinite(eye["pupil"]) & ~eye["blink"]
                eye["pupil_z"] = np.nan
                if valid.sum() > 10:
                    mu = eye.loc[valid, "pupil"].mean()
                    sd = eye.loc[valid, "pupil"].std() + 1e-9
                    eye.loc[valid, "pupil_z"] = (eye.loc[valid, "pupil"] - mu) / sd
                dt = float(np.median(np.diff(eye["t"]))) if len(eye) > 1 else 0.016
                dt = dt if dt > 0 else 0.016
                filled = eye["pupil"].interpolate(limit_direction="both").fillna(0).values
                eye["pupil_vel"] = np.gradient(filled, dt)
                eye.to_parquet(out / "eye_features.parquet", index=False)
                print(f"    eye: {len(eye)} samples, {int(blink.sum())} blinks")

        # --- Licks + rewards ---
        if "licking" in nwb.processing:
            licking_mod = nwb.processing["licking"]
            if "licks" in licking_mod.data_interfaces:
                lick_ts = licking_mod.data_interfaces["licks"].timestamps[:]
                pd.DataFrame({"t": lick_ts.astype(float)}).to_parquet(
                    out / "licks.parquet", index=False
                )
                print(f"    licks: {len(lick_ts)}")

        if "rewards" in nwb.processing:
            reward_mod = nwb.processing["rewards"]
            if "volume" in reward_mod.data_interfaces:
                rw = reward_mod.data_interfaces["volume"]
                t = rw.timestamps[:]
                vol = rw.data[:]
                pd.DataFrame({"t": t.astype(float), "volume_ml": vol.astype(float)}).to_parquet(
                    out / "rewards.parquet", index=False
                )
                print(f"    rewards: {len(t)}")

    flag.touch()
    return out


def _detect_active_passive_blocks(stim: pd.DataFrame) -> tuple[int, int]:
    """Find active vs passive natural-image blocks.

    Allen VBN convention: active = block 0 (first natural-images block, with rewards),
    passive = last natural-images block (re-shown passively).
    """
    # Blocks with natural images (has is_change) and sufficient duration
    blocks = (
        stim.groupby("stimulus_block")
        .agg(n=("id", "count"),
             active_mean=("active", lambda x: x.dropna().astype(bool).mean() if len(x.dropna()) else np.nan),
             has_change=("is_change", lambda x: x.sum() > 0))
        .reset_index()
    )
    # Natural-image blocks are the ones with is_change events
    natural = blocks[blocks["has_change"]]
    if len(natural) < 2:
        raise ValueError(f"Expected ≥2 natural-image blocks, found {len(natural)}")
    active_block = natural.iloc[0]["stimulus_block"]
    passive_block = natural.iloc[-1]["stimulus_block"]
    return int(active_block), int(passive_block)


def load_session_bundle(session_id: int) -> dict:
    """Load all per-session artifacts as a dict of DataFrames / arrays."""
    extract_session(session_id)
    d = _session_dir(session_id)
    units = pd.read_parquet(d / "units.parquet")
    units["ecephys_structure_acronym"] = units["ecephys_structure_acronym"].astype(str)
    spikes_npz = np.load(d / "spike_times.npz", allow_pickle=True)
    spikes = {k: spikes_npz[k] for k in spikes_npz.files}
    trials = pd.read_parquet(d / "trials.parquet") if (d / "trials.parquet").exists() else None
    stim = pd.read_parquet(d / "stimuli.parquet") if (d / "stimuli.parquet").exists() else None
    running = pd.read_parquet(d / "running.parquet") if (d / "running.parquet").exists() else None
    eye = pd.read_parquet(d / "eye_features.parquet") if (d / "eye_features.parquet").exists() else None
    licks = pd.read_parquet(d / "licks.parquet") if (d / "licks.parquet").exists() else None

    active_block, passive_block = _detect_active_passive_blocks(stim) if stim is not None else (0, 5)
    # Block time boundaries
    ab = stim[stim["stimulus_block"] == active_block]
    pb = stim[stim["stimulus_block"] == passive_block]
    active_range = (float(ab["t_start"].min()), float(ab["t_end"].max()))
    passive_range = (float(pb["t_start"].min()), float(pb["t_end"].max()))

    return dict(
        session_id=session_id,
        units=units,
        spikes=spikes,
        trials=trials,
        stim=stim,
        running=running,
        eye=eye,
        licks=licks,
        active_block=active_block,
        passive_block=passive_block,
        active_range=active_range,
        passive_range=passive_range,
        out_dir=d,
    )
