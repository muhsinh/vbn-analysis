"""Shared loading helpers for the full-session analysis pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import io_nwb  # noqa: E402

SESSION_ID = 1055240613
BIN_SIZE = 0.025

ACTIVE_BLOCK = 0
PASSIVE_BLOCK = 5

VISUAL_HIERARCHY = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam"]
MOTOR_AREAS = ["SCig", "SCiw", "MRN"]
HIPPOCAMPAL_AREAS = ["CA1", "CA3", "DG", "ProS", "SUB", "POST"]
THALAMIC_AREAS = ["LGd", "LP"]  # MGd/MGv/MGm dropped: session 1055240613 has
# off-target probe registration flagging MGd units as likely LP-adjacent rather
# than genuine medial geniculate. Per VBN insider critique (2026-04-16).
ALL_TARGET_AREAS = VISUAL_HIERARCHY + MOTOR_AREAS + HIPPOCAMPAL_AREAS[:3]

OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
REPORTS.mkdir(exist_ok=True)


def load_session() -> dict:
    """Load all cached artifacts for the primary session."""
    units = pd.read_parquet(OUTPUTS / "neural" / f"session_{SESSION_ID}_units.parquet")
    units["ecephys_structure_acronym"] = units["ecephys_structure_acronym"].astype(str)
    spikes = dict(io_nwb.load_spike_times_npz(
        OUTPUTS / "neural" / f"session_{SESSION_ID}_spike_times.npz"
    ))
    trials = pd.read_parquet(OUTPUTS / "behavior" / f"session_{SESSION_ID}_trials.parquet")
    stim = pd.read_parquet(OUTPUTS / "behavior" / f"session_{SESSION_ID}_stimuli.parquet")
    running = pd.read_parquet(OUTPUTS / "behavior" / f"session_{SESSION_ID}_running.parquet")
    pose = pd.read_parquet(OUTPUTS / "pose" / f"session_{SESSION_ID}_pose_features.parquet")
    return dict(units=units, spikes=spikes, trials=trials, stim=stim,
                running=running, pose=pose)


def block_bounds(stim: pd.DataFrame, block: int) -> tuple[float, float]:
    sub = stim[stim["stimulus_block"] == block]
    return float(sub["t_start"].min()), float(sub["t_end"].max())


def area_units(units: pd.DataFrame, spikes: dict, area: str) -> list[str]:
    ids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()]
    return [u for u in ids if u in spikes]


def time_grid_for_block(t_start: float, t_end: float, bin_size: float = BIN_SIZE) -> np.ndarray:
    return np.arange(t_start, t_end, bin_size)


def interp_to_grid(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    m = df[["t", col]].dropna().sort_values("t")
    return np.interp(grid, m["t"].values, m[col].values)


def bin_unit(spike_times: np.ndarray, grid: np.ndarray, bin_size: float = BIN_SIZE) -> np.ndarray:
    edges = np.append(grid, grid[-1] + bin_size)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts
