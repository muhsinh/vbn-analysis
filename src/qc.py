"""QC utilities for alignment and signal sanity checks."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def check_monotonic(times: np.ndarray) -> bool:
    return np.all(np.diff(times) > 0)


def detect_dropped_frames(times: np.ndarray, threshold_factor: float = 2.5) -> int:
    if len(times) < 3:
        return 0
    diffs = np.diff(times)
    med = np.median(diffs)
    return int(np.sum(diffs > threshold_factor * med))


def estimate_fps(times: np.ndarray) -> float | None:
    if len(times) < 2:
        return None
    dt = np.median(np.diff(times))
    if dt <= 0:
        return None
    return 1.0 / dt


def compute_video_qc(frame_times: pd.DataFrame, fps_nominal: float | None) -> Dict[str, Any]:
    times = frame_times["t"].to_numpy()
    monotonic = check_monotonic(times)
    dropped = detect_dropped_frames(times)
    fps_est = estimate_fps(times)
    drift = None
    if fps_nominal and fps_est:
        drift = float((fps_est - fps_nominal) / fps_nominal)
    return {
        "monotonic": monotonic,
        "dropped_frames": dropped,
        "fps_nominal": fps_nominal,
        "fps_estimated": fps_est,
        "drift_fraction": drift,
    }


def eye_qc_summary(eye_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    if eye_df is None or eye_df.empty:
        return {"available": False}
    summary["available"] = True
    summary["n_samples"] = int(len(eye_df))
    summary["missing_fraction"] = float(eye_df.isna().mean().mean())
    return summary
