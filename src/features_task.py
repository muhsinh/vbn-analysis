"""Task/behavior feature extraction."""
from __future__ import annotations

import pandas as pd


def derive_task_features(trials: pd.DataFrame | None, events: pd.DataFrame | None) -> pd.DataFrame | None:
    if trials is None or trials.empty:
        return None
    df = trials.copy()
    if "t_start" in df.columns:
        df["t"] = df["t_start"]
    elif "start_time" in df.columns:
        df["t"] = df["start_time"]
    else:
        df["t"] = range(len(df))
    # Keep a compact set of columns
    cols = ["t"] + [c for c in ["trial_type", "response", "rewarded", "stimulus_name"] if c in df.columns]
    return df[cols]
