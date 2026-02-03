"""Eye tracking feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def derive_eye_features(eye_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if eye_df is None or eye_df.empty:
        return None
    df = eye_df.copy()
    if "t" not in df.columns:
        # Assume index is time
        df = df.reset_index().rename(columns={"index": "t"})

    signal_cols = [c for c in df.columns if c != "t"]
    if not signal_cols:
        return df[["t"]]

    primary = signal_cols[0]
    df["pupil"] = df[primary]
    df["pupil_z"] = (df[primary] - np.nanmean(df[primary])) / (np.nanstd(df[primary]) + 1e-6)
    df["pupil_vel"] = np.gradient(df[primary].to_numpy(), df["t"].to_numpy())
    return df[["t", "pupil", "pupil_z", "pupil_vel"]]
