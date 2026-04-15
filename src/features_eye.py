"""Eye tracking feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def derive_eye_features(eye_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Derive analysis-ready eye features from raw VBN eye tracking data.

    Input is the output of io_nwb.extract_eye_tracking(), which returns
    structured columns: t, pupil_area, pupil_x, pupil_y, pupil_width,
    pupil_height, pupil_angle, likely_blink.

    Adds:
      - pupil_z        : z-scored pupil area (robust to blinks)
      - pupil_vel      : d/dt of pupil area
      - pupil_x_z      : z-scored pupil x position
      - pupil_y_z      : z-scored pupil y position
    """
    if eye_df is None or eye_df.empty:
        return None
    df = eye_df.copy()
    if "t" not in df.columns:
        df = df.reset_index().rename(columns={"index": "t"})

    signal_cols = [c for c in df.columns if c != "t"]
    if not signal_cols:
        return df[["t"]]

    t = df["t"].to_numpy(dtype=float)

    # Prefer the structured pupil_area column; fall back to first signal column
    # for backwards compatibility with non-VBN NWB files.
    area_col = "pupil_area" if "pupil_area" in df.columns else signal_cols[0]
    area = df[area_col].to_numpy(dtype=float)

    # Blink mask: exclude blink frames from statistics
    if "likely_blink" in df.columns:
        blink = df["likely_blink"].astype(bool).to_numpy()
        area_no_blink = area.copy()
        area_no_blink[blink] = np.nan
    else:
        area_no_blink = area

    df["pupil"] = area
    df["pupil_z"] = (area - np.nanmean(area_no_blink)) / (np.nanstd(area_no_blink) + 1e-6)
    df["pupil_vel"] = np.gradient(area, t)

    # Z-score position columns if available
    for pos_col in ("pupil_x", "pupil_y"):
        if pos_col in df.columns:
            vals = df[pos_col].to_numpy(dtype=float)
            df[f"{pos_col}_z"] = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-6)

    # Build output column list: always include all input columns plus derived ones
    derived = ["pupil", "pupil_z", "pupil_vel"]
    if "pupil_x" in df.columns:
        derived.append("pupil_x_z")
    if "pupil_y" in df.columns:
        derived.append("pupil_y_z")

    all_cols = ["t"] + signal_cols + [c for c in derived if c not in signal_cols]
    return df[[c for c in all_cols if c in df.columns]]
