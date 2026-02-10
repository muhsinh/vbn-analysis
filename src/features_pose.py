"""Pose feature extraction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def sample_frame_indices(frame_times: pd.DataFrame, n_samples: int = 50) -> np.ndarray:
    if frame_times is None or frame_times.empty:
        return np.array([], dtype=int)
    total = len(frame_times)
    n_samples = min(n_samples, total)
    return np.linspace(0, total - 1, n_samples, dtype=int)


def scaffold_pose_project(session_id: int, tool: str, base_dir: Path) -> Path:
    project_dir = base_dir / f"session_{session_id}" / tool
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "README.txt").write_text(
        f"Pose project scaffold for session {session_id} using {tool}.\n",
        encoding="utf-8",
    )
    return project_dir


def export_frame_samples(
    video_path: Path,
    frame_indices: np.ndarray,
    output_dir: Path,
    frame_times: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        return output_dir
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return output_dir
    rows = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        img_path = output_dir / f"frame_{int(idx):06d}.png"
        cv2.imwrite(str(img_path), frame)
        t = frame_times.loc[frame_times["frame_idx"] == idx, "t"]
        t_val = float(t.iloc[0]) if not t.empty else np.nan
        rows.append({"frame_idx": int(idx), "t": t_val, "image_path": str(img_path)})
    cap.release()
    if rows:
        samples_path = output_dir / "frame_samples.csv"
        pd.DataFrame(rows).to_csv(samples_path, index=False)
        from config import make_provenance
        meta = {
            "timebase": "nwb_seconds",
            "provenance": make_provenance(None, "nwb"),
        }
        meta_path = samples_path.with_suffix(samples_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_dir


def export_labeling_frames(
    video_path: Path,
    frame_indices: np.ndarray,
    output_dir: Path,
    frame_times: pd.DataFrame,
    session_id: int,
    camera: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        return output_dir
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return output_dir

    rows = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"frame_{int(idx):06d}.png"
        img_path = frames_dir / filename
        cv2.imwrite(str(img_path), frame)
        t = frame_times.loc[frame_times["frame_idx"] == idx, "t"]
        t_val = float(t.iloc[0]) if not t.empty else np.nan
        rows.append(
            {
                "image_path": str(Path("frames") / filename),
                "session_id": session_id,
                "camera": camera,
                "frame_idx": int(idx),
                "t": t_val,
            }
        )

    cap.release()

    if rows:
        labels_path = output_dir / "labels.csv"
        pd.DataFrame(rows).to_csv(labels_path, index=False)
    return output_dir


def derive_pose_features(pose_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if pose_df is None or pose_df.empty:
        return None
    df = pose_df.copy()
    if "t" not in df.columns:
        return None
    # Example: use first keypoint columns
    kp_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if not kp_cols:
        return df[["t"]]

    df["pose_speed"] = 0.0
    try:
        x_cols = [c for c in kp_cols if c.endswith("_x")]
        y_cols = [c for c in kp_cols if c.endswith("_y")]
        if x_cols and y_cols:
            x = df[x_cols[0]].to_numpy()
            y = df[y_cols[0]].to_numpy()
            dx = np.gradient(x, df["t"].to_numpy())
            dy = np.gradient(y, df["t"].to_numpy())
            df["pose_speed"] = np.sqrt(dx ** 2 + dy ** 2)
    except Exception:
        pass

    return df[["t", "pose_speed"]]
