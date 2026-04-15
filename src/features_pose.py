"""Pose feature extraction.

Expanded feature set for neural-behavior correlation:
- Per-keypoint velocity and acceleration
- Joint angles and body geometry
- Inter-keypoint distances
- Confidence-weighted filtering
- Behavioral state features (stillness, locomotion)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_frame_indices(frame_indices: np.ndarray | list[int] | None) -> np.ndarray:
    if frame_indices is None:
        return np.array([], dtype=int)
    arr = np.asarray(frame_indices, dtype=int)
    if arr.size == 0:
        return arr
    arr = np.unique(arr)
    arr.sort()
    return arr


def _build_time_map(frame_times: pd.DataFrame | None) -> dict[int, float]:
    t_map: dict[int, float] = {}
    if (
        frame_times is None
        or frame_times.empty
        or "frame_idx" not in frame_times.columns
        or "t" not in frame_times.columns
    ):
        return t_map
    ft = frame_times[["frame_idx", "t"]].copy()
    ft["frame_idx"] = pd.to_numeric(ft["frame_idx"], errors="coerce")
    ft["t"] = pd.to_numeric(ft["t"], errors="coerce")
    ft = ft[np.isfinite(ft["frame_idx"]) & np.isfinite(ft["t"])]
    t_map = {int(fi): float(tv) for fi, tv in zip(ft["frame_idx"].astype(int), ft["t"].astype(float))}
    return t_map


def sample_frame_indices(frame_times: pd.DataFrame, n_samples: int = 50) -> np.ndarray:
    """Return a set of frame indices sampled across the available frame_times."""
    if frame_times is None or frame_times.empty:
        return np.array([], dtype=int)
    df = frame_times.copy()
    if "frame_idx" not in df.columns:
        return np.array([], dtype=int)
    if "t" in df.columns:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df[np.isfinite(df["t"])].reset_index(drop=True)
    if df.empty:
        return np.array([], dtype=int)
    total = len(df)
    n_samples = min(int(n_samples), total)
    row_idx = np.linspace(0, total - 1, n_samples, dtype=int)
    sampled = pd.to_numeric(df.iloc[row_idx]["frame_idx"], errors="coerce").dropna().astype(int).to_numpy()
    sampled = np.unique(sampled)
    sampled.sort()
    return sampled


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
        raise ImportError(
            "OpenCV is required to export labeling frames.\n"
            "Install via conda-forge (recommended): `conda install -c conda-forge opencv`\n"
            "or via pip: `python -m pip install opencv-python`.\n"
            "Then restart your Jupyter kernel and re-run Notebook 06."
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for labeling export: {video_path}")

    frame_indices = _normalize_frame_indices(frame_indices)
    if frame_indices is None or len(frame_indices) == 0:
        cap.release()
        raise RuntimeError(f"No frame indices provided for session_id={session_id} camera={camera}.")

    t_map = _build_time_map(frame_times)

    rows = []
    for seq_i, idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"{seq_i:06d}.png"
        img_path = frames_dir / filename
        cv2.imwrite(str(img_path), frame)
        t_val = t_map.get(int(idx), np.nan)
        rows.append(
            {
                "image_path": str(Path("frames") / filename),
                "session_id": session_id,
                "camera": camera,
                "seq_idx": int(seq_i),
                "frame_idx": int(idx),
                "t": t_val,
            }
        )

    cap.release()

    if rows:
        labels_path = output_dir / "labels.csv"
        pd.DataFrame(rows).to_csv(labels_path, index=False)
    else:
        raise RuntimeError(
            f"No frames were exported for session_id={session_id} camera={camera}. "
            "This usually means OpenCV could not decode/seeking failed for the requested frames."
        )
    return output_dir


def export_labeling_video(
    video_path: Path,
    frame_indices: np.ndarray,
    output_dir: Path,
    frame_times: pd.DataFrame,
    session_id: int,
    camera: str,
    label_fps: float = 30.0,
    write_pngs: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required to export labeling videos.\n"
            "Install via conda-forge (recommended): `conda install -c conda-forge opencv`\n"
            "or via pip: `python -m pip install opencv-python`.\n"
            "Then restart your Jupyter kernel and re-run Notebook 06."
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for labeling export: {video_path}")

    frame_indices = _normalize_frame_indices(frame_indices)
    if frame_indices is None or len(frame_indices) == 0:
        cap.release()
        raise RuntimeError(f"No frame indices provided for session_id={session_id} camera={camera}.")

    first_frame = None
    first_idx = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            first_frame = frame
            first_idx = int(idx)
            break
    if first_frame is None:
        cap.release()
        raise RuntimeError(
            f"Failed to read any frames for session_id={session_id} camera={camera} from {video_path}."
        )

    height, width = first_frame.shape[:2]

    def _open_writer(path: Path, fourcc: int) -> "cv2.VideoWriter":
        return cv2.VideoWriter(str(path), fourcc, float(label_fps), (width, height))

    mp4_path = output_dir / "labeling.mp4"
    avi_path = output_dir / "labeling.avi"
    writer = _open_writer(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"))
    video_out = mp4_path
    if not writer.isOpened():
        writer.release()
        writer = _open_writer(avi_path, cv2.VideoWriter_fourcc(*"XVID"))
        video_out = avi_path
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to open VideoWriter for MP4 or AVI output.")

    frames_dir = None
    if write_pngs:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    t_map = _build_time_map(frame_times)
    rows = []
    seq_i = 0
    preloaded = {first_idx: first_frame}

    for idx in frame_indices:
        frame = preloaded.get(int(idx))
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        seq_i += 1

        image_path = None
        if frames_dir is not None:
            filename = f"{seq_i:06d}.png"
            img_path = frames_dir / filename
            cv2.imwrite(str(img_path), frame)
            image_path = str(Path("frames") / filename)

        t_val = t_map.get(int(idx), np.nan)
        row = {
            "video_path": str(Path(video_out.name)),
            "session_id": session_id,
            "camera": camera,
            "seq_idx": int(seq_i),
            "frame_idx": int(idx),
            "t": t_val,
        }
        if image_path is not None:
            row["image_path"] = image_path
        rows.append(row)

    writer.release()
    cap.release()

    if rows:
        labels_path = output_dir / "labels.csv"
        pd.DataFrame(rows).to_csv(labels_path, index=False)
    else:
        raise RuntimeError(
            f"No frames were exported for session_id={session_id} camera={camera}. "
            "This usually means OpenCV could not decode/seeking failed for the requested frames."
        )
    return output_dir


def _find_keypoints(df: pd.DataFrame) -> list[str]:
    """Find keypoint names from columns ending in _x / _y."""
    x_cols = [c for c in df.columns if c.endswith("_x")]
    names = []
    for c in x_cols:
        name = c[:-2]  # strip _x
        if f"{name}_y" in df.columns:
            names.append(name)
    return names


def _get_keypoint_xy(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    return df[f"{name}_x"].to_numpy(dtype=float), df[f"{name}_y"].to_numpy(dtype=float)


def filter_by_confidence(
    df: pd.DataFrame,
    threshold: float = 0.3,
    method: str = "nan",
) -> pd.DataFrame:
    """Filter low-confidence keypoint detections.

    Parameters
    ----------
    df : DataFrame with keypoint columns (name_x, name_y, name_score)
    threshold : minimum confidence score
    method : 'nan' (replace with NaN) or 'drop' (drop entire row)

    Returns
    -------
    Filtered DataFrame
    """
    df = df.copy()
    keypoints = _find_keypoints(df)

    if method == "nan":
        for kp in keypoints:
            score_col = f"{kp}_score"
            if score_col in df.columns:
                low = df[score_col] < threshold
                df.loc[low, f"{kp}_x"] = np.nan
                df.loc[low, f"{kp}_y"] = np.nan
    elif method == "drop":
        score_cols = [f"{kp}_score" for kp in keypoints if f"{kp}_score" in df.columns]
        if score_cols:
            mean_score = df[score_cols].mean(axis=1)
            df = df[mean_score >= threshold].reset_index(drop=True)

    return df


def derive_pose_features(
    pose_df: pd.DataFrame | None,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame | None:
    """Extract rich behavioral features from pose predictions.

    Features computed:
    - pose_speed: overall body speed (mean of all keypoint speeds)
    - per-keypoint velocities: {name}_vel
    - per-keypoint accelerations: {name}_accel
    - body_length: distance between first and last keypoint (proxy for stretch)
    - head_angle: angle of first keypoint pair (if >= 2 keypoints)
    - stillness: binary flag (pose_speed < threshold)
    """
    if pose_df is None or pose_df.empty:
        return None
    df = pose_df.copy()
    if "t" not in df.columns:
        return None

    t = df["t"].to_numpy(dtype=float)
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6  # avoid division by zero

    if confidence_threshold > 0:
        df = filter_by_confidence(df, confidence_threshold, method="nan")

    keypoints = _find_keypoints(df)
    if not keypoints:
        return df[["t"]]

    all_speeds = []
    for kp in keypoints:
        x, y = _get_keypoint_xy(df, kp)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        speed = np.sqrt(vx**2 + vy**2)
        accel = np.gradient(speed, t)

        df[f"{kp}_vel"] = speed
        df[f"{kp}_accel"] = accel
        all_speeds.append(speed)

    speed_matrix = np.column_stack(all_speeds)
    with np.errstate(all="ignore"):
        df["pose_speed"] = np.nanmean(speed_matrix, axis=1)
        df["pose_speed_std"] = np.nanstd(speed_matrix, axis=1)

    if len(keypoints) >= 2:
        x0, y0 = _get_keypoint_xy(df, keypoints[0])
        x1, y1 = _get_keypoint_xy(df, keypoints[-1])
        df["body_length"] = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    if len(keypoints) >= 2:
        x0, y0 = _get_keypoint_xy(df, keypoints[0])
        x1, y1 = _get_keypoint_xy(df, keypoints[1])
        df["head_angle"] = np.arctan2(y0 - y1, x0 - x1)
        df["head_angular_vel"] = np.gradient(np.unwrap(df["head_angle"].to_numpy()), t)

    for i in range(len(keypoints) - 1):
        xi, yi = _get_keypoint_xy(df, keypoints[i])
        xj, yj = _get_keypoint_xy(df, keypoints[i + 1])
        df[f"dist_{keypoints[i]}_{keypoints[i+1]}"] = np.sqrt((xj - xi)**2 + (yj - yi)**2)

    speed_threshold = np.nanpercentile(df["pose_speed"].to_numpy(), 10)
    df["is_still"] = (df["pose_speed"] < max(speed_threshold, 1.0)).astype(int)

    output_cols = ["t", "pose_speed", "pose_speed_std"]
    for kp in keypoints:
        output_cols.extend([f"{kp}_vel", f"{kp}_accel"])
    optional = ["body_length", "head_angle", "head_angular_vel", "is_still"]
    output_cols.extend([c for c in optional if c in df.columns])
    output_cols.extend([c for c in df.columns if c.startswith("dist_")])

    return df[[c for c in output_cols if c in df.columns]]


def _load_camera_timestamps(session_id: int, camera: str) -> np.ndarray | None:
    from config import get_config
    cfg = get_config()
    base = cfg.video_cache_dir / str(session_id) / "behavior_videos"
    candidates = [
        base / f"{camera}_timestamps.npy",
        base / f"{camera}_timestamps.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".npz":
            data = np.load(path)
            if data.files:
                return data[data.files[0]]
    return None


def _attach_timestamps(
    df: pd.DataFrame,
    frame_times: pd.DataFrame | None,
    timestamps: np.ndarray | None,
) -> pd.DataFrame:
    df = df.copy()
    if "frame_idx" not in df.columns:
        raise ValueError("Expected 'frame_idx' column in SLEAP CSV export.")

    if frame_times is not None and not frame_times.empty and "t" in frame_times.columns:
        ft = frame_times[["frame_idx", "t"]].copy()
        ft["frame_idx"] = pd.to_numeric(ft["frame_idx"], errors="coerce")
        ft["t"] = pd.to_numeric(ft["t"], errors="coerce")
        ft = ft[np.isfinite(ft["frame_idx"]) & np.isfinite(ft["t"])]
        ft["frame_idx"] = ft["frame_idx"].astype(int)
        df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
        df = df.merge(ft, on="frame_idx", how="left")
        return df

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float)
        frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce")
        t_vals = np.full(len(df), np.nan)
        valid = frame_idx.notna()
        frame_idx = frame_idx.fillna(-1).astype(int)
        in_bounds = valid & (frame_idx >= 0) & (frame_idx < len(ts))
        if in_bounds.any():
            t_vals[in_bounds.to_numpy()] = ts[frame_idx[in_bounds].to_numpy()]
        df["t"] = t_vals
        return df

    df["t"] = np.nan
    return df


def export_pose_predictions_from_sleap_csv(
    csv_path: Path,
    session_id: int,
    camera: str,
    frame_times: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """Convert SLEAP CSV export (wide format) into pose_predictions.parquet."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"SLEAP CSV export not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "frame_idx" not in df.columns:
        if "frame" in df.columns:
            df = df.rename(columns={"frame": "frame_idx"})
        else:
            raise ValueError("SLEAP CSV export must include a 'frame_idx' or 'frame' column.")

    rename_map = {col: col.replace(".", "_") for col in df.columns if "." in col}
    if rename_map:
        df = df.rename(columns=rename_map)

    df["session_id"] = session_id
    df["camera"] = camera

    timestamps = None
    if frame_times is None or frame_times.empty:
        timestamps = _load_camera_timestamps(session_id, camera)
    df = _attach_timestamps(df, frame_times, timestamps)

    front = ["session_id", "camera", "frame_idx", "t"]
    remaining = [c for c in df.columns if c not in front]
    df = df[front + remaining]

    if output_path is None:
        from config import get_config
        cfg = get_config()
        output_path = cfg.outputs_dir / "pose" / f"session_{session_id}_pose_predictions.parquet"

    from config import make_provenance
    from timebase import write_parquet_with_timebase

    write_parquet_with_timebase(
        df,
        output_path,
        provenance=make_provenance(session_id, "nwb"),
        required_columns=["t", "frame_idx"],
    )
    return output_path
