"""Video discovery and frame time alignment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import get_config
from qc import compute_video_qc
from timebase import write_parquet_with_timebase


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_video_files(session_id: int, search_dirs: List[Path]) -> List[Path]:
    hits: List[Path] = []
    session_str = str(session_id)
    for root in search_dirs:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS and session_str in path.name:
                hits.append(path)
    return hits


def get_video_metadata(path: Path) -> Tuple[float | None, int | None]:
    try:
        import cv2
    except ImportError:
        return None, None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    fps_val = float(fps) if fps and fps > 0 else None
    frame_val = int(frame_count) if frame_count and frame_count > 0 else None
    return fps_val, frame_val


def create_preview_clip(video_path: Path, output_path: Path, max_seconds: int = 5) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-t",
                str(max_seconds),
                "-c",
                "copy",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_path
    except Exception:
        try:
            import cv2
        except ImportError:
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        max_frames = int(fps * max_seconds)
        count = 0
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
        cap.release()
        writer.release()
        return output_path


def load_timestamps(path: Path) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        # take first array
        if data.files:
            return data[data.files[0]]
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        if "t" in df.columns:
            return df["t"].to_numpy()
        return df.iloc[:, 0].to_numpy()
    return None


def align_frame_times(
    frame_count: int | None,
    fps: float | None,
    t0: float | None,
    timestamps: np.ndarray | None,
    anchors: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame | None, str, List[str]]:
    """Return frame times dataframe, alignment_method, qc_flags."""
    qc_flags: List[str] = []

    if timestamps is not None and len(timestamps) > 0:
        frame_idx = np.arange(len(timestamps))
        df = pd.DataFrame({"frame_idx": frame_idx, "t": timestamps})
        return df, "direct_timestamps", qc_flags

    if anchors:
        try:
            anchor_frames = np.asarray(anchors["frame_idx"], dtype=float)
            anchor_times = np.asarray(anchors["t"], dtype=float)
            if frame_count is None:
                frame_count = int(anchor_frames.max()) + 1
            frame_idx = np.arange(frame_count)
            mapped = np.interp(frame_idx, anchor_frames, anchor_times)
            df = pd.DataFrame({"frame_idx": frame_idx, "t": mapped})
            return df, "sync_anchors", qc_flags
        except Exception:
            pass

    if frame_count is not None and fps is not None and t0 is not None:
        frame_idx = np.arange(frame_count)
        times = t0 + frame_idx / fps
        df = pd.DataFrame({"frame_idx": frame_idx, "t": times})
        qc_flags.append("LOW_CONFIDENCE_ALIGNMENT")
        return df, "fps_fallback", qc_flags

    # Tier 4: unavailable
    qc_flags.append("NO_FRAME_TIMES")
    return None, "unavailable", qc_flags


def build_video_manifest(
    session_id: int,
    nwb_path: Path | None,
    video_dir: Path | None,
    access_mode: str,
    outputs_dir: Path,
    prefer_download: bool = False,
) -> Dict[str, Any]:
    cfg = get_config()
    search_dirs: List[Path] = []
    if video_dir:
        search_dirs.append(video_dir)
    # fallback: look under outputs/video
    search_dirs.append(outputs_dir)

    videos = discover_video_files(session_id, search_dirs)
    streams: List[Dict[str, Any]] = []
    frame_times_df: pd.DataFrame | None = None
    alignment_method = "unavailable"
    qc_flags: List[str] = []

    selected_video = videos[0] if videos else None
    fps = None
    frame_count = None
    if selected_video:
        fps, frame_count = get_video_metadata(selected_video)

    timestamps = None
    timestamps_path = None
    if selected_video:
        # look for sibling timestamp file
        for suffix in [".npy", ".npz", ".csv"]:
            candidate = selected_video.with_suffix(selected_video.suffix + suffix)
            if candidate.exists():
                timestamps_path = candidate
                timestamps = load_timestamps(candidate)
                break

    frame_times_df, alignment_method, qc_flags = align_frame_times(
        frame_count=frame_count,
        fps=fps,
        t0=0.0 if frame_count is not None and fps is not None else None,
        timestamps=timestamps,
        anchors=None,
    )

    if selected_video:
        streams.append(
            {
                "stream_name": "primary",
                "file_path": str(selected_video),
                "fps": fps,
                "frame_count": frame_count,
                "timestamps_path": str(timestamps_path) if timestamps_path else None,
                "alignment_method": alignment_method,
                "qc_flags": qc_flags,
            }
        )

    from config import make_provenance

    manifest = {
        "session_id": session_id,
        "streams": streams,
        "alignment_method": alignment_method,
        "qc_flags": qc_flags,
        "timebase": "nwb_seconds",
        "provenance": make_provenance(session_id, alignment_method),
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / f"session_{session_id}_video_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Frame times output when available
    if frame_times_df is not None and selected_video is not None:
        frame_times_df["stream_name"] = "primary"
        frame_path = outputs_dir / f"session_{session_id}_frame_times.parquet"
        write_parquet_with_timebase(
            frame_times_df,
            frame_path,
            timebase="nwb_seconds",
            provenance=manifest["provenance"],
            required_columns=["frame_idx", "t", "stream_name"],
        )
        # QC output
        qc = compute_video_qc(frame_times_df, fps)
        qc_path = outputs_dir / f"session_{session_id}_video_qc.json"
        with qc_path.open("w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2)

    # Preview clip
    if selected_video is not None:
        preview_path = outputs_dir / f"session_{session_id}_preview.mp4"
        create_preview_clip(selected_video, preview_path)

    return manifest
