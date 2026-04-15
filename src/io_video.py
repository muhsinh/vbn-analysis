"""Video asset discovery and frame time alignment (download-first)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import get_config, make_provenance
from io_s3 import list_video_assets as list_s3_assets
from io_s3 import download_asset
from timebase import write_parquet_with_timebase



def load_timestamps(path: Path) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if data.files:
            return data[data.files[0]]
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        if "t" in df.columns:
            return df["t"].to_numpy()
        return df.iloc[:, 0].to_numpy()
    return None


def _candidate_roots(session_id: int, video_dir: Path | None, cache_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if video_dir:
        video_dir = Path(video_dir)
        roots.extend(
            [
                video_dir,
                video_dir / str(session_id),
                video_dir / str(session_id) / "behavior_videos",
            ]
        )
    if cache_dir:
        roots.append(Path(cache_dir) / str(session_id) / "behavior_videos")
    # De-dup while preserving order
    seen = set()
    uniq: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _find_first(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _resolve_local_assets(
    session_id: int,
    camera: str,
    video_dir: Path | None,
    cache_dir: Path | None,
) -> dict[str, Path | None]:
    roots = _candidate_roots(session_id, video_dir, cache_dir)
    video_path = None
    ts_path = None
    meta_path = None

    for root in roots:
        if video_path is None:
            video_path = _find_first(root, [f"{camera}.mp4"])
        if ts_path is None:
            ts_path = _find_first(root, [f"{camera}_timestamps.npy", f"{camera}_timestamps.npz"])
        if meta_path is None:
            meta_path = _find_first(root, [f"{camera}_metadata.json", f"{camera}_metadata.mp4"])

    return {
        "video": video_path,
        "timestamps": ts_path,
        "metadata": meta_path,
    }


def _compute_frame_metrics(
    session_id: int,
    camera: str,
    timestamps: np.ndarray | None,
) -> tuple[pd.DataFrame | None, dict[str, Any], list[str]]:
    qc_flags: list[str] = []
    metrics: dict[str, Any] = {
        "n_frames": None,
        "fps_est": None,
        "t0": None,
        "tN": None,
    }
    if timestamps is None:
        qc_flags.append("NO_TIMESTAMPS")
        return None, metrics, qc_flags

    ts = np.asarray(timestamps, dtype=float)
    finite = np.isfinite(ts)
    if not np.all(finite):
        qc_flags.append("TIMESTAMP_NAN_PRESENT")

    valid_ts = ts[finite]
    if valid_ts.size == 0:
        qc_flags.append("NO_VALID_TIMESTAMPS")
        return None, metrics, qc_flags

    frame_idx = np.arange(len(ts))[finite]
    frame_times_df = pd.DataFrame(
        {
            "session_id": session_id,
            "camera": camera,
            "frame_idx": frame_idx,
            "t": valid_ts,
        }
    )

    metrics["n_frames"] = int(valid_ts.size)
    metrics["t0"] = float(valid_ts[0])
    metrics["tN"] = float(valid_ts[-1])

    if valid_ts.size >= 2:
        diffs = np.diff(valid_ts)
        med = float(np.median(diffs))
        if med > 0:
            metrics["fps_est"] = 1.0 / med

    from qc import compute_video_qc  # local import: qc is a leaf module; kept here to avoid io_video → qc top-level coupling
    qc = compute_video_qc(frame_times_df[["frame_idx", "t"]], metrics["fps_est"])
    if qc.get("monotonic") is False:
        qc_flags.append("NON_MONOTONIC")
    if (qc.get("dropped_frames") or 0) > 0:
        qc_flags.append("DROPPED_FRAMES")

    return frame_times_df, metrics, qc_flags


def _join_flags(flags: list[str]) -> str:
    uniq = []
    seen = set()
    for flag in flags:
        if not flag or flag in seen:
            continue
        seen.add(flag)
        uniq.append(flag)
    return "|".join(uniq)


def build_video_assets(
    session_id: int,
    video_dir: Path | None = None,
    outputs_dir: Path | None = None,
    download_missing: bool | None = None,
) -> pd.DataFrame:
    cfg = get_config()
    if outputs_dir is None:
        outputs_dir = cfg.outputs_dir / "video"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if download_missing is None:
        download_missing = cfg.video_source in {"auto", "s3"}

    s3_assets = list_s3_assets(session_id, cameras=cfg.video_cameras)

    asset_rows: list[dict[str, Any]] = []
    frame_times_rows: list[pd.DataFrame] = []

    for camera in cfg.video_cameras:
        local_assets = _resolve_local_assets(
            session_id=session_id,
            camera=camera,
            video_dir=video_dir,
            cache_dir=cfg.video_cache_dir,
        )
        qc_flags: list[str] = []

        local_video = local_assets["video"]
        local_ts = local_assets["timestamps"]
        local_meta = local_assets["metadata"]

        downloaded = False
        if download_missing:
            if local_video is None:
                try:
                    local_video = (
                        Path(cfg.video_cache_dir)
                        / str(session_id)
                        / "behavior_videos"
                        / f"{camera}.mp4"
                    )
                    download_asset(s3_assets[camera]["s3_uri_video"], local_video)
                    downloaded = True
                except Exception:
                    local_video = None
                    qc_flags.append("DOWNLOAD_FAILED_VIDEO")

            if local_ts is None:
                try:
                    local_ts = (
                        Path(cfg.video_cache_dir)
                        / str(session_id)
                        / "behavior_videos"
                        / f"{camera}_timestamps.npy"
                    )
                    download_asset(s3_assets[camera]["s3_uri_timestamps"], local_ts)
                    downloaded = True
                except Exception:
                    local_ts = None
                    qc_flags.append("DOWNLOAD_FAILED_TIMESTAMPS")

            if local_meta is None:
                try:
                    local_meta = (
                        Path(cfg.video_cache_dir)
                        / str(session_id)
                        / "behavior_videos"
                        / f"{camera}_metadata.json"
                    )
                    download_asset(s3_assets[camera]["s3_uri_metadata"], local_meta)
                    downloaded = True
                except Exception:
                    local_meta = None
                    qc_flags.append("DOWNLOAD_FAILED_METADATA")

        timestamps = load_timestamps(local_ts) if local_ts else None
        frame_times_df, metrics, ts_flags = _compute_frame_metrics(session_id, camera, timestamps)
        qc_flags.extend(ts_flags)

        if frame_times_df is not None:
            frame_times_rows.append(frame_times_df)

        source = "local"
        if downloaded or (download_missing and local_video is None and local_ts is None):
            source = "s3"

        asset_rows.append(
            {
                "session_id": session_id,
                "camera": camera,
                "source": source,
                "s3_uri_video": s3_assets[camera]["s3_uri_video"],
                "s3_uri_timestamps": s3_assets[camera]["s3_uri_timestamps"],
                "s3_uri_metadata": s3_assets[camera]["s3_uri_metadata"],
                "http_url_video": s3_assets[camera]["http_url_video"],
                "local_video_path": str(local_video) if local_video else None,
                "local_timestamps_path": str(local_ts) if local_ts else None,
                "local_metadata_path": str(local_meta) if local_meta else None,
                "n_frames": metrics["n_frames"],
                "fps_est": metrics["fps_est"],
                "t0": metrics["t0"],
                "tN": metrics["tN"],
                "qc_flags": _join_flags(qc_flags),
            }
        )

    assets_df = pd.DataFrame(asset_rows)
    assets_path = outputs_dir / "video_assets.parquet"
    _upsert_assets(assets_df, assets_path)

    if frame_times_rows:
        frames_df = pd.concat(frame_times_rows, ignore_index=True)
        frames_path = outputs_dir / "frame_times.parquet"
        _upsert_frame_times(frames_df, frames_path)

    return assets_df


def _drop_existing_session_camera_keys(existing: pd.DataFrame, keys: set) -> pd.DataFrame:
    if existing.empty:
        return existing
    existing_keys = existing.set_index(["session_id", "camera"]).index
    mask = [key not in keys for key in existing_keys]
    return existing.loc[mask].reset_index(drop=True)


def _upsert_assets(new_rows: pd.DataFrame, path: Path) -> Path:
    if new_rows is None or new_rows.empty:
        return path
    if path.exists():
        existing = pd.read_parquet(path)
        keys = set(map(tuple, new_rows[["session_id", "camera"]].values))
        existing = _drop_existing_session_camera_keys(existing, keys)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_parquet(path, index=False)
    return path


def _upsert_frame_times(new_rows: pd.DataFrame, path: Path) -> Path:
    if new_rows is None or new_rows.empty:
        return path
    if path.exists():
        existing = pd.read_parquet(path)
        keys = set(map(tuple, new_rows[["session_id", "camera"]].drop_duplicates().values))
        existing = _drop_existing_session_camera_keys(existing, keys)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    write_parquet_with_timebase(
        combined,
        path,
        timebase="nwb_seconds",
        provenance=make_provenance(None, "timestamps"),
        required_columns=["session_id", "camera", "frame_idx", "t"],
    )
    return path


def load_video_assets(session_id: int | None = None, camera: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    path = cfg.outputs_dir / "video" / "video_assets.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if session_id is not None:
        df = df[df["session_id"] == session_id]
    if camera is not None:
        df = df[df["camera"] == camera]
    return df.reset_index(drop=True)


def load_frame_times(session_id: int | None = None, camera: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    path = cfg.outputs_dir / "video" / "frame_times.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if session_id is not None:
        df = df[df["session_id"] == session_id]
    if camera is not None:
        df = df[df["camera"] == camera]
    return df.reset_index(drop=True)
