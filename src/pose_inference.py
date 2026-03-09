"""Automated SLEAP pose inference pipeline.

Eliminates the manual label-export-paste cycle by:
1. Discovering trained SLEAP models in pose_projects/ or user-specified paths
2. Running batch inference on videos programmatically
3. Auto-loading predictions into the standard pose_predictions.parquet format

With ~150 labeled frames you can train a usable SLEAP model.  This module
runs inference on *all locally-cached videos* so you don't need to download
the full 500 GB dataset -- just the sessions you've already fetched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_sleap_models(search_dirs: List[Path] | None = None) -> List[Dict[str, Any]]:
    """Find trained SLEAP models on disk.

    Searches pose_projects/, outputs/models/, and any user-specified dirs.
    A SLEAP model is identified by a directory containing a `training_config.json`
    or a `.h5`/`.keras` weights file alongside a `confmap` or `centroid` folder.
    Also supports single-file `.zip` or `.pkg.slp` model packages.
    """
    from config import get_config
    cfg = get_config()

    if search_dirs is None:
        search_dirs = [
            cfg.pose_projects_dir,
            cfg.outputs_dir / "models",
            cfg.data_dir / "sleap_models",
        ]

    models: List[Dict[str, Any]] = []

    for base in search_dirs:
        if not base.exists():
            continue

        # Pattern 1: directory-based models (training_config.json)
        for config_path in base.rglob("training_config.json"):
            model_dir = config_path.parent
            meta = _parse_training_config(config_path)
            models.append({
                "path": str(model_dir),
                "type": "directory",
                "config_path": str(config_path),
                **meta,
            })

        # Pattern 2: exported .zip or .pkg.slp model packages
        for ext in ("*.zip", "*.pkg.slp"):
            for pkg in base.rglob(ext):
                models.append({
                    "path": str(pkg),
                    "type": "package",
                    "name": pkg.stem,
                })

        # Pattern 3: .h5 / .keras weight files (bare exports)
        for ext in ("*.h5", "*.keras"):
            for wf in base.rglob(ext):
                if any(wf.is_relative_to(Path(m["path"])) for m in models):
                    continue  # already covered by a directory model
                models.append({
                    "path": str(wf),
                    "type": "weights",
                    "name": wf.stem,
                })

    return models


def _parse_training_config(config_path: Path) -> Dict[str, Any]:
    """Extract useful metadata from a SLEAP training_config.json."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "name": data.get("model", {}).get("backbone", {}).get("type", "unknown"),
            "skeleton": data.get("data", {}).get("labels", {}).get("skeleton", None),
            "n_keypoints": len(
                data.get("data", {}).get("labels", {}).get("skeletons", [{}])[0].get("nodes", [])
            ) if data.get("data", {}).get("labels", {}).get("skeletons") else None,
        }
    except Exception:
        return {"name": config_path.parent.name}


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_sleap_inference(
    video_path: Path,
    model_paths: List[str] | str,
    output_path: Path | None = None,
    batch_size: int = 4,
    peak_threshold: float = 0.2,
) -> Path:
    """Run SLEAP inference on a single video.

    Parameters
    ----------
    video_path : Path
        Path to the .mp4 video file.
    model_paths : list of str, or str
        Path(s) to trained SLEAP model directories/packages.
        For top-down: [centroid_model, instance_model].
        For single-animal bottom-up: [single_model].
    output_path : Path, optional
        Where to write the .slp predictions file.
    batch_size : int
        GPU batch size (reduce if OOM).
    peak_threshold : float
        Confidence threshold for peak detection.

    Returns
    -------
    Path to the .slp predictions file.
    """
    try:
        import sleap
    except ImportError:
        raise ImportError(
            "SLEAP is required for automated inference.\n"
            "Install: conda install -c conda-forge -c nvidia -c sleap sleap\n"
            "Or: pip install sleap"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if isinstance(model_paths, (str, Path)):
        model_paths = [str(model_paths)]
    else:
        model_paths = [str(p) for p in model_paths]

    if output_path is None:
        output_path = video_path.with_suffix(".predictions.slp")

    # Build the SLEAP CLI args
    args = [
        str(video_path),
        "--batch_size", str(batch_size),
        "--peak_threshold", str(peak_threshold),
        "-o", str(output_path),
    ]
    for mp in model_paths:
        args.extend(["-m", mp])

    logger.info(f"Running SLEAP inference: {video_path.name} -> {output_path.name}")

    # Use SLEAP's inference API directly
    sleap.nn.inference.main(args)

    return output_path


def run_batch_inference(
    session_ids: List[int] | None = None,
    cameras: List[str] | None = None,
    model_paths: List[str] | str | None = None,
    batch_size: int = 4,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Run SLEAP inference on all locally-cached videos.

    This is the main entry point. It:
    1. Finds all videos in the cache (no downloading)
    2. Runs inference with the specified (or auto-discovered) model
    3. Converts predictions to parquet
    4. Returns a summary DataFrame

    Parameters
    ----------
    session_ids : list of int, optional
        Specific sessions to process. If None, processes all cached.
    cameras : list of str, optional
        Which cameras to process. Defaults to config.video_cameras.
    model_paths : list/str, optional
        Explicit model path(s). If None, auto-discovers.
    batch_size : int
        GPU batch size.
    skip_existing : bool
        Skip videos that already have predictions.

    Returns
    -------
    DataFrame with columns: session_id, camera, video_path, slp_path,
    parquet_path, n_frames, status
    """
    from config import get_config
    from io_video import load_video_assets

    cfg = get_config()
    if cameras is None:
        cameras = cfg.video_cameras

    # Auto-discover model if not specified
    if model_paths is None:
        models = discover_sleap_models()
        if not models:
            raise FileNotFoundError(
                "No SLEAP models found. Train a model first using your ~150 labeled frames,\n"
                "then place it in pose_projects/ or data/sleap_models/.\n"
                "To train: sleap-train <config.json> <labels.slp>"
            )
        # Use the first discovered model
        model_paths = [models[0]["path"]]
        logger.info(f"Auto-discovered model: {model_paths[0]}")

    # Find all cached videos
    video_assets = load_video_assets()
    if video_assets.empty:
        logger.warning("No video assets found. Run Notebook 05 first.")
        return pd.DataFrame()

    if session_ids is not None:
        video_assets = video_assets[video_assets["session_id"].isin(session_ids)]

    results = []
    for _, row in video_assets.iterrows():
        sid = int(row["session_id"])
        cam = str(row["camera"])

        if cam not in cameras:
            continue

        video_path_str = row.get("local_video_path")
        if not video_path_str or not Path(video_path_str).exists():
            results.append({
                "session_id": sid, "camera": cam,
                "video_path": video_path_str,
                "slp_path": None, "parquet_path": None,
                "n_frames": 0, "status": "NO_LOCAL_VIDEO",
            })
            continue

        video_path = Path(video_path_str)
        slp_path = (
            cfg.outputs_dir / "pose" / "predictions"
            / f"session_{sid}_{cam}.predictions.slp"
        )
        parquet_path = (
            cfg.outputs_dir / "pose"
            / f"session_{sid}_pose_predictions.parquet"
        )

        if skip_existing and slp_path.exists():
            results.append({
                "session_id": sid, "camera": cam,
                "video_path": str(video_path),
                "slp_path": str(slp_path),
                "parquet_path": str(parquet_path),
                "n_frames": _count_slp_frames(slp_path),
                "status": "SKIPPED_EXISTS",
            })
            continue

        try:
            slp_path.parent.mkdir(parents=True, exist_ok=True)
            run_sleap_inference(
                video_path=video_path,
                model_paths=model_paths,
                output_path=slp_path,
                batch_size=batch_size,
            )
            # Convert to parquet
            n_frames = slp_to_parquet(
                slp_path=slp_path,
                session_id=sid,
                camera=cam,
                output_path=parquet_path,
            )
            results.append({
                "session_id": sid, "camera": cam,
                "video_path": str(video_path),
                "slp_path": str(slp_path),
                "parquet_path": str(parquet_path),
                "n_frames": n_frames,
                "status": "SUCCESS",
            })
            logger.info(f"Inference complete: session={sid} camera={cam} frames={n_frames}")
        except Exception as exc:
            logger.error(f"Inference failed for session={sid} camera={cam}: {exc}")
            results.append({
                "session_id": sid, "camera": cam,
                "video_path": str(video_path),
                "slp_path": None, "parquet_path": None,
                "n_frames": 0, "status": f"FAILED: {exc}",
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# SLP -> Parquet conversion
# ---------------------------------------------------------------------------

def slp_to_parquet(
    slp_path: Path,
    session_id: int,
    camera: str,
    output_path: Path | None = None,
    confidence_threshold: float = 0.0,
) -> int:
    """Convert SLEAP .slp predictions to pose_predictions.parquet.

    Returns the number of frames converted.
    """
    try:
        import sleap
    except ImportError:
        raise ImportError("SLEAP is required to read .slp files")

    from config import get_config, make_provenance
    from io_video import load_frame_times
    from timebase import write_parquet_with_timebase

    cfg = get_config()
    slp_path = Path(slp_path)
    if not slp_path.exists():
        raise FileNotFoundError(f"SLP file not found: {slp_path}")

    labels = sleap.load_file(str(slp_path))

    # Extract predictions into a flat table
    rows = []
    skeleton = labels.skeletons[0] if labels.skeletons else None
    node_names = [n.name for n in skeleton.nodes] if skeleton else []

    for lf in labels.labeled_frames:
        frame_idx = lf.frame_idx
        for inst_idx, inst in enumerate(lf.instances):
            row = {
                "frame_idx": int(frame_idx),
                "instance": int(inst_idx),
            }
            # Instance-level score
            if hasattr(inst, "score") and inst.score is not None:
                row["instance_score"] = float(inst.score)

            for node_name, point in zip(node_names, inst.points):
                if point is None or not hasattr(point, "x"):
                    row[f"{node_name}_x"] = np.nan
                    row[f"{node_name}_y"] = np.nan
                    row[f"{node_name}_score"] = 0.0
                else:
                    row[f"{node_name}_x"] = float(point.x)
                    row[f"{node_name}_y"] = float(point.y)
                    score = float(point.score) if hasattr(point, "score") and point.score is not None else 1.0
                    row[f"{node_name}_score"] = score

            rows.append(row)

    if not rows:
        logger.warning(f"No predictions found in {slp_path}")
        return 0

    df = pd.DataFrame(rows)

    # Filter by confidence if requested
    if confidence_threshold > 0:
        score_cols = [c for c in df.columns if c.endswith("_score") and c != "instance_score"]
        if score_cols:
            mean_score = df[score_cols].mean(axis=1)
            df = df[mean_score >= confidence_threshold].reset_index(drop=True)

    # Attach timestamps from frame_times
    frame_times = load_frame_times(session_id=session_id, camera=camera)
    if not frame_times.empty:
        ft = frame_times[["frame_idx", "t"]].drop_duplicates("frame_idx")
        df = df.merge(ft, on="frame_idx", how="left")
    else:
        # Try loading camera timestamps directly
        from features_pose import _load_camera_timestamps
        ts = _load_camera_timestamps(session_id, camera)
        if ts is not None:
            t_vals = np.full(len(df), np.nan)
            fi = df["frame_idx"].to_numpy()
            valid = (fi >= 0) & (fi < len(ts))
            t_vals[valid] = ts[fi[valid]]
            df["t"] = t_vals
        else:
            df["t"] = np.nan

    df["session_id"] = session_id
    df["camera"] = camera

    # Reorder columns
    front = ["session_id", "camera", "frame_idx", "instance", "t"]
    rest = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + rest]

    if output_path is None:
        output_path = cfg.outputs_dir / "pose" / f"session_{session_id}_pose_predictions.parquet"

    write_parquet_with_timebase(
        df, output_path,
        provenance=make_provenance(session_id, "sleap_inference"),
        required_columns=["frame_idx", "t"],
    )

    return len(df)


def _count_slp_frames(slp_path: Path) -> int:
    """Quick count of frames in an SLP file without full parsing."""
    try:
        import sleap
        labels = sleap.load_file(str(slp_path))
        return len(labels.labeled_frames)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# SLEAP CSV import (for existing hand-labeled exports)
# ---------------------------------------------------------------------------

def auto_discover_sleap_csvs(
    session_id: int | None = None,
) -> List[Dict[str, Any]]:
    """Find all SLEAP CSV exports in outputs/pose/ and outputs/labeling/.

    Returns list of dicts with keys: path, session_id, camera (inferred from filename).
    """
    from config import get_config
    cfg = get_config()

    csvs = []
    search_dirs = [
        cfg.outputs_dir / "pose",
        cfg.outputs_dir / "labeling",
    ]

    for base in search_dirs:
        if not base.exists():
            continue
        for csv_path in base.rglob("*.csv"):
            # Try to infer session_id and camera from filename/path
            name = csv_path.stem.lower()
            parts = str(csv_path).lower()
            sid = _extract_session_id(parts)
            cam = _extract_camera(parts, name)

            if session_id is not None and sid != session_id:
                continue

            csvs.append({
                "path": str(csv_path),
                "session_id": sid,
                "camera": cam,
                "filename": csv_path.name,
            })

    return csvs


def _extract_session_id(text: str) -> int | None:
    """Try to extract a session ID (10-digit number) from a path string."""
    import re
    match = re.search(r"session[_]?(\d{9,11})", text)
    if match:
        return int(match.group(1))
    # Fallback: look for any 10-digit number
    match = re.search(r"(\d{10})", text)
    if match:
        return int(match.group(1))
    return None


def _extract_camera(path_text: str, filename: str) -> str | None:
    """Try to infer camera name from path or filename."""
    for cam in ("side", "face", "eye"):
        if cam in filename or cam in path_text:
            return cam
    return None


# ---------------------------------------------------------------------------
# Training helper (wraps SLEAP CLI)
# ---------------------------------------------------------------------------

def train_sleap_model(
    labels_path: Path,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 100,
    batch_size: int = 4,
) -> Path:
    """Train a SLEAP model from labeled data.

    Parameters
    ----------
    labels_path : Path
        Path to the .slp labels file with your ~150 labeled frames.
    config_path : Path, optional
        Path to a SLEAP training config JSON. If None, uses a sensible
        default for single-animal bottom-up pose estimation.
    output_dir : Path, optional
        Where to save the trained model.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.

    Returns
    -------
    Path to the trained model directory.
    """
    try:
        import sleap
    except ImportError:
        raise ImportError("SLEAP is required for model training")

    from config import get_config
    cfg = get_config()

    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if output_dir is None:
        output_dir = cfg.pose_projects_dir / "trained_models" / labels_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_path is not None and Path(config_path).exists():
        # Use user-provided config
        args = [
            str(config_path),
            str(labels_path),
            "--run_path", str(output_dir),
        ]
    else:
        # Generate a sensible default single-instance config
        args = [
            str(labels_path),
            "--run_path", str(output_dir),
            "--default_single",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]

    logger.info(f"Training SLEAP model: {labels_path} -> {output_dir}")
    sleap.nn.training.main(args)

    return output_dir


# ---------------------------------------------------------------------------
# Active learning: suggest which frames to label next
# ---------------------------------------------------------------------------

def suggest_frames_to_label(
    slp_path: Path,
    n_suggestions: int = 20,
    strategy: str = "low_confidence",
) -> np.ndarray:
    """Suggest frames that would benefit most from manual labeling.

    Uses prediction confidence to find uncertain frames -- labeling these
    gives you the biggest improvement per frame labeled.

    Parameters
    ----------
    slp_path : Path
        Path to .slp predictions file from inference.
    n_suggestions : int
        How many frames to suggest.
    strategy : str
        "low_confidence" - frames with lowest mean keypoint confidence
        "spread" - low-confidence frames spread across the session

    Returns
    -------
    Array of frame indices to label.
    """
    try:
        import sleap
    except ImportError:
        raise ImportError("SLEAP is required")

    labels = sleap.load_file(str(slp_path))

    frame_scores = []
    for lf in labels.labeled_frames:
        scores = []
        for inst in lf.instances:
            for pt in inst.points:
                if pt is not None and hasattr(pt, "score") and pt.score is not None:
                    scores.append(float(pt.score))
        mean_score = np.mean(scores) if scores else 0.0
        frame_scores.append((lf.frame_idx, mean_score))

    if not frame_scores:
        return np.array([], dtype=int)

    frame_scores.sort(key=lambda x: x[1])  # lowest confidence first

    if strategy == "low_confidence":
        return np.array([fs[0] for fs in frame_scores[:n_suggestions]], dtype=int)

    elif strategy == "spread":
        # Pick from low-confidence but spread across the video
        candidates = frame_scores[:n_suggestions * 3]
        if len(candidates) <= n_suggestions:
            return np.array([fs[0] for fs in candidates], dtype=int)
        indices = np.linspace(0, len(candidates) - 1, n_suggestions, dtype=int)
        return np.array([candidates[i][0] for i in indices], dtype=int)

    return np.array([fs[0] for fs in frame_scores[:n_suggestions]], dtype=int)
