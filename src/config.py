"""Configuration and paths for the VBN analysis suite."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}

def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


@dataclass
class Config:
    access_mode: str = "manual"
    pose_tool: str = "sleap"
    model_name: str = "xgboost"
    bin_size_s: float = 0.025
    categorical_cols: list[str] = field(
        default_factory=lambda: ["motif_id", "trial_type", "stimulus_name"]
    )
    mock_mode: bool = False

    outputs_dir: Path = ROOT_DIR / "outputs"
    cache_dir: Path = ROOT_DIR / "outputs" / "cache"
    pose_projects_dir: Path = ROOT_DIR / "pose_projects"
    data_dir: Path = ROOT_DIR / "data"
    video_source: str = "auto"
    video_cache_dir: Path = ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"
    video_bucket: str = "allen-brain-observatory"
    video_base_path: str = "visual-behavior-neuropixels/raw-data"
    video_cameras: list[str] = field(default_factory=lambda: ["eye", "face", "side"])
    sessions_csv: Path = ROOT_DIR / "sessions.csv"
    legacy_dir: Path = ROOT_DIR / "legacy"

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports" / "logs").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.pose_projects_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.video_cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_mode": self.access_mode,
            "pose_tool": self.pose_tool,
            "model_name": self.model_name,
            "bin_size_s": self.bin_size_s,
            "categorical_cols": self.categorical_cols,
            "mock_mode": self.mock_mode,
            "outputs_dir": str(self.outputs_dir),
            "cache_dir": str(self.cache_dir),
            "pose_projects_dir": str(self.pose_projects_dir),
            "data_dir": str(self.data_dir),
            "video_source": self.video_source,
            "video_cache_dir": str(self.video_cache_dir),
            "video_bucket": self.video_bucket,
            "video_base_path": self.video_base_path,
            "video_cameras": self.video_cameras,
            "sessions_csv": str(self.sessions_csv),
            "legacy_dir": str(self.legacy_dir),
            "code_version": get_code_version(),
        }


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(
            access_mode=_get_env("ACCESS_MODE", "manual"),
            pose_tool=_get_env("POSE_TOOL", "sleap"),
            model_name=_get_env("MODEL_NAME", "xgboost"),
            bin_size_s=float(_get_env("BIN_SIZE_S", "0.025")),
            mock_mode=_as_bool(_get_env("MOCK_MODE"), False),
            video_source=_get_env("VIDEO_SOURCE", "auto"),
            video_cache_dir=Path(
                _get_env(
                    "VIDEO_CACHE_DIR",
                    str(ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"),
                )
            ),
            video_bucket=_get_env("VIDEO_BUCKET", "allen-brain-observatory"),
            video_base_path=_get_env("VIDEO_BASE_PATH", "visual-behavior-neuropixels/raw-data"),
            video_cameras=_parse_csv(_get_env("VIDEO_CAMERAS"), ["eye", "face", "side"]),
        )
    return _CONFIG


def get_code_version() -> str:
    """Return a git hash if available; otherwise 'unknown'."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if result:
            return result
    except Exception:
        pass
    return "unknown"


def write_config_snapshot(path: Path | None = None) -> Path:
    config = get_config()
    config.ensure_dirs()
    snapshot = config.to_dict()
    if path is None:
        path = config.outputs_dir / "reports" / "config_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return path


def make_provenance(session_id: int | None, alignment_method: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "code_version": get_code_version(),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "alignment_method": alignment_method,
    }
