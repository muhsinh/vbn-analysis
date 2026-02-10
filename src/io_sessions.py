"""Session discovery and bundle orchestration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

import io_nwb
from config import get_config
from reports import setup_session_logger


REQUIRED_SESSIONS_COLUMNS = ["session_id", "nwb_path", "video_dir", "notes"]


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _cache_path(session_id: int, step: str, params: dict[str, Any], ext: str = "joblib") -> Path:
    cfg = get_config()
    cache_dir = cfg.cache_dir / f"session_{session_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_params(params)
    return cache_dir / f"{step}_{key}.{ext}"


def load_sessions_csv(path: Path | None = None, create_if_missing: bool = True) -> pd.DataFrame:
    cfg = get_config()
    if path is None:
        path = cfg.sessions_csv

    if path.exists():
        df = pd.read_csv(path)
        return _normalize_sessions_df(df, path)

    if not create_if_missing:
        raise FileNotFoundError(f"sessions.csv not found at {path}")

    # Fallback to sessions.txt (root or legacy)
    txt_candidates = [cfg.sessions_csv.with_suffix(".txt"), cfg.legacy_dir / "sessions.txt"]
    txt_path = next((p for p in txt_candidates if p.exists()), None)
    if txt_path is None:
        # Create empty template
        df = pd.DataFrame(columns=REQUIRED_SESSIONS_COLUMNS)
        df.to_csv(path, index=False)
        return df

    df = generate_sessions_csv_from_txt(txt_path, path)
    return df


def generate_sessions_csv_from_txt(txt_path: Path, output_path: Path) -> pd.DataFrame:
    session_ids: list[int] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            session_ids.append(int(line))
        except ValueError:
            continue

    df = pd.DataFrame({"session_id": session_ids})
    df["nwb_path"] = ""
    df["video_dir"] = ""
    df["notes"] = ""
    df = df[REQUIRED_SESSIONS_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _normalize_sessions_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [col for col in REQUIRED_SESSIONS_COLUMNS if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = ""
        df = df[REQUIRED_SESSIONS_COLUMNS]
        df.to_csv(path, index=False)
    return df


@dataclass
class SessionBundle:
    session_id: int
    nwb_path: Path | None
    video_dir: Path | None
    access_mode: str
    modalities_present: dict[str, bool] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)
    alignment_qc: dict[str, Any] = field(default_factory=dict)

    def ensure_logger(self):
        return setup_session_logger(self.session_id)

    def load_spikes(self) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
        cfg = get_config()
        logger = self.ensure_logger()
        outputs_dir = cfg.outputs_dir / "neural"
        units_path = outputs_dir / f"session_{self.session_id}_units.parquet"
        spikes_path = outputs_dir / f"session_{self.session_id}_spike_times.npz"

        if units_path.exists() and spikes_path.exists():
            units = pd.read_parquet(units_path)
            spikes = dict(io_nwb.load_spike_times_npz(spikes_path))
            return units, spikes

        def _compute():
            with io_nwb.open_nwb_handle(self.nwb_path, mock_mode=cfg.mock_mode) as nwb:
                return io_nwb.extract_units_and_spikes(nwb)

        units, spikes = _compute()
        if units is not None:
            outputs_dir.mkdir(parents=True, exist_ok=True)
            io_nwb.save_units_and_spikes(
                units,
                spikes,
                units_path,
                spikes_path,
                session_id=self.session_id,
                alignment_method="nwb",
            )
            logger.info(f"Saved units to {units_path}")
        else:
            if "neural_unavailable" not in self.qc_flags:
                self.qc_flags.append("neural_unavailable")
        return units, spikes

    def load_trials_and_events(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        cfg = get_config()
        logger = self.ensure_logger()
        outputs_dir = cfg.outputs_dir / "behavior"
        trials_path = outputs_dir / f"session_{self.session_id}_trials.parquet"
        events_path = outputs_dir / f"session_{self.session_id}_events.parquet"

        if trials_path.exists() and events_path.exists():
            return pd.read_parquet(trials_path), pd.read_parquet(events_path)

        with io_nwb.open_nwb_handle(self.nwb_path, mock_mode=cfg.mock_mode) as nwb:
            trials = io_nwb.extract_trials(nwb)
            events = io_nwb.extract_behavior_events(nwb)

        outputs_dir.mkdir(parents=True, exist_ok=True)
        io_nwb.save_behavior_tables(
            trials,
            events,
            trials_path,
            events_path,
            session_id=self.session_id,
            alignment_method="nwb",
        )
        logger.info(f"Saved behavior tables to {outputs_dir}")
        return trials, events

    def load_eye_features(self) -> pd.DataFrame | None:
        cfg = get_config()
        logger = self.ensure_logger()
        outputs_dir = cfg.outputs_dir / "eye"
        eye_path = outputs_dir / f"session_{self.session_id}_eye_features.parquet"

        if eye_path.exists():
            return pd.read_parquet(eye_path)

        from features_eye import derive_eye_features

        with io_nwb.open_nwb_handle(self.nwb_path, mock_mode=cfg.mock_mode) as nwb:
            eye_raw = io_nwb.extract_eye_tracking(nwb)

        if eye_raw is None:
            logger.warning("Eye tracking data missing")
            if "eye_unavailable" not in self.qc_flags:
                self.qc_flags.append("eye_unavailable")
            return None

        eye_features = derive_eye_features(eye_raw)
        if eye_features is None:
            return None

        outputs_dir.mkdir(parents=True, exist_ok=True)
        io_nwb.save_eye_table(
            eye_features,
            eye_path,
            session_id=self.session_id,
            alignment_method="nwb",
        )
        logger.info(f"Saved eye features to {eye_path}")
        return eye_features

    def load_video_assets(self) -> pd.DataFrame:
        from io_video import build_video_assets
        cfg = get_config()
        logger = self.ensure_logger()
        outputs_dir = cfg.outputs_dir / "video"
        assets_path = outputs_dir / "video_assets.parquet"

        assets = build_video_assets(
            session_id=self.session_id,
            video_dir=self.video_dir,
            outputs_dir=outputs_dir,
        )
        logger.info(f"Updated video assets at {assets_path}")
        if assets.empty:
            if "video_unavailable" not in self.qc_flags:
                self.qc_flags.append("video_unavailable")
        return assets

    def load_frame_times(self, camera: str | None = None) -> pd.DataFrame:
        from io_video import load_frame_times
        return load_frame_times(session_id=self.session_id, camera=camera)


def get_session_bundle(session_id: int, sessions_df: pd.DataFrame | None = None) -> SessionBundle:
    cfg = get_config()
    if sessions_df is None:
        sessions_df = load_sessions_csv()

    row = sessions_df[sessions_df["session_id"] == session_id]
    if row.empty:
        nwb_path = None
        video_dir = None
    else:
        nwb_path_str = str(row.iloc[0]["nwb_path"]).strip()
        video_dir_str = str(row.iloc[0]["video_dir"]).strip()
        nwb_path = Path(nwb_path_str) if nwb_path_str else None
        video_dir = Path(video_dir_str) if video_dir_str else None

    resolved_nwb_path = io_nwb.resolve_nwb_path(
        session_id=session_id,
        access_mode=cfg.access_mode,
        nwb_path_override=nwb_path,
    )

    modalities = {}
    if resolved_nwb_path is not None:
        with io_nwb.open_nwb_handle(resolved_nwb_path, mock_mode=cfg.mock_mode) as nwb:
            modalities = io_nwb.inspect_modalities(nwb)

    bundle = SessionBundle(
        session_id=session_id,
        nwb_path=resolved_nwb_path,
        video_dir=video_dir,
        access_mode=cfg.access_mode,
        modalities_present=modalities,
    )
    return bundle


def cache_step(
    session_id: int,
    step: str,
    params: dict[str, Any],
    compute_fn: Callable[[], Any],
) -> Any:
    cache_path = _cache_path(session_id, step, params)
    if cache_path.exists():
        return joblib.load(cache_path)
    result = compute_fn()
    joblib.dump(result, cache_path)
    return result
