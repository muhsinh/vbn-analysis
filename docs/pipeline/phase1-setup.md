# Phase 1: Setup & Discovery

Phase 1 spans **Notebook 00** (environment bootstrap) and **Notebook 01** (session discovery). By the end of this phase you have a validated environment, a registry of sessions, and a `SessionBundle` object for each session that orchestrates all downstream data loading.

---

## Notebook 00: Configuration, Directories, and Dependencies

### What happens step by step

#### 1. Load the `Config` dataclass

The entire pipeline is controlled by a single `Config` object built from environment variables with sensible defaults:

```python title="src/config.py: Config dataclass (abbreviated)"
@dataclass
class Config:
    access_mode: str = "sdk"           # "sdk" | "manual"
    pose_tool: str = "sleap"           # "sleap" | "dlc"
    model_name: str = "xgboost"        # encoding/decoding model backend
    bin_size_s: float = 0.025          # 25 ms time bins
    mock_mode: bool = False            # True = synthetic data, no NWB needed

    outputs_dir: Path = ROOT_DIR / "outputs"
    cache_dir: Path = ROOT_DIR / "outputs" / "cache"
    pose_projects_dir: Path = ROOT_DIR / "pose_projects"
    data_dir: Path = ROOT_DIR / "data"
    video_cache_dir: Path = ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"
    video_cameras: list[str] = field(
        default_factory=lambda: ["eye", "face", "side"]
    )
    sessions_csv: Path = ROOT_DIR / "sessions.csv"
```

!!! tip "Environment variable overrides"
    Every config field can be overridden via environment variable before the first `get_config()` call:

    ```bash
    export ACCESS_MODE=manual
    export MOCK_MODE=true
    export BIN_SIZE_S=0.05
    export VIDEO_CAMERAS=eye,face
    ```

The singleton is created lazily:

```python title="src/config.py: get_config()"
_CONFIG: Config | None = None

def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(
            access_mode=_get_env("ACCESS_MODE", "sdk"),
            pose_tool=_get_env("POSE_TOOL", "sleap"),
            model_name=_get_env("MODEL_NAME", "xgboost"),
            bin_size_s=float(_get_env("BIN_SIZE_S", "0.025")),
            mock_mode=_as_bool(_get_env("MOCK_MODE"), False),
            video_source=_get_env("VIDEO_SOURCE", "auto"),
            video_cache_dir=Path(_get_env(
                "VIDEO_CACHE_DIR",
                str(ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"),
            )),
            video_cameras=_parse_csv(
                _get_env("VIDEO_CAMERAS"), ["eye", "face", "side"]
            ),
        )
    return _CONFIG
```

#### 2. Create directory structure

`Config.ensure_dirs()` creates the full directory tree:

```python title="src/config.py: ensure_dirs()"
def ensure_dirs(self) -> None:
    self.outputs_dir.mkdir(parents=True, exist_ok=True)
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    (self.outputs_dir / "reports" / "logs").mkdir(parents=True, exist_ok=True)
    (self.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
    self.pose_projects_dir.mkdir(parents=True, exist_ok=True)
    self.data_dir.mkdir(parents=True, exist_ok=True)
    self.video_cache_dir.mkdir(parents=True, exist_ok=True)
```

This produces the following layout:

```
vbn-analysis/
  outputs/
    cache/
    reports/
      logs/
    neural/          (created later by Phase 2)
    behavior/        (created later by Phase 2)
    eye/             (created later by Phase 2)
    video/           (created later by Phase 3)
    pose/            (created later by Phase 3)
  pose_projects/
  data/
    raw/
      visual-behavior-neuropixels/
```

#### 3. Dependency check

Notebook 00 typically imports key packages and reports versions. The pipeline is designed to run in **mock mode** when heavy dependencies like `pynwb` or `allensdk` are not installed; the `open_nwb_handle` context manager falls back to synthetic data automatically.

#### 4. Write config snapshot

```python title="src/config.py: write_config_snapshot()"
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
```

The snapshot includes a `code_version` field that captures the current git commit hash, providing full reproducibility tracing:

```json title="outputs/reports/config_snapshot.json"
{
  "access_mode": "sdk",
  "pose_tool": "sleap",
  "model_name": "xgboost",
  "bin_size_s": 0.025,
  "mock_mode": false,
  "outputs_dir": "/path/to/vbn-analysis/outputs",
  "video_cameras": ["eye", "face", "side"],
  "code_version": "9e576ad"
}
```

---

## Notebook 01: Session Discovery and Metadata

### What happens step by step

#### 1. Load (or create) `sessions.csv`

The function `load_sessions_csv()` implements a three-level fallback:

```python title="src/io_sessions.py:load_sessions_csv()"
REQUIRED_SESSIONS_COLUMNS = ["session_id", "nwb_path", "video_dir", "notes"]

def load_sessions_csv(
    path: Path | None = None,
    create_if_missing: bool = True,
) -> pd.DataFrame:
    cfg = get_config()
    if path is None:
        path = cfg.sessions_csv                    # (1)!

    if path.exists():
        df = pd.read_csv(path)
        return _normalize_sessions_df(df, path)    # (2)!

    if not create_if_missing:
        raise FileNotFoundError(...)

    # Fallback: look for legacy sessions.txt
    txt_candidates = [
        cfg.sessions_csv.with_suffix(".txt"),      # (3)!
        cfg.legacy_dir / "sessions.txt",
    ]
    txt_path = next((p for p in txt_candidates if p.exists()), None)
    if txt_path is None:
        # Create empty template
        df = pd.DataFrame(columns=REQUIRED_SESSIONS_COLUMNS)
        df.to_csv(path, index=False)
        return df

    df = generate_sessions_csv_from_txt(txt_path, path)
    return df
```

1. Default path: `<project_root>/sessions.csv`
2. Ensures all four required columns exist, adding blanks if missing
3. Tries `sessions.txt` in root, then `legacy/sessions.txt`

!!! info "Legacy `sessions.txt` migration"
    If you have a plain-text file with one session ID per line, the pipeline auto-converts it into a proper CSV:

    ```python title="src/io_sessions.py:generate_sessions_csv_from_txt()"
    def generate_sessions_csv_from_txt(txt_path, output_path):
        session_ids = []
        for line in txt_path.read_text().splitlines():
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
        df.to_csv(output_path, index=False)
        return df
    ```

#### Schema of `sessions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | `int` | Allen Institute ecephys session ID (10-digit integer) |
| `nwb_path` | `str` | Optional explicit path to NWB file; blank = use AllenSDK resolution |
| `video_dir` | `str` | Optional path to directory containing behavior videos |
| `notes` | `str` | Free-form notes for this session |

#### 2. Normalize the DataFrame

`_normalize_sessions_df()` ensures the required columns exist. If any are missing, they are added as empty strings and the CSV is re-written:

```python title="src/io_sessions.py:_normalize_sessions_df()"
def _normalize_sessions_df(df, path):
    missing = [col for col in REQUIRED_SESSIONS_COLUMNS if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = ""
        df = df[REQUIRED_SESSIONS_COLUMNS]
        df.to_csv(path, index=False)
    return df
```

---

## The `SessionBundle` Dataclass

`SessionBundle` is the central orchestration object. Each session gets one bundle, and it provides lazy-loading methods for every data modality.

```python title="src/io_sessions.py:SessionBundle"
@dataclass
class SessionBundle:
    session_id: int
    nwb_path: Path | None
    video_dir: Path | None
    access_mode: str
    modalities_present: dict[str, bool] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)
    alignment_qc: dict[str, Any] = field(default_factory=dict)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `int` | The 10-digit Allen session identifier |
| `nwb_path` | `Path \| None` | Resolved path to the NWB file on disk |
| `video_dir` | `Path \| None` | Root directory for behavior video assets |
| `access_mode` | `str` | `"sdk"` (AllenSDK auto-download) or `"manual"` (user-supplied paths) |
| `modalities_present` | `dict[str, bool]` | Which data streams exist: `spikes`, `trials`, `eye`, `behavior`, `stimulus` |
| `qc_flags` | `list[str]` | Accumulates issues: `"neural_unavailable"`, `"eye_unavailable"`, `"video_unavailable"` |
| `alignment_qc` | `dict` | Populated during cross-modal alignment checks |

### Lazy-loading methods

Each method follows the same pattern: check for cached output on disk, and if missing, extract from NWB and save.

| Method | Returns | Output artifact(s) |
|--------|---------|---------------------|
| `load_spikes()` | `(units_df, spike_times_dict)` | `outputs/neural/session_<id>_units.parquet`, `session_<id>_spike_times.npz` |
| `load_trials_and_events()` | `(trials_df, events_df)` | `outputs/behavior/session_<id>_trials.parquet`, `session_<id>_events.parquet` |
| `load_eye_features()` | `eye_features_df` | `outputs/eye/session_<id>_eye_features.parquet` |
| `load_video_assets()` | `assets_df` | `outputs/video/video_assets.parquet` |
| `load_frame_times(camera)` | `frame_times_df` | `outputs/video/frame_times.parquet` |

---

## How `get_session_bundle()` Works

This is the factory function that notebooks call to obtain a fully-initialized bundle:

```python title="src/io_sessions.py:get_session_bundle()"
def get_session_bundle(
    session_id: int,
    sessions_df: pd.DataFrame | None = None,
    *,
    resolve_nwb: bool = True,
    inspect_modalities: bool = True,
) -> SessionBundle:
    cfg = get_config()
    if sessions_df is None:
        sessions_df = load_sessions_csv()          # (1)!

    row = sessions_df[sessions_df["session_id"] == session_id]
    if row.empty:
        nwb_path = None
        video_dir = None
    else:
        nwb_path_str = str(row.iloc[0]["nwb_path"]).strip()
        video_dir_str = str(row.iloc[0]["video_dir"]).strip()
        nwb_path = Path(nwb_path_str) if nwb_path_str else None
        video_dir = Path(video_dir_str) if video_dir_str else None

    resolved_nwb_path = None
    if resolve_nwb:
        resolved_nwb_path = io_nwb.resolve_nwb_path(  # (2)!
            session_id=session_id,
            access_mode=cfg.access_mode,
            nwb_path_override=nwb_path,
        )
    else:
        resolved_nwb_path = nwb_path               # (3)!

    modalities = {}
    if inspect_modalities and resolved_nwb_path is not None:
        with io_nwb.open_nwb_handle(               # (4)!
            resolved_nwb_path, mock_mode=cfg.mock_mode
        ) as nwb:
            modalities = io_nwb.inspect_modalities(nwb)

    bundle = SessionBundle(
        session_id=session_id,
        nwb_path=resolved_nwb_path,
        video_dir=video_dir,
        access_mode=cfg.access_mode,
        modalities_present=modalities,
    )
    return bundle
```

1. Auto-loads or creates the session registry
2. In SDK mode, uses `VisualBehaviorNeuropixelsProjectCache.from_s3_cache()` to download/locate the NWB file
3. In video-only workflows, skip NWB resolution entirely to avoid unnecessary downloads
4. Opens the NWB file (or mock) to check which modalities are present

### NWB path resolution

```python title="src/io_nwb.py:resolve_nwb_path()"
def resolve_nwb_path(session_id, access_mode, nwb_path_override=None):
    if access_mode == "manual":
        return nwb_path_override               # (1)!

    try:
        from allensdk.brain_observatory ... import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError:
        return nwb_path_override               # (2)!

    cfg = get_config()
    cache_dir = cfg.data_dir / "allensdk_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=str(cache_dir)
        )
        session = cache.get_ecephys_session(session_id)
        nwb_path = Path(session.nwb_path)
        return nwb_path                        # (3)!
    except Exception:
        return nwb_path_override               # (4)!
```

1. Manual mode: trust the user-supplied path
2. AllenSDK not installed: fall back gracefully
3. SDK success: NWB is now guaranteed to be on disk
4. SDK failure (network, permissions): fall back to override

### Modality inspection

Once the NWB handle is open, `inspect_modalities()` probes for each data stream:

```python title="src/io_nwb.py:inspect_modalities()"
def inspect_modalities(nwb) -> Dict[str, bool]:
    modalities = {
        "spikes": False,
        "trials": False,
        "eye": False,
        "behavior": False,
        "stimulus": False,
    }
    modalities["spikes"] = hasattr(nwb, "units") and nwb.units is not None
    modalities["trials"] = hasattr(nwb, "trials") and nwb.trials is not None
    modalities["eye"] = "eye_tracking" in getattr(nwb, "processing", {})
    modalities["behavior"] = "behavior" in getattr(nwb, "processing", {})
    modalities["stimulus"] = hasattr(nwb, "stimulus") and nwb.stimulus is not None
    return modalities
```

!!! example "Typical modalities_present output"
    ```python
    {
        "spikes": True,
        "trials": True,
        "eye": True,
        "behavior": True,
        "stimulus": False,
    }
    ```

---

## Step-Level Caching

For expensive intermediate computations that do not fit neatly into the per-modality artifact pattern, the pipeline provides a generic `cache_step()` function:

```python title="src/io_sessions.py:cache_step()"
def cache_step(session_id, step, params, compute_fn):
    cache_path = _cache_path(session_id, step, params)
    if cache_path.exists():
        return joblib.load(cache_path)
    result = compute_fn()
    joblib.dump(result, cache_path)
    return result
```

The cache key is an MD5 hash of the parameters dict, ensuring that changing any parameter triggers a recomputation:

```python title="src/io_sessions.py:_cache_path()"
def _cache_path(session_id, step, params, ext="joblib"):
    cfg = get_config()
    cache_dir = cfg.cache_dir / f"session_{session_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_params(params)
    return cache_dir / f"{step}_{key}.{ext}"

def _hash_params(params):
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()
```

!!! note "Cache file naming"
    A cache file for session `1234567890` running the step `"feature_extraction"` with parameters `{"bin_size": 0.025}` would be saved as:

    ```
    outputs/cache/session_1234567890/feature_extraction_a1b2c3d4.joblib
    ```

---

## Files Produced by Phase 1

| File | Location | Format | Purpose |
|------|----------|--------|---------|
| `config_snapshot.json` | `outputs/reports/` | JSON | Full configuration for reproducibility |
| `sessions.csv` | Project root | CSV | Session registry with IDs and paths |

Phase 1 does **not** extract any data from NWB files; it only resolves paths and inspects which modalities are available. Extraction begins in [Phase 2](phase2-signals.md).
