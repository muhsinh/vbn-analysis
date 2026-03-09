# io_sessions

Session discovery, CSV manifest management, and the `SessionBundle`
orchestrator that ties together all data modalities for a single session.

**Source:** `src/io_sessions.py`

---

## Constants

### `REQUIRED_SESSIONS_COLUMNS`

```python
REQUIRED_SESSIONS_COLUMNS = ["session_id", "nwb_path", "video_dir", "notes"]
```

Column names that every `sessions.csv` must contain. Missing columns are
auto-added with empty string values.

---

## Functions

### `load_sessions_csv`

```python
def load_sessions_csv(
    path: Path | None = None,
    create_if_missing: bool = True,
) -> pd.DataFrame
```

Load the session manifest CSV. This is the starting point of every pipeline
run; it lists which sessions to process and where to find their data.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path \| None` | `None` | Path to the CSV file. If `None`, uses `config.sessions_csv`. |
| `create_if_missing` | `bool` | `True` | If `True` and the CSV does not exist, attempt to generate one from a legacy `sessions.txt` file or create an empty template. If `False`, raise `FileNotFoundError`. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with at least the columns `session_id`, `nwb_path`, `video_dir`, `notes`. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | `create_if_missing=False` and the file does not exist. |

**Behavior details:**

1. If the CSV exists, it is read and normalized (missing required columns are
   added).
2. If the CSV does not exist and `create_if_missing=True`, the function looks
   for `sessions.txt` in the project root or `legacy/` directory and converts
   it.
3. If no `.txt` fallback is found, an empty template CSV is created.

**Example:**

```python
from io_sessions import load_sessions_csv

# Load default sessions.csv
sessions = load_sessions_csv()
print(sessions.columns.tolist())
# ['session_id', 'nwb_path', 'video_dir', 'notes']

# Load from a custom path, fail if missing
sessions = load_sessions_csv(Path("my_sessions.csv"), create_if_missing=False)
```

---

### `get_session_bundle`

```python
def get_session_bundle(
    session_id: int,
    sessions_df: pd.DataFrame | None = None,
    *,
    resolve_nwb: bool = True,
    inspect_modalities: bool = True,
) -> SessionBundle
```

Create a fully-initialized `SessionBundle` for a given session. This is the
primary entry point for accessing all data associated with a session.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int` | -- | The Allen Brain Observatory session ID (typically a 10-digit integer). |
| `sessions_df` | `pd.DataFrame \| None` | `None` | Pre-loaded sessions manifest. If `None`, calls `load_sessions_csv()`. |
| `resolve_nwb` | `bool` | `True` | Whether to resolve the NWB file path (may trigger AllenSDK download in `"sdk"` mode). Set to `False` for video-only or labeling workflows. |
| `inspect_modalities` | `bool` | `True` | Whether to open the NWB file and check which data modalities are available. |

**Returns:**

| Type | Description |
|------|-------------|
| `SessionBundle` | A bundle object with resolved paths and modality flags. |

**Example:**

```python
from io_sessions import get_session_bundle

# Full bundle with NWB resolution and modality inspection
bundle = get_session_bundle(1064644573)
print(bundle.modalities_present)
# {'spikes': True, 'trials': True, 'eye': False, 'behavior': True, 'stimulus': True}

# Lightweight bundle for video-only work (no NWB download)
bundle = get_session_bundle(1064644573, resolve_nwb=False, inspect_modalities=False)
```

---

### `cache_step`

```python
def cache_step(
    session_id: int,
    step: str,
    params: dict[str, Any],
    compute_fn: Callable[[], Any],
) -> Any
```

Execute a computation with on-disk caching. Results are stored as joblib
files under `outputs/cache/session_{id}/`. The cache key is an MD5 hash of the
serialized `params` dictionary, so changing any parameter triggers a fresh
computation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int` | -- | Session identifier (used to organize the cache directory). |
| `step` | `str` | -- | Human-readable step name (e.g., `"extract_spikes"`, `"pose_features"`). Used in the cache filename. |
| `params` | `dict[str, Any]` | -- | Parameters that affect the computation result. Any change invalidates the cache. |
| `compute_fn` | `Callable[[], Any]` | -- | Zero-argument callable that performs the computation. Only called on cache miss. |

**Returns:**

| Type | Description |
|------|-------------|
| `Any` | The result of `compute_fn()`, either freshly computed or loaded from cache. |

**Example:**

```python
from io_sessions import cache_step

result = cache_step(
    session_id=1064644573,
    step="compute_peth",
    params={"bin_size": 0.01, "window": (-0.5, 1.0)},
    compute_fn=lambda: expensive_peth_computation(),
)
# Second call with same params returns cached result instantly.
```

---

## Classes

### `SessionBundle`

```python
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

A session bundle groups all data access for a single experimental session.
It lazily loads and caches neural, behavioral, eye-tracking, and video data.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `int` | Allen Brain Observatory session ID. |
| `nwb_path` | `Path \| None` | Resolved path to the NWB file (may be `None` if unavailable). |
| `video_dir` | `Path \| None` | Directory containing video files for this session. |
| `access_mode` | `str` | Data access mode (`"sdk"` or `"manual"`). |
| `modalities_present` | `dict[str, bool]` | Which data modalities exist in the NWB file (keys: `spikes`, `trials`, `eye`, `behavior`, `stimulus`). |
| `qc_flags` | `list[str]` | Accumulates QC warning flags during data loading (e.g., `"neural_unavailable"`, `"video_unavailable"`). |
| `alignment_qc` | `dict[str, Any]` | Alignment quality metrics populated during loading. |

#### Methods

##### `ensure_logger()`

```python
def ensure_logger(self) -> logging.Logger
```

Return a session-specific logger (writes to `outputs/reports/logs/`).

---

##### `load_spikes()`

```python
def load_spikes(self) -> tuple[pd.DataFrame | None, dict[str, Any] | None]
```

Load or extract spike data for this session.

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[pd.DataFrame \| None, dict[str, Any] \| None]` | `(units_df, spike_times_dict)`. `units_df` has one row per unit with metadata columns. `spike_times_dict` maps unit ID strings to `np.ndarray` of spike times in NWB seconds. Returns `(None, None)` if neural data is unavailable. |

**Side effects:**

- On first call, extracts data from NWB and saves to
  `outputs/neural/session_{id}_units.parquet` and
  `outputs/neural/session_{id}_spike_times.npz`.
- Subsequent calls load from disk.
- Appends `"neural_unavailable"` to `qc_flags` if extraction fails.

**Example:**

```python
bundle = get_session_bundle(1064644573)
units, spikes = bundle.load_spikes()

if units is not None:
    print(f"{len(units)} units, {len(spikes)} spike trains")
    print(f"First unit has {len(spikes[list(spikes.keys())[0]])} spikes")
```

---

##### `load_trials_and_events()`

```python
def load_trials_and_events(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]
```

Load or extract trial structure and behavioral events.

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[pd.DataFrame \| None, pd.DataFrame \| None]` | `(trials_df, events_df)`. `trials_df` has columns `t` (or `t_start`/`t_end`), `trial_type`, etc. `events_df` merges all behavioral time series from the NWB processing module. |

**Side effects:**

- Saves to `outputs/behavior/session_{id}_trials.parquet` and
  `outputs/behavior/session_{id}_events.parquet`.

**Example:**

```python
bundle = get_session_bundle(1064644573)
trials, events = bundle.load_trials_and_events()

if trials is not None:
    print(trials[["t", "trial_type"]].head())
```

---

##### `load_eye_features()`

```python
def load_eye_features(self) -> pd.DataFrame | None
```

Load or extract eye-tracking features (pupil area, gaze position, etc.).

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame \| None` | Eye feature DataFrame with a `t` column, or `None` if eye tracking is unavailable. |

**Side effects:**

- Calls `features_eye.derive_eye_features()` on first extraction.
- Saves to `outputs/eye/session_{id}_eye_features.parquet`.
- Appends `"eye_unavailable"` to `qc_flags` if data is missing.

**Example:**

```python
bundle = get_session_bundle(1064644573)
eye = bundle.load_eye_features()

if eye is not None:
    print(eye.columns.tolist())
```

---

##### `load_video_assets()`

```python
def load_video_assets(self) -> pd.DataFrame
```

Discover, download (if configured), and catalog video assets for this session.

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Asset catalog with columns: `session_id`, `camera`, `source`, `s3_uri_video`, `local_video_path`, `n_frames`, `fps_est`, `qc_flags`, and more. |

**Side effects:**

- Delegates to `io_video.build_video_assets()`.
- Appends `"video_unavailable"` to `qc_flags` if the result is empty.

**Example:**

```python
bundle = get_session_bundle(1064644573)
assets = bundle.load_video_assets()
print(assets[["camera", "n_frames", "fps_est"]])
```

---

##### `load_frame_times()`

```python
def load_frame_times(self, camera: str | None = None) -> pd.DataFrame
```

Load per-frame timestamps for one or all cameras.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera` | `str \| None` | `None` | Camera name to filter by (e.g., `"eye"`). If `None`, returns all cameras. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Frame-time table with columns `session_id`, `camera`, `frame_idx`, `t`. |

**Example:**

```python
bundle = get_session_bundle(1064644573)
ft = bundle.load_frame_times(camera="side")
print(ft[["frame_idx", "t"]].head())
```
