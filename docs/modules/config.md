# config

::: src.config

Global configuration, directory management, and provenance tracking for the
VBN Analysis Suite.

**Source:** `src/config.py`

---

## Constants

### `ROOT_DIR`

```python
ROOT_DIR: Path
```

Absolute path to the project root directory (parent of `src/`). Computed at
import time via `Path(__file__).resolve().parents[1]`.

---

## Classes

### `Config`

```python
@dataclass
class Config:
    access_mode: str = "sdk"
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
```

Central dataclass holding every configurable parameter. Created once by
[`get_config()`](#get_config) and cached for the lifetime of the process.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `access_mode` | `str` | `"sdk"` | How to resolve NWB files. `"sdk"` uses AllenSDK S3 cache; `"manual"` expects user-provided paths. |
| `pose_tool` | `str` | `"sleap"` | Pose estimation backend (`"sleap"` or `"dlc"`). |
| `model_name` | `str` | `"xgboost"` | Default model for neural-behavior fusion in the modeling module. |
| `bin_size_s` | `float` | `0.025` | Default time-bin width in seconds (25 ms). |
| `categorical_cols` | `list[str]` | `["motif_id", "trial_type", "stimulus_name"]` | Columns treated as categorical in the modeling pipeline. |
| `mock_mode` | `bool` | `False` | When `True`, NWB reads return synthetic data so notebooks run without real files. |
| `outputs_dir` | `Path` | `ROOT_DIR / "outputs"` | Root directory for all pipeline outputs. |
| `cache_dir` | `Path` | `ROOT_DIR / "outputs" / "cache"` | Per-session computation cache (joblib). |
| `pose_projects_dir` | `Path` | `ROOT_DIR / "pose_projects"` | Directory for SLEAP/DLC project files and trained models. |
| `data_dir` | `Path` | `ROOT_DIR / "data"` | Root for raw and downloaded data. |
| `video_source` | `str` | `"auto"` | Video resolution strategy: `"auto"`, `"s3"`, or `"local"`. |
| `video_cache_dir` | `Path` | `ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"` | Local cache for downloaded video files. |
| `video_bucket` | `str` | `"allen-brain-observatory"` | S3 bucket name for Allen Brain Observatory videos. |
| `video_base_path` | `str` | `"visual-behavior-neuropixels/raw-data"` | Base key prefix inside the S3 bucket. |
| `video_cameras` | `list[str]` | `["eye", "face", "side"]` | Camera names to process. |
| `sessions_csv` | `Path` | `ROOT_DIR / "sessions.csv"` | Path to the session manifest CSV. |
| `legacy_dir` | `Path` | `ROOT_DIR / "legacy"` | Directory for legacy session list files. |

#### Methods

##### `ensure_dirs()`

```python
def ensure_dirs(self) -> None
```

Create all required output directories if they do not exist. Called
automatically by [`write_config_snapshot()`](#write_config_snapshot).

**Example:**

```python
cfg = get_config()
cfg.ensure_dirs()
# outputs/, outputs/cache/, outputs/reports/, outputs/reports/logs/,
# pose_projects/, data/, data/raw/visual-behavior-neuropixels/
# are now guaranteed to exist.
```

##### `to_dict()`

```python
def to_dict(self) -> Dict[str, Any]
```

Serialize the configuration to a plain dictionary. `Path` values are converted
to strings. The current `code_version` (git hash) is included.

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Flat dictionary of all configuration values plus `code_version`. |

**Example:**

```python
cfg = get_config()
snapshot = cfg.to_dict()
print(snapshot["bin_size_s"])   # 0.025
print(snapshot["code_version"])  # e.g. "9e576ad..."
```

---

## Functions

### `get_config`

```python
def get_config() -> Config
```

Return the global `Config` singleton. On first call, reads environment
variables to populate fields; subsequent calls return the cached instance.

#### Environment Variables

| Variable | Maps to | Example |
|----------|---------|---------|
| `ACCESS_MODE` | `access_mode` | `"manual"` |
| `POSE_TOOL` | `pose_tool` | `"sleap"` |
| `MODEL_NAME` | `model_name` | `"xgboost"` |
| `BIN_SIZE_S` | `bin_size_s` | `"0.01"` |
| `MOCK_MODE` | `mock_mode` | `"true"` |
| `VIDEO_SOURCE` | `video_source` | `"s3"` |
| `VIDEO_CACHE_DIR` | `video_cache_dir` | `"/data/videos"` |
| `VIDEO_BUCKET` | `video_bucket` | `"my-bucket"` |
| `VIDEO_BASE_PATH` | `video_base_path` | `"my/prefix"` |
| `VIDEO_CAMERAS` | `video_cameras` | `"eye,face"` |

**Returns:**

| Type | Description |
|------|-------------|
| `Config` | The global configuration instance. |

**Example:**

```python
from config import get_config

cfg = get_config()
print(cfg.bin_size_s)    # 0.025
print(cfg.mock_mode)     # False
print(cfg.outputs_dir)   # /path/to/project/outputs
```

---

### `get_code_version`

```python
def get_code_version() -> str
```

Return the current git commit hash (full SHA) by running `git rev-parse HEAD`
in the project root. Returns `"unknown"` if git is unavailable or the
directory is not a repository.

**Returns:**

| Type | Description |
|------|-------------|
| `str` | 40-character git SHA, or `"unknown"`. |

**Example:**

```python
from config import get_code_version

version = get_code_version()
print(version)  # "9e576adc3f..."
```

---

### `write_config_snapshot`

```python
def write_config_snapshot(path: Path | None = None) -> Path
```

Write a JSON snapshot of the current configuration to disk. Useful for
reproducibility -- embed the config in every experiment run.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path \| None` | `None` | Output file path. If `None`, writes to `outputs/reports/config_snapshot.json`. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the written JSON file. |

**Example:**

```python
from config import write_config_snapshot

path = write_config_snapshot()
print(f"Config saved to {path}")

# Or write to a custom location:
path = write_config_snapshot(Path("runs/exp01/config.json"))
```

---

### `make_provenance`

```python
def make_provenance(
    session_id: int | None,
    alignment_method: str,
) -> Dict[str, Any]
```

Build a provenance dictionary for embedding in artifact metadata. Every
Parquet and NPZ file produced by the pipeline includes this information.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int \| None` | -- | The session this artifact belongs to, or `None` for cross-session artifacts. |
| `alignment_method` | `str` | -- | How timestamps were aligned (e.g., `"nwb"`, `"timestamps"`, `"sleap_inference"`). |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary with keys `session_id`, `code_version`, `created_at` (ISO 8601 UTC), and `alignment_method`. |

**Example:**

```python
from config import make_provenance

prov = make_provenance(session_id=1064644573, alignment_method="nwb")
print(prov)
# {
#     "session_id": 1064644573,
#     "code_version": "9e576ad...",
#     "created_at": "2026-03-09T14:22:01.123456+00:00",
#     "alignment_method": "nwb"
# }
```
