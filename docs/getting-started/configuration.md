# Configuration Reference

The VBN Analysis Suite is configured through **environment variables** that populate a **`Config` dataclass** singleton. This page is the complete reference for every configuration knob, how values are loaded, and how provenance tracking works.

---

## How Configuration is Loaded

```
Environment Variables
        |
        v
   _get_env()          Read from os.environ, fall back to defaults
        |
        v
   get_config()        Build a Config dataclass (singleton)
        |
        v
   _CONFIG             Module-level singleton, created once per process
```

The configuration is loaded **lazily** on the first call to `get_config()`. Once created, the same `Config` instance is returned for all subsequent calls within the same Python process.

```python
from config import get_config

cfg = get_config()  # Creates the singleton
cfg2 = get_config()  # Returns the same object
assert cfg is cfg2  # True
```

!!! note "Singleton pattern"

    The `Config` object is stored in a module-level variable `_CONFIG`. This means:

    - Environment variables are read **once**, at first access
    - Changing `os.environ` after the first `get_config()` call has **no effect**
    - To force a reload, you must reset the module-level variable (not recommended in production)

    ```python
    # Force reload (testing/debugging only)
    import config
    config._CONFIG = None
    cfg = config.get_config()  # Re-reads environment variables
    ```

---

## Environment Variables

All environment variables are optional. If unset, the default value applies.

### Core Pipeline Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ACCESS_MODE` | `str` | `"sdk"` | How NWB files are resolved. `"sdk"` uses AllenSDK to download/cache. `"manual"` uses paths from `sessions.csv`. |
| `POSE_TOOL` | `str` | `"sleap"` | Pose estimation backend. `"sleap"` or `"dlc"` (DeepLabCut). |
| `MODEL_NAME` | `str` | `"xgboost"` | Model for neural-behavior fusion. Currently only `"xgboost"` is implemented. |
| `BIN_SIZE_S` | `float` | `0.025` | Bin width in seconds for time-aligned fusion tables. Controls temporal resolution. |
| `MOCK_MODE` | `bool` | `false` | When `"true"`, all data loading returns synthetic data. No NWB files or downloads required. |

### Video Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VIDEO_SOURCE` | `str` | `"auto"` | Video fetch strategy. `"auto"` tries S3 then local. `"local"` skips S3. `"s3"` forces S3 download. |
| `VIDEO_CACHE_DIR` | `path` | `data/raw/visual-behavior-neuropixels` | Local directory for cached S3 video downloads. |
| `VIDEO_BUCKET` | `str` | `"allen-brain-observatory"` | S3 bucket name for video assets. |
| `VIDEO_BASE_PATH` | `str` | `"visual-behavior-neuropixels/raw-data"` | S3 key prefix within the bucket. |
| `VIDEO_CAMERAS` | `csv` | `"eye,face,side"` | Comma-separated list of camera names to process. |

---

## Setting Environment Variables

=== "Shell export"

    ```bash
    export ACCESS_MODE=manual
    export MOCK_MODE=true
    export BIN_SIZE_S=0.05
    export VIDEO_CAMERAS=eye,face
    jupyter lab
    ```

=== ".env file"

    Create a `.env` file in the project root:

    ```ini
    # .env
    ACCESS_MODE=sdk
    MOCK_MODE=false
    BIN_SIZE_S=0.025
    POSE_TOOL=sleap
    VIDEO_SOURCE=auto
    VIDEO_CAMERAS=eye,face,side
    ```

    The pipeline uses `python-dotenv` to load this automatically if present.

=== "In-notebook override"

    ```python
    import os
    os.environ["MOCK_MODE"] = "true"
    os.environ["BIN_SIZE_S"] = "0.05"

    # IMPORTANT: Reset the singleton before calling get_config()
    import config
    config._CONFIG = None

    cfg = config.get_config()
    print(cfg.mock_mode)   # True
    print(cfg.bin_size_s)  # 0.05
    ```

!!! warning "Order matters for in-notebook overrides"

    You must set `os.environ` **before** the first call to `get_config()`, or reset `config._CONFIG = None` to force a re-read. Any code that imports from `config` at module load time will trigger the singleton creation.

---

## The `Config` Dataclass

The full dataclass definition with types and defaults:

```python
@dataclass
class Config:
    # --- Core pipeline settings ---
    access_mode: str = "sdk"
    """'sdk' uses AllenSDK to download/cache NWB files.
       'manual' uses paths from sessions.csv."""

    pose_tool: str = "sleap"
    """Pose estimation backend: 'sleap' or 'dlc'."""

    model_name: str = "xgboost"
    """Model for neural-behavior fusion modeling."""

    bin_size_s: float = 0.025
    """Bin width in seconds (25 ms default = 40 Hz resolution)."""

    categorical_cols: list[str] = field(
        default_factory=lambda: ["motif_id", "trial_type", "stimulus_name"]
    )
    """Columns treated as categorical in modeling (one-hot encoded for XGBoost)."""

    mock_mode: bool = False
    """Generate synthetic data instead of loading real NWB files."""

    # --- Paths ---
    outputs_dir: Path = ROOT_DIR / "outputs"
    """Root directory for all pipeline outputs."""

    cache_dir: Path = ROOT_DIR / "outputs" / "cache"
    """Intermediate computation cache (joblib)."""

    pose_projects_dir: Path = ROOT_DIR / "pose_projects"
    """SLEAP/DLC project files and trained models."""

    data_dir: Path = ROOT_DIR / "data"
    """Raw data directory (NWB files, etc.)."""

    video_source: str = "auto"
    """Video fetch strategy: 'auto', 'local', or 's3'."""

    video_cache_dir: Path = ROOT_DIR / "data" / "raw" / "visual-behavior-neuropixels"
    """Local cache for S3 video downloads."""

    video_bucket: str = "allen-brain-observatory"
    """S3 bucket name for video assets."""

    video_base_path: str = "visual-behavior-neuropixels/raw-data"
    """S3 key prefix within the bucket."""

    video_cameras: list[str] = field(default_factory=lambda: ["eye", "face", "side"])
    """Camera names to process."""

    sessions_csv: Path = ROOT_DIR / "sessions.csv"
    """Path to the session inventory CSV."""

    legacy_dir: Path = ROOT_DIR / "legacy"
    """Directory for legacy session files (sessions.txt fallback)."""
```

### Accessing Config Fields

```python
from config import get_config

cfg = get_config()

# All fields are plain Python attributes
print(cfg.access_mode)       # "sdk"
print(cfg.bin_size_s)        # 0.025
print(cfg.outputs_dir)       # PosixPath('/path/to/vbn-analysis/outputs')
print(cfg.video_cameras)     # ['eye', 'face', 'side']
print(cfg.categorical_cols)  # ['motif_id', 'trial_type', 'stimulus_name']
```

### Serializing Config to a Dictionary

```python
cfg_dict = cfg.to_dict()
# Returns a JSON-serializable dict with all fields (Paths converted to strings)
# Also includes 'code_version' (git hash)
```

### Creating Output Directories

```python
cfg.ensure_dirs()
# Creates:
#   outputs/
#   outputs/cache/
#   outputs/reports/
#   outputs/reports/logs/
#   pose_projects/
#   data/
#   data/raw/visual-behavior-neuropixels/
```

---

## Detailed Variable Reference

### `ACCESS_MODE`

Controls how NWB files are located and loaded.

=== "sdk (default)"

    The AllenSDK resolves the session ID to a cached NWB file, downloading it from the Allen Institute S3 bucket if not already cached.

    ```python
    # Internally calls:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=...)
    session = cache.get_ecephys_session(session_id)
    ```

    **Pros**: No manual file management. Always gets the canonical file.
    **Cons**: First run downloads large files (10--40 GB per session).

=== "manual"

    Uses the `nwb_path` column from `sessions.csv` directly. No SDK calls, no downloads.

    ```csv
    session_id,nwb_path,video_dir,notes
    1064644573,/data/nwb/session_1064644573.nwb,/data/video/1064644573,
    ```

    **Pros**: Full control over file locations. Works offline.
    **Cons**: You must manage NWB files yourself.

### `POSE_TOOL`

| Value | Description |
|-------|-------------|
| `sleap` | Use SLEAP for pose estimation. Notebooks 06--07 will look for `.slp` models and CSV exports. |
| `dlc` | Use DeepLabCut. The feature extraction code works with either format. |

### `BIN_SIZE_S`

The temporal resolution for the fusion table. This is the bin width used when aligning neural spikes, behavioral features, and pose features onto a common time grid.

| Value | Resolution | Typical use case |
|-------|-----------|-----------------|
| `0.001` | 1 ms | Fine-grained spike analysis |
| `0.010` | 10 ms | PETH computation |
| `0.025` | 25 ms (default) | General-purpose fusion |
| `0.050` | 50 ms | Coarser analysis, faster computation |
| `0.100` | 100 ms | Slow behavioral features |

!!! tip "Choosing a bin size"

    The default of 25 ms (40 Hz) is a good balance between temporal resolution and statistical power. For fast neural dynamics, use 10 ms. For slow behavioral signals (e.g., pupil diameter), 50--100 ms is sufficient and reduces computational cost.

### `MOCK_MODE`

Accepts any of: `"true"`, `"1"`, `"yes"`, `"y"` (case-insensitive) to enable. Everything else is treated as `false`.

```python
def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}
```

When enabled, all `io_nwb.open_nwb_handle()` calls return a mock NWB object with synthetic data. This cascades through the entire pipeline --- every modality gets plausible fake data.

### `VIDEO_SOURCE`

| Value | Behavior |
|-------|----------|
| `auto` | Try S3 download first. If it fails (no network, bucket unreachable), fall back to local files. |
| `local` | Only use files already present in `VIDEO_CACHE_DIR` or `video_dir` from `sessions.csv`. No network access. |
| `s3` | Always download from S3. Fail if download fails. |

### `VIDEO_CAMERAS`

A comma-separated string parsed into a list:

```python
# Environment variable
VIDEO_CAMERAS=eye,face

# Parsed by _parse_csv() into:
["eye", "face"]
```

The Allen VBN dataset provides three cameras per session: `eye`, `face`, and `side`. You can restrict processing to a subset to save disk space and time.

### `MODEL_NAME`

Currently only `"xgboost"` is implemented in `modeling.py`. The architecture is designed for extension:

```python
def make_model(name: str, task: str, **kwargs) -> Any:
    if name == "xgboost":
        return xgb.XGBRegressor(...)
    if name == "catboost":
        raise NotImplementedError("CatBoost not yet implemented")
    raise ValueError(f"Unsupported model: {name}")
```

---

## Provenance Tracking

Every artifact written by the pipeline includes provenance metadata. This ensures full reproducibility --- you can always trace an output back to the code and data that produced it.

### `make_provenance()`

```python
from config import make_provenance

prov = make_provenance(session_id=1064644573, alignment_method="nwb")
```

Returns:

```python
{
    "session_id": 1064644573,
    "code_version": "9e576adf1234567890abcdef...",  # git HEAD hash
    "created_at": "2026-03-09T12:34:56.789000+00:00",
    "alignment_method": "nwb",
}
```

| Field | Source | Description |
|-------|--------|-------------|
| `session_id` | Caller passes it | The session being processed (or `None` for global artifacts) |
| `code_version` | `get_code_version()` | Git HEAD commit hash, or `"unknown"` if not in a git repo |
| `created_at` | `pd.Timestamp.utcnow()` | UTC timestamp of artifact creation |
| `alignment_method` | Caller passes it | How timestamps were aligned (usually `"nwb"`) |

### `get_code_version()`

```python
from config import get_code_version

version = get_code_version()
# Returns: "9e576adf1234567890abcdef..."  (git HEAD hash)
# Or:      "unknown"                       (not a git repo)
```

Internally runs `git rev-parse HEAD` in a subprocess. Falls back gracefully if git is not available.

### Sidecar Metadata Files

Every `.parquet` artifact has a companion `.parquet.meta.json` sidecar:

```
outputs/neural/session_1064644573_units.parquet
outputs/neural/session_1064644573_units.parquet.meta.json
```

The sidecar contains:

```json
{
  "timebase": "nwb_seconds",
  "provenance": {
    "session_id": 1064644573,
    "code_version": "9e576adf...",
    "created_at": "2026-03-09T12:34:56.789000+00:00",
    "alignment_method": "nwb"
  }
}
```

!!! info "Why sidecar files?"

    Parquet metadata is embedded in the file schema, but not all tools can read it. The sidecar JSON provides tool-agnostic access to provenance information --- you can inspect it with `cat`, `jq`, or any JSON library.

---

## Config Snapshot

At the start of every pipeline run, the full configuration is serialized to JSON:

```python
from config import write_config_snapshot

path = write_config_snapshot()
# Writes to: outputs/reports/config_snapshot.json
```

### Example `config_snapshot.json`

```json
{
  "access_mode": "sdk",
  "pose_tool": "sleap",
  "model_name": "xgboost",
  "bin_size_s": 0.025,
  "categorical_cols": [
    "motif_id",
    "trial_type",
    "stimulus_name"
  ],
  "mock_mode": false,
  "outputs_dir": "/Users/you/projects/vbn-analysis/outputs",
  "cache_dir": "/Users/you/projects/vbn-analysis/outputs/cache",
  "pose_projects_dir": "/Users/you/projects/vbn-analysis/pose_projects",
  "data_dir": "/Users/you/projects/vbn-analysis/data",
  "video_source": "auto",
  "video_cache_dir": "/Users/you/projects/vbn-analysis/data/raw/visual-behavior-neuropixels",
  "video_bucket": "allen-brain-observatory",
  "video_base_path": "visual-behavior-neuropixels/raw-data",
  "video_cameras": [
    "eye",
    "face",
    "side"
  ],
  "sessions_csv": "/Users/you/projects/vbn-analysis/sessions.csv",
  "legacy_dir": "/Users/you/projects/vbn-analysis/legacy",
  "code_version": "9e576adf1234567890abcdef1234567890abcdef"
}
```

!!! tip "Version control your config snapshots"

    While `outputs/` is git-ignored, consider copying important `config_snapshot.json` files to a tracked location when you publish results. This ensures anyone can reproduce your exact configuration.

---

## Overriding Config Per-Notebook

Each notebook can override settings before importing the config module. The pattern is:

```python
# Cell 1: Override environment variables
import os
os.environ["BIN_SIZE_S"] = "0.010"   # Finer resolution for this notebook
os.environ["VIDEO_CAMERAS"] = "eye"  # Only process eye camera

# Cell 2: Import config (singleton is created here)
import sys
sys.path.insert(0, "../src")
from config import get_config

cfg = get_config()
print(cfg.bin_size_s)      # 0.01
print(cfg.video_cameras)   # ['eye']
```

!!! danger "Import order matters"

    If any cell or import statement triggers `get_config()` before your `os.environ` overrides, the default values will be locked in. Always set environment variables in the **first cell** of the notebook, before any `from config import ...` or `from io_sessions import ...` statements.

For notebooks that run after the singleton is already created (e.g., when Notebook 09 calls sub-notebooks), you can modify the config object directly:

```python
from config import get_config

cfg = get_config()
cfg.bin_size_s = 0.010            # Direct attribute mutation
cfg.video_cameras = ["eye"]       # Works, but not persisted to env
```

!!! warning "Direct mutation is not persisted"

    Modifying `cfg` attributes directly works for the current process but is **not** reflected in `config_snapshot.json` unless you call `write_config_snapshot()` again after the changes.

---

## Boolean Parsing

The `_as_bool()` helper is used for `MOCK_MODE` and any future boolean environment variables:

```python
def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}
```

| Input | Result |
|-------|--------|
| `"true"` | `True` |
| `"True"` | `True` |
| `"TRUE"` | `True` |
| `"1"` | `True` |
| `"yes"` | `True` |
| `"y"` | `True` |
| `"false"` | `False` |
| `"0"` | `False` |
| `"no"` | `False` |
| `""` | `False` |
| `None` (unset) | Uses `default` parameter |

---

## CSV List Parsing

The `_parse_csv()` helper is used for `VIDEO_CAMERAS` and any future list-valued environment variables:

```python
def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]
```

| Input | Result |
|-------|--------|
| `"eye,face,side"` | `["eye", "face", "side"]` |
| `"eye, face"` | `["eye", "face"]` |
| `"eye"` | `["eye"]` |
| `""` | `[]` |
| `None` (unset) | Uses `default` parameter |

---

## Path Resolution

All paths in the `Config` dataclass are resolved relative to `ROOT_DIR`, which is the project root (parent of `src/`):

```python
ROOT_DIR = Path(__file__).resolve().parents[1]
# e.g., /Users/you/projects/vbn-analysis
```

| Config field | Default path |
|-------------|-------------|
| `outputs_dir` | `<ROOT>/outputs` |
| `cache_dir` | `<ROOT>/outputs/cache` |
| `pose_projects_dir` | `<ROOT>/pose_projects` |
| `data_dir` | `<ROOT>/data` |
| `video_cache_dir` | `<ROOT>/data/raw/visual-behavior-neuropixels` |
| `sessions_csv` | `<ROOT>/sessions.csv` |
| `legacy_dir` | `<ROOT>/legacy` |

You can override `video_cache_dir` via the `VIDEO_CACHE_DIR` environment variable to point to a different disk (e.g., an external SSD with more space):

```bash
export VIDEO_CACHE_DIR=/mnt/fast-ssd/vbn-video-cache
```

---

## Artifact Registry

The artifact registry (`outputs/reports/artifact_registry.parquet`) provides a single inventory of every file produced by the pipeline:

```python
from reports import build_artifact_registry, write_artifact_registry

# Build the registry (scans outputs/ recursively)
registry = build_artifact_registry()
print(registry[["step", "artifact_path", "session_id"]].head())

# Write to disk
path = write_artifact_registry()
```

| Column | Description |
|--------|-------------|
| `step` | Parent directory name (e.g., `"neural"`, `"behavior"`, `"pose"`) |
| `artifact_path` | Absolute path to the artifact |
| `exists` | Whether the file exists at scan time |
| `last_modified` | File modification timestamp |
| `session_id` | Extracted from filename (e.g., `session_1064644573`) or `None` |
| `notes` | Free-text (currently empty, available for manual annotation) |

---

## Caching

The pipeline caches intermediate results using `joblib` to avoid redundant computation:

```python
from io_sessions import cache_step

result = cache_step(
    session_id=1064644573,
    step="spike_extraction",
    params={"access_mode": "sdk"},
    compute_fn=lambda: expensive_computation(),
)
```

Cache files are stored under `outputs/cache/session_{id}/` with filenames that include an MD5 hash of the parameters. This means changing any parameter automatically invalidates the cache.

!!! tip "Clearing the cache"

    To force recomputation, delete the cache directory:

    ```bash
    rm -rf outputs/cache/session_1064644573/
    ```

    Or delete all caches:

    ```bash
    rm -rf outputs/cache/
    ```
