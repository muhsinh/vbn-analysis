# pose_inference

Automated SLEAP pose inference pipeline. Discovers trained models, runs batch
inference on locally-cached videos, converts predictions to Parquet, and
provides active-learning utilities for iterative labeling.

**Source:** `src/pose_inference.py`

---

## Functions

### `discover_sleap_models`

```python
def discover_sleap_models(
    search_dirs: List[Path] | None = None,
) -> List[Dict[str, Any]]
```

Find trained SLEAP models on disk by scanning standard directories.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_dirs` | `List[Path] \| None` | `None` | Directories to search. If `None`, searches `pose_projects/`, `outputs/models/`, and `data/sleap_models/`. |

**Returns:**

| Type | Description |
|------|-------------|
| `List[Dict[str, Any]]` | List of model descriptors, each with keys: |

**Model descriptor keys:**

| Key | Type | Description |
|-----|------|-------------|
| `path` | `str` | Filesystem path to the model directory, package, or weights file. |
| `type` | `str` | `"directory"`, `"package"`, or `"weights"`. |
| `name` | `str` | Model name (backbone type for directory models, stem for packages/weights). |
| `config_path` | `str` | Path to `training_config.json` (directory models only). |
| `skeleton` | `str \| None` | Skeleton name if found in config. |
| `n_keypoints` | `int \| None` | Number of keypoints if found in config. |

**Discovery patterns:**

1. **Directory models**: contains `training_config.json`.
2. **Package models**: `.zip` or `.pkg.slp` files.
3. **Weight files**: `.h5` or `.keras` files not already covered by a directory model.

**Example:**

```python
from pose_inference import discover_sleap_models

models = discover_sleap_models()
for m in models:
    print(f"{m['type']}: {m['path']} ({m.get('name', '?')})")
# directory: pose_projects/centroid_model (unet)
# package: data/sleap_models/mouse_topdown.pkg.slp
```

---

### `run_sleap_inference`

```python
def run_sleap_inference(
    video_path: Path,
    model_paths: List[str] | str,
    output_path: Path | None = None,
    batch_size: int = 4,
    peak_threshold: float = 0.2,
) -> Path
```

Run SLEAP inference on a single video file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `Path` | -- | Path to the `.mp4` video file. |
| `model_paths` | `List[str] \| str` | -- | Path(s) to trained SLEAP model directories or packages. For top-down pipelines, provide `[centroid_model, instance_model]`. For single-animal bottom-up, provide `[single_model]`. |
| `output_path` | `Path \| None` | `None` | Where to write the `.slp` predictions file. Defaults to `{video_path}.predictions.slp`. |
| `batch_size` | `int` | `4` | GPU batch size. Reduce if encountering out-of-memory errors. |
| `peak_threshold` | `float` | `0.2` | Confidence threshold for peak detection during inference. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the `.slp` predictions file. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | SLEAP is not installed. |
| `FileNotFoundError` | The video file does not exist. |

**Example:**

```python
from pose_inference import run_sleap_inference
from pathlib import Path

slp_path = run_sleap_inference(
    video_path=Path("data/side.mp4"),
    model_paths=["pose_projects/centroid", "pose_projects/instance"],
    batch_size=8,
)
print(f"Predictions written to {slp_path}")
```

---

### `run_batch_inference`

```python
def run_batch_inference(
    session_ids: List[int] | None = None,
    cameras: List[str] | None = None,
    model_paths: List[str] | str | None = None,
    batch_size: int = 4,
    skip_existing: bool = True,
) -> pd.DataFrame
```

Run SLEAP inference on all locally-cached videos. This is the main entry
point for batch processing. It discovers videos from the asset catalog, runs
inference, and converts predictions to Parquet.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_ids` | `List[int] \| None` | `None` | Specific sessions to process. If `None`, processes all sessions in the video asset catalog. |
| `cameras` | `List[str] \| None` | `None` | Which cameras to process. Defaults to `config.video_cameras`. |
| `model_paths` | `List[str] \| str \| None` | `None` | Explicit model path(s). If `None`, auto-discovers via `discover_sleap_models()`. |
| `batch_size` | `int` | `4` | GPU batch size. |
| `skip_existing` | `bool` | `True` | Skip videos that already have `.slp` prediction files. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Summary table with one row per video processed. |

**Output columns:**

| Column | Description |
|--------|-------------|
| `session_id` | Session identifier. |
| `camera` | Camera name. |
| `video_path` | Path to the source video. |
| `slp_path` | Path to the `.slp` predictions file. |
| `parquet_path` | Path to the converted Parquet file. |
| `n_frames` | Number of frames with predictions. |
| `status` | `"SUCCESS"`, `"SKIPPED_EXISTS"`, `"NO_LOCAL_VIDEO"`, or `"FAILED: {error}"`. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | No SLEAP models found and `model_paths` is `None`. |

**Example:**

```python
from pose_inference import run_batch_inference

# Process all cached videos
results = run_batch_inference(batch_size=8)
print(results[["session_id", "camera", "n_frames", "status"]])

# Process specific sessions
results = run_batch_inference(
    session_ids=[1064644573, 1064644574],
    cameras=["side"],
)
```

---

### `slp_to_parquet`

```python
def slp_to_parquet(
    slp_path: Path,
    session_id: int,
    camera: str,
    output_path: Path | None = None,
    confidence_threshold: float = 0.0,
) -> int
```

Convert a SLEAP `.slp` predictions file to the standardized
`pose_predictions.parquet` format. Automatically attaches timestamps from
the frame-times catalog or raw camera timestamp files.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slp_path` | `Path` | -- | Path to the `.slp` file. |
| `session_id` | `int` | -- | Session ID. |
| `camera` | `str` | -- | Camera name. |
| `output_path` | `Path \| None` | `None` | Output Parquet path. Defaults to `outputs/pose/session_{id}_pose_predictions.parquet`. |
| `confidence_threshold` | `float` | `0.0` | If > 0, drop instances whose mean keypoint score is below this threshold. |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of frames (rows) converted. Returns `0` if no predictions are found. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | SLEAP is not installed. |
| `FileNotFoundError` | The `.slp` file does not exist. |

**Output columns:**

`session_id`, `camera`, `frame_idx`, `instance`, `t`, `instance_score`,
`{keypoint}_x`, `{keypoint}_y`, `{keypoint}_score` for each skeleton node.

**Example:**

```python
from pose_inference import slp_to_parquet

n = slp_to_parquet(
    slp_path=Path("outputs/pose/predictions/session_123_side.predictions.slp"),
    session_id=123,
    camera="side",
    confidence_threshold=0.3,
)
print(f"Converted {n} prediction rows")
```

---

### `auto_discover_sleap_csvs`

```python
def auto_discover_sleap_csvs(
    session_id: int | None = None,
) -> List[Dict[str, Any]]
```

Find all SLEAP CSV exports in `outputs/pose/` and `outputs/labeling/`.
Infers session ID and camera name from filenames and directory structure.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int \| None` | `None` | If provided, only return CSVs matching this session. |

**Returns:**

| Type | Description |
|------|-------------|
| `List[Dict[str, Any]]` | List of dicts with keys `path`, `session_id`, `camera`, `filename`. |

**Example:**

```python
from pose_inference import auto_discover_sleap_csvs

csvs = auto_discover_sleap_csvs(session_id=1064644573)
for c in csvs:
    print(f"{c['filename']}: session={c['session_id']} camera={c['camera']}")
```

---

### `train_sleap_model`

```python
def train_sleap_model(
    labels_path: Path,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 100,
    batch_size: int = 4,
) -> Path
```

Train a SLEAP model from labeled data. Wraps the SLEAP training CLI.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels_path` | `Path` | -- | Path to the `.slp` labels file containing your labeled frames (typically ~150 frames for a usable model). |
| `config_path` | `Path \| None` | `None` | Path to a SLEAP training config JSON. If `None`, uses a sensible default for single-animal bottom-up pose estimation. |
| `output_dir` | `Path \| None` | `None` | Where to save the trained model. Defaults to `pose_projects/trained_models/{labels_stem}/`. |
| `epochs` | `int` | `100` | Number of training epochs. |
| `batch_size` | `int` | `4` | Training batch size. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the trained model directory. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | SLEAP is not installed. |
| `FileNotFoundError` | The labels file does not exist. |

**Example:**

```python
from pose_inference import train_sleap_model

model_dir = train_sleap_model(
    labels_path=Path("pose_projects/my_labels.slp"),
    epochs=200,
    batch_size=8,
)
print(f"Model saved to {model_dir}")
```

---

### `suggest_frames_to_label`

```python
def suggest_frames_to_label(
    slp_path: Path,
    n_suggestions: int = 20,
    strategy: str = "low_confidence",
) -> np.ndarray
```

Suggest frames that would benefit most from manual labeling, using prediction
confidence as a signal. This supports active-learning workflows where you
iteratively label the most uncertain frames.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slp_path` | `Path` | -- | Path to the `.slp` predictions file from inference. |
| `n_suggestions` | `int` | `20` | Number of frames to suggest. |
| `strategy` | `str` | `"low_confidence"` | Selection strategy. `"low_confidence"`: pick the N frames with the lowest mean keypoint confidence. `"spread"`: pick low-confidence frames but spread them across the session temporally. |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray` | Array of frame indices (dtype `int`) to label next. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | SLEAP is not installed. |

**Example:**

```python
from pose_inference import suggest_frames_to_label

frames = suggest_frames_to_label(
    slp_path=Path("outputs/pose/predictions/session_123_side.predictions.slp"),
    n_suggestions=30,
    strategy="spread",
)
print(f"Label these {len(frames)} frames next: {frames}")
```
