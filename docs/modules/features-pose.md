# features_pose

Pose feature extraction from keypoint predictions. Computes rich behavioral
features (velocity, acceleration, body geometry, inter-keypoint distances,
stillness detection) for neural-behavior correlation analysis. Also provides
utilities for confidence filtering, frame sampling, and exporting data for
SLEAP labeling workflows.

**Source:** `src/features_pose.py`

---

## Functions

### `derive_pose_features`

```python
def derive_pose_features(
    pose_df: pd.DataFrame | None,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame | None
```

Extract a comprehensive set of behavioral features from pose keypoint
predictions. This is the main feature-engineering function for the
pose-to-motifs pipeline.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pose_df` | `pd.DataFrame \| None` | -- | Pose predictions table. Must contain a `t` column and keypoint columns in `{name}_x` / `{name}_y` format. May also include `{name}_score` columns. |
| `confidence_threshold` | `float` | `0.0` | Minimum confidence score. If > 0, low-confidence detections are replaced with NaN before feature computation. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame \| None` | Feature table with `t` and all computed columns, or `None` if input is `None`/empty or missing a `t` column. |

**Computed features:**

| Feature | Description |
|---------|-------------|
| `pose_speed` | Mean speed across all keypoints (pixels/s). |
| `pose_speed_std` | Standard deviation of per-keypoint speeds. |
| `{name}_vel` | Per-keypoint speed (pixels/s). |
| `{name}_accel` | Per-keypoint acceleration (pixels/s^2). |
| `body_length` | Distance between first and last keypoint (proxy for body stretch). Requires >= 2 keypoints. |
| `head_angle` | Angle of the vector from the second to the first keypoint (radians). Requires >= 2 keypoints. |
| `head_angular_vel` | Angular velocity of the head direction (rad/s). |
| `dist_{kp_i}_{kp_j}` | Distance between adjacent keypoint pairs. |
| `is_still` | Binary flag: `1` if `pose_speed` is below the 10th percentile (minimum 1.0 px/s). |

**Example:**

```python
from features_pose import derive_pose_features
import pandas as pd
import numpy as np

pose = pd.DataFrame({
    "t": np.linspace(0, 1, 100),
    "nose_x": np.random.randn(100).cumsum(),
    "nose_y": np.random.randn(100).cumsum(),
    "nose_score": np.ones(100) * 0.9,
    "tail_x": np.random.randn(100).cumsum() + 50,
    "tail_y": np.random.randn(100).cumsum(),
    "tail_score": np.ones(100) * 0.8,
})

features = derive_pose_features(pose, confidence_threshold=0.3)
print(features.columns.tolist())
# ['t', 'pose_speed', 'pose_speed_std', 'nose_vel', 'nose_accel',
#  'tail_vel', 'tail_accel', 'body_length', 'head_angle',
#  'head_angular_vel', 'is_still', 'dist_nose_tail']
```

---

### `filter_by_confidence`

```python
def filter_by_confidence(
    df: pd.DataFrame,
    threshold: float = 0.3,
    method: str = "nan",
) -> pd.DataFrame
```

Filter low-confidence keypoint detections from a pose DataFrame.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | Pose DataFrame with `{name}_x`, `{name}_y`, and `{name}_score` columns. |
| `threshold` | `float` | `0.3` | Minimum acceptable confidence score. |
| `method` | `str` | `"nan"` | Filtering strategy. `"nan"`: replace low-confidence `_x`/`_y` values with NaN (preserves row count). `"drop"`: drop entire rows where the mean score across keypoints falls below the threshold. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Filtered copy of the input DataFrame. |

**Example:**

```python
from features_pose import filter_by_confidence

# Replace low-confidence coordinates with NaN
filtered = filter_by_confidence(pose_df, threshold=0.5, method="nan")

# Drop entire rows below threshold
filtered = filter_by_confidence(pose_df, threshold=0.5, method="drop")
print(f"Kept {len(filtered)} / {len(pose_df)} rows")
```

---

### `sample_frame_indices`

```python
def sample_frame_indices(
    frame_times: pd.DataFrame,
    n_samples: int = 50,
) -> np.ndarray
```

Sample frame indices uniformly spread across the session duration. Useful
for selecting representative frames for SLEAP labeling.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_times` | `pd.DataFrame` | -- | Frame-time table with `frame_idx` and (optionally) `t` columns. |
| `n_samples` | `int` | `50` | Desired number of sample frames. Clamped to the total available frames. |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray` | Sorted, unique array of frame indices (dtype `int`). May be empty if no valid frames exist. |

**Example:**

```python
from features_pose import sample_frame_indices
from io_video import load_frame_times

ft = load_frame_times(session_id=1064644573, camera="side")
indices = sample_frame_indices(ft, n_samples=100)
print(f"Sampled {len(indices)} frames spanning [{indices[0]}, {indices[-1]}]")
```

---

### `export_labeling_frames`

```python
def export_labeling_frames(
    video_path: Path,
    frame_indices: np.ndarray,
    output_dir: Path,
    frame_times: pd.DataFrame,
    session_id: int,
    camera: str,
) -> Path
```

Export individual PNG frames from a video for manual labeling in SLEAP.
Creates a `frames/` subdirectory with sequentially named images and a
`labels.csv` manifest.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `Path` | -- | Path to the source `.mp4` video. |
| `frame_indices` | `np.ndarray` | -- | Array of frame indices to extract. |
| `output_dir` | `Path` | -- | Root output directory. |
| `frame_times` | `pd.DataFrame` | -- | Frame-time table for timestamp lookup. |
| `session_id` | `int` | -- | Session ID (written to `labels.csv`). |
| `camera` | `str` | -- | Camera name (written to `labels.csv`). |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the `output_dir` (which now contains `frames/` and `labels.csv`). |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | OpenCV (`cv2`) is not installed. |
| `RuntimeError` | Video cannot be opened, no frame indices provided, or no frames could be decoded. |

**Output structure:**

```
output_dir/
  frames/
    000001.png
    000002.png
    ...
  labels.csv    # columns: image_path, session_id, camera, seq_idx, frame_idx, t
```

**Example:**

```python
from features_pose import export_labeling_frames, sample_frame_indices
from io_video import load_frame_times
from pathlib import Path

ft = load_frame_times(session_id=1064644573, camera="side")
indices = sample_frame_indices(ft, n_samples=50)

output = export_labeling_frames(
    video_path=Path("data/side.mp4"),
    frame_indices=indices,
    output_dir=Path("outputs/labeling/session_1064644573_side"),
    frame_times=ft,
    session_id=1064644573,
    camera="side",
)
print(f"Frames exported to {output}")
```

---

### `export_labeling_video`

```python
def export_labeling_video(
    video_path: Path,
    frame_indices: np.ndarray,
    output_dir: Path,
    frame_times: pd.DataFrame,
    session_id: int,
    camera: str,
    label_fps: float = 30.0,
    write_pngs: bool = False,
) -> Path
```

Export a labeling video (MP4 or AVI) containing only the selected frames,
plus a `labels.csv` manifest. Optionally also writes individual PNG frames.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `Path` | -- | Source video file. |
| `frame_indices` | `np.ndarray` | -- | Frame indices to include. |
| `output_dir` | `Path` | -- | Root output directory. |
| `frame_times` | `pd.DataFrame` | -- | Frame-time table for timestamp lookup. |
| `session_id` | `int` | -- | Session ID. |
| `camera` | `str` | -- | Camera name. |
| `label_fps` | `float` | `30.0` | Frame rate for the output video. |
| `write_pngs` | `bool` | `False` | If `True`, also write individual PNG frames to `frames/`. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the `output_dir`. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | OpenCV is not installed. |
| `RuntimeError` | Video cannot be opened, no frame indices provided, no frames decoded, or VideoWriter fails. |

**Output structure:**

```
output_dir/
  labeling.mp4       # or labeling.avi as fallback
  labels.csv         # columns: video_path, session_id, camera, seq_idx, frame_idx, t [, image_path]
  frames/            # only if write_pngs=True
    000001.png
    ...
```

**Example:**

```python
from features_pose import export_labeling_video, sample_frame_indices

indices = sample_frame_indices(ft, n_samples=150)
output = export_labeling_video(
    video_path=Path("data/side.mp4"),
    frame_indices=indices,
    output_dir=Path("outputs/labeling/session_123_side"),
    frame_times=ft,
    session_id=123,
    camera="side",
    label_fps=15.0,
    write_pngs=True,
)
```

---

### `export_pose_predictions_from_sleap_csv`

```python
def export_pose_predictions_from_sleap_csv(
    csv_path: Path,
    session_id: int,
    camera: str,
    frame_times: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path
```

Convert a SLEAP CSV export (wide format) into the standardized
`pose_predictions.parquet` format used by the rest of the pipeline.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `Path` | -- | Path to the SLEAP CSV export file. |
| `session_id` | `int` | -- | Session ID to tag the predictions with. |
| `camera` | `str` | -- | Camera name. |
| `frame_times` | `pd.DataFrame \| None` | `None` | Frame-time table for timestamp attachment. If `None`, attempts to load camera timestamps from the video cache. |
| `output_path` | `Path \| None` | `None` | Output Parquet path. Defaults to `outputs/pose/session_{id}_pose_predictions.parquet`. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Path to the written Parquet file. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | The CSV file does not exist. |
| `ValueError` | The CSV has no `frame_idx` or `frame` column. |

!!! note "Column normalization"
    Dots in column names (e.g., `nose.x`) are replaced with underscores
    (`nose_x`) to match the convention used by other modules.

**Example:**

```python
from features_pose import export_pose_predictions_from_sleap_csv

parquet_path = export_pose_predictions_from_sleap_csv(
    csv_path=Path("outputs/labeling/predictions.csv"),
    session_id=1064644573,
    camera="side",
)
print(f"Pose predictions saved to {parquet_path}")
```

---

### `_find_keypoints`

```python
def _find_keypoints(df: pd.DataFrame) -> list[str]
```

Discover keypoint names from DataFrame columns. A keypoint is identified by
the presence of both `{name}_x` and `{name}_y` columns.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | DataFrame with keypoint coordinate columns. |

**Returns:**

| Type | Description |
|------|-------------|
| `list[str]` | List of keypoint base names (e.g., `["nose", "left_ear", "tail_base"]`). |

!!! note "Internal function"
    This is a private helper prefixed with `_`, but it is documented here
    because it is useful for understanding how keypoints are auto-discovered
    across the codebase.

**Example:**

```python
from features_pose import _find_keypoints
import pandas as pd

df = pd.DataFrame({
    "nose_x": [1], "nose_y": [2], "nose_score": [0.9],
    "tail_x": [3], "tail_y": [4],
    "orphan_x": [5],  # no matching _y -> not a keypoint
})
print(_find_keypoints(df))  # ['nose', 'tail']
```
