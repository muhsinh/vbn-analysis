# io_video

Video asset discovery, S3 downloading, frame-time alignment, QC computation,
and preview clip generation. Implements a download-first strategy so that
video files are available locally before any analysis begins.

**Source:** `src/io_video.py`

---

## Functions

### `build_video_assets`

```python
def build_video_assets(
    session_id: int,
    video_dir: Path | None = None,
    outputs_dir: Path | None = None,
    download_missing: bool | None = None,
) -> pd.DataFrame
```

Discover, download, and catalog all video assets for a session. This is the
main entry point for the video pipeline. It scans local directories, fetches
missing files from S3 (if configured), computes frame-time metrics and QC
flags, and upserts everything into persistent Parquet catalogs.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int` | -- | Session ID to process. |
| `video_dir` | `Path \| None` | `None` | Custom directory containing video files. If `None`, searches the default `video_cache_dir`. |
| `outputs_dir` | `Path \| None` | `None` | Where to write output catalogs. Defaults to `outputs/video/`. |
| `download_missing` | `bool \| None` | `None` | Whether to download missing files from S3. If `None`, auto-detects from `config.video_source` (downloads when `"auto"` or `"s3"`). |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Asset catalog with one row per camera. |

**Output columns:**

| Column | Description |
|--------|-------------|
| `session_id` | Session identifier. |
| `camera` | Camera name (`"eye"`, `"face"`, `"side"`). |
| `source` | `"local"` or `"s3"`. |
| `s3_uri_video` | S3 URI for the video file. |
| `s3_uri_timestamps` | S3 URI for the timestamp file. |
| `s3_uri_metadata` | S3 URI for the metadata file. |
| `http_url_video` | HTTP URL for the video file. |
| `local_video_path` | Local path to the downloaded/found video. |
| `local_timestamps_path` | Local path to the timestamp file. |
| `local_metadata_path` | Local path to the metadata file. |
| `n_frames` | Number of valid frames. |
| `fps_est` | Estimated frames per second. |
| `t0` | Timestamp of the first frame. |
| `tN` | Timestamp of the last frame. |
| `qc_flags` | Pipe-separated QC flag string. |

**Side effects:**

- Downloads video, timestamp, and metadata files from S3 if `download_missing` is true and files are not found locally.
- Writes/updates `outputs/video/video_assets.parquet`.
- Writes/updates `outputs/video/frame_times.parquet`.

**Example:**

```python
from io_video import build_video_assets

assets = build_video_assets(session_id=1064644573)
print(assets[["camera", "n_frames", "fps_est", "qc_flags"]])
#   camera  n_frames     fps_est       qc_flags
# 0    eye    612345   29.997001
# 1   face    612301   29.996998
# 2   side    612340   29.997002  DROPPED_FRAMES
```

---

### `load_video_assets`

```python
def load_video_assets(
    session_id: int | None = None,
    camera: str | None = None,
) -> pd.DataFrame
```

Load the video asset catalog from disk, optionally filtered by session
and/or camera.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int \| None` | `None` | Filter to this session. If `None`, returns all sessions. |
| `camera` | `str \| None` | `None` | Filter to this camera. If `None`, returns all cameras. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Asset catalog rows matching the filters. Empty DataFrame if the catalog file does not exist. |

**Example:**

```python
from io_video import load_video_assets

# All assets for one session
assets = load_video_assets(session_id=1064644573)

# Just the side camera
side = load_video_assets(session_id=1064644573, camera="side")
print(side["local_video_path"].iloc[0])
```

---

### `load_frame_times`

```python
def load_frame_times(
    session_id: int | None = None,
    camera: str | None = None,
) -> pd.DataFrame
```

Load per-frame timestamps from the frame-times catalog.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int \| None` | `None` | Filter to this session. |
| `camera` | `str \| None` | `None` | Filter to this camera. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with columns `session_id`, `camera`, `frame_idx`, `t`. Empty if the catalog does not exist. |

**Example:**

```python
from io_video import load_frame_times

ft = load_frame_times(session_id=1064644573, camera="eye")
print(ft.head())
#    session_id camera  frame_idx         t
# 0  1064644573    eye          0  0.003344
# 1  1064644573    eye          1  0.036678
# 2  1064644573    eye          2  0.070012
```

---

### `load_timestamps`

```python
def load_timestamps(path: Path) -> np.ndarray | None
```

Load raw timestamp data from a file. Supports `.npy`, `.npz`, `.csv`, and
`.tsv` formats.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path` | -- | Path to the timestamp file. |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray \| None` | 1-D array of timestamps, or `None` if the file does not exist or the format is unrecognized. |

**Format handling:**

| Extension | Behavior |
|-----------|----------|
| `.npy` | `np.load(path)` |
| `.npz` | Loads the first array in the archive. |
| `.csv` / `.tsv` | Reads with `pd.read_csv()`. Uses the `"t"` column if present, otherwise the first column. |

**Example:**

```python
from io_video import load_timestamps
from pathlib import Path

ts = load_timestamps(Path("data/eye_timestamps.npy"))
if ts is not None:
    print(f"{len(ts)} timestamps, range [{ts[0]:.3f}, {ts[-1]:.3f}]")
```

---

### `create_preview_clip`

```python
def create_preview_clip(
    video_path: Path,
    output_path: Path,
    max_seconds: int = 5,
) -> Path | None
```

Create a short preview clip from a video file. Tries `ffmpeg` first (fast,
stream-copy), then falls back to OpenCV frame-by-frame writing.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `Path` | -- | Source video file. |
| `output_path` | `Path` | -- | Where to write the preview clip. |
| `max_seconds` | `int` | `5` | Maximum duration of the preview in seconds. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path \| None` | Path to the created clip, or `None` if both ffmpeg and OpenCV fail. |

!!! note "Dependencies"
    Requires either `ffmpeg` on `PATH` or `opencv-python` (`cv2`). If neither
    is available, returns `None` without raising.

**Example:**

```python
from io_video import create_preview_clip
from pathlib import Path

clip = create_preview_clip(
    Path("data/side.mp4"),
    Path("outputs/previews/side_preview.mp4"),
    max_seconds=3,
)
if clip:
    print(f"Preview saved to {clip}")
```
