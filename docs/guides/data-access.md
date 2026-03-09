# Data Access Guide

This guide explains how the Allen Institute Visual Behavior Neuropixels (VBN) dataset is organized, how the pipeline accesses it, and how to configure data access for different scenarios (full SDK download, manual local files, limited disk space, and development mock mode).

---

## Dataset Overview

The Visual Behavior Neuropixels dataset contains recordings from mice performing a visual change-detection task. Each **session** includes:

| Data Type | Format | Typical Size | Source |
|---|---|---|---|
| Neural data (spikes, units, LFP) | NWB / HDF5 | 1--5 GB | AllenSDK or direct NWB |
| Trial/behavior events | Inside NWB | -- | Extracted from NWB |
| Eye tracking | Inside NWB | -- | Extracted from NWB |
| Behavior videos | `.mp4` + `.npy` timestamps | 200 MB -- 2 GB per camera | S3 bucket |
| Video metadata | `.json` | < 1 KB | S3 bucket |

Each session has up to **three cameras**:

- **`eye`** -- close-up of the eye (for pupil tracking)
- **`face`** -- front-facing view of the mouse's face
- **`side`** -- lateral view of the full body (best for pose estimation)

---

## Access Modes

The pipeline supports two access modes, controlled by the `ACCESS_MODE` environment variable or Notebook 00 configuration.

### SDK Mode (Default)

```python
ACCESS_MODE = "sdk"
```

Uses the AllenSDK `VisualBehaviorNeuropixelsProjectCache` to:

1. Download and cache NWB files locally
2. Resolve session IDs to local file paths
3. Provide a session table with metadata

The SDK caches data to `data/allensdk_cache/` by default.

```python
from io_nwb import resolve_nwb_path

nwb_path = resolve_nwb_path(
    session_id=1098595957,
    access_mode="sdk",
    nwb_path_override=None,  # SDK will download/cache
)
```

!!! note "AllenSDK Installation"
    The SDK must be installed separately. If it is not available, the pipeline falls back to the manual path from `sessions.csv`.

    ```bash
    pip install allensdk
    ```

### Manual Mode

```python
ACCESS_MODE = "manual"
```

Uses paths directly from `sessions.csv` without any SDK calls or automatic downloads. This is useful when:

- You have NWB files on a shared filesystem
- You downloaded files manually
- The AllenSDK is not available in your environment

```csv
session_id,nwb_path,video_dir,notes
1098595957,/data/vbn/session_1098595957.nwb,/data/vbn/videos/1098595957,first session
1099234091,/data/vbn/session_1099234091.nwb,/data/vbn/videos/1099234091,
```

---

## Session Discovery

### sessions.csv

The pipeline's session inventory lives in `sessions.csv` at the project root. Required columns:

| Column | Type | Description |
|---|---|---|
| `session_id` | int | Allen Institute session ID (10-digit number) |
| `nwb_path` | str | Path to the NWB file (for manual mode) |
| `video_dir` | str | Directory containing video files (for manual mode) |
| `notes` | str | Free-text notes |

```python
from io_sessions import load_sessions_csv

sessions = load_sessions_csv()
print(sessions)
```

### Auto-Generation from sessions.txt

If `sessions.csv` does not exist but `sessions.txt` does (one session ID per line), the pipeline generates a template CSV automatically:

```python
# sessions.txt content:
# 1098595957
# 1099234091
# 1100123456

sessions = load_sessions_csv()  # creates sessions.csv from sessions.txt
```

### SessionBundle

Each session is wrapped in a `SessionBundle` object that provides lazy-loading methods for all data types:

```python
from io_sessions import get_session_bundle

bundle = get_session_bundle(
    session_id=1098595957,
    resolve_nwb=True,           # resolve NWB path via SDK or sessions.csv
    inspect_modalities=True,    # check which data types are present
)

print(f"NWB path: {bundle.nwb_path}")
print(f"Modalities: {bundle.modalities_present}")
# {'spikes': True, 'trials': True, 'eye': True, 'behavior': True, 'stimulus': False}
```

Loading data through the bundle caches results to disk automatically:

```python
# First call: extracts from NWB and saves to outputs/neural/
units, spikes = bundle.load_spikes()

# Second call: loads from cached parquet/npz (fast)
units, spikes = bundle.load_spikes()
```

Available bundle methods:

| Method | Returns | Cached To |
|---|---|---|
| `load_spikes()` | `(units_df, spike_times_dict)` | `outputs/neural/` |
| `load_trials_and_events()` | `(trials_df, events_df)` | `outputs/behavior/` |
| `load_eye_features()` | `eye_features_df` | `outputs/eye/` |
| `load_video_assets()` | `video_assets_df` | `outputs/video/` |
| `load_frame_times(camera)` | `frame_times_df` | `outputs/video/` |

---

## S3 Video Downloads

### How It Works

Behavior videos are stored in a **public S3 bucket** (`allen-brain-observatory`) and are downloaded with unsigned (anonymous) access. No AWS credentials are needed.

The download strategy is controlled by `VIDEO_SOURCE`:

| Value | Behavior |
|---|---|
| `auto` (default) | Download from S3 if local files are not found |
| `s3` | Always attempt S3 download |
| `local` | Never download; use only local files |

### S3 Bucket Structure

```
s3://allen-brain-observatory/
  visual-behavior-neuropixels/
    raw-data/
      {session_id}/
        behavior_videos/
          eye.mp4
          eye_timestamps.npy
          eye_metadata.json
          face.mp4
          face_timestamps.npy
          face_metadata.json
          side.mp4
          side_timestamps.npy
          side_metadata.json
```

### Download Size Estimates

| Camera | Typical File Size | Notes |
|---|---|---|
| `eye.mp4` | 100--400 MB | Close-up, lower resolution |
| `face.mp4` | 200--600 MB | Medium resolution |
| `side.mp4` | 300--800 MB | Full body, highest resolution |
| `*_timestamps.npy` | 1--5 MB | NumPy array of timestamps |
| `*_metadata.json` | < 1 KB | Camera metadata |

**Total per session**: approximately **0.6--1.8 GB** for all three cameras.

### Configuration

```python
# Environment variables (set before running notebooks)
VIDEO_SOURCE = "auto"                           # auto, s3, or local
VIDEO_CACHE_DIR = "data/raw/visual-behavior-neuropixels"  # where to save downloads
VIDEO_BUCKET = "allen-brain-observatory"         # S3 bucket name
VIDEO_BASE_PATH = "visual-behavior-neuropixels/raw-data"  # S3 prefix
VIDEO_CAMERAS = "eye,face,side"                  # which cameras to download
```

### Programmatic Download

```python
from io_s3 import download_asset, s3_uri

# Download a single video
uri = s3_uri(session_id=1098595957, camera="side", kind="video")
local_path = download_asset(uri, Path("data/raw/.../side.mp4"))

# Download timestamps
ts_uri = s3_uri(session_id=1098595957, camera="side", kind="timestamps")
ts_path = download_asset(ts_uri, Path("data/raw/.../side_timestamps.npy"))
```

### Video Asset Registry

After downloading, the pipeline maintains a flat parquet registry:

```python
from io_video import load_video_assets, load_frame_times

# All video assets across all sessions
assets = load_video_assets()
print(assets[["session_id", "camera", "source", "n_frames", "fps_est"]])

# Frame times for a specific session/camera
ft = load_frame_times(session_id=1098595957, camera="side")
print(f"Frames: {len(ft)}, t range: [{ft['t'].min():.1f}, {ft['t'].max():.1f}] s")
```

---

## Working with Limited Disk Space

### Strategy 1: Process One Camera at a Time

Set `VIDEO_CAMERAS` to only the camera you need (usually `side` for pose estimation):

```bash
export VIDEO_CAMERAS="side"
```

This reduces download size by ~60%.

### Strategy 2: Disable Video Downloads

```bash
export VIDEO_SOURCE="local"
```

The pipeline will skip video-dependent notebooks (05--07) but neural and behavioral analyses (02--04, partial 08) will still work.

### Strategy 3: Selective Session Processing

Only process sessions you care about. In Notebook 09, set:

```python
SESSION_IDS = [1098595957, 1099234091]  # only these two
```

### Strategy 4: Clean Up After Processing

Once pose predictions are saved as parquet, you can delete the raw video files:

```bash
# Keep predictions, remove raw video
rm data/raw/visual-behavior-neuropixels/1098595957/behavior_videos/*.mp4
```

The pipeline will detect existing parquet predictions and skip video processing.

---

## NWB File Structure

### What Is Inside an NWB File?

NWB (Neurodata Without Borders) is an HDF5-based format. The VBN NWB files contain:

```
NWBFile
  units/                    # Spike-sorted units
    unit_id                 # Unit identifiers
    spike_times             # Array of spike times per unit (seconds)
    ...quality metrics...
  trials/                   # Trial information
    start_time, stop_time   # Trial boundaries (seconds)
    trial_type              # go, no-go, catch, etc.
    rewarded                # Whether the mouse was rewarded
    response                # Mouse response type
  processing/
    behavior/               # Behavioral time series
      lick_times            # Lick timestamps
      running_speed         # Running wheel data
    eye_tracking/           # Eye tracking data
      pupil_area            # Pupil size over time
      eye_position          # Gaze direction
  stimulus/                 # Stimulus presentation info
```

### Inspecting an NWB File

```python
from io_nwb import open_nwb_handle, inspect_modalities
from pathlib import Path

with open_nwb_handle(Path("path/to/session.nwb")) as nwb:
    modalities = inspect_modalities(nwb)
    print(modalities)
    # {'spikes': True, 'trials': True, 'eye': True, 'behavior': True, 'stimulus': False}
```

### Extracting Data Programmatically

```python
from io_nwb import (
    extract_units_and_spikes,
    extract_trials,
    extract_behavior_events,
    extract_eye_tracking,
    open_nwb_handle,
)

with open_nwb_handle(Path("path/to/session.nwb")) as nwb:
    # Neural data
    units_df, spike_times = extract_units_and_spikes(nwb)
    # units_df: DataFrame with unit metadata
    # spike_times: dict of {unit_id: np.array of spike times in seconds}

    # Trial data
    trials = extract_trials(nwb)
    # DataFrame with t_start, t_end, trial_type, rewarded, etc.

    # Behavioral events
    events = extract_behavior_events(nwb)
    # DataFrame with t, lick_times, running_speed, etc.

    # Eye tracking
    eye = extract_eye_tracking(nwb)
    # DataFrame with t, pupil_area, eye_position, etc.
```

!!! tip "All timestamps are in NWB seconds"
    Every timestamp extracted from the NWB file is already in the canonical `nwb_seconds` timebase. No conversion is needed. See the [Timebase Reference](../reference/timebase.md) for details.

---

## Mock Mode

For development and testing without real data, enable mock mode:

```bash
export MOCK_MODE=true
```

Or in Notebook 00:

```python
MOCK_MODE = True
```

### What Mock Mode Does

- `open_nwb_handle()` returns a synthetic NWB-like object with:
    - 3 mock units with short spike trains
    - 2 mock trials (go and no-go)
    - No eye tracking or behavior processing modules
- Video-related functions return empty DataFrames (no download attempts)
- All notebooks can run end-to-end without any real data

### When to Use Mock Mode

- Testing notebook execution in CI/CD
- Developing new analysis code without waiting for data downloads
- Verifying that output formats and schemas are correct

!!! info "Mock NWB Object Structure"
    The mock object provides:

    ```python
    mock.units     # DataFrame with unit_id, spike_times (3 units)
    mock.trials    # DataFrame with start_time, stop_time, trial_type (2 trials)
    mock.processing  # Empty dict (no eye/behavior data)
    mock.stimulus    # None
    ```

---

## Accessing Data Programmatically (Summary)

```python
from config import get_config
from io_sessions import load_sessions_csv, get_session_bundle
from io_video import load_video_assets, load_frame_times
from io_nwb import open_nwb_handle

# 1. Get configuration
cfg = get_config()
print(f"Access mode: {cfg.access_mode}")
print(f"Video source: {cfg.video_source}")
print(f"Outputs dir: {cfg.outputs_dir}")

# 2. List sessions
sessions = load_sessions_csv()
print(f"Sessions: {sessions['session_id'].tolist()}")

# 3. Get a session bundle
bundle = get_session_bundle(session_id=1098595957)

# 4. Load neural data (extracts from NWB if not cached)
units, spikes = bundle.load_spikes()

# 5. Load behavior data
trials, events = bundle.load_trials_and_events()

# 6. Load video assets (downloads from S3 if needed)
video_assets = bundle.load_video_assets()

# 7. Load eye features
eye = bundle.load_eye_features()

# 8. Load frame times for a specific camera
ft = bundle.load_frame_times(camera="side")
```
