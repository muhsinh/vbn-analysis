# Timebase Contract

This page defines the canonical timebase used throughout the VBN Analysis Suite, explains why it matters, and describes how every data source is aligned to it.

---

## What Is `nwb_seconds`?

**`nwb_seconds`** is the canonical timebase for all artifacts produced by this pipeline. It means:

> The `t` column in any artifact contains timestamps in **seconds**, measured from the **same clock** as the NWB file's internal timestamps (i.e., the Neuropixels recording system clock).

This is the clock that the Allen Institute hardware synchronization system uses. All neural data (spike times, LFP), behavioral events (trial times, lick times), and eye tracking data within the NWB file are already on this clock.

The challenge is aligning **external** data sources -- primarily video frame timestamps -- to this same clock.

---

## Why a Canonical Timebase Matters

Without a shared timebase, you cannot meaningfully compare neural activity to behavior:

| Scenario | Problem |
|---|---|
| Neural spike at t=100.0 s, video frame at frame 3000 | Are these the same moment? Only if you know the mapping. |
| Pupil size measured at "sample 45000" | What time is that in NWB seconds? |
| Pose speed computed from frame differences | What is the actual velocity in real seconds? |

The pipeline guarantees that **every artifact with a `t` column uses `nwb_seconds`**. This means you can merge any two artifacts on `t` and they will be temporally aligned.

!!! danger "Breaking the Timebase Contract"
    If you write a custom artifact with a `t` column that is NOT in NWB seconds (e.g., video frame indices, or seconds-since-video-start), downstream analyses will silently produce **wrong results**. Always use `write_parquet_with_timebase()` to write artifacts.

---

## How Each Data Source Gets Aligned

### Neural Data (Spikes, Units)

**Alignment method**: `"nwb"` -- already in NWB seconds.

Spike times are extracted directly from the NWB file's `units` table. These timestamps are already on the Neuropixels clock and require no conversion.

```python
from io_nwb import extract_units_and_spikes

with open_nwb_handle(nwb_path) as nwb:
    units_df, spike_times = extract_units_and_spikes(nwb)
    # spike_times["unit_42"] -> array([0.1, 0.5, 1.0, ...]) in NWB seconds
```

### Trials and Behavioral Events

**Alignment method**: `"nwb"` -- already in NWB seconds.

Trial start/stop times and behavioral event timestamps (licks, running speed) are extracted from the NWB file and are already on the same clock.

```python
from io_nwb import extract_trials

with open_nwb_handle(nwb_path) as nwb:
    trials = extract_trials(nwb)
    # trials["t_start"] -> NWB seconds
    # trials["t"] = trials["t_start"] (alias)
```

### Eye Tracking

**Alignment method**: `"nwb"` -- already in NWB seconds.

Eye tracking data lives in the NWB `processing/eye_tracking` module and uses the NWB timestamp array.

```python
from io_nwb import extract_eye_tracking

with open_nwb_handle(nwb_path) as nwb:
    eye = extract_eye_tracking(nwb)
    # eye["t"] -> NWB seconds
```

### Video Frame Timestamps

**Alignment method**: `"timestamps"` -- from `.npy` timestamp files.

Video timestamps come from `.npy` files alongside the video (e.g., `side_timestamps.npy`). These files contain one timestamp per frame, already in NWB seconds (the Allen Institute synchronization hardware records camera frame times on the Neuropixels clock).

```python
from io_video import load_timestamps, load_frame_times

# Raw timestamps
ts = load_timestamps(Path("data/raw/.../side_timestamps.npy"))
# ts[0] -> first frame time in NWB seconds
# ts[-1] -> last frame time in NWB seconds

# Structured frame times table
ft = load_frame_times(session_id=1098595957, camera="side")
# ft["t"] -> NWB seconds
# ft["frame_idx"] -> integer frame number in the video
```

### Pose Predictions

**Alignment method**: `"sleap_inference"` or `"nwb"` (depending on source).

Pose predictions get their timestamps by mapping `frame_idx` to `t` using the frame times table:

```
pose_prediction.frame_idx -> frame_times[frame_idx] -> t (NWB seconds)
```

This happens automatically during `slp_to_parquet()` or `export_pose_predictions_from_sleap_csv()`.

```python
from pose_inference import slp_to_parquet

n = slp_to_parquet(
    slp_path=Path("predictions.slp"),
    session_id=1098595957,
    camera="side",
)
# Output parquet has t column in NWB seconds
```

If frame times are unavailable, the pipeline attempts to load camera timestamps directly from `.npy` files. If neither source is available, `t` is set to NaN.

!!! warning "NaN Timestamps in Pose Data"
    If your pose predictions have NaN in the `t` column, it means the frame-to-time mapping failed. Check that:

    1. `outputs/video/frame_times.parquet` exists and contains entries for your session/camera
    2. The `.npy` timestamp file exists in the video cache directory
    3. The `frame_idx` values in your predictions match the frame indices in the timestamp file

### Pose Features

**Alignment method**: Inherited from pose predictions.

Pose features (`pose_speed`, `head_angle`, etc.) are derived from pose predictions and inherit the same `t` column.

### Motifs

**Alignment method**: Inherited from pose features.

Motif assignments carry the `t` column from the pose features they were computed from.

### Fusion Table

**Alignment method**: `build_time_grid()` -- constructed from NWB-second boundaries.

The fusion table is built by:

1. Creating a regular time grid in NWB seconds using `build_time_grid(t_start, t_end, bin_size)`
2. Binning spike times into this grid
3. Binning continuous behavioral features into the same grid
4. Merging everything on the `t` column

```python
from timebase import build_time_grid, bin_spike_times, bin_continuous_features

time_grid = build_time_grid(0.0, 3600.0, bin_size_s=0.025)
# time_grid[0] = 0.0, time_grid[1] = 0.025, ...

spike_counts = bin_spike_times(spike_times, time_grid, bin_size_s=0.025)
# spike_counts["t"] = time_grid (NWB seconds)

behavior_binned = bin_continuous_features(pose_features, time_grid)
# behavior_binned["t"] = time_grid (NWB seconds)
```

---

## The Provenance Metadata Structure

Every artifact's sidecar `.meta.json` contains provenance information:

```json
{
  "timebase": "nwb_seconds",
  "provenance": {
    "session_id": 1098595957,
    "code_version": "9e576ad",
    "created_at": "2026-03-09T12:34:56.789000+00:00",
    "alignment_method": "nwb"
  }
}
```

| Field | Values | Meaning |
|---|---|---|
| `timebase` | `"nwb_seconds"` | Confirms the `t` column uses the canonical timebase |
| `alignment_method` | `"nwb"` | Data was extracted directly from NWB (already aligned) |
| | `"timestamps"` | Aligned via camera timestamp `.npy` files |
| | `"sleap_inference"` | Timestamps attached during SLEAP prediction conversion |
| `code_version` | git hash | Which code version produced this artifact |
| `created_at` | ISO 8601 | When the artifact was written |
| `session_id` | int or null | Which session (null for cross-session artifacts like the registry) |

### Creating Provenance

```python
from config import make_provenance

prov = make_provenance(session_id=1098595957, alignment_method="nwb")
# {
#     "session_id": 1098595957,
#     "code_version": "9e576ad...",
#     "created_at": "2026-03-09T...",
#     "alignment_method": "nwb"
# }
```

---

## How to Verify Timebase Alignment

### Check 1: Sidecar Metadata

```python
import json
from pathlib import Path

sidecar = Path("outputs/pose/session_1098595957_pose_predictions.parquet.meta.json")
meta = json.loads(sidecar.read_text())

assert meta["timebase"] == "nwb_seconds", f"Wrong timebase: {meta['timebase']}"
print(f"Alignment method: {meta['provenance']['alignment_method']}")
```

### Check 2: Time Range Overlap

Two artifacts for the same session should have overlapping time ranges:

```python
import pandas as pd

spikes_unit0 = spikes["0"]
pose = pd.read_parquet("outputs/pose/session_1098595957_pose_predictions.parquet")

spike_range = (spikes_unit0.min(), spikes_unit0.max())
pose_range = (pose["t"].min(), pose["t"].max())

print(f"Spikes: {spike_range[0]:.1f} -- {spike_range[1]:.1f} s")
print(f"Pose:   {pose_range[0]:.1f} -- {pose_range[1]:.1f} s")

overlap = max(0, min(spike_range[1], pose_range[1]) - max(spike_range[0], pose_range[0]))
print(f"Overlap: {overlap:.1f} s")

if overlap < 10:
    print("WARNING: Very little temporal overlap. Check alignment!")
```

### Check 3: Validate with the Reports Module

```python
from reports import validate_artifact_schema
from pathlib import Path

artifacts = [
    ("outputs/neural/session_1098595957_units.parquet", ["unit_id"]),
    ("outputs/behavior/session_1098595957_trials.parquet", ["t"]),
    ("outputs/video/frame_times.parquet", ["session_id", "camera", "frame_idx", "t"]),
    ("outputs/pose/session_1098595957_pose_predictions.parquet", ["frame_idx", "t"]),
]

for path_str, required_cols in artifacts:
    path = Path(path_str)
    valid = validate_artifact_schema(path, required_cols)
    status = "OK" if valid else "INVALID"
    print(f"  [{status}] {path.name}")
```

### Check 4: Frame Time Monotonicity

Video frame timestamps must be strictly increasing:

```python
from qc import check_monotonic, detect_dropped_frames
import pandas as pd

ft = pd.read_parquet("outputs/video/frame_times.parquet")
ft_session = ft[ft["session_id"] == 1098595957]

for camera, group in ft_session.groupby("camera"):
    times = group.sort_values("frame_idx")["t"].to_numpy()
    mono = check_monotonic(times)
    drops = detect_dropped_frames(times)
    print(f"  {camera}: monotonic={mono}, dropped_frames={drops}")
```

---

## What to Do If Timestamps Do Not Match

### Symptom: Pose Times Are All NaN

**Cause**: Frame times were not available when pose predictions were converted.

**Fix**:

1. Run Notebook 05 to generate `frame_times.parquet`
2. Re-run the pose conversion (Notebook 07 or `slp_to_parquet()`)

### Symptom: Neural and Behavior Time Ranges Do Not Overlap

**Cause**: The NWB file and video timestamps use different clocks (this should not happen with properly synchronized Allen Institute data).

**Fix**:

1. Check that you are using the correct NWB file for the session
2. Verify the session ID matches between the NWB and video files
3. Check the `.npy` timestamp file is for the correct session

### Symptom: Cross-Correlation Peak Is at a Large Lag

**Cause**: If the peak lag is much larger than expected (e.g., > 5 seconds), the timestamps may be offset.

**Fix**: Check for a constant offset between data sources:

```python
import numpy as np

# Compare first timestamps
neural_t0 = min(times.min() for times in spike_times.values())
video_t0 = float(frame_times["t"].min())
offset = neural_t0 - video_t0
print(f"Offset: {offset:.3f} s")

# If offset is large, investigate the timestamp source
```

---

## Writing Timebase-Annotated Artifacts

Always use the provided helper functions to write artifacts. These ensure the timebase metadata is written correctly.

### Parquet Files

```python
from timebase import write_parquet_with_timebase
from config import make_provenance

write_parquet_with_timebase(
    df=my_dataframe,
    path=Path("outputs/my_step/my_artifact.parquet"),
    timebase="nwb_seconds",       # always use this value
    provenance=make_provenance(session_id=1098595957, alignment_method="nwb"),
    required_columns=["t"],       # validate before writing
)
```

This function:

1. Validates required columns exist
2. Writes the parquet file with embedded schema metadata
3. Writes a companion `.meta.json` sidecar

### NPZ Files

```python
from timebase import write_npz_with_provenance
from config import make_provenance

write_npz_with_provenance(
    data={"unit_0": spike_times_array, "unit_1": another_array},
    path=Path("outputs/neural/session_1098595957_spike_times.npz"),
    provenance=make_provenance(session_id=1098595957, alignment_method="nwb"),
)
```

### Time Grid Construction

When you need a regular time grid for binning:

```python
from timebase import build_time_grid

grid = build_time_grid(start=100.0, end=3700.0, bin_size_s=0.025)
# array([100.0, 100.025, 100.05, ...])
# All values are in NWB seconds
```

### Binning Spike Times

```python
from timebase import bin_spike_times

counts_df = bin_spike_times(
    spike_times={"u0": np.array([100.1, 100.5]), "u1": np.array([100.2])},
    time_grid=grid,
    bin_size_s=0.025,
)
# DataFrame with columns: t, u0, u1
# t is the grid (NWB seconds), u0/u1 are spike counts per bin
```

### Binning Continuous Features

```python
from timebase import bin_continuous_features

binned = bin_continuous_features(
    df=pose_features,    # must have 't' column in NWB seconds
    time_grid=grid,
    agg="mean",          # aggregation function
)
# DataFrame with columns: t, pose_speed, head_angle, ...
# t is the grid (NWB seconds), features are aggregated per bin
```

---

## Constants

The canonical timebase string is defined in `src/timebase.py`:

```python
CANONICAL_TIMEBASE = "nwb_seconds"
```

All validation code compares against this constant. Do not hardcode the string `"nwb_seconds"` elsewhere -- import and use the constant:

```python
from timebase import CANONICAL_TIMEBASE

assert meta["timebase"] == CANONICAL_TIMEBASE
```
