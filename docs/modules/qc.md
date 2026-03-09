# qc

Quality-control utilities for timestamp validation, frame-drop detection,
FPS estimation, and eye-tracking data summary. These functions are used
throughout the pipeline to flag potential data issues early.

**Source:** `src/qc.py`

---

## Functions

### `check_monotonic`

```python
def check_monotonic(times: np.ndarray) -> bool
```

Check whether a timestamp array is strictly monotonically increasing.
Non-monotonic timestamps indicate clock resets, duplicate frames, or
data corruption.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `times` | `np.ndarray` | -- | 1-D array of timestamps. |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | `True` if every consecutive pair satisfies `t[i+1] > t[i]`, `False` otherwise. |

!!! warning
    An array of length 0 or 1 returns `True` (vacuously monotonic). This
    function does not check for NaN values -- ensure timestamps are finite
    before calling.

**Example:**

```python
from qc import check_monotonic
import numpy as np

good = np.array([0.0, 0.033, 0.066, 0.100])
bad = np.array([0.0, 0.033, 0.020, 0.066])  # time goes backward

print(check_monotonic(good))  # True
print(check_monotonic(bad))   # False
```

---

### `detect_dropped_frames`

```python
def detect_dropped_frames(
    times: np.ndarray,
    threshold_factor: float = 2.5,
) -> int
```

Detect the number of frame-to-frame intervals that exceed a threshold,
indicating dropped or missing frames.

The threshold is `threshold_factor * median(diff(times))`. An interval
larger than this is counted as a "drop event" (which may represent one or
more missing frames).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `times` | `np.ndarray` | -- | 1-D array of timestamps. |
| `threshold_factor` | `float` | `2.5` | Multiplier on the median inter-frame interval. Higher values are more lenient. |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of intervals exceeding the threshold. Returns `0` if fewer than 3 timestamps are provided. |

!!! note
    This counts the number of **gap events**, not the number of missing frames.
    A single gap event at 10x the median interval could represent ~9 missing
    frames. Use the video alignment plot (`viz.plot_video_alignment`) for a
    detailed per-gap breakdown.

**Example:**

```python
from qc import detect_dropped_frames
import numpy as np

# Normal 30 fps with one dropped frame
dt = 1.0 / 30.0
times = np.array([0, dt, 2*dt, 3*dt, 6*dt, 7*dt])  # gap at index 3->4
n_drops = detect_dropped_frames(times)
print(n_drops)  # 1
```

---

### `estimate_fps`

```python
def estimate_fps(times: np.ndarray) -> float | None
```

Estimate the frame rate from timestamps using the median inter-frame
interval. More robust than using the mean, which is skewed by dropped frames.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `times` | `np.ndarray` | -- | 1-D array of timestamps. |

**Returns:**

| Type | Description |
|------|-------------|
| `float \| None` | Estimated FPS (frames per second), or `None` if fewer than 2 timestamps are provided or the median interval is zero or negative. |

**Example:**

```python
from qc import estimate_fps
import numpy as np

times = np.arange(0, 10, 1.0/30.0)  # 10 seconds at 30 fps
fps = estimate_fps(times)
print(f"Estimated FPS: {fps:.1f}")  # ~30.0
```

---

### `compute_video_qc`

```python
def compute_video_qc(
    frame_times: pd.DataFrame,
    fps_nominal: float | None,
) -> Dict[str, Any]
```

Compute a comprehensive QC report for video frame timestamps. Combines
monotonicity check, drop detection, FPS estimation, and clock drift
measurement.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_times` | `pd.DataFrame` | -- | DataFrame with a `t` column containing frame timestamps. |
| `fps_nominal` | `float \| None` | -- | Expected/nominal frame rate. Used to compute drift. If `None`, drift is not computed. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | QC report dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `monotonic` | `bool` | Whether timestamps are strictly increasing. |
| `dropped_frames` | `int` | Number of detected frame-drop events. |
| `fps_nominal` | `float \| None` | The expected frame rate (pass-through). |
| `fps_estimated` | `float \| None` | Estimated frame rate from timestamps. |
| `drift_fraction` | `float \| None` | Fractional drift: `(fps_estimated - fps_nominal) / fps_nominal`. Positive means the camera ran faster than nominal. `None` if either FPS is unavailable. |

**Example:**

```python
from qc import compute_video_qc
import pandas as pd
import numpy as np

ft = pd.DataFrame({"t": np.arange(0, 10, 1.0/30.0)})
qc = compute_video_qc(ft, fps_nominal=30.0)
print(qc)
# {
#     'monotonic': True,
#     'dropped_frames': 0,
#     'fps_nominal': 30.0,
#     'fps_estimated': 30.0,
#     'drift_fraction': 0.0,
# }

# With a drifting clock:
ft_drift = pd.DataFrame({"t": np.arange(0, 10, 1.0/29.95)})
qc = compute_video_qc(ft_drift, fps_nominal=30.0)
print(f"Drift: {qc['drift_fraction']:.6f}")
# Drift: -0.001667  (camera is 0.17% slower than nominal)
```

---

### `eye_qc_summary`

```python
def eye_qc_summary(
    eye_df: pd.DataFrame,
) -> Dict[str, Any]
```

Compute a quick QC summary for eye-tracking data.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eye_df` | `pd.DataFrame` | -- | Eye-tracking feature DataFrame. May be `None` or empty. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Summary dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `available` | `bool` | `True` if data is present and non-empty. |
| `n_samples` | `int` | Number of rows (only if `available`). |
| `missing_fraction` | `float` | Mean fraction of NaN values across all columns (only if `available`). |

**Example:**

```python
from qc import eye_qc_summary

summary = eye_qc_summary(eye_features)
if summary["available"]:
    print(f"Eye data: {summary['n_samples']} samples, "
          f"{summary['missing_fraction']*100:.1f}% missing")
else:
    print("Eye tracking data not available")
```
