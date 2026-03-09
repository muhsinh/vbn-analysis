# Artifact Reference

Every file produced by the VBN Analysis Suite pipeline, with format details, column schemas, storage locations, and code for reading each artifact type.

---

## Artifact Map by Notebook

| Notebook | Artifact | Format | Location |
|---|---|---|---|
| 00 | Config snapshot | JSON | `outputs/reports/config_snapshot.json` |
| 01 | Session metadata | CSV | `sessions.csv` |
| 02 | Unit table | Parquet | `outputs/neural/session_{id}_units.parquet` |
| 02 | Spike times | NPZ | `outputs/neural/session_{id}_spike_times.npz` |
| 03 | Trials table | Parquet | `outputs/behavior/session_{id}_trials.parquet` |
| 03 | Behavior events | Parquet | `outputs/behavior/session_{id}_events.parquet` |
| 04 | Eye features | Parquet | `outputs/eye/session_{id}_eye_features.parquet` |
| 05 | Video asset registry | Parquet | `outputs/video/video_assets.parquet` |
| 05 | Frame times | Parquet | `outputs/video/frame_times.parquet` |
| 06 | Labeling frames | PNG + CSV | `outputs/labeling/sleap/{id}/{camera}/frames/` |
| 06 | Labeling video | MP4 + CSV | `outputs/labeling/sleap/{id}/{camera}/labeling.mp4` |
| 07 | Pose predictions | Parquet | `outputs/pose/session_{id}_pose_predictions.parquet` |
| 07 | SLEAP predictions | SLP | `outputs/pose/predictions/session_{id}_{camera}.predictions.slp` |
| 07 | Pose features | Parquet | `outputs/pose/session_{id}_pose_features.parquet` |
| 07 | Motifs | Parquet | `outputs/pose/session_{id}_motifs.parquet` |
| 08 | Fusion table | Parquet | `outputs/fusion/session_{id}_fusion.parquet` |
| 08 | Model metrics | JSON | `outputs/models/session_{id}_metrics.json` |
| 08 | Alignment report | JSON | `outputs/models/session_{id}_alignment.json` |
| 08 | Selectivity screen | Parquet | `outputs/models/session_{id}_selectivity.parquet` |
| 09 | Artifact registry | Parquet | `outputs/reports/artifact_registry.parquet` |
| 09 | Run summary | Parquet | `outputs/reports/run_summary.parquet` |
| 09 | QC checklist | JSON | `outputs/reports/qc_checklist.json` |
| -- | Session logs | Log | `outputs/reports/logs/session_{id}.log` |
| -- | Metadata sidecars | JSON | `*.meta.json` (next to each parquet/npz) |

---

## Parquet File Schemas

### Unit Table (`outputs/neural/session_{id}_units.parquet`)

| Column | Type | Description |
|---|---|---|
| `unit_id` | int/str | Unique unit identifier |
| `firing_rate` | float | Mean firing rate (Hz) |
| `snr` | float | Signal-to-noise ratio |
| `isi_violations` | float | ISI violation rate |
| `presence_ratio` | float | Fraction of session with spikes |
| *(additional quality metrics)* | float | Varies by NWB file |

!!! note "Column Variability"
    The exact columns depend on what the NWB file provides. `unit_id` is always present. Quality metrics vary by session.

### Spike Times (`outputs/neural/session_{id}_spike_times.npz`)

NPZ archive where each key is a unit ID and the value is a 1D float64 array of spike times in NWB seconds.

```python
from io_nwb import load_spike_times_npz
from pathlib import Path

spikes = load_spike_times_npz(Path("outputs/neural/session_1098595957_spike_times.npz"))
# spikes["0"] -> array([0.1, 0.5, 1.0, ...])
# spikes["1"] -> array([0.2, 0.7, 1.4, ...])
```

### Trials Table (`outputs/behavior/session_{id}_trials.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Trial start time (NWB seconds), alias for `t_start` |
| `t_start` | float | Trial start time (NWB seconds) |
| `t_end` | float | Trial end time (NWB seconds) |
| `trial_type` | str | go, no-go, catch, etc. |
| `rewarded` | bool | Whether the mouse received a reward |
| `response` | str | Mouse response type (hit, miss, false_alarm, correct_reject) |
| `stimulus_name` | str | Stimulus identifier |

### Behavior Events (`outputs/behavior/session_{id}_events.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Event timestamp (NWB seconds) |
| *(signal columns)* | float | Behavioral time series (lick_times, running_speed, etc.) |

The exact signal columns depend on what the NWB file's behavior processing module contains.

### Eye Features (`outputs/eye/session_{id}_eye_features.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Timestamp (NWB seconds) |
| `pupil` | float | Raw pupil area |
| `pupil_z` | float | Z-scored pupil area |
| `pupil_vel` | float | Pupil area velocity (derivative) |

### Video Asset Registry (`outputs/video/video_assets.parquet`)

| Column | Type | Description |
|---|---|---|
| `session_id` | int | Session identifier |
| `camera` | str | Camera name (eye, face, side) |
| `source` | str | "local" or "s3" |
| `s3_uri_video` | str | S3 URI for the video file |
| `s3_uri_timestamps` | str | S3 URI for timestamps |
| `s3_uri_metadata` | str | S3 URI for metadata |
| `http_url_video` | str | HTTPS URL for the video |
| `local_video_path` | str | Local path to downloaded video |
| `local_timestamps_path` | str | Local path to timestamps file |
| `local_metadata_path` | str | Local path to metadata file |
| `n_frames` | int | Number of valid frames |
| `fps_est` | float | Estimated FPS from timestamps |
| `t0` | float | First timestamp (NWB seconds) |
| `tN` | float | Last timestamp (NWB seconds) |
| `qc_flags` | str | Pipe-separated QC flags |

QC flag values:

| Flag | Meaning |
|---|---|
| `NO_TIMESTAMPS` | Timestamp file not found |
| `TIMESTAMP_NAN_PRESENT` | Some timestamps are NaN |
| `NO_VALID_TIMESTAMPS` | All timestamps are NaN |
| `NON_MONOTONIC` | Timestamps are not strictly increasing |
| `DROPPED_FRAMES` | Large gaps detected between frames |
| `DOWNLOAD_FAILED_VIDEO` | S3 download failed for video |
| `DOWNLOAD_FAILED_TIMESTAMPS` | S3 download failed for timestamps |
| `DOWNLOAD_FAILED_METADATA` | S3 download failed for metadata |

### Frame Times (`outputs/video/frame_times.parquet`)

| Column | Type | Description |
|---|---|---|
| `session_id` | int | Session identifier |
| `camera` | str | Camera name |
| `frame_idx` | int | Frame number in the video |
| `t` | float | Timestamp (NWB seconds) |

### Pose Predictions (`outputs/pose/session_{id}_pose_predictions.parquet`)

| Column | Type | Description |
|---|---|---|
| `session_id` | int | Session identifier |
| `camera` | str | Camera name |
| `frame_idx` | int | Frame number |
| `instance` | int | Instance index (0 for single-animal) |
| `t` | float | Timestamp (NWB seconds) |
| `{keypoint}_x` | float | X coordinate (pixels) |
| `{keypoint}_y` | float | Y coordinate (pixels) |
| `{keypoint}_score` | float | Confidence score (0-1) |
| `instance_score` | float | Overall instance confidence |

Keypoint names depend on the SLEAP skeleton definition. Common examples: `nose`, `left_ear`, `right_ear`, `neck`, `body_center`, `tail_base`, `tail_tip`.

### Pose Features (`outputs/pose/session_{id}_pose_features.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Timestamp (NWB seconds) |
| `pose_speed` | float | Mean speed across all keypoints (px/s) |
| `pose_speed_std` | float | Std dev of keypoint speeds |
| `{kp}_vel` | float | Per-keypoint velocity (px/s) |
| `{kp}_accel` | float | Per-keypoint acceleration (px/s^2) |
| `body_length` | float | Distance between first and last keypoint (px) |
| `head_angle` | float | Head direction angle (radians) |
| `head_angular_vel` | float | Head angular velocity (rad/s) |
| `dist_{kp1}_{kp2}` | float | Distance between adjacent keypoints (px) |
| `is_still` | int | 1 if pose_speed < 10th percentile threshold |

### Motifs (`outputs/pose/session_{id}_motifs.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Timestamp (NWB seconds) |
| `motif_id` | int | Cluster/state assignment |

### Fusion Table (`outputs/fusion/session_{id}_fusion.parquet`)

| Column | Type | Description |
|---|---|---|
| `t` | float | Time bin center (NWB seconds) |
| `{unit_id}` | int | Spike count per bin per unit |
| `motif_id` | int | Behavioral motif assignment |
| `pose_speed` | float | Binned pose speed |
| *(other behavioral features)* | float | Binned behavioral signals |

### Selectivity Screen (`outputs/models/session_{id}_selectivity.parquet`)

| Column | Type | Description |
|---|---|---|
| `unit_id` | str | Unit identifier |
| `d_prime` | float | Effect size (signed) |
| `abs_d_prime` | float | Absolute effect size |
| `rate_diff` | float | Mean rate difference (Hz) |
| `mean_rate_a` | float | Mean rate for condition A (Hz) |
| `mean_rate_b` | float | Mean rate for condition B (Hz) |
| `p_value` | float | Mann-Whitney U test p-value |
| `significant` | bool | p_value < threshold |

### Artifact Registry (`outputs/reports/artifact_registry.parquet`)

| Column | Type | Description |
|---|---|---|
| `step` | str | Pipeline step / parent directory name |
| `artifact_path` | str | Absolute path to the artifact |
| `exists` | bool | Whether the file exists |
| `last_modified` | datetime | Last modification timestamp |
| `session_id` | int | Session ID (extracted from filename, may be null) |
| `notes` | str | Free-text notes |

### Run Summary (`outputs/reports/run_summary.parquet`)

Contains per-session status information from the end-to-end pipeline run.

---

## Metadata Sidecar Format

Every parquet and NPZ artifact has a companion `.meta.json` sidecar file. The sidecar is located at `{artifact_path}.meta.json` (e.g., `session_1098595957_units.parquet.meta.json`).

### Structure

```json
{
  "timebase": "nwb_seconds",
  "provenance": {
    "session_id": 1098595957,
    "code_version": "9e576ad...",
    "created_at": "2026-03-09T12:34:56.789000+00:00",
    "alignment_method": "nwb"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `timebase` | str | Always `"nwb_seconds"` for valid artifacts |
| `provenance.session_id` | int | Session this artifact belongs to (null for cross-session artifacts) |
| `provenance.code_version` | str | Git commit hash when the artifact was created |
| `provenance.created_at` | str | ISO 8601 timestamp of creation |
| `provenance.alignment_method` | str | How timestamps were aligned (e.g., "nwb", "timestamps", "sleap_inference") |

### Parquet-Embedded Metadata

In addition to the sidecar, the `timebase` and `provenance` are embedded in the parquet file's schema metadata (via PyArrow). This ensures the metadata travels with the file even if the sidecar is lost:

```python
import pyarrow.parquet as pq

table = pq.read_table("outputs/neural/session_1098595957_units.parquet")
meta = table.schema.metadata
print(meta[b"timebase"])       # b"nwb_seconds"
print(meta[b"provenance"])     # JSON bytes
```

---

## Reading Artifacts Programmatically

### Parquet Files

```python
import pandas as pd

# Standard read
df = pd.read_parquet("outputs/neural/session_1098595957_units.parquet")

# With metadata validation
from reports import validate_artifact_schema
valid = validate_artifact_schema(
    Path("outputs/neural/session_1098595957_units.parquet"),
    required_columns=["unit_id"],
)
print(f"Valid: {valid}")
```

### NPZ Files

```python
import numpy as np
from io_nwb import load_spike_times_npz

spikes = load_spike_times_npz(Path("outputs/neural/session_1098595957_spike_times.npz"))
for unit_id, times in spikes.items():
    print(f"Unit {unit_id}: {len(times)} spikes")
```

### JSON Files

```python
import json
from pathlib import Path

with open("outputs/reports/config_snapshot.json") as f:
    config = json.load(f)
```

### Metadata Sidecars

```python
import json
from pathlib import Path

artifact = Path("outputs/neural/session_1098595957_units.parquet")
sidecar = artifact.with_suffix(artifact.suffix + ".meta.json")

with open(sidecar) as f:
    meta = json.load(f)

print(f"Timebase: {meta['timebase']}")
print(f"Created: {meta['provenance']['created_at']}")
```

---

## The Artifact Registry

The registry is a single parquet file that indexes all artifacts in the `outputs/` directory.

### Building the Registry

```python
from reports import build_artifact_registry, write_artifact_registry

# Build in memory
registry = build_artifact_registry()
print(registry[["step", "artifact_path", "session_id"]].head(20))

# Write to disk
path = write_artifact_registry()
print(f"Registry written to: {path}")
```

### Querying the Registry

```python
import pandas as pd

registry = pd.read_parquet("outputs/reports/artifact_registry.parquet")

# All artifacts for a specific session
session_artifacts = registry[registry["session_id"] == 1098595957]

# All artifacts from a specific step
neural_artifacts = registry[registry["step"] == "neural"]

# Missing artifacts
missing = registry[~registry["exists"]]
```

### Validating Artifacts

```python
from reports import validate_artifact_schema
from pathlib import Path

# Check if an artifact has correct schema and metadata
valid = validate_artifact_schema(
    path=Path("outputs/video/frame_times.parquet"),
    required_columns=["session_id", "camera", "frame_idx", "t"],
)

if not valid:
    print("Artifact is invalid -- re-run Notebook 05")
```

The validation checks:

1. File exists
2. All required columns are present
3. Sidecar `.meta.json` exists
4. Sidecar contains `"timebase": "nwb_seconds"`
5. Sidecar contains `"provenance"` key
