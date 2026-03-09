# io_nwb

NWB file I/O and data extraction utilities. Handles opening NWB files (or
generating mock data), inspecting available modalities, extracting neural
and behavioral data, and saving standardized Parquet/NPZ artifacts.

**Source:** `src/io_nwb.py`

---

## Functions

### `open_nwb_handle`

```python
@contextlib.contextmanager
def open_nwb_handle(
    nwb_path: Path | None,
    mock_mode: bool = False,
) -> Iterator[Any]
```

Context manager that opens an NWB file and yields the `NWBFile` object. The
file handle is automatically closed when the context exits.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb_path` | `Path \| None` | -- | Path to the `.nwb` file on disk. |
| `mock_mode` | `bool` | `False` | If `True`, or if `nwb_path` is `None` or does not exist, yields a synthetic `MockNWB` object instead. |

**Yields:**

| Type | Description |
|------|-------------|
| `NWBFile` or `MockNWB` | The opened NWB file object, or a mock for testing. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | `pynwb` is not installed and a real NWB file is requested. |

!!! warning "Resource management"
    Always use this function as a context manager (`with` statement) to ensure
    the HDF5 file handle is properly closed.

**Example:**

```python
from io_nwb import open_nwb_handle
from pathlib import Path

# Open a real NWB file
with open_nwb_handle(Path("session.nwb")) as nwb:
    print(type(nwb))  # pynwb.NWBFile

# Use mock mode for testing
with open_nwb_handle(None, mock_mode=True) as nwb:
    print(nwb)  # <MockNWB>
    print(nwb.units)  # DataFrame with 3 synthetic units
```

---

### `resolve_nwb_path`

```python
def resolve_nwb_path(
    session_id: int,
    access_mode: str,
    nwb_path_override: Path | None = None,
) -> Path | None
```

Resolve the local path to an NWB file, optionally downloading it via
the Allen SDK.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `int` | -- | Allen Brain Observatory session ID. |
| `access_mode` | `str` | -- | `"sdk"` to use AllenSDK's S3 cache (auto-downloads if needed), or `"manual"` to return `nwb_path_override` as-is. |
| `nwb_path_override` | `Path \| None` | `None` | User-provided NWB path. Used directly in `"manual"` mode; used as fallback in `"sdk"` mode if SDK resolution fails. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path \| None` | Local filesystem path to the NWB file, or `None` if it cannot be resolved. |

**Example:**

```python
from io_nwb import resolve_nwb_path

# SDK mode: auto-downloads if needed
path = resolve_nwb_path(1064644573, access_mode="sdk")

# Manual mode: just pass through the override
path = resolve_nwb_path(1064644573, access_mode="manual",
                        nwb_path_override=Path("/data/my_session.nwb"))
```

---

### `inspect_modalities`

```python
def inspect_modalities(nwb: Any) -> Dict[str, bool]
```

Check which data modalities are present in an NWB file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb` | `Any` | -- | An opened `NWBFile` object (or `MockNWB`). |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, bool]` | Dictionary with keys `"spikes"`, `"trials"`, `"eye"`, `"behavior"`, `"stimulus"`, each mapping to `True`/`False`. |

**Example:**

```python
from io_nwb import open_nwb_handle, inspect_modalities

with open_nwb_handle(nwb_path) as nwb:
    mods = inspect_modalities(nwb)
    print(mods)
    # {'spikes': True, 'trials': True, 'eye': True, 'behavior': True, 'stimulus': False}
```

---

### `extract_units_and_spikes`

```python
def extract_units_and_spikes(
    nwb: Any,
) -> Tuple[pd.DataFrame | None, Dict[str, Any] | None]
```

Extract the units table and spike-time arrays from an NWB file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb` | `Any` | -- | An opened `NWBFile` object. |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[pd.DataFrame \| None, Dict[str, Any] \| None]` | `(units_df, spike_times_dict)`. `units_df` contains unit metadata (with the index reset). `spike_times_dict` maps string unit IDs to `np.ndarray` of spike times. Returns `(None, None)` if no units are available. |

!!! note
    The `spike_times` column is removed from `units_df` and placed into the
    dictionary to keep the DataFrame lightweight.

**Example:**

```python
from io_nwb import open_nwb_handle, extract_units_and_spikes

with open_nwb_handle(nwb_path) as nwb:
    units, spikes = extract_units_and_spikes(nwb)
    if units is not None:
        print(f"Found {len(units)} units")
        for uid, times in list(spikes.items())[:3]:
            print(f"  Unit {uid}: {len(times)} spikes")
```

---

### `extract_trials`

```python
def extract_trials(nwb: Any) -> pd.DataFrame | None
```

Extract the trials table from an NWB file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb` | `Any` | -- | An opened `NWBFile` object. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame \| None` | Trial table with columns `t` (or `t_start`/`t_end`), `trial_type`, and any other trial-level metadata. `start_time`/`stop_time` are renamed to `t_start`/`t_end`. A `t` column is added as an alias for `t_start`. Returns `None` if no trials exist. |

**Example:**

```python
from io_nwb import open_nwb_handle, extract_trials

with open_nwb_handle(nwb_path) as nwb:
    trials = extract_trials(nwb)
    if trials is not None:
        print(trials[["t_start", "t_end", "trial_type"]].head())
```

---

### `extract_behavior_events`

```python
def extract_behavior_events(nwb: Any) -> pd.DataFrame | None
```

Extract behavioral event time series from the NWB `processing["behavior"]`
module. Merges all data interfaces (e.g., running speed, lick times) into a
single DataFrame via `merge_asof` on the `t` column.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb` | `Any` | -- | An opened `NWBFile` object. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame \| None` | Merged behavior DataFrame with a `t` column and one column per behavioral signal. Returns `None` if no behavior data exists. |

**Example:**

```python
from io_nwb import open_nwb_handle, extract_behavior_events

with open_nwb_handle(nwb_path) as nwb:
    events = extract_behavior_events(nwb)
    if events is not None:
        print(events.columns.tolist())
        # e.g., ['t', 'running_speed', 'lick_times', ...]
```

---

### `extract_eye_tracking`

```python
def extract_eye_tracking(nwb: Any) -> pd.DataFrame | None
```

Extract eye-tracking data from the NWB `processing["eye_tracking"]` module.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwb` | `Any` | -- | An opened `NWBFile` object. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame \| None` | Eye-tracking DataFrame with `t` and signal columns (e.g., `pupil_area`, `gaze_x_0`, `gaze_x_1`). Returns `None` if no eye-tracking processing module exists. |

**Example:**

```python
from io_nwb import open_nwb_handle, extract_eye_tracking

with open_nwb_handle(nwb_path) as nwb:
    eye = extract_eye_tracking(nwb)
    if eye is not None:
        print(eye[["t"]].describe())
```

---

### `save_units_and_spikes`

```python
def save_units_and_spikes(
    units: pd.DataFrame,
    spikes: Dict[str, Any],
    units_path: Path,
    spikes_path: Path,
    session_id: int,
    alignment_method: str,
) -> None
```

Save extracted units and spike times to disk with provenance metadata.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `units` | `pd.DataFrame` | -- | Units metadata table. |
| `spikes` | `Dict[str, Any]` | -- | Spike times dictionary (unit ID -> array). |
| `units_path` | `Path` | -- | Output path for the Parquet units file. |
| `spikes_path` | `Path` | -- | Output path for the NPZ spike times file. |
| `session_id` | `int` | -- | Session ID for provenance. |
| `alignment_method` | `str` | -- | Alignment method string for provenance. |

**Side effects:**

- Writes `units_path` as Parquet with timebase and provenance metadata.
- Writes `spikes_path` as NPZ with a JSON sidecar.
- Ensures a `unit_id` column exists in the units table.

---

### `save_behavior_tables`

```python
def save_behavior_tables(
    trials: pd.DataFrame | None,
    events: pd.DataFrame | None,
    trials_path: Path,
    events_path: Path,
    session_id: int,
    alignment_method: str,
) -> None
```

Save trial and event tables to Parquet with provenance. Skips `None` inputs.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trials` | `pd.DataFrame \| None` | -- | Trial table (may be `None`). |
| `events` | `pd.DataFrame \| None` | -- | Events table (may be `None`). |
| `trials_path` | `Path` | -- | Output path for trials Parquet. |
| `events_path` | `Path` | -- | Output path for events Parquet. |
| `session_id` | `int` | -- | Session ID for provenance. |
| `alignment_method` | `str` | -- | Alignment method for provenance. |

---

### `save_eye_table`

```python
def save_eye_table(
    eye_df: pd.DataFrame,
    eye_path: Path,
    session_id: int,
    alignment_method: str,
) -> None
```

Save eye-tracking features to Parquet with provenance.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eye_df` | `pd.DataFrame` | -- | Eye feature DataFrame (must have a `t` column). |
| `eye_path` | `Path` | -- | Output path. |
| `session_id` | `int` | -- | Session ID for provenance. |
| `alignment_method` | `str` | -- | Alignment method for provenance. |

---

### `load_spike_times_npz`

```python
def load_spike_times_npz(path: Path) -> Dict[str, Any]
```

Load spike times from a previously saved NPZ file.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path` | -- | Path to the `.npz` file. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary mapping unit ID strings to `np.ndarray` spike-time arrays. |

**Example:**

```python
from io_nwb import load_spike_times_npz

spikes = load_spike_times_npz(Path("outputs/neural/session_123_spike_times.npz"))
for uid, times in list(spikes.items())[:3]:
    print(f"Unit {uid}: {len(times)} spikes")
```

---

## Classes

### `MockNWB`

```python
class MockNWB
```

A minimal synthetic NWB-like object used when `mock_mode=True` or when no
real NWB file is available. Keeps notebooks executable without real data.

#### Fields

| Field | Type | Content |
|-------|------|---------|
| `units` | `pd.DataFrame` | 3 synthetic units with `unit_id` and `spike_times` columns. |
| `trials` | `pd.DataFrame` | 2 synthetic trials with `start_time`, `stop_time`, `trial_type`. |
| `processing` | `dict` | Empty dictionary (no processing modules). |
| `stimulus` | `None` | No stimulus data. |

**Example:**

```python
from io_nwb import open_nwb_handle

with open_nwb_handle(None, mock_mode=True) as nwb:
    print(nwb)          # <MockNWB>
    print(len(nwb.units))  # 3
    print(nwb.trials["trial_type"].tolist())  # ['go', 'no-go']
```
