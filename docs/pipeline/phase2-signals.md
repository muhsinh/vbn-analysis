# Phase 2: Signal Extraction

Phase 2 spans **Notebooks 02-04** and extracts three classes of signals from NWB files: neural spike data, behavioral/task events, and eye tracking. Each extraction follows the same pattern: open NWB handle, extract into DataFrames, save as parquet with timebase metadata.

---

## NWB File Access: Real vs Mock

Every extraction begins with `open_nwb_handle()`, a context manager that transparently handles three scenarios:

```python title="src/io_nwb.py:open_nwb_handle()"
@contextlib.contextmanager
def open_nwb_handle(nwb_path, mock_mode=False):
    if mock_mode or nwb_path is None or not Path(nwb_path).exists():
        yield _mock_nwb()                      # (1)!
        return

    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(str(nwb_path), "r")
    try:
        nwb = io.read()
        yield nwb                              # (2)!
    finally:
        io.close()                             # (3)!
```

1. Mock mode: yields a synthetic NWB-like object with 3 units and 2 trials -- notebooks run without any data on disk
2. Real mode: yields the actual `pynwb.NWBFile` object
3. File handle is always closed, even if the notebook raises an exception

!!! info "The mock NWB object"
    When `mock_mode=True` or the NWB file is missing, the pipeline yields a `MockNWB` instance:

    ```python title="src/io_nwb.py:_mock_nwb()"
    class MockNWB:
        def __init__(self):
            self.units = pd.DataFrame({
                "unit_id": [1, 2, 3],
                "spike_times": [
                    np.array([0.1, 0.5, 1.0]),
                    np.array([0.2, 0.7, 1.4]),
                    np.array([0.3, 0.9, 1.8]),
                ],
            })
            self.trials = pd.DataFrame({
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "trial_type": ["go", "no-go"],
            })
            self.processing = {}
            self.stimulus = None
    ```

    This keeps all notebooks runnable for testing and development without requiring access to the 1+ TB VBN dataset.

---

## Notebook 02: Neural Data (Spikes and Events)

### `extract_units_and_spikes()`

This function converts the NWB `units` table into two outputs: a metadata DataFrame and a dictionary of spike time arrays.

```python title="src/io_nwb.py:extract_units_and_spikes()"
def extract_units_and_spikes(nwb):
    if nwb is None or not hasattr(nwb, "units") or nwb.units is None:
        return None, None

    units_table = nwb.units
    if hasattr(units_table, "to_dataframe"):
        units_df = units_table.to_dataframe()  # (1)!
    else:
        units_df = pd.DataFrame(units_table)

    spike_times = {}
    if "spike_times" in units_df.columns:
        for unit_id, times in units_df["spike_times"].items():
            spike_times[str(unit_id)] = np.asarray(times)  # (2)!
        units_df = units_df.drop(columns=["spike_times"])  # (3)!

    return units_df.reset_index(drop=False), spike_times
```

1. The NWB units table becomes a pandas DataFrame with columns like `unit_id`, `peak_channel_id`, `waveform_mean`, `isi_violations`, etc.
2. Each unit's spike times are stored as a numpy array, keyed by string unit ID
3. The `spike_times` column is dropped from the metadata DataFrame because it contains variable-length arrays that do not serialize well to parquet

### What the units DataFrame contains

| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | `int` | Unique identifier for this unit |
| `peak_channel_id` | `int` | Neuropixels channel with largest waveform amplitude |
| `waveform_mean` | `array` | Mean spike waveform shape |
| `firing_rate` | `float` | Baseline firing rate (Hz) |
| `isi_violations` | `float` | Fraction of inter-spike intervals below 1.5 ms (quality metric) |
| `snr` | `float` | Signal-to-noise ratio |
| `presence_ratio` | `float` | Fraction of session with detected spikes |
| *(varies by NWB)* | | Additional columns depend on the NWB file version |

### How `spike_times` dict is structured

```python
# spike_times is a dict: str(unit_id) -> np.ndarray of float64
spike_times = {
    "0": np.array([0.1, 0.5, 1.0, 1.7, ...]),   # times in NWB seconds
    "1": np.array([0.2, 0.7, 1.4, 2.1, ...]),
    "2": np.array([0.3, 0.9, 1.8, 3.2, ...]),
}
```

### How the `SessionBundle` orchestrates loading

```python title="src/io_sessions.py:SessionBundle.load_spikes()"
def load_spikes(self):
    cfg = get_config()
    outputs_dir = cfg.outputs_dir / "neural"
    units_path = outputs_dir / f"session_{self.session_id}_units.parquet"
    spikes_path = outputs_dir / f"session_{self.session_id}_spike_times.npz"

    if units_path.exists() and spikes_path.exists():     # (1)!
        units = pd.read_parquet(units_path)
        spikes = dict(io_nwb.load_spike_times_npz(spikes_path))
        return units, spikes

    with io_nwb.open_nwb_handle(                         # (2)!
        self.nwb_path, mock_mode=cfg.mock_mode
    ) as nwb:
        units, spikes = io_nwb.extract_units_and_spikes(nwb)

    if units is not None:                                # (3)!
        outputs_dir.mkdir(parents=True, exist_ok=True)
        io_nwb.save_units_and_spikes(
            units, spikes, units_path, spikes_path,
            session_id=self.session_id,
            alignment_method="nwb",
        )
    else:
        self.qc_flags.append("neural_unavailable")       # (4)!

    return units, spikes
```

1. Cache hit: load from parquet + npz
2. Cache miss: open NWB and extract
3. Save for next time
4. Flag missing data for QC reporting

### Saving with timebase metadata

```python title="src/io_nwb.py:save_units_and_spikes()"
def save_units_and_spikes(units, spikes, units_path, spikes_path,
                          session_id, alignment_method):
    provenance = _provenance(session_id, alignment_method)
    units_df = units.copy()
    if "unit_id" not in units_df.columns:
        units_df.insert(0, "unit_id", range(len(units_df)))

    write_parquet_with_timebase(                   # (1)!
        units_df, units_path,
        timebase="nwb_seconds",
        provenance=provenance,
        required_columns=["unit_id"],
    )
    write_npz_with_provenance(spikes, spikes_path, provenance)  # (2)!
```

1. Parquet file gets `timebase` and `provenance` embedded in both the Parquet schema metadata and a sidecar `.meta.json`
2. NPZ file gets a sidecar `.meta.json` with the same provenance

### Output files

| File | Format | Key columns |
|------|--------|-------------|
| `outputs/neural/session_<id>_units.parquet` | Parquet | `unit_id`, plus all NWB unit metadata |
| `outputs/neural/session_<id>_spike_times.npz` | NumPy compressed | One array per unit, keyed by unit ID string |

---

## Notebook 03: Behavior and Task Alignment

### `extract_trials()`

Extracts the NWB trials table and normalizes column names:

```python title="src/io_nwb.py:extract_trials()"
def extract_trials(nwb):
    if nwb is None or not hasattr(nwb, "trials") or nwb.trials is None:
        return None

    trials_table = nwb.trials
    if hasattr(trials_table, "to_dataframe"):
        df = trials_table.to_dataframe()
    else:
        df = pd.DataFrame(trials_table)
    df = df.reset_index(drop=False)

    if "start_time" in df.columns:
        df = df.rename(columns={
            "start_time": "t_start",               # (1)!
            "stop_time": "t_end",
        })
    if "t_start" in df.columns and "t" not in df.columns:
        df["t"] = df["t_start"]                    # (2)!
    return df
```

1. Rename NWB convention (`start_time`) to pipeline convention (`t_start`)
2. Add a convenience `t` column for event-alignment (points to trial start)

### Schema of `trials.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `t` | `float64` | Trial start time (NWB seconds), used for PETH alignment |
| `t_start` | `float64` | Trial start time |
| `t_end` | `float64` | Trial end time |
| `trial_type` | `str` | `"go"`, `"no-go"`, `"catch"`, etc. |
| `response` | `str` | `"hit"`, `"miss"`, `"false_alarm"`, `"correct_reject"` |
| `rewarded` | `bool` | Whether the animal received a reward |
| `stimulus_name` | `str` | Name of the visual stimulus presented |

### `extract_behavior_events()`

Extracts continuous behavior signals from the NWB `processing["behavior"]` module:

```python title="src/io_nwb.py:extract_behavior_events()"
def extract_behavior_events(nwb):
    if nwb is None:
        return None

    events = []
    processing = getattr(nwb, "processing", {}) or {}
    if "behavior" in processing:
        behavior_module = processing["behavior"]
        for name, ts in behavior_module.data_interfaces.items():
            try:
                times = np.asarray(ts.timestamps)
                data = np.asarray(ts.data)
                if data.ndim == 1:                     # (1)!
                    df = pd.DataFrame({"t": times, name: data})
                else:
                    cols = [f"{name}_{i}" for i in range(data.shape[1])]  # (2)!
                    df = pd.DataFrame(data, columns=cols)
                    df.insert(0, "t", times)
                events.append(df)
            except Exception:
                continue

    if not events:
        return None

    merged = events[0]
    for df in events[1:]:
        merged = pd.merge_asof(                        # (3)!
            merged.sort_values("t"),
            df.sort_values("t"),
            on="t",
        )
    return merged
```

1. Scalar time series (e.g., running speed) become a single column
2. Vector time series (e.g., position x/y) become multiple columns
3. All behavior streams are merged on the time column using `merge_asof` (nearest-time join)

### Schema of `events.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `t` | `float64` | Timestamp (NWB seconds) |
| `running_speed` | `float64` | Running wheel speed (cm/s) |
| `lick_times` | `float64` | Lick sensor activation times |
| *(varies)* | | Additional columns depend on what the NWB `behavior` module contains |

### `derive_task_features()`

Produces a compact task-feature table for downstream modeling:

```python title="src/features_task.py:derive_task_features()"
def derive_task_features(trials, events):
    if trials is None or trials.empty:
        return None
    df = trials.copy()

    if "t_start" in df.columns:
        df["t"] = df["t_start"]
    elif "start_time" in df.columns:
        df["t"] = df["start_time"]
    else:
        df["t"] = range(len(df))

    cols = ["t"] + [
        c for c in ["trial_type", "response", "rewarded", "stimulus_name"]
        if c in df.columns
    ]
    return df[cols]
```

### Output files

| File | Format | Key columns |
|------|--------|-------------|
| `outputs/behavior/session_<id>_trials.parquet` | Parquet | `t`, `t_start`, `t_end`, `trial_type`, `response`, `rewarded` |
| `outputs/behavior/session_<id>_events.parquet` | Parquet | `t`, `running_speed`, `lick_times`, ... |

---

## Notebook 04: Eye Tracking QC and Features

### `extract_eye_tracking()`

Extracts raw eye tracking data from the NWB `processing["eye_tracking"]` module:

```python title="src/io_nwb.py:extract_eye_tracking()"
def extract_eye_tracking(nwb):
    if nwb is None:
        return None
    processing = getattr(nwb, "processing", {}) or {}
    if "eye_tracking" not in processing:
        return None

    eye_module = processing["eye_tracking"]
    for name, ts in eye_module.data_interfaces.items():
        try:
            times = np.asarray(ts.timestamps)
            data = np.asarray(ts.data)
            if data.ndim == 1:
                df = pd.DataFrame({"t": times, name: data})
            else:
                cols = [f"{name}_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=cols)
                df.insert(0, "t", times)
            return df                          # (1)!
        except Exception:
            continue
    return None
```

1. Returns the **first** successfully parsed data interface, typically the pupil area time series

### `derive_eye_features()`

Computes three derived features from the raw eye tracking signal:

```python title="src/features_eye.py:derive_eye_features()"
def derive_eye_features(eye_df):
    if eye_df is None or eye_df.empty:
        return None
    df = eye_df.copy()
    if "t" not in df.columns:
        df = df.reset_index().rename(columns={"index": "t"})

    signal_cols = [c for c in df.columns if c != "t"]
    if not signal_cols:
        return df[["t"]]

    primary = signal_cols[0]                           # (1)!

    df["pupil"] = df[primary]                          # (2)!

    df["pupil_z"] = (
        df[primary] - np.nanmean(df[primary])
    ) / (np.nanstd(df[primary]) + 1e-6)               # (3)!

    df["pupil_vel"] = np.gradient(
        df[primary].to_numpy(),
        df["t"].to_numpy(),
    )                                                  # (4)!

    return df[["t", "pupil", "pupil_z", "pupil_vel"]]
```

1. Uses the first non-time column as the primary signal (typically pupil area in pixels^2)
2. Raw pupil area, renamed for consistency
3. Z-scored pupil area: zero-mean, unit-variance normalization (epsilon avoids division by zero)
4. Instantaneous pupil velocity via `np.gradient`, capturing the rate of change of pupil area with respect to time

### Mathematical details

The **z-scored pupil** removes session-level baseline differences:

$$
\text{pupil\_z}(t) = \frac{\text{pupil}(t) - \bar{\mu}}{\sigma + \epsilon}
$$

where $\bar{\mu}$ and $\sigma$ are the session-wide mean and standard deviation, and $\epsilon = 10^{-6}$.

The **pupil velocity** captures arousal dynamics (dilation/constriction rate):

$$
\text{pupil\_vel}(t) = \frac{d}{dt}\text{pupil}(t)
$$

computed via second-order central differences using `np.gradient`.

### Schema of `eye_features.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `t` | `float64` | Timestamp (NWB seconds) |
| `pupil` | `float64` | Raw pupil area (pixels^2) |
| `pupil_z` | `float64` | Z-scored pupil area (dimensionless) |
| `pupil_vel` | `float64` | Pupil area velocity (pixels^2 / second) |

### How `SessionBundle.load_eye_features()` works

```python title="src/io_sessions.py:SessionBundle.load_eye_features()"
def load_eye_features(self):
    cfg = get_config()
    outputs_dir = cfg.outputs_dir / "eye"
    eye_path = outputs_dir / f"session_{self.session_id}_eye_features.parquet"

    if eye_path.exists():
        return pd.read_parquet(eye_path)       # (1)!

    from features_eye import derive_eye_features

    with io_nwb.open_nwb_handle(self.nwb_path, mock_mode=cfg.mock_mode) as nwb:
        eye_raw = io_nwb.extract_eye_tracking(nwb)  # (2)!

    if eye_raw is None:
        self.qc_flags.append("eye_unavailable")     # (3)!
        return None

    eye_features = derive_eye_features(eye_raw)     # (4)!
    if eye_features is None:
        return None

    outputs_dir.mkdir(parents=True, exist_ok=True)
    io_nwb.save_eye_table(                          # (5)!
        eye_features, eye_path,
        session_id=self.session_id,
        alignment_method="nwb",
    )
    return eye_features
```

1. Cache hit
2. Extract raw signal from NWB
3. Flag missing data
4. Compute derived features (pupil, pupil_z, pupil_vel)
5. Save with timebase metadata

---

## Timebase Enforcement

Every artifact written in Phase 2 goes through `write_parquet_with_timebase()`, which does three things:

```python title="src/timebase.py:write_parquet_with_timebase()"
def write_parquet_with_timebase(df, path, timebase="nwb_seconds",
                                provenance=None, required_columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if required_columns:                               # (1)!
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing}")

    metadata = {"timebase": timebase}
    if provenance:
        metadata["provenance"] = provenance

    # Embed in Parquet schema metadata via PyArrow      (2)!
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    merged = {**existing}
    merged[b"timebase"] = timebase.encode("utf-8")
    if provenance:
        merged[b"provenance"] = json.dumps(provenance).encode("utf-8")
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, path)

    # Write sidecar .meta.json                          (3)!
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    with sidecar.open("w") as f:
        json.dump(metadata, f, indent=2)
    return path
```

1. **Schema validation**: ensures required columns exist before writing
2. **Parquet metadata**: timebase and provenance are embedded in the Parquet file itself
3. **Sidecar JSON**: a human-readable `.meta.json` file is written alongside for tool-agnostic access

!!! warning "Required columns are enforced"
    If you try to write a trials parquet without a `t` column, the pipeline raises:

    ```
    ValueError: Missing required columns ['t'] for artifact session_1234_trials.parquet
    ```
