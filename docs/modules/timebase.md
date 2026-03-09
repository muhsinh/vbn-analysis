# timebase

Canonical timebase enforcement, standardized artifact writing, time-grid
construction, and binning utilities. Every time column in the pipeline flows
through this module to ensure consistent temporal alignment.

**Source:** `src/timebase.py`

---

## Constants

### `CANONICAL_TIMEBASE`

```python
CANONICAL_TIMEBASE: str = "nwb_seconds"
```

The canonical time reference for all artifacts in the pipeline. All `t`
columns are in **NWB seconds**, the same clock used by the Allen Brain
Observatory NWB files. This string is embedded in every Parquet and NPZ
artifact's metadata.

---

## Functions

### `ensure_time_column`

```python
def ensure_time_column(
    df: pd.DataFrame,
    time_col: str = "t",
) -> pd.DataFrame
```

Validate that a DataFrame has a time column and coerce it to numeric type.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | Input DataFrame. |
| `time_col` | `str` | `"t"` | Name of the expected time column. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Copy of `df` with `time_col` cast to numeric (non-numeric values become NaN). |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | The specified time column does not exist in the DataFrame. |

**Example:**

```python
from timebase import ensure_time_column

df = ensure_time_column(raw_df, time_col="t")
# Guaranteed: df["t"] is numeric
```

---

### `write_parquet_with_timebase`

```python
def write_parquet_with_timebase(
    df: pd.DataFrame,
    path: Path,
    timebase: str = CANONICAL_TIMEBASE,
    provenance: Dict[str, Any] | None = None,
    required_columns: Iterable[str] | None = None,
) -> Path
```

Write a DataFrame to Parquet with embedded timebase and provenance metadata.
This is the standard artifact writer used throughout the pipeline.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | Data to write. |
| `path` | `Path` | -- | Output file path. Parent directories are created automatically. |
| `timebase` | `str` | `"nwb_seconds"` | Timebase identifier embedded in Parquet schema metadata. |
| `provenance` | `Dict[str, Any] \| None` | `None` | Provenance dictionary (from `config.make_provenance()`). Stored as JSON in Parquet schema metadata. |
| `required_columns` | `Iterable[str] \| None` | `None` | Columns that must exist in `df`. Raises `ValueError` if any are missing. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | The output file path. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | One or more `required_columns` are missing from the DataFrame. |

**Side effects:**

- Creates the parent directory if needed.
- Writes a `.meta.json` sidecar file alongside the Parquet file with
  `timebase` and `provenance` for tool-agnostic metadata access.
- Attempts to embed metadata in the Parquet schema via PyArrow; falls
  back to plain `pd.to_parquet()` if PyArrow is unavailable.

**Example:**

```python
from timebase import write_parquet_with_timebase
from config import make_provenance
from pathlib import Path

write_parquet_with_timebase(
    df=my_data,
    path=Path("outputs/behavior/session_123_trials.parquet"),
    provenance=make_provenance(session_id=123, alignment_method="nwb"),
    required_columns=["t", "trial_type"],
)
# Also creates: outputs/behavior/session_123_trials.parquet.meta.json
```

---

### `write_npz_with_provenance`

```python
def write_npz_with_provenance(
    data: Dict[str, Any],
    path: Path,
    provenance: Dict[str, Any],
) -> Path
```

Write arrays to an NPZ file with a provenance sidecar.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Dict[str, Any]` | -- | Dictionary mapping names to arrays (passed to `np.savez`). |
| `path` | `Path` | -- | Output `.npz` file path. |
| `provenance` | `Dict[str, Any]` | -- | Provenance dictionary. |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | The output file path. |

**Side effects:**

- Creates parent directories.
- Writes a `.meta.json` sidecar with `provenance` and `timebase`.

**Example:**

```python
from timebase import write_npz_with_provenance

write_npz_with_provenance(
    data={"unit_0": spike_times_array, "unit_1": other_array},
    path=Path("outputs/neural/session_123_spike_times.npz"),
    provenance=make_provenance(123, "nwb"),
)
```

---

### `build_time_grid`

```python
def build_time_grid(
    start: float,
    end: float,
    bin_size_s: float,
) -> np.ndarray
```

Construct a uniform time grid (array of bin left-edges).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `float` | -- | Start time in seconds. |
| `end` | `float` | -- | End time in seconds. |
| `bin_size_s` | `float` | -- | Bin width in seconds. |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray` | 1-D array of bin left-edge times. Empty array if `end <= start`. |

**Example:**

```python
from timebase import build_time_grid

grid = build_time_grid(0.0, 10.0, 0.025)
print(f"{len(grid)} bins, [{grid[0]:.3f}, {grid[-1]:.3f}]")
# 400 bins, [0.000, 9.975]
```

---

### `bin_spike_times`

```python
def bin_spike_times(
    spike_times: Dict[str, np.ndarray],
    time_grid: np.ndarray,
    bin_size_s: float,
) -> pd.DataFrame
```

Bin spike times into counts on a uniform time grid.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times mapping. |
| `time_grid` | `np.ndarray` | -- | Bin left-edge times (from `build_time_grid`). |
| `bin_size_s` | `float` | -- | Bin width in seconds. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with a `t` column (bin left-edges) and one column per unit containing integer spike counts. Empty DataFrame if `spike_times` is `None`. |

**Example:**

```python
from timebase import build_time_grid, bin_spike_times

grid = build_time_grid(0.0, 100.0, 0.025)
counts = bin_spike_times(spikes, grid, 0.025)
print(counts.head())
#        t  unit_0  unit_1  unit_2
# 0  0.000       0       1       0
# 1  0.025       1       0       0
# 2  0.050       0       0       1
```

---

### `bin_continuous_features`

```python
def bin_continuous_features(
    df: pd.DataFrame,
    time_grid: np.ndarray,
    agg: str = "mean",
) -> pd.DataFrame
```

Bin continuous features onto a uniform time grid using a specified
aggregation function.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | Feature DataFrame with a `t` column. |
| `time_grid` | `np.ndarray` | -- | Bin left-edge times. |
| `agg` | `str` | `"mean"` | Aggregation function name (any valid argument to `DataFrame.agg()`, e.g., `"mean"`, `"median"`, `"sum"`, `"max"`). |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Binned DataFrame with a `t` column and one row per time bin. Bins with no data are filled with NaN. |

!!! note
    The original `t` column in `df` is used for bin assignment but is replaced
    by the `time_grid` values in the output. Feature columns are aggregated
    within each bin.

**Example:**

```python
from timebase import build_time_grid, bin_continuous_features

grid = build_time_grid(0.0, 100.0, 0.025)
binned = bin_continuous_features(pose_features, grid, agg="mean")
print(binned[["t", "pose_speed"]].head())
#        t  pose_speed
# 0  0.000       12.34
# 1  0.025       11.98
# 2  0.050       13.01
```
