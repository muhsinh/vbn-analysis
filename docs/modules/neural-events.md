# neural_events

Event-aligned neural analysis. Provides the core analyses for correlating
behavior with neural activity: peri-event time histograms (PETHs),
trial-averaged firing rates, population activity vectors, dimensionality
reduction, and single-unit selectivity screening.

**Source:** `src/neural_events.py`

---

## Functions

### `compute_peth`

```python
def compute_peth(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
) -> Dict[str, Any]
```

Compute a peri-event time histogram (PETH) for a single unit aligned to
behavioral events.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `np.ndarray` | -- | All spike times for this unit (seconds, NWB timebase). |
| `event_times` | `np.ndarray` | -- | Times of behavioral events to align to (e.g., stimulus onset, lick times). |
| `window` | `Tuple[float, float]` | `(-0.5, 1.0)` | `(pre, post)` time window relative to each event in seconds. Negative values are before the event. |
| `bin_size` | `float` | `0.01` | Bin width in seconds (10 ms default). |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary with the following keys: |

| Key | Type | Description |
|-----|------|-------------|
| `time_bins` | `np.ndarray` | Bin centers (seconds relative to event). |
| `mean_rate` | `np.ndarray` | Mean firing rate per bin (Hz) averaged across trials. |
| `sem_rate` | `np.ndarray` | Standard error of the mean across trials. |
| `trial_spikes` | `list[np.ndarray]` | Per-trial spike times relative to the event. |
| `n_trials` | `int` | Number of events/trials. |

**Example:**

```python
from neural_events import compute_peth
import numpy as np

spike_times = np.sort(np.random.uniform(0, 100, 5000))
event_times = np.array([10.0, 30.0, 50.0, 70.0, 90.0])

peth = compute_peth(spike_times, event_times, window=(-0.2, 0.5), bin_size=0.01)
print(f"n_trials={peth['n_trials']}, n_bins={len(peth['time_bins'])}")
print(f"Peak rate: {peth['mean_rate'].max():.1f} Hz")
```

---

### `compute_population_peth`

```python
def compute_population_peth(
    spike_times_dict: Dict[str, np.ndarray],
    event_times: np.ndarray,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    unit_ids: List[str] | None = None,
) -> Dict[str, Any]
```

Compute PETHs for all units in a population simultaneously.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times_dict` | `Dict[str, np.ndarray]` | -- | Mapping of unit ID strings to spike-time arrays. |
| `event_times` | `np.ndarray` | -- | Event times to align to. |
| `window` | `Tuple[float, float]` | `(-0.5, 1.0)` | Time window around events. |
| `bin_size` | `float` | `0.01` | Bin width in seconds. |
| `unit_ids` | `List[str] \| None` | `None` | Subset of unit IDs to include. If `None`, uses all keys in `spike_times_dict`. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `time_bins` | `np.ndarray` | Bin centers. |
| `population_matrix` | `np.ndarray` | `(n_units, n_bins)` matrix of mean firing rates. |
| `unit_ids` | `list[str]` | Unit IDs in the same order as `population_matrix` rows. |
| `peths` | `Dict[str, Dict]` | Individual PETH results keyed by unit ID. |

**Example:**

```python
from neural_events import compute_population_peth

pop = compute_population_peth(spikes, event_times, window=(-0.5, 1.0))
print(f"Population matrix shape: {pop['population_matrix'].shape}")
# (50, 150)  -- 50 units, 150 time bins
```

---

### `trial_averaged_rates`

```python
def trial_averaged_rates(
    spike_times_dict: Dict[str, np.ndarray],
    trials: pd.DataFrame,
    group_col: str = "trial_type",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.025,
) -> Dict[str, Dict[str, Any]]
```

Compute mean firing rates per condition per unit, grouped by a trial-level
categorical variable.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times_dict` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times mapping. |
| `trials` | `pd.DataFrame` | -- | Trial table. Must have a `t` column (event time) and `group_col`. |
| `group_col` | `str` | `"trial_type"` | Column to group trials by (e.g., `"trial_type"`, `"rewarded"`, `"stimulus_name"`). |
| `window` | `Tuple[float, float]` | `(-0.5, 1.0)` | Time window. |
| `bin_size` | `float` | `0.025` | Bin width. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Dict[str, Any]]` | Mapping of condition value (string) to population PETH result (as returned by `compute_population_peth`). If `group_col` is not in `trials`, all trials are grouped under the key `"all"`. |

!!! note
    Conditions with fewer than 2 trials are skipped.

**Example:**

```python
from neural_events import trial_averaged_rates

cond_peths = trial_averaged_rates(spikes, trials, group_col="trial_type")
for cond, result in cond_peths.items():
    n = result["peths"][list(result["peths"].keys())[0]]["n_trials"]
    print(f"Condition '{cond}': {n} trials, {len(result['unit_ids'])} units")
```

---

### `build_population_vectors`

```python
def build_population_vectors(
    spike_times_dict: Dict[str, np.ndarray],
    time_grid: np.ndarray,
    bin_size: float,
) -> np.ndarray
```

Build a population activity matrix by binning spike times on a uniform
time grid.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times_dict` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times. |
| `time_grid` | `np.ndarray` | -- | Left edges of time bins (from `timebase.build_time_grid`). |
| `bin_size` | `float` | -- | Bin width in seconds. |

**Returns:**

| Type | Description |
|------|-------------|
| `np.ndarray` | `(n_timepoints, n_units)` matrix of spike counts. Each entry is the number of spikes in that bin for that unit. |

**Example:**

```python
from neural_events import build_population_vectors
from timebase import build_time_grid

grid = build_time_grid(0.0, 100.0, 0.025)
pop = build_population_vectors(spikes, grid, 0.025)
print(f"Population matrix: {pop.shape}")  # (4000, 50)
```

---

### `reduce_population`

```python
def reduce_population(
    pop_matrix: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
) -> Tuple[np.ndarray, Any]
```

Reduce dimensionality of a population activity matrix.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop_matrix` | `np.ndarray` | -- | `(n_timepoints, n_units)` activity matrix. |
| `method` | `str` | `"pca"` | Dimensionality reduction method. `"pca"` uses sklearn PCA; `"umap"` uses umap-learn. |
| `n_components` | `int` | `3` | Target number of dimensions. |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[np.ndarray, Any]` | `(reduced_matrix, model)` where `reduced_matrix` is `(n_timepoints, n_components)` and `model` is the fitted PCA or UMAP object. |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ImportError` | `umap-learn` not installed (for `method="umap"`). |
| `ValueError` | Unknown method name. |

!!! note
    The input is z-scored per unit before reduction.

**Example:**

```python
from neural_events import reduce_population

reduced, pca = reduce_population(pop, method="pca", n_components=3)
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

---

### `compute_selectivity_index`

```python
def compute_selectivity_index(
    spike_times: np.ndarray,
    condition_a_times: np.ndarray,
    condition_b_times: np.ndarray,
    window: Tuple[float, float] = (0.0, 0.5),
) -> Dict[str, float]
```

Compute how selective a single unit is between two experimental conditions.
Uses d-prime and a Mann-Whitney U test.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `np.ndarray` | -- | Spike times for this unit. |
| `condition_a_times` | `np.ndarray` | -- | Event times for condition A. |
| `condition_b_times` | `np.ndarray` | -- | Event times for condition B. |
| `window` | `Tuple[float, float]` | `(0.0, 0.5)` | Time window after each event in which to count spikes. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, float]` | Dictionary with keys: |

| Key | Description |
|-----|-------------|
| `d_prime` | Cohen's d-prime: `(mean_a - mean_b) / pooled_std`. |
| `rate_diff` | Raw difference in mean firing rates (Hz). |
| `mean_rate_a` | Mean firing rate for condition A. |
| `mean_rate_b` | Mean firing rate for condition B. |
| `p_value` | Two-sided Mann-Whitney U p-value. |

**Example:**

```python
from neural_events import compute_selectivity_index

sel = compute_selectivity_index(
    spike_times=spikes["unit_0"],
    condition_a_times=go_trial_times,
    condition_b_times=nogo_trial_times,
    window=(0.0, 0.3),
)
print(f"d'={sel['d_prime']:.2f}, p={sel['p_value']:.4f}")
```

---

### `screen_selective_units`

```python
def screen_selective_units(
    spike_times_dict: Dict[str, np.ndarray],
    condition_a_times: np.ndarray,
    condition_b_times: np.ndarray,
    window: Tuple[float, float] = (0.0, 0.5),
    p_threshold: float = 0.05,
) -> pd.DataFrame
```

Screen all units in a population for selectivity between two conditions.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times_dict` | `Dict[str, np.ndarray]` | -- | All unit spike times. |
| `condition_a_times` | `np.ndarray` | -- | Event times for condition A. |
| `condition_b_times` | `np.ndarray` | -- | Event times for condition B. |
| `window` | `Tuple[float, float]` | `(0.0, 0.5)` | Response window. |
| `p_threshold` | `float` | `0.05` | Significance threshold. |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Table sorted by `|d_prime|` (descending) with columns: `unit_id`, `d_prime`, `rate_diff`, `mean_rate_a`, `mean_rate_b`, `p_value`, `significant` (bool), `abs_d_prime`. |

**Example:**

```python
from neural_events import screen_selective_units

sel_df = screen_selective_units(
    spikes,
    condition_a_times=go_times,
    condition_b_times=nogo_times,
)
sig = sel_df[sel_df["significant"]]
print(f"{len(sig)} / {len(sel_df)} units are significantly selective")
print(sel_df.head())
```
