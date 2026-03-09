# viz

Matplotlib visualization helpers for every analysis stage. Each function
creates a ready-to-display figure -- they are designed to be called directly
from Jupyter notebook cells.

**Source:** `src/viz.py`

---

## Functions

### `plot_raster`

```python
def plot_raster(
    spike_times: Dict[str, np.ndarray],
    max_units: int = 50,
) -> None
```

Plot a spike raster for a subset of units. Each unit is a horizontal row;
spikes are vertical ticks.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times mapping. |
| `max_units` | `int` | `50` | Maximum number of units to display. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_raster

plot_raster(spikes, max_units=30)
```

---

### `plot_firing_rate_summary`

```python
def plot_firing_rate_summary(
    spike_times: Dict[str, np.ndarray],
) -> None
```

Plot a histogram of mean firing rates across all units.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times mapping. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_firing_rate_summary

plot_firing_rate_summary(spikes)
# Shows distribution of firing rates in Hz
```

---

### `plot_behavior_summary`

```python
def plot_behavior_summary(
    trials: pd.DataFrame | None,
) -> None
```

Plot a trial-type summary with counts, percentages, and a timeline. If the
`trial_type` column exists, shows a horizontal bar chart of trial counts with
percentages and (if `t_start`/`t_end` exist) a scatter-and-segment timeline.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trials` | `pd.DataFrame \| None` | -- | Trial table. Should have `trial_type` and optionally `t_start`/`t_end` columns. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_behavior_summary

plot_behavior_summary(trials)
# Left panel: trial type counts with percentages
# Right panel: trial timeline showing when each trial occurred
```

---

### `plot_eye_qc`

```python
def plot_eye_qc(
    eye_df: pd.DataFrame | None,
) -> None
```

Plot the first signal column from an eye-tracking DataFrame over time.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eye_df` | `pd.DataFrame \| None` | -- | Eye-tracking DataFrame with a `t` column and at least one signal column. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_eye_qc

plot_eye_qc(eye_features)
```

---

### `plot_video_alignment`

```python
def plot_video_alignment(
    frame_times: pd.DataFrame | None,
) -> None
```

Plot frame-to-frame timing (dt) and identify temporal gaps. Produces a
two-panel figure:

- **Left:** dt vs frame index (downsampled for performance), with median dt
  and gap-detection threshold lines.
- **Right:** Bar chart of the 10 largest gaps by estimated missing frames.

Also prints a summary line and details of the largest gaps.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_times` | `pd.DataFrame \| None` | -- | Frame-time table with `frame_idx` and `t` columns. |

**Returns:** `None` (displays a matplotlib figure and prints summary text).

**Example:**

```python
from viz import plot_video_alignment

plot_video_alignment(frame_times)
# Prints: frames_valid=612345 frames_est_total=612400 lost_est=55 lost_pct=0.009% fps_est=29.997
```

---

### `plot_motif_transition`

```python
def plot_motif_transition(
    motifs: pd.DataFrame | None,
) -> None
```

Plot the behavioral motif transition matrix as a heatmap. Shows how
frequently each motif transitions to every other motif.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `motifs` | `pd.DataFrame \| None` | -- | Motif table with a `motif_id` column. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_motif_transition

plot_motif_transition(motif_df)
# Heatmap: rows = current motif, columns = next motif
```

---

### `plot_model_performance`

```python
def plot_model_performance(
    metrics: Dict[str, Any],
) -> None
```

Plot model evaluation metrics as a bar chart. Only numeric metric values are
displayed.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `Dict[str, Any]` | -- | Metrics dictionary (e.g., `{"r2": 0.45, "pseudo_r2": 0.38}`). |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_model_performance

plot_model_performance(result["metrics"])
```

---

### `plot_fusion_sanity`

```python
def plot_fusion_sanity(
    fusion: pd.DataFrame,
    target_col: str,
) -> None
```

Quick sanity-check plot of a target column from the fusion table over time.
Shows the first 1000 rows.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fusion` | `pd.DataFrame` | -- | Fusion table with `t` and `target_col`. |
| `target_col` | `str` | -- | Column name to plot. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_fusion_sanity

plot_fusion_sanity(fusion, target_col="unit_0")
```

---

### `plot_peth`

```python
def plot_peth(
    peth_result: Dict[str, Any],
    unit_id: str = "",
    ax: Any = None,
) -> None
```

Plot a peri-event time histogram with SEM shading. Shows the mean firing
rate over time aligned to an event, with a vertical line at time zero.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `peth_result` | `Dict[str, Any]` | -- | PETH result from `neural_events.compute_peth()`. Must contain `time_bins`, `mean_rate`, `sem_rate`, `n_trials`. |
| `unit_id` | `str` | `""` | Unit identifier for the plot title. |
| `ax` | `matplotlib.axes.Axes \| None` | `None` | Axes to plot on. If `None`, creates a new figure. |

**Returns:** `None` (displays or updates a matplotlib figure).

**Example:**

```python
from viz import plot_peth
from neural_events import compute_peth

peth = compute_peth(spikes["unit_0"], event_times)
plot_peth(peth, unit_id="unit_0")
```

---

### `plot_population_peth`

```python
def plot_population_peth(
    pop_result: Dict[str, Any],
    title: str = "Population PETH",
) -> None
```

Plot a population PETH as a heatmap (units on the y-axis, time on the
x-axis). Color intensity represents firing rate.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop_result` | `Dict[str, Any]` | -- | Population PETH result from `neural_events.compute_population_peth()`. Must contain `population_matrix` and `time_bins`. |
| `title` | `str` | `"Population PETH"` | Plot title. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_population_peth
from neural_events import compute_population_peth

pop = compute_population_peth(spikes, event_times)
plot_population_peth(pop, title="Stimulus-aligned population response")
```

---

### `plot_trial_comparison`

```python
def plot_trial_comparison(
    condition_peths: Dict[str, Dict[str, Any]],
    unit_id: str | None = None,
) -> None
```

Plot overlaid PETHs for different trial conditions on a single axis. Each
condition is shown in a different color with SEM shading.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition_peths` | `Dict[str, Dict[str, Any]]` | -- | Mapping of condition name to population PETH result (from `neural_events.trial_averaged_rates()`). |
| `unit_id` | `str \| None` | `None` | If provided, plots the PETH for that specific unit. If `None`, plots the population-averaged response. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_trial_comparison
from neural_events import trial_averaged_rates

cond_peths = trial_averaged_rates(spikes, trials, group_col="trial_type")
plot_trial_comparison(cond_peths)

# For a specific unit:
plot_trial_comparison(cond_peths, unit_id="unit_0")
```

---

### `plot_crosscorrelation`

```python
def plot_crosscorrelation(
    xcorr: Dict[str, Any],
    bin_size: float = 0.025,
) -> None
```

Plot the cross-correlation function between neural and behavioral signals.
Shows the correlation as a filled line plot with the peak lag annotated.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xcorr` | `Dict[str, Any]` | -- | Cross-correlation result from `cross_correlation.crosscorrelation()`. |
| `bin_size` | `float` | `0.025` | Bin size in seconds (for converting lag bins to seconds on the x-axis). |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_crosscorrelation
from cross_correlation import crosscorrelation

xcorr = crosscorrelation(pop_rate, behavior, max_lag=40)
plot_crosscorrelation(xcorr, bin_size=0.025)
```

---

### `plot_sliding_correlation`

```python
def plot_sliding_correlation(
    slide: Dict[str, Any],
    bin_size: float = 0.025,
) -> None
```

Plot sliding-window correlation over time. Two-panel figure:

- **Top:** Pearson r values over time with fill.
- **Bottom:** Significance scatter (-log10 p-value), with a horizontal line
  at p = 0.05.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slide` | `Dict[str, Any]` | -- | Sliding correlation result from `cross_correlation.sliding_correlation()`. |
| `bin_size` | `float` | `0.025` | Bin size for converting window centers to seconds. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_sliding_correlation
from cross_correlation import sliding_correlation

slide = sliding_correlation(pop_rate, behavior, window_size=200, step=50)
plot_sliding_correlation(slide, bin_size=0.025)
```

---

### `plot_encoding_decoding`

```python
def plot_encoding_decoding(
    enc_result: Dict[str, Any],
    dec_result: Dict[str, Any],
) -> None
```

Plot encoding vs decoding model performance as side-by-side box plots with
individual CV fold scores overlaid.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enc_result` | `Dict[str, Any]` | -- | Encoding model result (must have `cv_scores` key). |
| `dec_result` | `Dict[str, Any]` | -- | Decoding model result (must have `cv_scores` key). |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_encoding_decoding

plot_encoding_decoding(
    enc_result=results["encoding"],
    dec_result=results["decoding"],
)
```

---

### `plot_granger_summary`

```python
def plot_granger_summary(
    gc_n2b: Dict[str, Any],
    gc_b2n: Dict[str, Any],
) -> None
```

Plot Granger causality results in both directions as horizontal bars.
Significant results (p < 0.05) are highlighted in red.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gc_n2b` | `Dict[str, Any]` | -- | Granger test result for neural-to-behavior direction. |
| `gc_b2n` | `Dict[str, Any]` | -- | Granger test result for behavior-to-neural direction. |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_granger_summary

plot_granger_summary(
    gc_n2b=results["granger_neural_to_behavior"],
    gc_b2n=results["granger_behavior_to_neural"],
)
```

---

### `plot_unit_lag_distribution`

```python
def plot_unit_lag_distribution(
    unit_xcorr: pd.DataFrame,
    bin_size: float = 0.025,
) -> None
```

Plot the distribution of peak lags across all units. Two-panel figure:

- **Left:** Histogram of peak lags (seconds). A vertical line at zero
  separates units that lead vs lag behavior.
- **Right:** Scatter plot of lag vs absolute correlation strength, colored by
  sign of correlation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unit_xcorr` | `pd.DataFrame` | -- | Per-unit cross-correlation summary from `cross_correlation.population_crosscorrelation()`. Must have `peak_lag_s` and `peak_corr` columns. |
| `bin_size` | `float` | `0.025` | Bin size (not used in computation, included for API consistency). |

**Returns:** `None` (displays a matplotlib figure).

**Example:**

```python
from viz import plot_unit_lag_distribution

plot_unit_lag_distribution(results["unit_crosscorrelation"])
```
