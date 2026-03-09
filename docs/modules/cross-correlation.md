# cross_correlation

Cross-modal correlation between neural activity and behavior. Answers the
core question: do changes in behavior align with changes in neural activity?
Provides time-lagged cross-correlation, sliding-window analysis, encoding
models (behavior to neural), decoding models (neural to behavior),
Granger causality testing, and a full alignment pipeline.

**Source:** `src/cross_correlation.py`

---

## Functions

### `crosscorrelation`

```python
def crosscorrelation(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 50,
    normalize: bool = True,
) -> Dict[str, np.ndarray]
```

Compute the time-lagged cross-correlation between a neural signal and a
behavioral signal.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neural` | `np.ndarray` | -- | 1-D array of neural activity (e.g., population firing rate or single-unit binned counts). Shape `(n_timepoints,)`. |
| `behavior` | `np.ndarray` | -- | 1-D array of behavioral variable (e.g., pose speed, pupil diameter). Shape `(n_timepoints,)`. |
| `max_lag` | `int` | `50` | Maximum lag in bins (both positive and negative directions). |
| `normalize` | `bool` | `True` | If `True`, normalize to Pearson correlation coefficients in range `[-1, 1]`. If `False`, compute raw cross-covariance. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, np.ndarray]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `lags` | `np.ndarray` | Lag values from `-max_lag` to `+max_lag`. Negative lag means neural leads behavior. |
| `correlation` | `np.ndarray` | Cross-correlation values at each lag. |
| `peak_lag` | `int` | Lag of maximum absolute correlation. |
| `peak_corr` | `float` | Correlation value at the peak lag. |

!!! info "Lag sign convention"
    Negative lag means neural activity **leads** behavior (neural changes
    precede behavioral changes). Positive lag means behavior leads neural.

**Example:**

```python
from cross_correlation import crosscorrelation
import numpy as np

neural = np.random.randn(10000)
behavior = np.roll(neural, 5) + 0.5 * np.random.randn(10000)

xcorr = crosscorrelation(neural, behavior, max_lag=20)
print(f"Peak lag: {xcorr['peak_lag']} bins, r={xcorr['peak_corr']:.3f}")
# Peak lag: -5 bins, r=0.85  (neural leads by 5 bins)
```

---

### `population_crosscorrelation`

```python
def population_crosscorrelation(
    pop_matrix: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 50,
    bin_size: float = 0.025,
) -> pd.DataFrame
```

Compute cross-correlation between each unit in a population matrix and a
single behavioral signal.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop_matrix` | `np.ndarray` | -- | Population activity matrix, shape `(n_timepoints, n_units)`. |
| `behavior` | `np.ndarray` | -- | Behavioral signal, shape `(n_timepoints,)`. |
| `max_lag` | `int` | `50` | Maximum lag in bins. |
| `bin_size` | `float` | `0.025` | Bin size in seconds (for converting lag to seconds). |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | One row per unit with columns: `unit_idx`, `peak_lag_bins`, `peak_lag_s`, `peak_corr`. |

**Example:**

```python
from cross_correlation import population_crosscorrelation

unit_xcorr = population_crosscorrelation(pop, behavior, max_lag=40, bin_size=0.025)
print(unit_xcorr.describe())
# Shows distribution of peak lags and correlations across all units
```

---

### `sliding_correlation`

```python
def sliding_correlation(
    neural: np.ndarray,
    behavior: np.ndarray,
    window_size: int = 100,
    step: int = 10,
) -> Dict[str, np.ndarray]
```

Compute Pearson correlation in sliding windows over time. Reveals **when**
neural-behavior coupling is strong versus weak during a session.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neural` | `np.ndarray` | -- | Neural signal, shape `(n_timepoints,)`. |
| `behavior` | `np.ndarray` | -- | Behavioral signal, shape `(n_timepoints,)`. |
| `window_size` | `int` | `100` | Number of bins per window. |
| `step` | `int` | `10` | Step size between consecutive windows. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, np.ndarray]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `window_centers` | `np.ndarray` | Center index of each window. |
| `correlations` | `np.ndarray` | Pearson r value for each window. |
| `p_values` | `np.ndarray` | p-value for each window. |

!!! note
    Windows where either signal has zero variance or fewer than 10 valid
    (non-NaN) data points are silently skipped.

**Example:**

```python
from cross_correlation import sliding_correlation

slide = sliding_correlation(pop_rate, behavior, window_size=200, step=50)
sig_windows = np.sum(slide["p_values"] < 0.05)
print(f"{sig_windows} / {len(slide['p_values'])} windows significantly correlated")
```

---

### `fit_encoding_model`

```python
def fit_encoding_model(
    behavior_features: pd.DataFrame,
    neural_target: np.ndarray,
    model_type: str = "poisson",
    n_folds: int = 5,
    lags: List[int] | None = None,
) -> Dict[str, Any]
```

Fit an encoding model: predict neural activity from behavioral features.
Answers: "Given what the animal is doing, can we predict neural firing?"

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `behavior_features` | `pd.DataFrame` | -- | Behavioral variables (columns like `pose_speed`, `pupil_area`, etc.). Should **not** include `t`, `session_id`, or `camera` columns (they are auto-dropped). |
| `neural_target` | `np.ndarray` | -- | 1-D target array (spike count or firing rate). |
| `model_type` | `str` | `"poisson"` | `"poisson"` for Poisson regression (GLM, suited for spike counts) or `"ridge"` for Ridge regression (suited for continuous rates). |
| `n_folds` | `int` | `5` | Number of time-blocked cross-validation folds. |
| `lags` | `List[int] \| None` | `None` | If provided, adds time-lagged copies of all features. E.g., `[-2, -1, 0, 1, 2]` adds 4 lagged versions of each feature. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `model` | `sklearn estimator \| None` | Final model fitted on all data. |
| `cv_scores` | `list[float]` | Per-fold R-squared values. |
| `feature_importance` | `np.ndarray \| None` | Absolute coefficient values of the final model. |
| `mean_r2` | `float` | Mean R-squared across CV folds (`NaN` if no folds succeeded). |

**Example:**

```python
from cross_correlation import fit_encoding_model

enc = fit_encoding_model(
    behavior_features=beh_df[["pose_speed", "pupil_area"]],
    neural_target=pop_rate,
    model_type="ridge",
    lags=[-2, -1, 0, 1, 2],
)
print(f"Encoding R2 = {enc['mean_r2']:.3f}")
print(f"CV scores: {enc['cv_scores']}")
```

---

### `fit_decoding_model`

```python
def fit_decoding_model(
    pop_matrix: np.ndarray,
    behavior_target: np.ndarray,
    n_folds: int = 5,
    lags: List[int] | None = None,
) -> Dict[str, Any]
```

Fit a decoding model: predict behavior from neural activity. Answers:
"Given neural activity, can we predict what the animal is doing?"

Uses Ridge regression with time-blocked cross-validation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop_matrix` | `np.ndarray` | -- | Population activity matrix, shape `(n_timepoints, n_units)`. |
| `behavior_target` | `np.ndarray` | -- | 1-D behavioral variable to predict. |
| `n_folds` | `int` | `5` | Number of CV folds. |
| `lags` | `List[int] \| None` | `None` | Time lags to add (same as `fit_encoding_model`). |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary with keys `model`, `cv_scores`, `mean_r2`, `feature_importance`. Same structure as `fit_encoding_model`. |

**Example:**

```python
from cross_correlation import fit_decoding_model

dec = fit_decoding_model(
    pop_matrix=pop,
    behavior_target=behavior_signal,
    lags=[-2, -1, 0, 1, 2],
)
print(f"Decoding R2 = {dec['mean_r2']:.3f}")
```

---

### `granger_test`

```python
def granger_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 10,
) -> Dict[str, Any]
```

Simple Granger causality test. Tests whether `cause` helps predict `effect`
beyond the effect's own history, by comparing a restricted autoregressive
model to a full model that includes lagged values of `cause`.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cause` | `np.ndarray` | -- | Putative causal signal (1-D). |
| `effect` | `np.ndarray` | -- | Effect signal (1-D). |
| `max_lag` | `int` | `10` | Number of lag terms in the autoregressive models. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Result dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `f_statistic` | `float` | F-statistic comparing the full vs restricted model. |
| `p_value` | `float` | p-value from the F-distribution. |
| `r2_restricted` | `float` | R-squared of the autoregressive-only model. |
| `r2_full` | `float` | R-squared of the full model (with cause lags). |
| `improvement` | `float` | `r2_full - r2_restricted`, the R-squared gain from adding the cause. |

!!! info "Interpretation"
    - Low p-value (< 0.05) suggests the `cause` signal Granger-causes the
      `effect`.
    - Run in both directions to check if neural causes behavior or vice versa.
    - Uses Ridge regression (alpha=0.1) for numerical stability.

**Example:**

```python
from cross_correlation import granger_test

# Does neural activity Granger-cause behavior?
gc_n2b = granger_test(pop_rate, behavior, max_lag=10)
print(f"Neural -> Behavior: F={gc_n2b['f_statistic']:.2f}, p={gc_n2b['p_value']:.4f}")

# Does behavior Granger-cause neural activity?
gc_b2n = granger_test(behavior, pop_rate, max_lag=10)
print(f"Behavior -> Neural: F={gc_b2n['f_statistic']:.2f}, p={gc_b2n['p_value']:.4f}")
```

---

### `compute_neural_behavior_alignment`

```python
def compute_neural_behavior_alignment(
    spike_times_dict: Dict[str, np.ndarray],
    behavior_df: pd.DataFrame,
    trials: pd.DataFrame | None = None,
    bin_size: float = 0.025,
    behavior_col: str = "pose_speed",
    max_lag_bins: int = 40,
) -> Dict[str, Any]
```

Run the **full** neural-behavior correlation analysis pipeline. This is the
top-level convenience function that combines cross-correlation, sliding
correlation, encoding, decoding, and Granger causality into a single call.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spike_times_dict` | `Dict[str, np.ndarray]` | -- | Unit ID to spike times mapping. |
| `behavior_df` | `pd.DataFrame` | -- | Behavioral features DataFrame with `t` and behavior columns. |
| `trials` | `pd.DataFrame \| None` | `None` | Trial table with `t` column (for trial-averaged analysis). |
| `bin_size` | `float` | `0.025` | Time-bin width in seconds. |
| `behavior_col` | `str` | `"pose_speed"` | Which column in `behavior_df` to use as the primary behavioral signal. |
| `max_lag_bins` | `int` | `40` | Maximum cross-correlation lag in bins. |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Comprehensive results dictionary: |

| Key | Type | Description |
|-----|------|-------------|
| `bin_size` | `float` | Bin size used. |
| `n_units` | `int` | Number of neural units. |
| `n_timebins` | `int` | Number of time bins. |
| `time_range` | `tuple` | `(t_start, t_end)`. |
| `crosscorrelation` | `dict` | Population-level cross-correlation result. |
| `peak_lag_s` | `float` | Peak lag in seconds. |
| `peak_corr` | `float` | Peak correlation value. |
| `unit_crosscorrelation` | `pd.DataFrame` | Per-unit cross-correlation summary. |
| `sliding_correlation` | `dict` | Sliding-window result (if enough data). |
| `encoding` | `dict` | Encoding model results (`mean_r2`, `cv_scores`). |
| `decoding` | `dict` | Decoding model results (`mean_r2`, `cv_scores`). |
| `granger_neural_to_behavior` | `dict` | Granger test: neural causes behavior. |
| `granger_behavior_to_neural` | `dict` | Granger test: behavior causes neural. |
| `trial_averaged_*` | `dict` | Trial-averaged analysis (if trials provided, for each grouping column). |

**Example:**

```python
from cross_correlation import compute_neural_behavior_alignment

results = compute_neural_behavior_alignment(
    spike_times_dict=spikes,
    behavior_df=pose_features,
    trials=trials,
    behavior_col="pose_speed",
)

print(f"Peak correlation: {results['peak_corr']:.3f} at lag {results['peak_lag_s']:.3f}s")
print(f"Encoding R2:  {results['encoding']['mean_r2']:.3f}")
print(f"Decoding R2:  {results['decoding']['mean_r2']:.3f}")
print(f"Granger N->B: p={results['granger_neural_to_behavior']['p_value']:.4f}")
print(f"Granger B->N: p={results['granger_behavior_to_neural']['p_value']:.4f}")
```
