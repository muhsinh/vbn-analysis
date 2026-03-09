# Phase 4: Neural-Behavior Correlation

Phase 4 spans **Notebooks 08-09** and answers the central scientific question: **do changes in behavior align with changes in neural activity?** This phase fuses all upstream artifacts into a common time grid and applies a battery of statistical and modeling analyses.

---

## Fusion Table Construction

Before any analysis can run, neural spike times and behavior features must be aligned to a **common time grid**. The `build_fusion_table()` function in `modeling.py` handles this:

```python title="src/modeling.py:build_fusion_table()"
def build_fusion_table(spike_times, motifs, bin_size_s):
    # Determine time range from neural data (preferred) or behavior
    if spike_times:
        all_times = np.concatenate(list(spike_times.values()))
        t_start, t_end = float(all_times.min()), float(all_times.max())
    elif motifs is not None and not motifs.empty:
        t_start, t_end = float(motifs["t"].min()), float(motifs["t"].max())
    else:
        t_start, t_end = 0.0, 10.0

    time_grid = build_time_grid(t_start, t_end, bin_size_s)     # (1)!
    spike_counts = bin_spike_times(                              # (2)!
        spike_times or {}, time_grid, bin_size_s
    )

    if motifs is not None and not motifs.empty:
        motifs_binned = bin_continuous_features(motifs, time_grid)  # (3)!
    else:
        motifs_binned = pd.DataFrame({"t": time_grid})

    fusion = spike_counts.merge(motifs_binned, on="t", how="left")  # (4)!
    return fusion
```

1. Creates a uniform time grid: `t_start, t_start + bin_size, t_start + 2*bin_size, ...`
2. For each unit, counts spikes falling in each bin using `np.histogram`
3. Bins continuous features (pose speed, pupil, etc.) by assigning each sample to the nearest time bin and averaging within each bin
4. Left-joins neural and behavioral columns on the `t` column

### How `build_time_grid` works

```python title="src/timebase.py:build_time_grid()"
def build_time_grid(start, end, bin_size_s):
    n_bins = int(np.floor((end - start) / bin_size_s))
    return start + np.arange(n_bins) * bin_size_s
```

With the default `bin_size_s=0.025` (25 ms), a 1-hour session produces approximately 144,000 time bins.

### How `bin_spike_times` works

```python title="src/timebase.py:bin_spike_times()"
def bin_spike_times(spike_times, time_grid, bin_size_s):
    counts = {}
    for unit_id, times in spike_times.items():
        bins = np.append(time_grid, time_grid[-1] + bin_size_s)
        counts[unit_id], _ = np.histogram(times, bins=bins)  # (1)!
    df = pd.DataFrame(counts)
    df.insert(0, "t", time_grid)
    return df
```

1. Each bin edge is `[t_i, t_i + bin_size)`. A spike at time 1.037 with bin edges `[1.025, 1.050)` falls in that bin. The result is a spike **count** (integer) per bin per unit.

### How `bin_continuous_features` works

```python title="src/timebase.py:bin_continuous_features()"
def bin_continuous_features(df, time_grid, agg="mean"):
    df = df.copy()
    df["bin"] = np.searchsorted(time_grid, df["t"].to_numpy(),
                                side="right") - 1          # (1)!
    df = df[df["bin"].between(0, len(time_grid) - 1)]
    grouped = df.groupby("bin").agg(agg)                   # (2)!
    grouped = grouped.reindex(range(len(time_grid)),
                              fill_value=np.nan)           # (3)!
    grouped.insert(0, "t", time_grid)
    return grouped.reset_index(drop=True)
```

1. `np.searchsorted` assigns each continuous sample to the bin whose left edge is closest
2. Multiple samples in the same bin are averaged (default) or aggregated by the specified function
3. Bins with no data get NaN

!!! note "Fusion table shape"
    The resulting DataFrame has one row per time bin and columns for:

    - `t`: bin center time
    - One column per neural unit (spike counts)
    - Behavioral feature columns (`pose_speed`, `pupil_z`, `is_still`, etc.)

---

## Peri-Event Time Histograms (PETHs)

A PETH answers: **how does a neuron's firing rate change around a behavioral event?** It is the neural analog of an event-related potential (ERP).

### What a PETH is, mathematically

Given a set of event times $\{e_1, e_2, \ldots, e_N\}$ and a unit's spike train $S$:

1. For each event $e_k$, extract all spikes in the window $[e_k + t_{\text{pre}},\; e_k + t_{\text{post}})$
2. Compute relative spike times: $\tau = t_{\text{spike}} - e_k$
3. Histogram these relative times into bins of width $\Delta t$
4. Divide by $\Delta t$ to convert counts to firing rate (Hz)
5. Average across all $N$ trials

The result is a trial-averaged firing rate as a function of time relative to the event.

### How `compute_peth()` works

```python title="src/neural_events.py:compute_peth()"
def compute_peth(spike_times, event_times,
                 window=(-0.5, 1.0), bin_size=0.01):
    pre, post = window
    n_bins = int(np.round((post - pre) / bin_size))
    bin_edges = np.linspace(pre, post, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2    # (1)!

    trial_counts = []
    trial_spikes_list = []

    for evt in event_times:
        relative = spike_times - evt                      # (2)!
        in_window = relative[(relative >= pre) & (relative < post)]
        trial_spikes_list.append(in_window)

        counts, _ = np.histogram(in_window, bins=bin_edges)  # (3)!
        trial_counts.append(counts)

    trial_counts = np.array(trial_counts)  # (n_trials, n_bins)
    rates = trial_counts / bin_size                       # (4)!

    return {
        "time_bins": bin_centers,
        "mean_rate": np.mean(rates, axis=0),              # (5)!
        "sem_rate": np.std(rates, axis=0) / np.sqrt(len(rates)),  # (6)!
        "trial_spikes": trial_spikes_list,
        "n_trials": len(event_times),
    }
```

1. Bin centers are used for plotting (e.g., -0.495, -0.485, ..., 0.995)
2. Subtract event time to get spike times relative to the event
3. Count spikes per bin for this single trial
4. Convert from counts to Hz: $\text{rate} = \text{count} / \Delta t$
5. Trial-averaged firing rate per bin
6. Standard error of the mean across trials

!!! example "Interpreting a PETH"
    ```python
    peth = compute_peth(
        spike_times=spikes["42"],
        event_times=trials["t"].to_numpy(),
        window=(-0.5, 1.0),  # 500 ms before to 1000 ms after
        bin_size=0.01,        # 10 ms bins
    )

    # peth["time_bins"]  -> array of 150 bin centers from -0.495 to +0.995
    # peth["mean_rate"]  -> array of 150 firing rates in Hz
    # peth["n_trials"]   -> number of trials averaged over
    ```

    A peak in `mean_rate` at time +0.1 means the neuron fires maximally ~100 ms after the event.

### Population PETHs

`compute_population_peth()` runs `compute_peth()` for every unit and stacks the results:

```python title="src/neural_events.py:compute_population_peth()"
def compute_population_peth(spike_times_dict, event_times,
                            window=(-0.5, 1.0), bin_size=0.01,
                            unit_ids=None):
    if unit_ids is None:
        unit_ids = list(spike_times_dict.keys())

    peths = {}
    pop_matrix = []

    for uid in unit_ids:
        st = spike_times_dict.get(uid, np.array([]))
        peth = compute_peth(st, event_times, window, bin_size)
        peths[uid] = peth
        pop_matrix.append(peth["mean_rate"])

    pop_matrix = np.array(pop_matrix)  # (n_units, n_bins)

    return {
        "time_bins": peths[unit_ids[0]]["time_bins"],
        "population_matrix": pop_matrix,
        "unit_ids": unit_ids,
        "peths": peths,
    }
```

The `population_matrix` is an `(n_units, n_bins)` array that can be visualized as a heatmap to see how the entire population responds to an event.

### Trial-Averaged Rates by Condition

```python title="src/neural_events.py:trial_averaged_rates()"
def trial_averaged_rates(spike_times_dict, trials,
                         group_col="trial_type",
                         window=(-0.5, 1.0), bin_size=0.025):
    results = {}
    for condition, group in trials.groupby(group_col):
        event_times = group["t"].dropna().to_numpy()
        if len(event_times) < 2:
            continue
        results[str(condition)] = compute_population_peth(
            spike_times_dict, event_times, window, bin_size
        )
    return results
```

This lets you compare neural responses across conditions (e.g., "go" vs "no-go" trials, rewarded vs unrewarded).

---

## Cross-Correlation

Cross-correlation measures the **linear relationship between two signals as a function of time lag**. It answers: when the behavior signal changes, does the neural signal change before, after, or simultaneously?

### How it is computed

```python title="src/cross_correlation.py:crosscorrelation()"
def crosscorrelation(neural, behavior, max_lag=50, normalize=True):
    neural = np.asarray(neural, dtype=float)
    behavior = np.asarray(behavior, dtype=float)

    # Remove means
    neural = neural - np.nanmean(neural)               # (1)!
    behavior = behavior - np.nanmean(behavior)

    # Replace NaNs with 0
    neural = np.nan_to_num(neural)
    behavior = np.nan_to_num(behavior)

    n = len(neural)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            n_slice = neural[lag:]                     # (2)!
            b_slice = behavior[:n - lag]
        else:
            n_slice = neural[:n + lag]
            b_slice = behavior[-lag:]

        if normalize:
            std_n = np.std(n_slice)
            std_b = np.std(b_slice)
            if std_n > 0 and std_b > 0:
                corr[i] = np.mean(n_slice * b_slice) / (std_n * std_b)  # (3)!
        else:
            corr[i] = np.mean(n_slice * b_slice)

    peak_idx = np.argmax(np.abs(corr))
    return {
        "lags": lags,
        "correlation": corr,
        "peak_lag": int(lags[peak_idx]),               # (4)!
        "peak_corr": float(corr[peak_idx]),
    }
```

1. Mean-subtraction ensures we measure covariance, not shared offset
2. At positive lag $k$, we compare `neural[k:]` with `behavior[:n-k]`; that is, the neural signal is shifted forward in time
3. Normalized cross-correlation gives Pearson $r$ values in $[-1, 1]$
4. The lag with the strongest absolute correlation

!!! warning "Interpreting `peak_lag`"
    The lag convention is:

    - **Negative `peak_lag`** (e.g., -3): neural activity leads behavior by 3 bins. The brain acts first, behavior follows.
    - **Positive `peak_lag`** (e.g., +5): behavior leads neural activity by 5 bins. Sensory feedback or re-afferent signal.
    - **Zero `peak_lag`**: simultaneous correlation.

    To convert lag from bins to seconds: `peak_lag_seconds = peak_lag * bin_size`

### Per-Unit Cross-Correlation

```python title="src/cross_correlation.py:population_crosscorrelation()"
def population_crosscorrelation(pop_matrix, behavior,
                                max_lag=50, bin_size=0.025):
    n_units = pop_matrix.shape[1]
    rows = []
    for i in range(n_units):
        result = crosscorrelation(pop_matrix[:, i], behavior, max_lag)
        rows.append({
            "unit_idx": i,
            "peak_lag_bins": result["peak_lag"],
            "peak_lag_s": result["peak_lag"] * bin_size,
            "peak_corr": result["peak_corr"],
        })
    return pd.DataFrame(rows)
```

The result is a DataFrame with one row per unit showing its peak correlation and lag. You can then:

- Sort by `|peak_corr|` to find the most behaviorally-coupled neurons
- Plot the distribution of `peak_lag_s` to see whether neural activity generally leads or lags behavior

---

## Sliding-Window Correlation

Cross-correlation gives you a single summary over the whole session. But neural-behavior coupling can change over time (e.g., stronger during active behavior, weaker during rest). Sliding-window correlation reveals **when** the coupling is strongest.

```python title="src/cross_correlation.py:sliding_correlation()"
def sliding_correlation(neural, behavior,
                        window_size=100, step=10):
    from scipy.stats import pearsonr

    n = min(len(neural), len(behavior))
    centers = []
    corrs = []
    pvals = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        n_win = neural[start:end]
        b_win = behavior[start:end]

        if np.std(n_win) < 1e-10 or np.std(b_win) < 1e-10:
            continue                                   # (1)!

        valid = np.isfinite(n_win) & np.isfinite(b_win)
        if valid.sum() < 10:
            continue

        r, p = pearsonr(n_win[valid], b_win[valid])    # (2)!
        centers.append((start + end) / 2)
        corrs.append(r)
        pvals.append(p)

    return {
        "window_centers": np.array(centers),
        "correlations": np.array(corrs),               # (3)!
        "p_values": np.array(pvals),
    }
```

1. Skip windows with no variance (e.g., periods of silence)
2. Pearson correlation within each window
3. Time series of correlation values; plot this to see coupling dynamics

!!! tip "Choosing window parameters"
    - `window_size=100` with 25 ms bins = 2.5 second windows
    - `step=10` = 250 ms step size
    - Larger windows give smoother but less temporally resolved estimates
    - The `p_values` array tells you which windows have statistically significant coupling

---

## Encoding Model: Behavior to Neural

The encoding model asks: **given what the animal is doing, can we predict neural firing?** This tests whether behavioral variables carry information about neural activity.

```python title="src/cross_correlation.py:fit_encoding_model()"
def fit_encoding_model(behavior_features, neural_target,
                       model_type="poisson", n_folds=5,
                       lags=None):
    X = behavior_features.copy()
    drop_cols = [c for c in X.columns if c in ("t", "session_id", "camera")]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    if lags is not None:
        X = _add_lags(X, lags)                         # (1)!

    X = X.fillna(0).to_numpy()
    y = np.asarray(neural_target, dtype=float)
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    # Time-blocked cross-validation
    fold_size = n // n_folds
    scores = []

    for i in range(n_folds):
        test_start = i * fold_size                     # (2)!
        test_end = min((i + 1) * fold_size, n)
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n),
        ])

        model = _make_encoding_model(model_type)       # (3)!
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        r2 = r2_score(y[test_idx], pred)
        scores.append(r2)

    # Final model on all data
    final_model = _make_encoding_model(model_type)
    final_model.fit(X, y)

    importance = np.abs(final_model.coef_) if hasattr(final_model, "coef_") else None

    return {
        "model": final_model,
        "cv_scores": scores,
        "feature_importance": importance,              # (4)!
        "mean_r2": float(np.mean(scores)),
    }
```

1. Adds time-lagged copies of each feature (e.g., `pose_speed_lag-2`, `pose_speed_lag-1`, `pose_speed_lag1`). This captures the fact that neural responses may lead or lag behavior.
2. **Time-blocked CV**: each fold is a contiguous time block, not random. This respects temporal autocorrelation.
3. `model_type="poisson"` uses `PoissonRegressor(alpha=0.01)` (for spike counts); `"ridge"` uses `Ridge(alpha=1.0)` (for firing rates)
4. Feature importance is the absolute value of model coefficients, which tells you which behavioral features are most predictive of neural activity

### How lag features are added

```python title="src/cross_correlation.py:_add_lags()"
def _add_lags(X, lags):
    dfs = [X]
    for lag in lags:
        if lag == 0:
            continue
        shifted = X.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in X.columns]
        dfs.append(shifted)
    result = pd.concat(dfs, axis=1)
    # Drop rows with NaN from shifting
    max_lag = max(abs(l) for l in lags)
    result = result.iloc[max_lag:-max_lag]
    return result.reset_index(drop=True)
```

With `lags=[-2, -1, 0, 1, 2]` and 25 ms bins, the model sees behavior from 50 ms before to 50 ms after each time point. This covers both predictive and feedback relationships.

!!! info "Interpreting encoding R^2"
    - `R^2 > 0.1`: meaningful encoding; behavior explains some variance in neural activity
    - `R^2 > 0.3`: strong encoding
    - `R^2 < 0.01`: behavior and neural activity are largely independent at this time scale

---

## Decoding Model: Neural to Behavior

The decoding model asks the complementary question: **given neural activity, can we predict what the animal is doing?**

```python title="src/cross_correlation.py:fit_decoding_model()"
def fit_decoding_model(pop_matrix, behavior_target,
                       n_folds=5, lags=None):
    X = pd.DataFrame(
        pop_matrix,
        columns=[f"unit_{i}" for i in range(pop_matrix.shape[1])]
    )

    if lags is not None:
        X = _add_lags(X, lags)

    X_arr = np.nan_to_num(X.to_numpy())
    y = np.asarray(behavior_target, dtype=float)

    # Remove NaN targets
    valid = np.isfinite(y)
    X_arr, y = X_arr[valid], y[valid]

    # Time-blocked CV (same structure as encoding)
    fold_size = len(y) // n_folds
    scores = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, len(y))
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, len(y)),
        ])
        model = Ridge(alpha=1.0)
        model.fit(X_arr[train_idx], y[train_idx])
        pred = model.predict(X_arr[test_idx])
        scores.append(r2_score(y[test_idx], pred))

    final_model = Ridge(alpha=1.0)
    final_model.fit(X_arr, y)
    importance = np.abs(final_model.coef_)

    return {
        "model": final_model,
        "cv_scores": scores,
        "mean_r2": float(np.mean(scores)),
        "feature_importance": importance,              # (1)!
    }
```

1. Feature importance here tells you which **units** are most informative for decoding behavior

!!! tip "Encoding vs Decoding: when to use which"
    | Question | Use |
    |----------|-----|
    | "Does this brain area represent locomotion speed?" | Decoding (neural -> pose_speed) |
    | "Does running speed modulate neural firing?" | Encoding (pose_speed -> neural) |
    | "Which units carry the most behavioral information?" | Decoding feature importance |
    | "Which behavioral features drive neural activity?" | Encoding feature importance |

---

## Granger Causality

Granger causality tests whether one signal **helps predict the future** of another signal beyond its own history. Unlike cross-correlation (which only measures co-occurrence), Granger causality measures **predictive information flow**.

### How it works

The test compares two models:

- **Restricted model**: predict the effect from its own past only

$$
\text{effect}_t = \sum_{k=1}^{K} \beta_k \cdot \text{effect}_{t-k} + \epsilon
$$

- **Full model**: predict the effect from its own past **plus** the cause's past

$$
\text{effect}_t = \sum_{k=1}^{K} \beta_k \cdot \text{effect}_{t-k} + \sum_{k=1}^{K} \gamma_k \cdot \text{cause}_{t-k} + \epsilon
$$

If the full model is significantly better, the cause "Granger-causes" the effect.

```python title="src/cross_correlation.py:granger_test()"
def granger_test(cause, effect, max_lag=10):
    cause = np.asarray(cause, dtype=float)
    effect = np.asarray(effect, dtype=float)
    n = min(len(cause), len(effect))

    y = effect[max_lag:]

    # Restricted: effect's own history
    X_restricted = np.column_stack([
        effect[max_lag - i - 1: n - i - 1]
        for i in range(max_lag)
    ])                                                 # (1)!

    # Full: effect's history + cause's history
    X_cause_lags = np.column_stack([
        cause[max_lag - i - 1: n - i - 1]
        for i in range(max_lag)
    ])
    X_full = np.column_stack([X_restricted, X_cause_lags])  # (2)!

    # Fit both models
    model_r = Ridge(alpha=0.1)
    model_f = Ridge(alpha=0.1)
    model_r.fit(X_restricted, y)
    model_f.fit(X_full, y)

    pred_r = model_r.predict(X_restricted)
    pred_f = model_f.predict(X_full)

    ss_r = np.sum((y - pred_r) ** 2)                   # (3)!
    ss_f = np.sum((y - pred_f) ** 2)

    r2_r = r2_score(y, pred_r)
    r2_f = r2_score(y, pred_f)

    # F-test
    df1 = max_lag                                      # (4)!
    df2 = len(y) - 2 * max_lag
    f_stat = ((ss_r - ss_f) / df1) / (ss_f / df2)

    from scipy.stats import f as f_dist
    p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)      # (5)!

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "r2_restricted": float(r2_r),
        "r2_full": float(r2_f),
        "improvement": float(r2_f - r2_r),            # (6)!
    }
```

1. Build the lagged design matrix for the restricted model: columns are `effect[t-1], effect[t-2], ..., effect[t-K]`
2. Full model has twice as many columns: the effect's own lags plus the cause's lags
3. Sum of squared residuals for each model (smaller = better fit)
4. Degrees of freedom: `df1 = K` (extra parameters in full model), `df2 = N - 2K`
5. F-test p-value: probability of seeing this improvement by chance
6. `improvement = R^2_full - R^2_restricted`: how much additional variance is explained by the cause

!!! info "The pipeline tests both directions"
    In `compute_neural_behavior_alignment()`, Granger causality is tested **both ways**:

    ```python
    gc_n2b = granger_test(pop_rate, beh_signal, max_lag=10)  # neural -> behavior
    gc_b2n = granger_test(beh_signal, pop_rate, max_lag=10)  # behavior -> neural
    ```

    | Result | Interpretation |
    |--------|----------------|
    | `gc_n2b` significant, `gc_b2n` not | Neural activity predicts future behavior (motor command) |
    | `gc_b2n` significant, `gc_n2b` not | Behavior predicts future neural activity (sensory feedback) |
    | Both significant | Bidirectional coupling (expected in many circuits) |
    | Neither significant | No predictive relationship at this lag/timescale |

---

## Unit Selectivity Screening

Before running expensive models, it is useful to screen which units are selective for specific behavioral conditions. The pipeline computes d-prime and a Mann-Whitney U test for each unit.

### `compute_selectivity_index()`

```python title="src/neural_events.py:compute_selectivity_index()"
def compute_selectivity_index(spike_times, condition_a_times,
                              condition_b_times, window=(0.0, 0.5)):
    def _mean_rate(event_times):
        rates = []
        for evt in event_times:
            n = np.sum(
                (spike_times >= evt + window[0]) &
                (spike_times < evt + window[1])
            )
            rates.append(n / (window[1] - window[0]))  # (1)!
        return np.array(rates)

    rates_a = _mean_rate(condition_a_times)
    rates_b = _mean_rate(condition_b_times)

    mean_a, mean_b = np.mean(rates_a), np.mean(rates_b)
    var_a = np.var(rates_a, ddof=1)
    var_b = np.var(rates_b, ddof=1)
    pooled_std = np.sqrt((var_a + var_b) / 2) + 1e-8

    d_prime = (mean_a - mean_b) / pooled_std           # (2)!

    from scipy.stats import mannwhitneyu
    _, p_val = mannwhitneyu(rates_a, rates_b,
                            alternative="two-sided")   # (3)!

    return {
        "d_prime": float(d_prime),
        "rate_diff": float(mean_a - mean_b),
        "mean_rate_a": float(mean_a),
        "mean_rate_b": float(mean_b),
        "p_value": float(p_val),
    }
```

1. For each event, count spikes in the post-event window and convert to firing rate (Hz)
2. **d-prime**: standardized difference between conditions. $d' = \frac{\mu_A - \mu_B}{\sigma_{\text{pooled}}}$
3. **Mann-Whitney U**: non-parametric test that does not assume normality

### Batch screening

```python title="src/neural_events.py:screen_selective_units()"
def screen_selective_units(spike_times_dict, condition_a_times,
                           condition_b_times, window=(0.0, 0.5),
                           p_threshold=0.05):
    rows = []
    for uid, st in spike_times_dict.items():
        sel = compute_selectivity_index(
            st, condition_a_times, condition_b_times, window
        )
        sel["unit_id"] = uid
        rows.append(sel)

    df = pd.DataFrame(rows)
    df["significant"] = df["p_value"] < p_threshold    # (1)!
    df["abs_d_prime"] = np.abs(df["d_prime"])
    df = df.sort_values("abs_d_prime", ascending=False)  # (2)!
    return df
```

1. Flag units with p < 0.05 (or your chosen threshold)
2. Sort by absolute d-prime: the most selective units come first

!!! tip "Interpreting d-prime"
    | |d'| | Interpretation |
    |------|----------------|
    | < 0.2 | Negligible selectivity |
    | 0.2 - 0.5 | Small selectivity |
    | 0.5 - 0.8 | Medium selectivity |
    | > 0.8 | Large selectivity |
    | > 2.0 | Extremely selective; this unit strongly discriminates the two conditions |

---

## The Full Pipeline: `compute_neural_behavior_alignment()`

The convenience function `compute_neural_behavior_alignment()` runs the complete analysis battery in a single call:

```python title="src/cross_correlation.py:compute_neural_behavior_alignment() (abbreviated)"
def compute_neural_behavior_alignment(
    spike_times_dict, behavior_df, trials=None,
    bin_size=0.025, behavior_col="pose_speed",
    max_lag_bins=40,
):
    # 1. Build time grid and population matrix
    time_grid = build_time_grid(t_start, t_end, bin_size)
    pop = build_population_vectors(spike_times_dict, time_grid, bin_size)
    pop_rate = pop.sum(axis=1)                         # (1)!

    # 2. Bin behavior signal
    beh_binned = bin_continuous_features(
        behavior_df[["t", behavior_col]], time_grid
    )
    beh_signal = beh_binned[behavior_col].to_numpy()

    # 3. Cross-correlation
    xcorr = crosscorrelation(pop_rate, beh_signal, max_lag=max_lag_bins)

    # 4. Per-unit cross-correlation
    unit_xcorr = population_crosscorrelation(
        pop, beh_signal, max_lag_bins, bin_size
    )

    # 5. Sliding-window correlation
    slide = sliding_correlation(pop_rate, beh_signal, ...)

    # 6. Encoding model: behavior -> neural
    enc = fit_encoding_model(
        beh_features, pop_rate,
        model_type="ridge", lags=[-2, -1, 0, 1, 2],
    )

    # 7. Decoding model: neural -> behavior
    dec = fit_decoding_model(
        pop, beh_signal, lags=[-2, -1, 0, 1, 2],
    )

    # 8. Granger causality (both directions)
    gc_n2b = granger_test(pop_rate, beh_signal, max_lag=10)
    gc_b2n = granger_test(beh_signal, pop_rate, max_lag=10)

    # 9. Trial-averaged PETHs
    for group_col in ["trial_type", "rewarded"]:
        tavg = trial_averaged_rates(spike_times_dict, trials, group_col)

    return results
```

1. `pop_rate` is the total population firing rate (sum across all units per bin), a single time series summarizing the population

### Results Dictionary Structure

```python
results = {
    "bin_size": 0.025,
    "n_units": 42,
    "n_timebins": 144000,
    "time_range": (0.0, 3600.0),

    # Cross-correlation
    "crosscorrelation": {"lags": ..., "correlation": ..., "peak_lag": -3, "peak_corr": 0.42},
    "peak_lag_s": -0.075,        # peak lag in seconds
    "peak_corr": 0.42,

    # Per-unit
    "unit_crosscorrelation": DataFrame(unit_idx, peak_lag_bins, peak_lag_s, peak_corr),

    # Sliding window
    "sliding_correlation": {"window_centers": ..., "correlations": ..., "p_values": ...},

    # Models
    "encoding": {"mean_r2": 0.15, "cv_scores": [0.12, 0.14, 0.16, 0.18, 0.13]},
    "decoding": {"mean_r2": 0.22, "cv_scores": [0.20, 0.21, 0.23, 0.24, 0.22]},

    # Granger causality
    "granger_neural_to_behavior": {"f_statistic": 8.3, "p_value": 0.001, "improvement": 0.04},
    "granger_behavior_to_neural": {"f_statistic": 3.1, "p_value": 0.02, "improvement": 0.02},

    # Trial-averaged (if trials provided)
    "trial_averaged_trial_type": {"go": {...}, "no-go": {...}},
    "trial_averaged_rewarded": {"True": {...}, "False": {...}},
}
```

---

## Notebook 09: QC Checklist

Notebook 09 consolidates all QC information into a final checklist. It does not run new analyses; it reviews the outputs of all previous phases.

### What it checks

| Check | Source | Pass criteria |
|-------|--------|---------------|
| Config snapshot exists | `outputs/reports/config_snapshot.json` | File exists and is valid JSON |
| Sessions loaded | `sessions.csv` | At least one session with a valid ID |
| Neural data extracted | `outputs/neural/` | Units parquet and spike_times.npz exist per session |
| Behavior extracted | `outputs/behavior/` | Trials and events parquet files exist |
| Eye features derived | `outputs/eye/` | Eye features parquet exists (or `eye_unavailable` QC flag) |
| Video assets registered | `outputs/video/video_assets.parquet` | At least one camera per session |
| Frame times valid | `outputs/video/frame_times.parquet` | No `NON_MONOTONIC` or `NO_VALID_TIMESTAMPS` flags |
| Pose predictions exist | `outputs/pose/` | Pose predictions parquet exists |
| Fusion table built | Computed in Notebook 08 | DataFrame is non-empty, has both neural and behavioral columns |
| Cross-correlation computed | Notebook 08 results | `peak_corr` is finite |
| Encoding/decoding R^2 | Notebook 08 results | `mean_r2` is finite and > 0 |
| Granger causality | Notebook 08 results | `p_value` is finite |
| QC flags | `SessionBundle.qc_flags` | List all accumulated flags for review |

---

## Interpreting Results: What Do These Numbers Mean?

### Cross-correlation peak

- **peak_lag = -3 bins (-75 ms)**: neural activity precedes the behavioral change by 75 ms. This is consistent with the brain driving motor output.
- **peak_corr = 0.42**: moderate positive correlation. When neural activity increases, behavior (e.g., movement speed) increases ~75 ms later.

### Encoding vs Decoding R^2

- **Encoding R^2 = 0.15**: behavior explains 15% of variance in population firing rate. The rest is driven by internal dynamics, other sensory inputs, or noise.
- **Decoding R^2 = 0.22**: neural activity explains 22% of variance in behavior. This is often higher than encoding because the population has more dimensions (one per unit) than the behavioral signal.

### Granger causality

- **Neural -> Behavior: F=8.3, p=0.001**: strong evidence that neural activity carries information about future behavior. Consistent with motor commands.
- **Behavior -> Neural: F=3.1, p=0.02**: weaker but significant evidence that behavior predicts future neural activity. Consistent with sensory re-afference or proprioceptive feedback.

### d-prime for unit selectivity

- A unit with **d' = 1.5, p < 0.001** between "go" and "no-go" trials is strongly selective for trial type. This unit fires significantly differently depending on the animal's decision.
- A unit with **d' = 0.1, p = 0.6** shows no preference. It responds similarly regardless of trial type.

!!! success "Putting it all together"
    A complete narrative might read:

    > The population shows a peak cross-correlation with locomotion speed at -75 ms (neural leads), with r = 0.42. An encoding model using pose speed and pupil diameter explains 15% of population firing rate variance (Ridge regression, 5-fold time-blocked CV). Granger causality analysis confirms bidirectional information flow, with stronger neural-to-behavior coupling (F=8.3) than behavior-to-neural (F=3.1). Of 42 recorded units, 12 show significant selectivity (p < 0.05) for trial type, with d' values ranging from 0.5 to 2.1.
