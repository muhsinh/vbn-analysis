# Neural-Behavior Correlation Guide

This guide explains how to interpret the correlation analyses produced by Notebook 08 and the `cross_correlation` module. It covers what each analysis tells you, what constitutes a meaningful result, and what to do when results are weak.

---

## The Core Question

> **Do changes in behavior align with changes in neural activity?**

Notebook 08 answers this question using seven complementary analyses. Each tells you something different about the neural-behavior relationship. No single analysis is sufficient on its own; you should look at all of them together to build a complete picture.

---

## Analysis Overview

| Analysis | What It Tells You | Key Output |
|---|---|---|
| **PETHs** | How firing rates change around behavioral events | Mean firing rate over time, locked to events |
| **Cross-correlation** | Time lag between neural and behavior signals | Peak lag (who leads?) and peak correlation |
| **Sliding-window correlation** | When during the session coupling is strongest | Time-varying correlation trace |
| **Encoding model** | Can behavior predict neural firing? | R-squared (cross-validated) |
| **Decoding model** | Can neural activity predict behavior? | R-squared (cross-validated) |
| **Granger causality** | Does one signal *cause* the other? | F-statistic and R-squared improvement |
| **Unit selectivity** | Which neurons differentiate trial types? | d-prime and p-value per unit |

---

## Peri-Event Time Histograms (PETHs)

### What It Does

PETHs show the **average firing rate of a neuron around a behavioral event** (e.g., stimulus onset, lick, reward delivery). They are the most intuitive way to visualize how a neuron responds to events.

### How to Run

```python
from neural_events import compute_peth, compute_population_peth
import numpy as np

# Single unit
peth = compute_peth(
    spike_times=spike_times["unit_42"],
    event_times=stimulus_onset_times,
    window=(-0.5, 1.0),   # 500 ms before to 1 s after event
    bin_size=0.01,         # 10 ms bins
)

# Population (all units at once)
pop_peth = compute_population_peth(
    spike_times_dict=spike_times,
    event_times=stimulus_onset_times,
    window=(-0.5, 1.0),
    bin_size=0.01,
)
```

### Interpreting Results

```python
from viz import plot_peth, plot_population_peth

plot_peth(peth, unit_id="unit_42")
plot_population_peth(pop_peth, title="Population response to stimulus")
```

| What You See | What It Means |
|---|---|
| Sharp increase after t=0 | Neuron responds to the event (excitatory response) |
| Sharp decrease after t=0 | Neuron is inhibited by the event |
| Gradual ramp before t=0 | Neuron may be anticipating the event (or responding to an earlier cue) |
| Flat line | No detectable response to this event type |
| Large SEM shading | Response is variable across trials (may need more trials) |

!!! tip "Choosing the Right Events"
    The VBN dataset includes multiple event types. Try aligning to:

    - **Stimulus onset**: `trials["t_start"]`, when the visual stimulus changes
    - **Lick times**: behavioral events from the NWB processing module
    - **Reward delivery**: filter trials where `rewarded == True`

    Different neurons may respond to different events. Try all of them.

### Trial-Averaged Comparison

Compare PETHs across conditions (e.g., hit vs miss, go vs no-go):

```python
from neural_events import trial_averaged_rates
from viz import plot_trial_comparison

tavg = trial_averaged_rates(
    spike_times_dict=spike_times,
    trials=trials_df,
    group_col="trial_type",    # or "rewarded"
    window=(-0.5, 1.0),
    bin_size=0.025,
)

plot_trial_comparison(tavg, unit_id="unit_42")
```

If the traces for different conditions diverge after the event, the neuron is **selective** for that task variable.

---

## Cross-Correlation

### What It Does

Cross-correlation measures the **time-lagged Pearson correlation** between a neural signal (e.g., population firing rate) and a behavioral signal (e.g., pose speed). The lag at which correlation peaks tells you **who leads**: the brain or the behavior.

### How to Run

```python
from cross_correlation import crosscorrelation, population_crosscorrelation
from viz import plot_crosscorrelation

# Population rate vs pose speed
xcorr = crosscorrelation(
    neural=population_firing_rate,
    behavior=pose_speed_binned,
    max_lag=50,       # +/- 50 bins
    normalize=True,   # correlation coefficients [-1, 1]
)

plot_crosscorrelation(xcorr, bin_size=0.025)
```

### Interpreting the Peak Lag

The `peak_lag` value tells you the temporal relationship:

| Peak Lag | Meaning |
|---|---|
| **Negative** (e.g., -5 bins = -125 ms) | **Neural leads behavior**: neural activity changes happen *before* the behavioral change. Suggests the brain is driving the behavior. |
| **Zero** (0 bins) | Neural and behavioral changes are **simultaneous** at this time resolution. Could indicate a shared external drive. |
| **Positive** (e.g., +3 bins = +75 ms) | **Behavior leads neural**: behavioral changes happen *before* the neural change. Suggests sensory feedback or reafference. |

!!! info "Lag Convention"
    In this pipeline, **negative lag = neural leads behavior**. This follows the convention: at lag $k$, we correlate `neural[t+k]` with `behavior[t]`.

### What Is a "Good" Peak Correlation?

| |peak_corr| | Interpretation |
|---|---|
| < 0.05 | No meaningful relationship |
| 0.05 - 0.15 | Weak relationship (common for single units) |
| 0.15 - 0.30 | Moderate relationship (typical for population rate) |
| > 0.30 | Strong relationship (notable finding) |

!!! warning "Context Matters"
    These thresholds are rough guidelines for VBN data. A correlation of 0.10 with a clear peak at a biologically plausible lag (-50 to -200 ms for motor commands) is more meaningful than a correlation of 0.25 with no clear peak structure.

### Per-Unit Cross-Correlation

```python
from viz import plot_unit_lag_distribution

unit_xcorr = population_crosscorrelation(
    pop_matrix=population_matrix,
    behavior=pose_speed_binned,
    max_lag=50,
    bin_size=0.025,
)

plot_unit_lag_distribution(unit_xcorr, bin_size=0.025)
```

The distribution of peak lags across units tells you:

- **Clustered near a single lag**: Most units have a consistent temporal relationship with behavior
- **Bimodal**: Two subpopulations with different timing (e.g., motor vs sensory neurons)
- **Uniformly spread**: No consistent population-level relationship (individual unit effects may still exist)

---

## Sliding-Window Correlation

### What It Does

Instead of computing a single correlation for the entire session, this analysis computes Pearson correlation in **sliding windows** across time. It reveals **when** neural-behavior coupling is strong vs weak.

### How to Run

```python
from cross_correlation import sliding_correlation
from viz import plot_sliding_correlation

slide = sliding_correlation(
    neural=population_firing_rate,
    behavior=pose_speed_binned,
    window_size=200,   # 200 bins = 5 seconds at 25 ms bins
    step=50,           # 50-bin step = 1.25 seconds
)

plot_sliding_correlation(slide, bin_size=0.025)
```

### Interpreting Results

| Pattern | Interpretation |
|---|---|
| Sustained high correlation | Neural-behavior coupling is stable throughout |
| Correlation peaks during task periods | Coupling is task-dependent (engaged vs disengaged) |
| Correlation starts high, decays | Animal may habituate or disengage over time |
| Random fluctuation around zero | No consistent relationship at this time scale |

The p-value plot (lower panel) shows which windows have statistically significant correlations (red dots, p < 0.05).

!!! tip "Window Size Selection"
    - **Small windows** (50-100 bins, 1-2.5 s): detect rapid changes in coupling, but noisier
    - **Large windows** (200-500 bins, 5-12.5 s): smoother estimates, but may miss brief coupling events
    - Start with 200 bins and adjust based on your session duration and event timing

---

## Encoding Model (Behavior -> Neural)

### What It Does

Fits a regression model to predict **neural activity from behavioral features**. This asks: "Given what the animal is doing (moving, still, grooming), how well can we predict neural firing?"

### How to Run

```python
from cross_correlation import fit_encoding_model
from viz import plot_encoding_decoding

enc = fit_encoding_model(
    behavior_features=behavior_df[["pose_speed", "head_angle", "body_length"]],
    neural_target=population_firing_rate,
    model_type="ridge",    # or "poisson" for spike counts
    n_folds=5,             # time-blocked CV
    lags=[-2, -1, 0, 1, 2],  # include lagged features
)

print(f"Encoding R2: {enc['mean_r2']:.4f}")
print(f"CV scores:   {enc['cv_scores']}")
```

### Model Types

| Model Type | When to Use | Target Variable |
|---|---|---|
| `"ridge"` | Continuous firing rate | Smoothed firing rate or binned population rate |
| `"poisson"` | Spike counts | Raw spike counts per bin (non-negative integers) |

### Interpreting R-squared

| Mean R2 | Interpretation |
|---|---|
| < 0.01 | Behavior does not predict neural activity (at this resolution) |
| 0.01 -- 0.05 | Weak encoding (common for single units with simple features) |
| 0.05 -- 0.15 | Moderate encoding (behavior explains some variance) |
| 0.15 -- 0.30 | Strong encoding (behavior is a substantial driver) |
| > 0.30 | Very strong encoding (rare without including stimulus features) |

!!! note "Negative R2 Is Possible"
    A negative R2 means the model is worse than predicting the mean. This happens when there is no real relationship and the model overfits to noise. It is not a bug.

### Feature Importance

```python
import numpy as np

if enc["feature_importance"] is not None:
    # Absolute coefficient magnitudes from the final Ridge model
    importance = enc["feature_importance"]
    # Feature names depend on whether lags were added
    print("Feature importances (absolute coefficients):")
    for i, val in enumerate(importance):
        print(f"  Feature {i}: {val:.4f}")
```

---

## Decoding Model (Neural -> Behavior)

### What It Does

The inverse of encoding: fits a Ridge regression to predict **behavioral variables from neural population activity**. This asks: "Given the pattern of neural firing, can we reconstruct what the animal was doing?"

### How to Run

```python
from cross_correlation import fit_decoding_model

dec = fit_decoding_model(
    pop_matrix=population_matrix,        # (n_timepoints, n_units)
    behavior_target=pose_speed_binned,   # what to predict
    n_folds=5,
    lags=[-2, -1, 0, 1, 2],
)

print(f"Decoding R2: {dec['mean_r2']:.4f}")
```

### Encoding vs Decoding: What the Comparison Tells You

```python
plot_encoding_decoding(enc, dec)
```

| Comparison | Interpretation |
|---|---|
| Encoding R2 >> Decoding R2 | Behavior provides a strong input to neural activity, but the neural code is distributed/redundant |
| Decoding R2 >> Encoding R2 | Neural population contains rich information about behavior, but few behavioral features capture the full drive |
| Both high | Strong bidirectional relationship |
| Both low | These signals are not strongly coupled at this time resolution |

---

## Granger Causality

### What It Does

Granger causality tests whether one signal helps **predict the future** of another, beyond its own history. It is run in both directions:

- **Neural -> Behavior**: Does past neural activity improve prediction of future behavior?
- **Behavior -> Neural**: Does past behavior improve prediction of future neural activity?

### How to Run

```python
from cross_correlation import granger_test
from viz import plot_granger_summary

gc_n2b = granger_test(
    cause=population_firing_rate,
    effect=pose_speed_binned,
    max_lag=10,   # use 10 lags of history
)

gc_b2n = granger_test(
    cause=pose_speed_binned,
    effect=population_firing_rate,
    max_lag=10,
)

plot_granger_summary(gc_n2b, gc_b2n)
```

### Interpreting Results

| Result | Meaning |
|---|---|
| Neural -> Behavior significant (p < 0.05) | Past neural activity helps predict future behavior beyond behavior's own history. Consistent with neural *causing* behavioral changes. |
| Behavior -> Neural significant (p < 0.05) | Past behavior helps predict future neural activity. Consistent with sensory feedback or reafference. |
| Both significant | Bidirectional causal relationship (common in real neural data). |
| Neither significant | No detectable causal relationship at this lag scale. |

Key metrics in the output:

| Metric | Description |
|---|---|
| `f_statistic` | F-test statistic comparing full vs restricted model |
| `p_value` | Statistical significance of the improvement |
| `r2_restricted` | R2 of the model using only the effect's own history |
| `r2_full` | R2 of the model adding the cause's history |
| `improvement` | R2 gain from adding the cause (= `r2_full - r2_restricted`) |

!!! warning "Granger Causality Is Not True Causality"
    Granger causality tests *predictive* relationships, not true causal mechanisms. A significant result means "X helps predict Y," not "X causes Y." Both could be driven by a third, unmeasured variable. Always combine Granger results with the other analyses.

!!! tip "Choosing max_lag"
    `max_lag=10` at a bin size of 25 ms tests causal effects up to 250 ms. Adjust based on expected neural-behavior latencies:

    - Motor commands: ~50-200 ms (max_lag=2-8)
    - Sensory feedback: ~30-150 ms (max_lag=1-6)
    - Cognitive/attentional: ~200-500 ms (max_lag=8-20)

---

## Unit Selectivity Screening

### What It Does

Tests each unit for **differential firing between two conditions** (e.g., go vs no-go trials, rewarded vs unrewarded). Returns d-prime (effect size) and Mann-Whitney U test p-value.

### How to Run

```python
from neural_events import screen_selective_units, compute_selectivity_index

# Screen all units
selectivity = screen_selective_units(
    spike_times_dict=spike_times,
    condition_a_times=go_trial_onset_times,
    condition_b_times=nogo_trial_onset_times,
    window=(0.0, 0.5),     # 0 to 500 ms after event
    p_threshold=0.05,
)

print(selectivity[["unit_id", "d_prime", "p_value", "significant"]].head(10))
```

### Interpreting d-prime

| |d-prime| | Interpretation |
|---|---|
| < 0.2 | Negligible selectivity |
| 0.2 -- 0.5 | Small selectivity |
| 0.5 -- 0.8 | Medium selectivity |
| 0.8 -- 1.2 | Large selectivity |
| > 1.2 | Very strong selectivity (highly task-modulated neuron) |

!!! info "Sign of d-prime"
    Positive d-prime means condition A has a higher firing rate than condition B. Negative means the reverse. The magnitude (|d-prime|) tells you the effect size.

---

## Running the Full Pipeline

The `compute_neural_behavior_alignment()` function runs all analyses at once:

```python
from cross_correlation import compute_neural_behavior_alignment

results = compute_neural_behavior_alignment(
    spike_times_dict=spike_times,
    behavior_df=behavior_features_df,  # must have 't' and behavior columns
    trials=trials_df,                  # optional, for trial-averaged analysis
    bin_size=0.025,
    behavior_col="pose_speed",         # which signal to correlate
    max_lag_bins=40,                   # +/- 40 bins = +/- 1 second
)

# Access individual results
print(f"Peak cross-correlation lag: {results['peak_lag_s']:.3f} s")
print(f"Peak correlation: {results['peak_corr']:.3f}")
print(f"Encoding R2: {results['encoding']['mean_r2']:.4f}")
print(f"Decoding R2: {results['decoding']['mean_r2']:.4f}")
print(f"Granger N->B p-value: {results['granger_neural_to_behavior']['p_value']:.4f}")
print(f"Granger B->N p-value: {results['granger_behavior_to_neural']['p_value']:.4f}")
```

---

## Choosing Which Behavior Signal to Correlate

The `behavior_col` parameter selects which behavioral variable to use. Options from the pose feature pipeline:

| Signal | When to Use |
|---|---|
| `pose_speed` | General locomotion/movement (recommended default) |
| `head_angle` | Head orientation relative to body |
| `head_angular_vel` | Head turning speed |
| `body_length` | Body stretch/posture changes |
| `is_still` | Binary movement state |
| `pupil` | Pupil diameter (from eye tracking, not pose) |
| `pupil_vel` | Rate of pupil change |

!!! tip "Try Multiple Signals"
    Different brain regions may correlate with different behavior variables. Run the analysis with several `behavior_col` values and compare results:

    ```python
    for col in ["pose_speed", "head_angular_vel", "body_length", "pupil"]:
        results = compute_neural_behavior_alignment(
            spike_times_dict=spike_times,
            behavior_df=merged_features,
            behavior_col=col,
        )
        print(f"{col:20s}  xcorr={results['peak_corr']:.3f}  "
              f"enc_R2={results['encoding']['mean_r2']:.3f}  "
              f"dec_R2={results['decoding']['mean_r2']:.3f}")
    ```

---

## Population vs Single-Unit Analysis

### Population-Level

The default pipeline uses **population firing rate** (summed spike counts across all units). This is the most powerful approach because:

- Averaging across units reduces noise
- Population activity is a stronger, more stable signal
- Encoding/decoding models benefit from more features (each unit is a feature)

### Single-Unit Analysis

For single-unit analysis, iterate over units:

```python
from cross_correlation import crosscorrelation

for unit_id, st in spike_times.items():
    # Bin this unit's spikes
    from timebase import bin_spike_times, build_time_grid
    time_grid = build_time_grid(t_start, t_end, bin_size=0.025)
    unit_binned = bin_spike_times({unit_id: st}, time_grid, 0.025)
    unit_rate = unit_binned[unit_id].to_numpy()

    xcorr = crosscorrelation(unit_rate, pose_speed_binned, max_lag=40)
    if abs(xcorr["peak_corr"]) > 0.1:
        print(f"{unit_id}: peak_lag={xcorr['peak_lag']} peak_corr={xcorr['peak_corr']:.3f}")
```

---

## What to Do If Results Are Weak

If correlations are near zero, encoding/decoding R2 is < 0.01, and Granger tests are non-significant:

### 1. Check Your Data

!!! danger "First, Rule Out Data Issues"
    - Is the timebase alignment correct? (Check for timestamp mismatches between neural and behavior data.)
    - Are spike times in NWB seconds? Are pose timestamps in NWB seconds?
    - Is the behavior signal non-constant? (A flat signal cannot correlate with anything.)
    - Are there enough data points? (< 1000 time bins may be insufficient.)

### 2. Adjust Parameters

| Parameter | Try | Rationale |
|---|---|---|
| `bin_size` | 0.01, 0.05, 0.1 | Finer bins capture fast dynamics; coarser bins average out noise |
| `max_lag_bins` | 20, 50, 100 | Wider lags if you expect slow relationships |
| `lags` (encoding/decoding) | `[-5, -3, -1, 0, 1, 3, 5]` | Include more temporal context |
| `window_size` (sliding) | 100, 300, 500 | Match to task epoch durations |

### 3. Try Different Behavior Signals

The pose speed may not be the right signal. Try pupil, head angle, or inter-keypoint distances.

### 4. Subset Your Data

Neural-behavior coupling may be task-dependent. Try analyzing only:

- Active engagement periods (exclude inter-trial intervals)
- Hit trials vs miss trials separately
- The first vs second half of the session

### 5. Use More Units

Single-unit correlations are inherently noisy. If you are running single-unit analysis, switch to population-level analysis.

---

## Statistical Considerations

### Multiple Comparisons

When screening many units for selectivity or running cross-correlation for each unit, you are performing many statistical tests. Apply correction:

```python
from scipy.stats import false_discovery_control

# Bonferroni (conservative)
p_corrected = selectivity["p_value"] * len(selectivity)
selectivity["significant_bonferroni"] = p_corrected < 0.05

# Benjamini-Hochberg FDR (less conservative, recommended)
selectivity["significant_fdr"] = false_discovery_control(
    selectivity["p_value"].to_numpy(), method="bh"
) < 0.05
```

### Temporal Autocorrelation

Neural and behavioral signals are temporally autocorrelated (nearby time points are similar). This inflates the effective sample size and makes p-values from standard tests overly optimistic.

The pipeline mitigates this by:

- Using **time-blocked cross-validation** (train and test on different time periods, not interleaved)
- Using **Ridge regression** (regularization reduces overfitting to autocorrelated noise)

!!! warning "Do Not Use Shuffled Cross-Validation"
    Standard k-fold cross-validation shuffles data randomly, which leaks temporal information between folds. The pipeline uses time-blocked splits exclusively. Do not change this unless you have a specific reason.

### Effect Sizes Over p-Values

With enough data, even tiny correlations become statistically significant. Focus on **effect sizes**:

- R-squared values from encoding/decoding
- |d-prime| from selectivity screening
- |peak_corr| from cross-correlation

A statistically significant result with R2 = 0.001 is not biologically meaningful.
