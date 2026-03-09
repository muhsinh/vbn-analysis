"""Cross-modal correlation between neural activity and behavior.

This module answers the core question: do changes in behavior align with
changes in neural activity, and what correlations exist?

Provides:
- Time-lagged cross-correlation
- Sliding-window correlation (detect WHEN correlations change)
- Encoding models: predict neural from behavior
- Decoding models: predict behavior from neural
- Granger-like causality testing
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Time-lagged cross-correlation
# ---------------------------------------------------------------------------

def crosscorrelation(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 50,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute cross-correlation between a neural and behavior signal.

    Parameters
    ----------
    neural : 1D array (n_timepoints,)
        E.g., population firing rate or a single unit's binned spike count.
    behavior : 1D array (n_timepoints,)
        E.g., pose speed, pupil diameter, running speed.
    max_lag : int
        Maximum lag in bins (both positive and negative).
    normalize : bool
        If True, normalize to correlation coefficients [-1, 1].

    Returns
    -------
    dict with:
        lags : array of lag values (negative = neural leads behavior)
        correlation : cross-correlation values
        peak_lag : lag of maximum absolute correlation
        peak_corr : correlation at peak lag
    """
    neural = np.asarray(neural, dtype=float)
    behavior = np.asarray(behavior, dtype=float)

    # Remove means
    neural = neural - np.nanmean(neural)
    behavior = behavior - np.nanmean(behavior)

    # Replace NaNs with 0 for correlation
    neural = np.nan_to_num(neural)
    behavior = np.nan_to_num(behavior)

    n = len(neural)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            n_slice = neural[lag:]
            b_slice = behavior[:n - lag]
        else:
            n_slice = neural[:n + lag]
            b_slice = behavior[-lag:]

        if len(n_slice) < 2:
            continue

        if normalize:
            std_n = np.std(n_slice)
            std_b = np.std(b_slice)
            if std_n > 0 and std_b > 0:
                corr[i] = np.mean(n_slice * b_slice) / (std_n * std_b)
        else:
            corr[i] = np.mean(n_slice * b_slice)

    peak_idx = np.argmax(np.abs(corr))
    return {
        "lags": lags,
        "correlation": corr,
        "peak_lag": int(lags[peak_idx]),
        "peak_corr": float(corr[peak_idx]),
    }


def population_crosscorrelation(
    pop_matrix: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 50,
    bin_size: float = 0.025,
) -> pd.DataFrame:
    """Cross-correlation between each unit and a behavior signal.

    Parameters
    ----------
    pop_matrix : (n_timepoints, n_units)
    behavior : (n_timepoints,)
    max_lag : int (in bins)
    bin_size : float (for converting lag to seconds)

    Returns
    -------
    DataFrame with columns: unit_idx, peak_lag_bins, peak_lag_s, peak_corr
    """
    n_units = pop_matrix.shape[1] if pop_matrix.ndim == 2 else 0
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


# ---------------------------------------------------------------------------
# Sliding-window correlation
# ---------------------------------------------------------------------------

def sliding_correlation(
    neural: np.ndarray,
    behavior: np.ndarray,
    window_size: int = 100,
    step: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute correlation in sliding windows.

    Reveals WHEN neural-behavior coupling is strong vs weak over the session.

    Parameters
    ----------
    neural, behavior : 1D arrays (same length)
    window_size : int, number of bins per window
    step : int, step size between windows

    Returns
    -------
    dict with:
        window_centers : array of center indices
        correlations : array of Pearson r values per window
        p_values : array of p-values per window
    """
    from scipy.stats import pearsonr

    n = min(len(neural), len(behavior))
    centers = []
    corrs = []
    pvals = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        n_win = neural[start:end]
        b_win = behavior[start:end]

        # Skip windows with no variance
        if np.std(n_win) < 1e-10 or np.std(b_win) < 1e-10:
            continue

        # Remove NaN pairs
        valid = np.isfinite(n_win) & np.isfinite(b_win)
        if valid.sum() < 10:
            continue

        r, p = pearsonr(n_win[valid], b_win[valid])
        centers.append((start + end) / 2)
        corrs.append(r)
        pvals.append(p)

    return {
        "window_centers": np.array(centers),
        "correlations": np.array(corrs),
        "p_values": np.array(pvals),
    }


# ---------------------------------------------------------------------------
# Encoding model: predict neural from behavior
# ---------------------------------------------------------------------------

def fit_encoding_model(
    behavior_features: pd.DataFrame,
    neural_target: np.ndarray,
    model_type: str = "poisson",
    n_folds: int = 5,
    lags: List[int] | None = None,
) -> Dict[str, Any]:
    """Fit an encoding model: behavior -> neural activity.

    "Given what the animal is doing, can we predict neural firing?"

    Parameters
    ----------
    behavior_features : DataFrame
        Columns are behavioral variables (pose_speed, pupil, etc.).
        Must NOT include 't' column.
    neural_target : 1D array
        Spike count or firing rate to predict.
    model_type : 'poisson' or 'ridge'
    n_folds : int
        Number of CV folds (time-blocked).
    lags : list of int, optional
        Add lagged versions of features (e.g., [-2, -1, 0, 1, 2]).

    Returns
    -------
    dict with: model, cv_scores, feature_importance, mean_r2
    """
    X = behavior_features.copy()

    # Drop non-numeric and time columns
    drop_cols = [c for c in X.columns if c in ("t", "session_id", "camera")]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Add lagged features
    if lags is not None:
        X = _add_lags(X, lags)

    X = X.fillna(0).to_numpy() if isinstance(X, pd.DataFrame) else np.nan_to_num(X)
    y = np.asarray(neural_target, dtype=float)

    # Align lengths
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    if n < 20:
        return {"model": None, "cv_scores": [], "feature_importance": None, "mean_r2": np.nan}

    # Time-blocked CV
    fold_size = n // n_folds
    scores = []
    models = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n)])

        if len(train_idx) < 10 or len(test_idx) < 5:
            continue

        model = _make_encoding_model(model_type)
        try:
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            r2 = r2_score(y[test_idx], pred)
            scores.append(r2)
            models.append(model)
        except Exception:
            continue

    # Final model on all data
    final_model = _make_encoding_model(model_type)
    try:
        final_model.fit(X, y)
    except Exception:
        final_model = None

    importance = None
    if final_model is not None and hasattr(final_model, "coef_"):
        importance = np.abs(final_model.coef_)

    return {
        "model": final_model,
        "cv_scores": scores,
        "feature_importance": importance,
        "mean_r2": float(np.mean(scores)) if scores else np.nan,
    }


def _make_encoding_model(model_type: str):
    if model_type == "poisson":
        return PoissonRegressor(alpha=0.01, max_iter=500)
    elif model_type == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unknown model type: {model_type}")


def _add_lags(X: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Add time-lagged copies of all columns."""
    dfs = [X]
    for lag in lags:
        if lag == 0:
            continue
        shifted = X.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in X.columns]
        dfs.append(shifted)
    result = pd.concat(dfs, axis=1)
    # Drop rows with NaN from shifting
    max_lag = max(abs(l) for l in lags) if lags else 0
    if max_lag > 0:
        result = result.iloc[max_lag:-max_lag] if max_lag < len(result) // 2 else result.dropna()
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Decoding model: predict behavior from neural
# ---------------------------------------------------------------------------

def fit_decoding_model(
    pop_matrix: np.ndarray,
    behavior_target: np.ndarray,
    n_folds: int = 5,
    lags: List[int] | None = None,
) -> Dict[str, Any]:
    """Fit a decoding model: neural -> behavior.

    "Given neural activity, can we predict what the animal is doing?"

    Parameters
    ----------
    pop_matrix : (n_timepoints, n_units)
        Population spike counts.
    behavior_target : 1D array
        Behavioral variable to predict (pose speed, pupil, etc.).
    n_folds : int
    lags : list of int, optional

    Returns
    -------
    dict with: model, cv_scores, mean_r2, feature_importance
    """
    X = pd.DataFrame(pop_matrix, columns=[f"unit_{i}" for i in range(pop_matrix.shape[1])])

    if lags is not None:
        X = _add_lags(X, lags)

    X_arr = np.nan_to_num(X.to_numpy() if isinstance(X, pd.DataFrame) else X)
    y = np.asarray(behavior_target, dtype=float)

    n = min(len(X_arr), len(y))
    X_arr, y = X_arr[:n], y[:n]

    # Remove NaN targets
    valid = np.isfinite(y)
    X_arr, y = X_arr[valid], y[valid]

    if len(y) < 20:
        return {"model": None, "cv_scores": [], "mean_r2": np.nan, "feature_importance": None}

    fold_size = len(y) // n_folds
    scores = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, len(y))
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, len(y))])

        if len(train_idx) < 10 or len(test_idx) < 5:
            continue

        model = Ridge(alpha=1.0)
        try:
            model.fit(X_arr[train_idx], y[train_idx])
            pred = model.predict(X_arr[test_idx])
            scores.append(r2_score(y[test_idx], pred))
        except Exception:
            continue

    final_model = Ridge(alpha=1.0)
    try:
        final_model.fit(X_arr, y)
    except Exception:
        final_model = None

    importance = np.abs(final_model.coef_) if final_model is not None else None

    return {
        "model": final_model,
        "cv_scores": scores,
        "mean_r2": float(np.mean(scores)) if scores else np.nan,
        "feature_importance": importance,
    }


# ---------------------------------------------------------------------------
# Granger-like causality
# ---------------------------------------------------------------------------

def granger_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 10,
) -> Dict[str, Any]:
    """Simple Granger causality test.

    Tests whether `cause` helps predict `effect` beyond the effect's own history.

    Uses a comparison of two Ridge models:
    - Restricted: effect_t ~ effect_{t-1}, ..., effect_{t-k}
    - Full: effect_t ~ effect_{t-1}, ..., effect_{t-k}, cause_{t-1}, ..., cause_{t-k}

    Returns
    -------
    dict with: f_statistic, p_value, r2_restricted, r2_full, improvement
    """
    cause = np.asarray(cause, dtype=float)
    effect = np.asarray(effect, dtype=float)

    n = min(len(cause), len(effect))
    if n < max_lag + 20:
        return {"f_statistic": 0.0, "p_value": 1.0, "r2_restricted": 0.0, "r2_full": 0.0, "improvement": 0.0}

    # Build lagged matrices
    y = effect[max_lag:]
    X_restricted = np.column_stack([effect[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
    X_cause_lags = np.column_stack([cause[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
    X_full = np.column_stack([X_restricted, X_cause_lags])

    y = y[:len(X_restricted)]

    # Fit models
    model_r = Ridge(alpha=0.1)
    model_f = Ridge(alpha=0.1)

    model_r.fit(X_restricted, y)
    model_f.fit(X_full, y)

    pred_r = model_r.predict(X_restricted)
    pred_f = model_f.predict(X_full)

    ss_r = np.sum((y - pred_r) ** 2)
    ss_f = np.sum((y - pred_f) ** 2)

    r2_r = r2_score(y, pred_r)
    r2_f = r2_score(y, pred_f)

    # F-test
    df1 = max_lag  # additional parameters in full model
    df2 = len(y) - 2 * max_lag
    if df2 <= 0 or ss_f <= 0:
        return {"f_statistic": 0.0, "p_value": 1.0, "r2_restricted": r2_r, "r2_full": r2_f, "improvement": r2_f - r2_r}

    f_stat = ((ss_r - ss_f) / df1) / (ss_f / df2)

    try:
        from scipy.stats import f as f_dist
        p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)
    except Exception:
        p_value = np.nan

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "r2_restricted": float(r2_r),
        "r2_full": float(r2_f),
        "improvement": float(r2_f - r2_r),
    }


# ---------------------------------------------------------------------------
# Convenience: full neural-behavior alignment report
# ---------------------------------------------------------------------------

def compute_neural_behavior_alignment(
    spike_times_dict: Dict[str, np.ndarray],
    behavior_df: pd.DataFrame,
    trials: pd.DataFrame | None = None,
    bin_size: float = 0.025,
    behavior_col: str = "pose_speed",
    max_lag_bins: int = 40,
) -> Dict[str, Any]:
    """Run the full correlation analysis pipeline.

    This is the main entry point that answers: "do behavior changes align
    with neural activity changes?"

    Parameters
    ----------
    spike_times_dict : unit_id -> spike_times
    behavior_df : DataFrame with 't' and behavior columns
    trials : DataFrame with trial info (optional, for trial-averaged analysis)
    bin_size : bin width in seconds
    behavior_col : which behavior column to correlate
    max_lag_bins : max cross-correlation lag

    Returns
    -------
    dict with all analysis results
    """
    from neural_events import build_population_vectors
    from timebase import build_time_grid

    results: Dict[str, Any] = {"bin_size": bin_size}

    # Determine time range from available data
    all_spike_times = np.concatenate(list(spike_times_dict.values())) if spike_times_dict else np.array([])
    if all_spike_times.size > 0:
        t_start = float(all_spike_times.min())
        t_end = float(all_spike_times.max())
    elif behavior_df is not None and "t" in behavior_df.columns:
        t_start = float(behavior_df["t"].min())
        t_end = float(behavior_df["t"].max())
    else:
        return results

    time_grid = build_time_grid(t_start, t_end, bin_size)
    if len(time_grid) < 20:
        results["error"] = "Not enough time bins for analysis"
        return results

    # Build population matrix
    pop = build_population_vectors(spike_times_dict, time_grid, bin_size)
    pop_rate = pop.sum(axis=1)  # total population firing rate
    results["n_units"] = pop.shape[1]
    results["n_timebins"] = len(time_grid)
    results["time_range"] = (float(t_start), float(t_end))

    # Bin behavior signal
    if behavior_df is not None and behavior_col in behavior_df.columns:
        from timebase import bin_continuous_features
        beh_binned = bin_continuous_features(behavior_df[["t", behavior_col]], time_grid)
        if beh_binned.empty or behavior_col not in beh_binned.columns:
            results["error"] = f"Could not bin behavior column '{behavior_col}'"
            return results
        beh_signal = beh_binned[behavior_col].to_numpy()
    else:
        results["error"] = f"Behavior column '{behavior_col}' not found"
        return results

    # 1. Cross-correlation: population rate vs behavior
    xcorr = crosscorrelation(pop_rate, beh_signal, max_lag=max_lag_bins)
    results["crosscorrelation"] = xcorr
    results["peak_lag_s"] = xcorr["peak_lag"] * bin_size
    results["peak_corr"] = xcorr["peak_corr"]

    # 2. Per-unit cross-correlation
    unit_xcorr = population_crosscorrelation(pop, beh_signal, max_lag_bins, bin_size)
    results["unit_crosscorrelation"] = unit_xcorr

    # 3. Sliding-window correlation
    window_size = min(200, len(time_grid) // 5)
    if window_size >= 20:
        slide = sliding_correlation(pop_rate, beh_signal, window_size=window_size, step=window_size // 4)
        results["sliding_correlation"] = slide

    # 4. Encoding model: behavior -> neural
    beh_features = beh_binned.drop(columns=["t"], errors="ignore")
    enc = fit_encoding_model(
        beh_features, pop_rate,
        model_type="ridge", lags=[-2, -1, 0, 1, 2],
    )
    results["encoding"] = {
        "mean_r2": enc["mean_r2"],
        "cv_scores": enc["cv_scores"],
    }

    # 5. Decoding model: neural -> behavior
    dec = fit_decoding_model(
        pop, beh_signal,
        lags=[-2, -1, 0, 1, 2],
    )
    results["decoding"] = {
        "mean_r2": dec["mean_r2"],
        "cv_scores": dec["cv_scores"],
    }

    # 6. Granger causality (both directions)
    gc_n2b = granger_test(pop_rate, beh_signal, max_lag=min(10, len(time_grid) // 10))
    gc_b2n = granger_test(beh_signal, pop_rate, max_lag=min(10, len(time_grid) // 10))
    results["granger_neural_to_behavior"] = gc_n2b
    results["granger_behavior_to_neural"] = gc_b2n

    # 7. Trial-averaged PETHs (if trials available)
    if trials is not None and not trials.empty and "t" in trials.columns:
        from neural_events import trial_averaged_rates
        for group_col in ["trial_type", "rewarded"]:
            if group_col in trials.columns:
                tavg = trial_averaged_rates(spike_times_dict, trials, group_col)
                results[f"trial_averaged_{group_col}"] = {
                    cond: {
                        "n_trials": res["peths"][list(res["peths"].keys())[0]]["n_trials"] if res["peths"] else 0,
                        "n_units": len(res["unit_ids"]),
                    }
                    for cond, res in tavg.items()
                }

    return results
