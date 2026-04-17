"""Cross-modal correlation between neural activity and behavior.

This module answers the core question: do changes in behavior align with
changes in neural activity, and what correlations exist?

Provides:
- Time-lagged cross-correlation
- Sliding-window correlation (detect WHEN correlations change)
- Encoding models: predict neural from behavior (single and multi-covariate)
- Decoding models: predict behavior from neural
- Granger-like causality testing
- Variance partitioning across behavioral covariates
- Permutation testing via circular shifts
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge, RidgeCV
from sklearn.metrics import r2_score

from vbn_types import SpikeTimesDict


def crosscorrelation(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 50,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
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

    neural = neural - np.nanmean(neural)
    behavior = behavior - np.nanmean(behavior)

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



def sliding_correlation(
    neural: np.ndarray,
    behavior: np.ndarray,
    window_size: int = 100,
    step: int = 10,
) -> dict[str, np.ndarray]:
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

        if np.std(n_win) < 1e-10 or np.std(b_win) < 1e-10:
            continue

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



def raised_cosine_basis(
    n_lag_bins: int,
    n_basis: int,
    log_spacing: bool = True,
) -> np.ndarray:
    """Raised cosine basis functions for temporal lag kernels (Pillow et al. 2008).

    Parameters
    ----------
    n_lag_bins : int
        Number of lag bins to cover (e.g. 40 bins at 25ms = 1s).
    n_basis : int
        Number of basis functions.
    log_spacing : bool
        If True, space basis peaks logarithmically (denser near lag=0).

    Returns
    -------
    B : (n_lag_bins, n_basis)
        Each column is one raised cosine bump.
    """
    lags = np.arange(n_lag_bins, dtype=float)
    if log_spacing:
        # log-compress the lag axis so early lags get finer resolution
        c = 1.0
        lags_c = np.log(lags + c)
        span = lags_c[-1] - lags_c[0]
    else:
        lags_c = lags
        span = lags_c[-1] - lags_c[0]

    centers = np.linspace(lags_c[0], lags_c[-1], n_basis)
    width = span / (n_basis - 1) if n_basis > 1 else span

    B = np.zeros((n_lag_bins, n_basis))
    for j, c_j in enumerate(centers):
        z = np.pi * (lags_c - c_j) / width
        bump = 0.5 * (np.cos(np.clip(z, -np.pi, np.pi)) + 1.0)
        B[:, j] = bump
    # Normalize each basis function to unit max
    col_max = B.max(axis=0)
    col_max[col_max == 0] = 1.0
    return B / col_max


def _add_raised_cosine_lags(
    X: np.ndarray,
    n_lag_bins: int = 40,
    n_basis: int = 10,
) -> np.ndarray:
    """Project each feature through a raised cosine lag basis.

    Parameters
    ----------
    X : (n_times, n_features)
    n_lag_bins : int
        How far back in time to reach (bins). 40 bins @ 25ms = 1 s.
    n_basis : int
        Number of raised cosine basis functions per feature.

    Returns
    -------
    X_lag : (n_times - n_lag_bins, n_features * n_basis)
    """
    B = raised_cosine_basis(n_lag_bins, n_basis)  # (n_lag_bins, n_basis)
    n_times, n_feat = X.shape
    n_out = n_times - n_lag_bins
    if n_out <= 0:
        return np.empty((0, n_feat * n_basis))

    # Vectorized: build lagged view (n_out, n_lag_bins, n_feat), then contract with B
    # Shape: (n_out, n_lag_bins, n_feat)
    idx = np.arange(n_lag_bins - 1, n_lag_bins - 1 + n_out)[:, None] - np.arange(n_lag_bins)[None, :]
    X_lagged = X[idx]  # (n_out, n_lag_bins, n_feat)
    # Contract lag axis with basis: (n_out, n_feat, n_basis)
    # Contract lag axis with basis, keep feature-major block ordering:
    # Output column order = [f0_b0, f0_b1, ..., f0_b{K-1}, f1_b0, ..., f{F-1}_b{K-1}]
    # which matches the coefficient-block extraction in fit_multi_covariate_encoding_model.
    X_lag = np.einsum("tlf,lb->tfb", X_lagged, B).reshape(n_out, n_feat * n_basis)
    return X_lag


def _forward_chain_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    gap_bins: int,
    alphas: list[float] | None = None,
    model_type: str = "ridge",
) -> list[float]:
    """Forward-chain CV returning a list of per-fold R² scores.

    Test folds are always in the future relative to training, with a
    `gap_bins` buffer to prevent autocorrelation leakage.
    """
    # Expanded alpha grid for design matrices with ~10^5 rows and ~10^2 cols.
    # The previous grid [0.01 ... 100] saturated on large problems.
    alphas = alphas or [1.0, 10.0, 100.0, 1000.0, 10000.0]
    n = len(y)
    block = n // (n_folds + 1)
    scores: list[float] = []
    for i in range(1, n_folds + 1):
        test_start = i * block
        train_end = test_start - gap_bins
        test_end = min((i + 1) * block, n)
        if train_end < 10 or test_end <= test_start:
            continue
        m = _make_encoding_model(model_type) if model_type != "ridge" else RidgeCV(alphas=alphas)
        m.fit(X[:train_end], y[:train_end])
        scores.append(r2_score(y[test_start:test_end], m.predict(X[test_start:test_end])))
    return scores


def fit_encoding_model(
    behavior_features: pd.DataFrame,
    neural_target: np.ndarray,
    model_type: str = "ridge",
    n_folds: int = 5,
    lags: list[int] | None = None,
    gap_bins: int = 40,
    use_raised_cosine: bool = False,
    n_lag_bins: int = 40,
    n_basis: int = 10,
) -> dict[str, Any]:
    """Fit an encoding model: behavior -> neural activity.

    Uses true forward-chaining cross-validation: test folds are always
    predicted from past data only, with a temporal gap to prevent
    autocorrelation leakage.

    Parameters
    ----------
    behavior_features : DataFrame
        Behavioral covariate columns. Must NOT include 't'.
    neural_target : 1D array
        Spike count or firing rate to predict.
    model_type : 'poisson' or 'ridge'
    n_folds : int
        Number of forward-chain CV folds.
    lags : list of int, optional
        Integer lag offsets to add as columns (simple shift method).
        Mutually exclusive with use_raised_cosine.
    gap_bins : int
        Temporal gap between end of training and start of test (bins).
        Prevents autocorrelation leakage. Default 20 bins = 500ms at 25ms bins.
    use_raised_cosine : bool
        If True, replace simple lag shifts with raised cosine basis projection.
    n_lag_bins : int
        Lag window depth for raised cosine (bins). Default 40 = 1s at 25ms.
    n_basis : int
        Number of raised cosine basis functions per feature.

    Returns
    -------
    dict with: model, cv_scores, feature_importance, mean_r2
    """
    X = behavior_features.copy()

    drop_cols = [c for c in X.columns if c in ("t", "session_id", "camera")]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    if lags is not None and not use_raised_cosine:
        X = _add_lags(X, lags)

    X_arr = X.fillna(0).to_numpy() if isinstance(X, pd.DataFrame) else np.nan_to_num(X)
    y = np.asarray(neural_target, dtype=float)

    # Z-score features: ridge needs commensurate scales across columns.
    x_mu = X_arr.mean(axis=0)
    x_sd = X_arr.std(axis=0)
    x_sd[x_sd == 0] = 1.0
    X_arr = (X_arr - x_mu) / x_sd

    if use_raised_cosine:
        X_arr = _add_raised_cosine_lags(X_arr, n_lag_bins=n_lag_bins, n_basis=n_basis)
        y = y[n_lag_bins:]  # align targets to trimmed X

    n = min(len(X_arr), len(y))
    X_arr, y = X_arr[:n], y[:n]

    # Z-score target so R² reflects explained normalized variance.
    y_mu, y_sd = float(np.nanmean(y)), float(np.nanstd(y))
    if y_sd == 0:
        y_sd = 1.0
    y = (y - y_mu) / y_sd

    if n < 40:
        return {"model": None, "cv_scores": [], "feature_importance": None, "mean_r2": np.nan}

    scores = _forward_chain_r2(X_arr, y, n_folds=n_folds, gap_bins=gap_bins, model_type=model_type)

    final_model = _make_encoding_model(model_type)
    final_model.fit(X_arr, y)

    importance = None
    if hasattr(final_model, "coef_"):
        importance = np.abs(final_model.coef_)

    return {
        "model": final_model,
        "cv_scores": scores,
        "feature_importance": importance,
        "mean_r2": float(np.mean(scores)) if scores else np.nan,
    }


def permutation_test(
    behavior_features: pd.DataFrame,
    neural_target: np.ndarray,
    observed_r2: float,
    n_permutations: int = 500,
    **fit_kwargs,
) -> dict[str, Any]:
    """Generate a null R² distribution via circular shifts of the neural signal.

    Circular shifts preserve the autocorrelation structure of the neural
    signal while destroying the temporal alignment with behavior.

    Parameters
    ----------
    behavior_features : DataFrame
        Same features used in the observed encoding model.
    neural_target : 1D array
        The neural signal that was fit (binned spike counts / rate).
    observed_r2 : float
        R² from the real (aligned) encoding model.
    n_permutations : int
        Number of circular shifts to perform.
    **fit_kwargs
        Passed to fit_encoding_model (model_type, gap_bins, etc.).

    Returns
    -------
    dict with: null_r2 (array), p_value, observed_r2, z_score
    """
    from modeling import circular_shift

    y = np.asarray(neural_target, dtype=float)
    n = len(y)
    null_r2 = np.zeros(n_permutations)

    rng = np.random.default_rng(42)
    shifts = rng.integers(n // 5, 4 * n // 5, size=n_permutations)

    for i, shift in enumerate(shifts):
        y_shifted = circular_shift(y, int(shift))
        result = fit_encoding_model(behavior_features, y_shifted, **fit_kwargs)
        null_r2[i] = result["mean_r2"] if not np.isnan(result["mean_r2"]) else 0.0

    p_value = float(np.mean(null_r2 >= observed_r2))
    null_mean = float(np.mean(null_r2))
    null_std = float(np.std(null_r2))
    z_score = (observed_r2 - null_mean) / (null_std + 1e-9)

    return {
        "null_r2": null_r2,
        "p_value": p_value,
        "observed_r2": observed_r2,
        "z_score": float(z_score),
        "null_mean": null_mean,
        "null_std": null_std,
    }


def fit_multi_covariate_encoding_model(
    covariate_dict: dict[str, np.ndarray],
    neural_target: np.ndarray,
    bin_size: float = 0.025,
    gap_bins: int = 40,
    n_lag_bins: int = 40,
    n_basis: int = 8,
    n_permutations: int = 200,
) -> dict[str, Any]:
    """Proper multi-covariate encoding model with variance partitioning.

    Fits a full model (all covariates) and reduced models (leaving out each
    covariate in turn). The unique variance explained by each covariate is
    the drop in R² when that covariate is removed.

    This is the correct approach for separating the contributions of
    running speed, pupil, licking, rewards, and stimulus drive.

    Parameters
    ----------
    covariate_dict : dict mapping covariate name -> 1D array (n_times,)
        Expected keys: 'running', 'pupil', 'licks', 'rewards', 'stimulus'
        (any subset is valid).
    neural_target : 1D array (n_times,)
        Binned spike counts for one unit or population rate.
    bin_size : float
        Bin width in seconds (used to label lag axes).
    gap_bins : int
        Temporal gap in CV to prevent autocorrelation leakage.
    n_lag_bins : int
        Raised cosine lag window depth (40 bins @ 25ms = 1 s).
    n_basis : int
        Number of raised cosine basis functions per covariate.
    n_permutations : int
        Number of circular shifts for null distribution. Set to 0 to skip.

    Returns
    -------
    dict with:
        full_r2      : R² of full model (all covariates)
        unique_r2    : dict {covariate_name: unique variance contribution}
        shared_r2    : variance explained only when all covariates present
        perm_results : permutation test result for the full model
        coef_dict    : regression coefficients per covariate block
    """
    n = len(neural_target)
    cov_names = list(covariate_dict.keys())

    X_raw = np.column_stack([
        np.asarray(covariate_dict[k], dtype=float)[:n] for k in cov_names
    ])
    X_raw = np.nan_to_num(X_raw)

    # Z-score each behavioral covariate so isotropic L2 penalty applies uniformly.
    x_mu = X_raw.mean(axis=0)
    x_sd = X_raw.std(axis=0)
    x_sd[x_sd == 0] = 1.0
    X_raw = (X_raw - x_mu) / x_sd

    X_full = _add_raised_cosine_lags(X_raw, n_lag_bins=n_lag_bins, n_basis=n_basis)
    y = np.asarray(neural_target, dtype=float)[n_lag_bins:n_lag_bins + len(X_full)]
    # Z-score target; raw spike counts are on a scale incommensurate with behavior.
    y_mu, y_sd = float(np.nanmean(y)), float(np.nanstd(y))
    if y_sd == 0:
        y_sd = 1.0
    y = (y - y_mu) / y_sd

    n_trimmed = len(y)
    coef_full = None

    scores_full = _forward_chain_r2(X_full, y, n_folds=5, gap_bins=gap_bins)
    full_r2 = float(np.mean(scores_full)) if scores_full else np.nan

    m_final = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    m_final.fit(X_full, y)
    coef_full = m_final.coef_

    unique_r2: dict[str, float] = {}
    coef_dict: dict[str, Any] = {}

    for drop_k in cov_names:
        keep = [k for k in cov_names if k != drop_k]
        if not keep:
            unique_r2[drop_k] = full_r2
            continue
        X_reduced_raw = np.column_stack([
            np.asarray(covariate_dict[k], dtype=float)[:n] for k in keep
        ])
        X_reduced_raw = np.nan_to_num(X_reduced_raw)
        X_reduced = _add_raised_cosine_lags(X_reduced_raw, n_lag_bins=n_lag_bins, n_basis=n_basis)
        y_r = y  # same trimmed target

        scores_red = _forward_chain_r2(X_reduced, y_r, n_folds=5, gap_bins=gap_bins)
        reduced_r2 = float(np.mean(scores_red)) if scores_red else np.nan
        unique_r2[drop_k] = float(full_r2 - reduced_r2) if not np.isnan(reduced_r2) else np.nan

    if coef_full is not None:
        for fi, k in enumerate(cov_names):
            start_col = fi * n_basis
            coef_dict[k] = coef_full[start_col: start_col + n_basis]

    perm_results: dict[str, Any] = {}
    if n_permutations > 0:
        from modeling import circular_shift
        rng = np.random.default_rng(0)
        shifts = rng.integers(n_trimmed // 5, 4 * n_trimmed // 5, size=n_permutations)
        null_r2 = np.zeros(n_permutations)
        for pi, sh in enumerate(shifts):
            y_sh = circular_shift(y, int(sh))
            s = _forward_chain_r2(X_full, y_sh, n_folds=5, gap_bins=gap_bins,
                                  alphas=[0.1, 1.0, 10.0])
            null_r2[pi] = float(np.mean(s)) if s else 0.0

        null_mean = float(np.nanmean(null_r2))
        null_std = float(np.nanstd(null_r2))
        perm_results = {
            "null_r2": null_r2,
            "p_value": float(np.mean(null_r2 >= full_r2)),
            "z_score": (full_r2 - null_mean) / (null_std + 1e-9),
            "null_mean": null_mean,
            "null_std": null_std,
        }

    return {
        "full_r2": full_r2,
        "unique_r2": unique_r2,
        "shared_r2": float(full_r2 - sum(v for v in unique_r2.values() if not np.isnan(v))),
        "perm_results": perm_results,
        "coef_dict": coef_dict,
        "covariate_names": cov_names,
        "n_lag_bins": n_lag_bins,
        "bin_size_s": bin_size,
    }


def _make_encoding_model(model_type: str) -> PoissonRegressor | RidgeCV:
    if model_type == "poisson":
        return PoissonRegressor(alpha=0.01, max_iter=500)
    elif model_type == "ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    raise ValueError(f"Unknown model type: {model_type}")


def _add_lags(X: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add time-lagged copies of all columns (simple integer shift method)."""
    dfs = [X]
    for lag in lags:
        if lag == 0:
            continue
        shifted = X.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in X.columns]
        dfs.append(shifted)
    result = pd.concat(dfs, axis=1)
    max_lag = max(abs(l) for l in lags) if lags else 0
    if max_lag > 0:
        result = result.iloc[max_lag:-max_lag] if max_lag < len(result) // 2 else result.dropna()
    return result.reset_index(drop=True)



def fit_decoding_model(
    pop_matrix: np.ndarray,
    behavior_target: np.ndarray,
    n_folds: int = 5,
    lags: list[int] | None = None,
    gap_bins: int = 40,
) -> dict[str, Any]:
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

    scores = _forward_chain_r2(X_arr, y, n_folds=n_folds, gap_bins=gap_bins)

    final_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    final_model.fit(X_arr, y)
    importance = np.abs(final_model.coef_) if hasattr(final_model, "coef_") else None

    return {
        "model": final_model,
        "cv_scores": scores,
        "mean_r2": float(np.mean(scores)) if scores else np.nan,
        "feature_importance": importance,
    }



def granger_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 10,
) -> dict[str, Any]:
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

    y = effect[max_lag:]
    X_restricted = np.column_stack([effect[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
    X_cause_lags = np.column_stack([cause[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
    X_full = np.column_stack([X_restricted, X_cause_lags])

    y = y[:len(X_restricted)]

    # Drop rows with NaN (from blink interpolation, missing data, etc.)
    valid = np.isfinite(y) & np.all(np.isfinite(X_full), axis=1)
    if valid.sum() < max_lag + 20:
        return {"f_statistic": 0.0, "p_value": 1.0, "r2_restricted": 0.0, "r2_full": 0.0, "improvement": 0.0}
    y, X_restricted, X_full = y[valid], X_restricted[valid], X_full[valid]

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

    df1 = max_lag  # additional parameters in full model
    df2 = len(y) - 2 * max_lag
    if df2 <= 0 or ss_f <= 0:
        return {"f_statistic": 0.0, "p_value": 1.0, "r2_restricted": r2_r, "r2_full": r2_f, "improvement": r2_f - r2_r}

    f_stat = ((ss_r - ss_f) / df1) / (ss_f / df2)

    try:
        from scipy.stats import f as f_dist
    except ImportError:
        p_value = np.nan
    else:
        p_value = 1.0 - f_dist.cdf(f_stat, df1, df2)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "r2_restricted": float(r2_r),
        "r2_full": float(r2_f),
        "improvement": float(r2_f - r2_r),
    }



def compute_alignment_by_area(
    spike_times_dict: SpikeTimesDict,
    units_df: "pd.DataFrame",
    behavior_df: "pd.DataFrame",
    trials: "pd.DataFrame | None" = None,
    bin_size: float = 0.025,
    behavior_cols: list[str] | None = None,
    behavior_col: str = "running",
    max_lag_bins: int = 40,
    gap_bins: int = 40,
    n_permutations: int = 200,
    area_col: str = "ecephys_structure_acronym",
    min_units: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run compute_neural_behavior_alignment separately for each brain area.

    Parameters
    ----------
    units_df : DataFrame with a unit_id column and an area column.
    area_col : Column in units_df that holds the brain area label.
    min_units : Skip areas with fewer than this many units.

    Returns
    -------
    dict mapping area_name -> alignment result dict
    """
    import pandas as _pd

    if area_col not in units_df.columns:
        return {"all": compute_neural_behavior_alignment(
            spike_times_dict, behavior_df, trials,
            bin_size=bin_size, behavior_col=behavior_col,
            behavior_cols=behavior_cols, max_lag_bins=max_lag_bins,
            gap_bins=gap_bins, n_permutations=n_permutations,
        )}

    areas = units_df[area_col].dropna().unique()
    results: dict[str, dict[str, Any]] = {}

    for area in sorted(areas):
        area_units = units_df[units_df[area_col] == area]
        uid_col = "unit_id" if "unit_id" in area_units.columns else area_units.index.name or "index"
        if uid_col == "index":
            unit_ids = {str(i) for i in area_units.index}
        else:
            unit_ids = {str(u) for u in area_units[uid_col]}

        area_spikes = {k: v for k, v in spike_times_dict.items() if k in unit_ids}
        if len(area_spikes) < min_units:
            continue

        area_result = compute_neural_behavior_alignment(
            area_spikes, behavior_df, trials,
            bin_size=bin_size, behavior_col=behavior_col,
            behavior_cols=behavior_cols, max_lag_bins=max_lag_bins,
            gap_bins=gap_bins, n_permutations=n_permutations,
            run_variance_partitioning=(behavior_cols is not None and len(behavior_cols) >= 2),
        )
        area_result["area"] = area
        area_result["n_area_units"] = len(area_spikes)
        results[area] = area_result

    return results


def compute_neural_behavior_alignment(
    spike_times_dict: SpikeTimesDict,
    behavior_df: pd.DataFrame,
    trials: pd.DataFrame | None = None,
    bin_size: float = 0.025,
    behavior_col: str = "pose_speed",
    behavior_cols: list[str] | None = None,
    max_lag_bins: int = 40,
    gap_bins: int = 40,
    n_permutations: int = 200,
    run_variance_partitioning: bool = True,
) -> dict[str, Any]:
    """Run the full neural-behavior alignment pipeline.

    Implements Pillow et al.-style multi-covariate encoding with:
    - Forward-chain CV with temporal gap (no data leakage)
    - Raised cosine lag basis (±1s temporal kernel)
    - Variance partitioning across covariates
    - Circular-shift permutation test for null distribution

    Parameters
    ----------
    spike_times_dict : unit_id -> spike_times
    behavior_df : DataFrame with 't' plus behavior columns
    trials : trial metadata (optional, for PETH analysis)
    bin_size : seconds per bin
    behavior_col : single column name (used if behavior_cols is None)
    behavior_cols : list of column names to include as simultaneous covariates.
        When provided, fits a multi-covariate model and returns unique
        variance per covariate. Recommended: ['running', 'pupil', ...].
    max_lag_bins : max lag for raw cross-correlation (bins)
    gap_bins : temporal gap between train and test in forward-chain CV
    n_permutations : circular-shift permutations for null distribution (0 = skip)
    run_variance_partitioning : if True and multiple behavior_cols provided,
        compute unique variance contribution of each covariate.

    Returns
    -------
    dict with all analysis results
    """
    from neural_events import build_population_vectors
    from timebase import build_time_grid, bin_continuous_features

    results: dict[str, Any] = {"bin_size": bin_size}

    cols_to_use = behavior_cols if behavior_cols is not None else [behavior_col]

    all_spike_times = np.concatenate(list(spike_times_dict.values())) if spike_times_dict else np.array([])
    if all_spike_times.size > 0:
        t_start, t_end = float(all_spike_times.min()), float(all_spike_times.max())
    elif behavior_df is not None and "t" in behavior_df.columns:
        t_start, t_end = float(behavior_df["t"].min()), float(behavior_df["t"].max())
    else:
        return results

    time_grid = build_time_grid(t_start, t_end, bin_size)
    if len(time_grid) < 40:
        results["error"] = "Not enough time bins for analysis"
        return results

    pop = build_population_vectors(spike_times_dict, time_grid, bin_size)
    pop_rate = pop.sum(axis=1)
    results["n_units"] = pop.shape[1]
    results["n_timebins"] = len(time_grid)
    results["time_range"] = (float(t_start), float(t_end))

    avail_cols = [c for c in cols_to_use if c in behavior_df.columns]
    if not avail_cols:
        results["error"] = f"None of {cols_to_use} found in behavior_df"
        return results

    beh_cols_with_t = ["t"] + avail_cols
    beh_binned = bin_continuous_features(behavior_df[beh_cols_with_t], time_grid)
    primary_col = avail_cols[0]
    beh_signal = beh_binned[primary_col].to_numpy() if primary_col in beh_binned.columns else np.zeros(len(time_grid))

    results["behavior_cols_used"] = avail_cols

    xcorr = crosscorrelation(pop_rate, beh_signal, max_lag=max_lag_bins)
    results["crosscorrelation"] = xcorr
    results["peak_lag_s"] = xcorr["peak_lag"] * bin_size
    results["peak_corr"] = xcorr["peak_corr"]

    unit_xcorr = population_crosscorrelation(pop, beh_signal, max_lag_bins, bin_size)
    results["unit_crosscorrelation"] = unit_xcorr

    window_size = min(200, len(time_grid) // 5)
    if window_size >= 20:
        slide = sliding_correlation(pop_rate, beh_signal, window_size=window_size, step=window_size // 4)
        results["sliding_correlation"] = slide

    covariate_dict: dict[str, np.ndarray] = {
        col: beh_binned[col].to_numpy()
        for col in avail_cols
        if col in beh_binned.columns
    }

    if len(covariate_dict) >= 2 and run_variance_partitioning:
        multi_enc = fit_multi_covariate_encoding_model(
            covariate_dict,
            pop_rate,
            bin_size=bin_size,
            gap_bins=gap_bins,
            n_lag_bins=max_lag_bins,
            n_basis=8,
            n_permutations=n_permutations,
        )
        results["encoding"] = {
            "full_r2": multi_enc["full_r2"],
            "unique_r2": multi_enc["unique_r2"],
            "shared_r2": multi_enc["shared_r2"],
            "perm_p_value": multi_enc["perm_results"].get("p_value"),
            "perm_z_score": multi_enc["perm_results"].get("z_score"),
            "covariate_names": multi_enc["covariate_names"],
        }
    else:
        beh_features = beh_binned[avail_cols].copy()
        enc = fit_encoding_model(
            beh_features, pop_rate,
            model_type="ridge",
            use_raised_cosine=True,
            n_lag_bins=max_lag_bins,
            n_basis=8,
            gap_bins=gap_bins,
        )
        results["encoding"] = {"mean_r2": enc["mean_r2"], "cv_scores": enc["cv_scores"]}

        # Permutation test
        if n_permutations > 0 and not np.isnan(enc["mean_r2"]):
            perm = permutation_test(
                beh_features, pop_rate, enc["mean_r2"],
                n_permutations=n_permutations,
                model_type="ridge",
                use_raised_cosine=True,
                n_lag_bins=max_lag_bins,
                n_basis=8,
                gap_bins=gap_bins,
            )
            results["encoding"]["perm_p_value"] = perm["p_value"]
            results["encoding"]["perm_z_score"] = perm["z_score"]

    dec = fit_decoding_model(pop, beh_signal, n_folds=5, gap_bins=gap_bins)
    results["decoding"] = {"mean_r2": dec["mean_r2"], "cv_scores": dec["cv_scores"]}

    max_gc_lag = min(10, len(time_grid) // 10)
    results["granger_neural_to_behavior"] = granger_test(pop_rate, beh_signal, max_lag=max_gc_lag)
    results["granger_behavior_to_neural"] = granger_test(beh_signal, pop_rate, max_lag=max_gc_lag)

    if trials is not None and not trials.empty:
        from neural_events import trial_averaged_rates
        for group_col in ["trial_type", "rewarded", "hit", "miss"]:
            if group_col in trials.columns:
                tavg = trial_averaged_rates(spike_times_dict, trials, group_col)
                results[f"trial_averaged_{group_col}"] = {
                    cond: {
                        "n_units": len(res["unit_ids"]),
                        "n_trials": res["peths"][list(res["peths"].keys())[0]]["n_trials"]
                        if res.get("peths") else 0,
                    }
                    for cond, res in tavg.items()
                }

    return results
