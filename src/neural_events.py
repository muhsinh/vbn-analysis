"""Event-aligned neural analysis.

Provides the core analyses for correlating behavior with neural activity:
- Peri-event time histograms (PETHs)
- Trial-averaged firing rates grouped by condition
- Population activity vectors
- Single-unit event-locked responses
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Peri-Event Time Histograms (PETHs)
# ---------------------------------------------------------------------------

def compute_peth(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
) -> Dict[str, Any]:
    """Compute a peri-event time histogram for a single unit.

    Parameters
    ----------
    spike_times : array
        All spike times for this unit (in seconds, NWB timebase).
    event_times : array
        Times of behavioral events to align to (e.g., stimulus onset).
    window : (pre, post)
        Time window relative to event. Negative = before event.
    bin_size : float
        Bin width in seconds.

    Returns
    -------
    dict with keys:
        time_bins : array of bin centers
        mean_rate : array of mean firing rates (Hz) per bin
        sem_rate  : array of SEM across trials
        trial_spikes : list of arrays, one per trial (spike times relative to event)
        n_trials : int
    """
    pre, post = window
    n_bins = int(np.round((post - pre) / bin_size))
    bin_edges = np.linspace(pre, post, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    trial_counts = []
    trial_spikes_list = []

    for evt in event_times:
        # Spikes relative to this event
        relative = spike_times - evt
        in_window = relative[(relative >= pre) & (relative < post)]
        trial_spikes_list.append(in_window)

        counts, _ = np.histogram(in_window, bins=bin_edges)
        trial_counts.append(counts)

    if not trial_counts:
        return {
            "time_bins": bin_centers,
            "mean_rate": np.zeros(n_bins),
            "sem_rate": np.zeros(n_bins),
            "trial_spikes": [],
            "n_trials": 0,
        }

    trial_counts = np.array(trial_counts)  # (n_trials, n_bins)
    rates = trial_counts / bin_size  # convert to Hz

    return {
        "time_bins": bin_centers,
        "mean_rate": np.mean(rates, axis=0),
        "sem_rate": np.std(rates, axis=0) / np.sqrt(len(rates)) if len(rates) > 1 else np.zeros(n_bins),
        "trial_spikes": trial_spikes_list,
        "n_trials": len(event_times),
    }


def compute_population_peth(
    spike_times_dict: Dict[str, np.ndarray],
    event_times: np.ndarray,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    unit_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """Compute PETHs for all units in the population.

    Returns
    -------
    dict with keys:
        time_bins : array of bin centers
        population_matrix : (n_units, n_bins) array of mean firing rates
        unit_ids : list of unit IDs in order
        peths : dict mapping unit_id -> individual PETH result
    """
    if unit_ids is None:
        unit_ids = list(spike_times_dict.keys())

    peths = {}
    pop_matrix = []

    for uid in unit_ids:
        st = spike_times_dict.get(uid, np.array([]))
        peth = compute_peth(st, event_times, window, bin_size)
        peths[uid] = peth
        pop_matrix.append(peth["mean_rate"])

    pop_matrix = np.array(pop_matrix) if pop_matrix else np.empty((0, 0))

    return {
        "time_bins": peths[unit_ids[0]]["time_bins"] if unit_ids else np.array([]),
        "population_matrix": pop_matrix,
        "unit_ids": unit_ids,
        "peths": peths,
    }


# ---------------------------------------------------------------------------
# Trial-averaged responses by condition
# ---------------------------------------------------------------------------

def trial_averaged_rates(
    spike_times_dict: Dict[str, np.ndarray],
    trials: pd.DataFrame,
    group_col: str = "trial_type",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.025,
) -> Dict[str, Dict[str, Any]]:
    """Compute mean firing rates per condition per unit.

    Parameters
    ----------
    spike_times_dict : dict
        Unit ID -> spike times array.
    trials : DataFrame
        Must have a 't' column (event time) and `group_col`.
    group_col : str
        Column to group trials by (e.g., 'trial_type', 'rewarded').
    window, bin_size : as in compute_peth

    Returns
    -------
    dict mapping condition_value -> population PETH result
    """
    if trials is None or trials.empty or "t" not in trials.columns:
        return {}

    if group_col not in trials.columns:
        # Treat all trials as one group
        event_times = trials["t"].dropna().to_numpy()
        return {"all": compute_population_peth(spike_times_dict, event_times, window, bin_size)}

    results = {}
    for condition, group in trials.groupby(group_col):
        event_times = group["t"].dropna().to_numpy()
        if len(event_times) < 2:
            continue
        results[str(condition)] = compute_population_peth(
            spike_times_dict, event_times, window, bin_size
        )

    return results


# ---------------------------------------------------------------------------
# Population vectors and dimensionality reduction
# ---------------------------------------------------------------------------

def build_population_vectors(
    spike_times_dict: Dict[str, np.ndarray],
    time_grid: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """Build (n_timepoints, n_units) population activity matrix.

    Each entry is the spike count in that bin for that unit.
    """
    n_units = len(spike_times_dict)
    n_bins = len(time_grid)
    pop = np.zeros((n_bins, n_units))

    bin_edges = np.append(time_grid, time_grid[-1] + bin_size)

    for i, (uid, st) in enumerate(spike_times_dict.items()):
        counts, _ = np.histogram(st, bins=bin_edges)
        pop[:, i] = counts

    return pop


def reduce_population(
    pop_matrix: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
) -> Tuple[np.ndarray, Any]:
    """Reduce dimensionality of population activity.

    Parameters
    ----------
    pop_matrix : (n_timepoints, n_units)
    method : 'pca' or 'umap'
    n_components : target dimensions

    Returns
    -------
    (reduced_matrix, model) where reduced_matrix is (n_timepoints, n_components)
    """
    if pop_matrix.size == 0:
        return np.empty((0, n_components)), None

    # Z-score each unit
    mean = np.mean(pop_matrix, axis=0, keepdims=True)
    std = np.std(pop_matrix, axis=0, keepdims=True) + 1e-8
    X = (pop_matrix - mean) / std

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=min(n_components, X.shape[1]))
        reduced = model.fit_transform(X)
        return reduced, model

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn required: pip install umap-learn")
        model = umap.UMAP(n_components=n_components)
        reduced = model.fit_transform(X)
        return reduced, model

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Single-unit selectivity
# ---------------------------------------------------------------------------

def compute_selectivity_index(
    spike_times: np.ndarray,
    condition_a_times: np.ndarray,
    condition_b_times: np.ndarray,
    window: Tuple[float, float] = (0.0, 0.5),
) -> Dict[str, float]:
    """Compute how selective a unit is between two conditions.

    Returns d-prime and mean rate difference.
    """
    def _mean_rate(event_times):
        if len(event_times) == 0:
            return np.array([])
        pre, post = window
        rates = []
        for evt in event_times:
            n = np.sum((spike_times >= evt + pre) & (spike_times < evt + post))
            rates.append(n / (post - pre))
        return np.array(rates)

    rates_a = _mean_rate(condition_a_times)
    rates_b = _mean_rate(condition_b_times)

    if rates_a.size == 0 or rates_b.size == 0:
        return {"d_prime": 0.0, "rate_diff": 0.0, "p_value": 1.0}

    mean_a, mean_b = np.mean(rates_a), np.mean(rates_b)
    var_a, var_b = np.var(rates_a, ddof=1) if len(rates_a) > 1 else 0, np.var(rates_b, ddof=1) if len(rates_b) > 1 else 0
    pooled_std = np.sqrt((var_a + var_b) / 2) + 1e-8
    d_prime = (mean_a - mean_b) / pooled_std

    # Mann-Whitney U test
    try:
        from scipy.stats import mannwhitneyu
        _, p_val = mannwhitneyu(rates_a, rates_b, alternative="two-sided")
    except Exception:
        p_val = 1.0

    return {
        "d_prime": float(d_prime),
        "rate_diff": float(mean_a - mean_b),
        "mean_rate_a": float(mean_a),
        "mean_rate_b": float(mean_b),
        "p_value": float(p_val),
    }


def screen_selective_units(
    spike_times_dict: Dict[str, np.ndarray],
    condition_a_times: np.ndarray,
    condition_b_times: np.ndarray,
    window: Tuple[float, float] = (0.0, 0.5),
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Screen all units for selectivity between two conditions.

    Returns a DataFrame sorted by |d_prime|, with significant units flagged.
    """
    rows = []
    for uid, st in spike_times_dict.items():
        sel = compute_selectivity_index(st, condition_a_times, condition_b_times, window)
        sel["unit_id"] = uid
        rows.append(sel)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["significant"] = df["p_value"] < p_threshold
    df["abs_d_prime"] = np.abs(df["d_prime"])
    df = df.sort_values("abs_d_prime", ascending=False).reset_index(drop=True)
    return df
