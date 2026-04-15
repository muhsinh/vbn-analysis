"""Modeling utilities for neural-behavior fusion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from timebase import build_time_grid, bin_spike_times, bin_continuous_features
from vbn_types import SpikeTimesDict


def time_blocked_splits(
    n_samples: int,
    n_splits: int = 5,
    gap_bins: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining time-blocked cross-validation splits.

    Each split trains only on data BEFORE the test block, with an optional
    temporal gap to prevent autocorrelation leakage.

    Parameters
    ----------
    n_samples : int
    n_splits : int
    gap_bins : int
        Number of bins to exclude between the end of training and the start
        of the test block. Use ~20 bins (500ms at 25ms bins) for behavioral
        signals with long autocorrelation.
    """
    if n_samples < 4:
        return []
    block = max(1, n_samples // (n_splits + 1))
    splits = []
    for i in range(1, n_splits + 1):
        test_start = i * block
        test_end = min((i + 1) * block, n_samples)
        train_end = max(0, test_start - gap_bins)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    shift = shift % len(arr)
    return np.concatenate([arr[-shift:], arr[:-shift]])



def build_fusion_table(
    spike_times: SpikeTimesDict | None,
    motifs: pd.DataFrame | None,
    bin_size_s: float,
) -> pd.DataFrame:
    if spike_times:
        all_times = np.concatenate(list(spike_times.values()))
        t_start, t_end = float(all_times.min()), float(all_times.max())
    elif motifs is not None and not motifs.empty:
        t_start, t_end = float(motifs["t"].min()), float(motifs["t"].max())
    else:
        t_start, t_end = 0.0, 10.0

    time_grid = build_time_grid(t_start, t_end, bin_size_s)
    spike_counts = bin_spike_times(spike_times or {}, time_grid, bin_size_s)

    if motifs is not None and not motifs.empty:
        motifs_binned = bin_continuous_features(motifs, time_grid)
    else:
        motifs_binned = pd.DataFrame({"t": time_grid})

    fusion = spike_counts.merge(motifs_binned, on="t", how="left")
    return fusion
