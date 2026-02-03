"""Visualization helpers for notebooks."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_raster(spike_times: Dict[str, np.ndarray], max_units: int = 50) -> None:
    if not spike_times:
        print("No spike times available for raster.")
        return
    plt.figure(figsize=(8, 4))
    for idx, (unit_id, times) in enumerate(list(spike_times.items())[:max_units]):
        plt.vlines(times, idx + 0.5, idx + 1.5)
    plt.title("Spike Raster (subset)")
    plt.xlabel("Time (s)")
    plt.ylabel("Unit")
    plt.tight_layout()


def plot_firing_rate_summary(spike_times: Dict[str, np.ndarray]) -> None:
    if not spike_times:
        print("No spike times available for firing rate summary.")
        return
    rates = [len(times) / (times.max() - times.min() + 1e-6) for times in spike_times.values()]
    plt.figure(figsize=(6, 4))
    plt.hist(rates, bins=20)
    plt.title("Firing Rate Distribution")
    plt.xlabel("Hz")
    plt.ylabel("Count")
    plt.tight_layout()


def plot_behavior_summary(trials: pd.DataFrame | None) -> None:
    if trials is None or trials.empty:
        print("No trials available for behavior summary.")
        return
    plt.figure(figsize=(6, 3))
    if "trial_type" in trials.columns:
        trials["trial_type"].value_counts().plot(kind="bar")
        plt.title("Trial Types")
    else:
        plt.plot(trials.index, trials.get("t_start", trials.index))
        plt.title("Trial Start Times")
    plt.tight_layout()


def plot_eye_qc(eye_df: pd.DataFrame | None) -> None:
    if eye_df is None or eye_df.empty:
        print("No eye data available for QC plot.")
        return
    plt.figure(figsize=(8, 3))
    cols = [c for c in eye_df.columns if c != "t"]
    if not cols:
        print("Eye dataframe has no signal columns.")
        return
    plt.plot(eye_df["t"], eye_df[cols[0]])
    plt.title(f"Eye Signal: {cols[0]}")
    plt.xlabel("Time (s)")
    plt.tight_layout()


def plot_video_alignment(frame_times: pd.DataFrame | None) -> None:
    if frame_times is None or frame_times.empty:
        print("No frame times available for alignment plot.")
        return
    plt.figure(figsize=(6, 3))
    plt.plot(frame_times["frame_idx"], frame_times["t"], linewidth=1)
    plt.title("Video Frame Times vs Frame Index")
    plt.xlabel("Frame Index")
    plt.ylabel("Time (s)")
    plt.tight_layout()


def plot_motif_transition(motifs: pd.DataFrame | None) -> None:
    if motifs is None or motifs.empty or "motif_id" not in motifs.columns:
        print("No motifs available for transition plot.")
        return
    motif_ids = motifs["motif_id"].to_numpy()
    n = int(np.max(motif_ids)) + 1 if len(motif_ids) > 0 else 0
    if n <= 1:
        print("Not enough motifs for transition matrix.")
        return
    mat = np.zeros((n, n))
    for a, b in zip(motif_ids[:-1], motif_ids[1:]):
        mat[int(a), int(b)] += 1
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="viridis")
    plt.title("Motif Transition Matrix")
    plt.xlabel("Next")
    plt.ylabel("Current")
    plt.colorbar()
    plt.tight_layout()


def plot_model_performance(metrics: Dict[str, Any]) -> None:
    if not metrics:
        print("No metrics available.")
        return
    keys = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
    vals = [metrics[k] for k in keys]
    plt.figure(figsize=(6, 3))
    plt.bar(keys, vals)
    plt.title("Model Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


def plot_fusion_sanity(fusion: pd.DataFrame, target_col: str) -> None:
    if fusion is None or fusion.empty or target_col not in fusion.columns:
        print("No fusion data available for QC plot.")
        return
    plt.figure(figsize=(8, 3))
    subset = fusion.iloc[: min(1000, len(fusion))]
    plt.plot(subset["t"], subset[target_col])
    plt.title("Fusion QC: Target Signal Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(target_col)
    plt.tight_layout()
