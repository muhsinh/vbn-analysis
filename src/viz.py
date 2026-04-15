"""Visualization helpers for notebooks."""
from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vbn_types import SpikeTimesDict


def plot_raster(spike_times: SpikeTimesDict, max_units: int = 50) -> None:
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


def plot_firing_rate_summary(spike_times: SpikeTimesDict) -> None:
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
    df = trials.copy()

    if "trial_type" not in df.columns:
        plt.figure(figsize=(7, 3))
        plt.plot(df.index, df.get("t_start", df.index))
        plt.title("Trial Start Times")
        plt.xlabel("Trial index")
        plt.ylabel("t_start (s)" if "t_start" in df.columns else "value")
        plt.tight_layout()
        return

    df["trial_type"] = df["trial_type"].astype(str).fillna("unknown")
    counts = df["trial_type"].value_counts(dropna=False)
    total = int(counts.sum())

    has_time = "t_start" in df.columns
    fig_w = 11 if has_time else 7
    fig, axes = plt.subplots(
        1,
        2 if has_time else 1,
        figsize=(fig_w, 3.6),
        gridspec_kw={"width_ratios": [1.1, 1.6]} if has_time else None,
    )
    ax_counts = axes[0] if has_time else axes

    # Count + percent (horizontal bars read better with long category names)
    labels = counts.index.tolist()
    y = np.arange(len(labels))
    ax_counts.barh(y, counts.to_numpy(), color="#3b82f6", alpha=0.9)
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(labels)
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("Count")
    ax_counts.set_title(f"Trial Types (n={total})")
    for yi, c in zip(y, counts.to_numpy()):
        pct = 100.0 * float(c) / float(total) if total else 0.0
        ax_counts.text(c + max(1, 0.01 * total), yi, f"{int(c)}  ({pct:.1f}%)", va="center", fontsize=9)

    if not has_time:
        plt.tight_layout()
        return

    ax_time = axes[1]
    df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce")
    df = df[df["t_start"].notna()].sort_values("t_start").reset_index(drop=True)
    if df.empty:
        ax_time.text(0.5, 0.5, "No valid t_start", ha="center", va="center")
        plt.tight_layout()
        return

    type_to_row = {t: i for i, t in enumerate(labels)}
    rows = df["trial_type"].map(type_to_row).to_numpy()
    t0 = df["t_start"].to_numpy()

    ax_time.scatter(t0, rows, s=10, c="#111827", alpha=0.6, linewidths=0)
    if "t_end" in df.columns:
        t_end = pd.to_numeric(df["t_end"], errors="coerce").to_numpy()
        ok = np.isfinite(t_end)
        for xs, xe, r in zip(t0[ok], t_end[ok], rows[ok]):
            if xe >= xs:
                ax_time.plot([xs, xe], [r, r], color="#111827", alpha=0.15, linewidth=3)

    ax_time.set_yticks(y)
    ax_time.set_yticklabels(labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Time (s, NWB seconds)")
    ax_time.set_title("Trial Timeline")
    ax_time.grid(True, axis="x", alpha=0.2)

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
    df = frame_times.copy()
    if "t" not in df.columns or "frame_idx" not in df.columns:
        print("Frame times missing required columns: frame_idx, t")
        return

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df = df[np.isfinite(df["t"]) & np.isfinite(df["frame_idx"])].sort_values("frame_idx").reset_index(drop=True)
    if len(df) < 3:
        print("Not enough valid frame times to summarize.")
        return

    t = df["t"].to_numpy(dtype=float)
    frame_idx = df["frame_idx"].to_numpy(dtype=int)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        print("No finite frame-to-frame deltas to summarize.")
        return

    dt_med = float(np.median(dt))
    fps_est = (1.0 / dt_med) if dt_med > 0 else None

    # Estimate "lost frames" from large timestamp gaps.
    # This is approximate (camera timing isn't perfect), but it's a useful sanity check.
    lost_est = 0
    gaps = []
    gap_threshold = 2.5 * dt_med if dt_med > 0 else None
    if dt_med > 0:
        gap_mask = dt > gap_threshold
        gap_idxs = np.where(gap_mask)[0]
        for i in gap_idxs:
            gap_s = float(t[i + 1] - t[i])
            missing = max(0, int(round(gap_s / dt_med)) - 1)
            lost_est += missing
            gaps.append(
                {
                    "after_frame_idx": int(frame_idx[i]),
                    "before_frame_idx": int(frame_idx[i + 1]),
                    "gap_s": gap_s,
                    "missing_est": int(missing),
                }
            )

    total_est = int(len(df) + lost_est)
    lost_pct = 100.0 * float(lost_est) / float(total_est) if total_est > 0 else 0.0

    print(
        "frames_valid="
        + str(int(len(df)))
        + " frames_est_total="
        + str(total_est)
        + " lost_est="
        + str(int(lost_est))
        + f" lost_pct={lost_pct:.3f}%"
        + (f" fps_est={fps_est:.3f}" if fps_est is not None else "")
        + f" frame_idx_range=[{int(frame_idx.min())},{int(frame_idx.max())}]"
    )

    # Plot: dt over time (downsampled) + top gaps bar chart.
    # This stays readable for ~600k frames.
    fig, (ax_dt, ax_gaps) = plt.subplots(
        1,
        2,
        figsize=(11.5, 3.6),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    x_dt = frame_idx[1:]
    y_dt = np.diff(t)
    # Downsample for plotting speed (keep overall texture)
    step = max(1, len(x_dt) // 8000)
    ax_dt.plot(x_dt[::step], y_dt[::step], color="#111827", linewidth=0.8, alpha=0.7)
    ax_dt.axhline(dt_med, color="#2563eb", linewidth=1.2, label=f"median dt={dt_med:.5f}s")
    if gap_threshold is not None:
        ax_dt.axhline(gap_threshold, color="#ef4444", linewidth=1.0, linestyle="--", label="drop threshold")

    if gaps:
        gap_x = np.array([g["after_frame_idx"] for g in gaps], dtype=int)
        gap_y = np.array([g["gap_s"] for g in gaps], dtype=float)
        gstep = max(1, len(gap_x) // 2000)
        ax_dt.scatter(gap_x[::gstep], gap_y[::gstep], s=10, color="#ef4444", alpha=0.6, linewidths=0)

    ax_dt.set_title(f"Frame-to-frame dt (lost~{lost_pct:.3f}%, fps~{fps_est:.3f})" if fps_est else f"Frame-to-frame dt (lost~{lost_pct:.3f}%)")
    ax_dt.set_xlabel("Frame index")
    ax_dt.set_ylabel("dt (s)")
    ax_dt.grid(True, alpha=0.15)
    ax_dt.legend(fontsize=8, frameon=False, loc="upper right")

    # Top gaps bar chart
    if gaps:
        gaps_sorted = sorted(gaps, key=lambda g: (g["missing_est"], g["gap_s"]), reverse=True)[:10]
        labels = [f"{g['after_frame_idx']}" for g in gaps_sorted]
        vals = [g["missing_est"] for g in gaps_sorted]
        ax_gaps.barh(np.arange(len(vals)), vals, color="#ef4444", alpha=0.85)
        ax_gaps.set_yticks(np.arange(len(vals)))
        ax_gaps.set_yticklabels(labels)
        ax_gaps.invert_yaxis()
        ax_gaps.set_xlabel("missing_est")
        ax_gaps.set_title("Largest gaps (after frame)")
        ax_gaps.grid(True, axis="x", alpha=0.15)
    else:
        ax_gaps.axis("off")
        ax_gaps.text(0.5, 0.5, "No large gaps", ha="center", va="center")

    plt.tight_layout()

    if gaps:
        print("largest_gaps(after->before, missing_est, gap_s):")
        for g in sorted(gaps, key=lambda g: (g["missing_est"], g["gap_s"]), reverse=True)[:10]:
            print(f"  {g['after_frame_idx']}->{g['before_frame_idx']}  missing_est={g['missing_est']}  gap_s={g['gap_s']:.4f}")


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


def plot_model_performance(metrics: dict[str, Any]) -> None:
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




def plot_peth(peth_result: dict[str, Any], unit_id: str = "", ax: Axes | None = None) -> None:
    """Plot a peri-event time histogram with SEM shading."""
    if not peth_result or peth_result.get("n_trials", 0) == 0:
        print("No PETH data available.")
        return
    t = peth_result["time_bins"]
    mean = peth_result["mean_rate"]
    sem = peth_result["sem_rate"]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, mean, color="#2563eb", linewidth=1.5)
    ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color="#2563eb")
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1, label="event")
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"PETH{' - ' + unit_id if unit_id else ''} (n={peth_result['n_trials']} trials)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.15)


def plot_population_peth(pop_result: dict[str, Any], title: str = "Population PETH") -> None:
    """Plot population PETH as a heatmap (units x time)."""
    if not pop_result or pop_result.get("population_matrix", np.empty(0)).size == 0:
        print("No population PETH data.")
        return
    mat = pop_result["population_matrix"]
    t = pop_result["time_bins"]

    fig, ax = plt.subplots(figsize=(8, max(3, mat.shape[0] * 0.15)))
    im = ax.imshow(
        mat, aspect="auto", cmap="viridis",
        extent=[t[0], t[-1], mat.shape[0] - 0.5, -0.5],
    )
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Unit")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
    plt.tight_layout()


def plot_trial_comparison(
    condition_peths: dict[str, dict[str, Any]],
    unit_id: str | None = None,
) -> None:
    """Plot overlaid PETHs for different trial conditions."""
    if not condition_peths:
        print("No trial-averaged data to plot.")
        return
    colors = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
    fig, ax = plt.subplots(figsize=(8, 3.5))

    for i, (cond, pop_result) in enumerate(condition_peths.items()):
        peths = pop_result.get("peths", {})
        if unit_id and unit_id in peths:
            peth = peths[unit_id]
        elif peths:
            # Average across all units
            all_rates = [p["mean_rate"] for p in peths.values()]
            t = list(peths.values())[0]["time_bins"]
            mean_rate = np.mean(all_rates, axis=0)
            n_trials = list(peths.values())[0]["n_trials"]
            peth = {"time_bins": t, "mean_rate": mean_rate, "sem_rate": np.std(all_rates, axis=0) / np.sqrt(len(all_rates)), "n_trials": n_trials}
        else:
            continue

        c = colors[i % len(colors)]
        ax.plot(peth["time_bins"], peth["mean_rate"], color=c, linewidth=1.5, label=f"{cond} (n={peth['n_trials']})")
        ax.fill_between(peth["time_bins"], peth["mean_rate"] - peth["sem_rate"], peth["mean_rate"] + peth["sem_rate"], alpha=0.15, color=c)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"Trial-averaged response{' - ' + unit_id if unit_id else ' (population mean)'}")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()


def plot_crosscorrelation(xcorr: dict[str, Any], bin_size: float = 0.025) -> None:
    """Plot cross-correlation function between neural and behavior."""
    if not xcorr:
        print("No cross-correlation data.")
        return
    lags = xcorr["lags"] * bin_size  # convert to seconds
    corr = xcorr["correlation"]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lags, corr, color="#2563eb", linewidth=1.5)
    ax.fill_between(lags, 0, corr, alpha=0.15, color="#2563eb")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    peak_s = xcorr["peak_lag"] * bin_size
    ax.axvline(peak_s, color="#ef4444", linestyle="--", linewidth=1, label=f"peak={peak_s:.3f}s (r={xcorr['peak_corr']:.3f})")
    ax.set_xlabel("Lag (s) [negative = neural leads]")
    ax.set_ylabel("Correlation")
    ax.set_title("Neural-Behavior Cross-Correlation")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()


def plot_sliding_correlation(slide: dict[str, Any], bin_size: float = 0.025) -> None:
    """Plot sliding-window correlation over time."""
    if not slide or len(slide.get("correlations", [])) == 0:
        print("No sliding correlation data.")
        return
    centers = slide["window_centers"] * bin_size
    corrs = slide["correlations"]
    pvals = slide["p_values"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(centers, corrs, color="#2563eb", linewidth=1.2)
    ax1.fill_between(centers, 0, corrs, alpha=0.15, color="#2563eb")
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax1.set_ylabel("Pearson r")
    ax1.set_title("Sliding-Window Neural-Behavior Correlation")
    ax1.grid(True, alpha=0.15)

    sig = np.array(pvals) < 0.05
    ax2.scatter(centers[sig], -np.log10(np.array(pvals)[sig]), s=8, color="#ef4444", alpha=0.7, label="p<0.05")
    ax2.scatter(centers[~sig], -np.log10(np.array(pvals)[~sig]), s=8, color="#94a3b8", alpha=0.4)
    ax2.axhline(-np.log10(0.05), color="#ef4444", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("-log10(p)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()


def plot_encoding_decoding(enc_result: dict[str, Any], dec_result: dict[str, Any]) -> None:
    """Plot encoding vs decoding model performance."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    labels = []
    values = []
    colors = []

    if enc_result.get("cv_scores"):
        labels.append("Encoding\n(behavior->neural)")
        values.append(enc_result["cv_scores"])
        colors.append("#2563eb")

    if dec_result.get("cv_scores"):
        labels.append("Decoding\n(neural->behavior)")
        values.append(dec_result["cv_scores"])
        colors.append("#10b981")

    if not values:
        print("No model results to plot.")
        return

    bp = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    for i, v in enumerate(values):
        ax.scatter([i + 1] * len(v), v, color=colors[i], s=30, zorder=3, alpha=0.7)
        ax.text(i + 1.25, np.mean(v), f"mean={np.mean(v):.3f}", fontsize=8, va="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("R-squared (CV)")
    ax.set_title("Encoding vs Decoding Performance")
    ax.grid(True, axis="y", alpha=0.15)
    plt.tight_layout()


def plot_granger_summary(gc_n2b: dict[str, Any], gc_b2n: dict[str, Any]) -> None:
    """Plot Granger causality results in both directions."""
    fig, ax = plt.subplots(figsize=(6, 3))

    labels = ["Neural -> Behavior", "Behavior -> Neural"]
    f_stats = [gc_n2b.get("f_statistic", 0), gc_b2n.get("f_statistic", 0)]
    p_vals = [gc_n2b.get("p_value", 1), gc_b2n.get("p_value", 1)]
    improvements = [gc_n2b.get("improvement", 0), gc_b2n.get("improvement", 0)]
    colors = ["#ef4444" if p < 0.05 else "#94a3b8" for p in p_vals]

    bars = ax.barh([0, 1], f_stats, color=colors, alpha=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("F-statistic")
    ax.set_title("Granger Causality")

    for i, (f, p, imp) in enumerate(zip(f_stats, p_vals, improvements)):
        sig = "*" if p < 0.05 else ""
        ax.text(f + 0.1, i, f"p={p:.4f}{sig}  R2 gain={imp:.4f}", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.15)
    plt.tight_layout()


def plot_unit_lag_distribution(unit_xcorr: "pd.DataFrame", bin_size: float = 0.025) -> None:
    """Plot distribution of peak lags across units."""
    if unit_xcorr is None or unit_xcorr.empty:
        print("No per-unit cross-correlation data.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    lags_s = unit_xcorr["peak_lag_s"].to_numpy()
    corrs = unit_xcorr["peak_corr"].to_numpy()

    ax1.hist(lags_s, bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="#ef4444", linestyle="--", linewidth=1)
    ax1.set_xlabel("Peak lag (s)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Neural-Behavior Lag")

    scatter_c = np.where(corrs > 0, "#2563eb", "#ef4444")
    ax2.scatter(lags_s, np.abs(corrs), c=scatter_c, s=15, alpha=0.6)
    ax2.set_xlabel("Peak lag (s)")
    ax2.set_ylabel("|Peak correlation|")
    ax2.set_title("Lag vs Correlation Strength")
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
