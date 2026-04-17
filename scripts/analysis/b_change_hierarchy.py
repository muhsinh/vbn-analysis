"""Deliverable B: change-detection signal across the visual hierarchy.

For each area, compute the population PETH for change flashes (is_change=1) vs
non-change repeat flashes, both within the active block. Quantify:
- Change-response magnitude (peak FR in [0,250]ms)
- Change-signal latency (first bin where change PSTH exceeds non-change by 2 SD of baseline)

Piet 2025 claim to test: LP leads change decoding at ~53ms, cortex follows at ~60ms.
(We don't have LP in this session but can rank LGd/cortex/MGd/SCig.)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import (ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, load_session)


def population_psth(spike_times_dict: dict, uids: list[str],
                    event_times: np.ndarray,
                    window: tuple[float, float], bin_size: float = 0.005,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-averaged, unit-averaged PSTH. Returns (time, mean_hz, sem_hz)."""
    pre, post = window
    edges = np.arange(pre, post + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    per_unit = []
    for uid in uids:
        st = np.sort(spike_times_dict[uid])
        counts = np.zeros(len(centers))
        for t in event_times:
            idx_lo = np.searchsorted(st, t + pre)
            idx_hi = np.searchsorted(st, t + post)
            rel = st[idx_lo:idx_hi] - t
            if rel.size:
                c, _ = np.histogram(rel, bins=edges)
                counts += c
        rate = counts / len(event_times) / bin_size
        per_unit.append(rate)
    M = np.array(per_unit) if per_unit else np.empty((0, len(centers)))
    mean_hz = M.mean(axis=0) if M.size else np.zeros(len(centers))
    sem_hz = M.std(axis=0, ddof=1) / np.sqrt(max(M.shape[0], 1)) if M.shape[0] > 1 else np.zeros(len(centers))
    return centers, mean_hz, sem_hz


def response_latency(time: np.ndarray, change_psth: np.ndarray,
                     ref_psth: np.ndarray, baseline_window: tuple[float, float] = (-0.25, -0.05),
                     n_std: float = 2.0, min_duration_s: float = 0.020,
                     search_window: tuple[float, float] = (0.0, 0.15)) -> float:
    """Latency: first time after 0 where (change - ref) exceeds n_std x baseline_std
    for at least min_duration_s consecutively."""
    bin_size = float(time[1] - time[0])
    diff = change_psth - ref_psth
    base_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
    std = diff[base_mask].std()
    if std == 0 or not np.isfinite(std):
        return np.nan
    threshold = n_std * std
    search_mask = (time >= search_window[0]) & (time <= search_window[1])
    above = (diff > threshold) & search_mask
    min_bins = int(min_duration_s / bin_size)
    # find first index with min_bins consecutive True
    for i in range(len(above) - min_bins):
        if above[i:i + min_bins].all():
            return float(time[i])
    return np.nan


def main() -> None:
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]

    active = stim[stim["stimulus_block"] == 0]
    t_change = active[active["is_change"] == 1]["t"].dropna().to_numpy()
    # Non-change repeat: match to same image set and active block, just not is_change
    t_repeat = active[(active["is_change"] == 0) & (active["is_omission"] == 0)]["t"].dropna().to_numpy()
    rng = np.random.default_rng(42)
    n = min(len(t_change), len(t_repeat), 500)
    t_change = np.sort(rng.choice(t_change, n, replace=False))
    t_repeat = np.sort(rng.choice(t_repeat, n, replace=False))
    print(f"Change events: {len(t_change)}, Repeat events: {len(t_repeat)}")

    WINDOW = (-0.25, 0.35)
    BIN = 0.005  # 5 ms for latency measurement

    out_rows = []
    psths = {}
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        t_c, mean_c, sem_c = population_psth(spikes, uids, t_change, WINDOW, BIN)
        _, mean_r, sem_r = population_psth(spikes, uids, t_repeat, WINDOW, BIN)

        lat = response_latency(t_c, mean_c, mean_r)
        base_mask = (t_c >= -0.25) & (t_c <= -0.05)
        evoked_mask = (t_c >= 0.03) & (t_c <= 0.20)
        peak_change = mean_c[evoked_mask].max() - mean_c[base_mask].mean()
        peak_repeat = mean_r[evoked_mask].max() - mean_r[base_mask].mean()
        change_ratio = peak_change / peak_repeat if peak_repeat > 0 else np.nan

        out_rows.append(dict(
            area=area, n_units=len(uids),
            peak_change_hz=peak_change, peak_repeat_hz=peak_repeat,
            change_amplification=change_ratio,
            change_latency_s=lat,
        ))
        psths[area] = dict(time=t_c, change=mean_c, repeat=mean_r,
                           sem_change=sem_c, sem_repeat=sem_r, n_units=len(uids))

    summary = pd.DataFrame(out_rows)
    order = [a for a in VISUAL_HIERARCHY + ["MGd", "CA1", "DG", "ProS", "SCig", "MRN"]
             if a in summary["area"].values]
    summary["area"] = pd.Categorical(summary["area"], categories=order, ordered=True)
    summary = summary.sort_values("area").reset_index(drop=True)
    summary.to_csv(REPORTS / "B_change_hierarchy.csv", index=False)
    print("\n=== Change response by area ===")
    print(summary.to_string(index=False))

    # Piet 2025 prediction: LGd 38ms image-ID; change-signal LGd/VISp ~60ms, LP leads at 53ms
    print("\nChange-signal latencies (Piet 2025 reports LGd/VISp/VISl/VISal ~60ms):")
    for _, r in summary.iterrows():
        if np.isfinite(r["change_latency_s"]):
            print(f"  {str(r['area']):8s}  {r['change_latency_s']*1000:5.0f} ms  (n={r['n_units']})")

    # === PLOTS ===
    n_plots = len(psths)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, (area, d) in zip(axes, psths.items()):
        ax.plot(d["time"] * 1000, d["change"], color="#d62728", lw=1.5, label="change")
        ax.fill_between(d["time"] * 1000, d["change"] - d["sem_change"], d["change"] + d["sem_change"],
                        color="#d62728", alpha=0.20)
        ax.plot(d["time"] * 1000, d["repeat"], color="#1f77b4", lw=1.5, label="repeat")
        ax.fill_between(d["time"] * 1000, d["repeat"] - d["sem_repeat"], d["repeat"] + d["sem_repeat"],
                        color="#1f77b4", alpha=0.20)
        # latency marker
        row = summary[summary["area"] == area]
        if not row.empty and np.isfinite(row["change_latency_s"].iloc[0]):
            lat_ms = row["change_latency_s"].iloc[0] * 1000
            ax.axvline(lat_ms, color="k", ls=":", alpha=0.6, lw=1)
            ax.text(lat_ms + 3, ax.get_ylim()[1] * 0.85, f"{lat_ms:.0f}ms",
                    fontsize=8, alpha=0.8)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(f"{area}  (n={d['n_units']} units)", fontsize=10)
        ax.set_xlabel("Time from flash (ms)", fontsize=8)
        ax.set_ylabel("FR (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(psths):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Deliverable B — Change vs Repeat PSTH, active block, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "B_change_psth_per_area.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Latency hierarchy barplot
    fig, ax = plt.subplots(figsize=(9, 4))
    lat_df = summary.dropna(subset=["change_latency_s"]).copy()
    lat_df["lat_ms"] = lat_df["change_latency_s"] * 1000
    ax.bar(range(len(lat_df)), lat_df["lat_ms"], color="#444")
    ax.axhline(53, color="#d62728", ls="--", label="Piet 2025 LP (53ms)")
    ax.axhline(60, color="#ff7f0e", ls="--", label="Piet 2025 cortex (60ms)")
    ax.set_xticks(range(len(lat_df)))
    ax.set_xticklabels(lat_df["area"], rotation=35, ha="right")
    ax.set_ylabel("Change-signal latency (ms)")
    ax.set_title("Change-signal onset latency (first sustained deviation from repeat)")
    ax.legend()
    fig.tight_layout()
    out = REPORTS / "B_change_latency_hierarchy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

    (REPORTS / "B_summary.json").write_text(json.dumps({
        "n_change": int(len(t_change)),
        "n_repeat": int(len(t_repeat)),
        "area_latencies_ms": {str(r["area"]): (float(r["change_latency_s"]) * 1000
                                              if np.isfinite(r["change_latency_s"]) else None)
                              for _, r in summary.iterrows()},
        "area_change_amplification": {str(r["area"]): (float(r["change_amplification"])
                                                       if np.isfinite(r["change_amplification"]) else None)
                                      for _, r in summary.iterrows()},
    }, indent=2))
    print(f"Summary: {REPORTS / 'B_summary.json'}")


if __name__ == "__main__":
    main()
