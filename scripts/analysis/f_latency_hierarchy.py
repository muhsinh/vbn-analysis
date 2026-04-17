"""Deliverable F: flash-onset latency across the visual hierarchy.

Separate from Deliverable B (which measured change-signal latency). Here we measure
the latency of the stimulus-evoked response itself on non-change flashes during
the active block. Piet 2025 reports:
  LGd: 38 ms
  VISp/VISl/VISal: 44-46 ms
  VISpm/VISam: ~53 ms

Method: per-area population PSTH; latency = first bin after 0 where PSTH exceeds
baseline by 4 SD for >=20 ms continuously (matches Allen/Piet method).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, load_session


def population_psth(spike_times_dict: dict, uids: list[str],
                    event_times: np.ndarray, window: tuple[float, float],
                    bin_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        per_unit.append(counts / len(event_times) / bin_size)
    M = np.array(per_unit) if per_unit else np.empty((0, len(centers)))
    mean_hz = M.mean(axis=0) if M.size else np.zeros(len(centers))
    sem_hz = M.std(axis=0, ddof=1) / np.sqrt(max(M.shape[0], 1)) if M.shape[0] > 1 else np.zeros(len(centers))
    return centers, mean_hz, sem_hz


def onset_latency(time: np.ndarray, psth: np.ndarray,
                  baseline_window: tuple[float, float] = (-0.20, -0.05),
                  n_std: float = 4.0, min_duration_s: float = 0.020,
                  search_window: tuple[float, float] = (0.0, 0.15)) -> float:
    """Latency = first time where PSTH > baseline_mean + n_std*baseline_sd
    for >= min_duration_s continuously."""
    bin_size = float(time[1] - time[0])
    base_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
    base_mean = psth[base_mask].mean()
    base_sd = psth[base_mask].std()
    if base_sd == 0 or not np.isfinite(base_sd):
        return np.nan
    threshold = base_mean + n_std * base_sd
    search_mask = (time >= search_window[0]) & (time <= search_window[1])
    above = (psth > threshold) & search_mask
    min_bins = max(int(min_duration_s / bin_size), 1)
    for i in range(len(above) - min_bins):
        if above[i:i + min_bins].all():
            return float(time[i])
    return np.nan


def main() -> None:
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]

    # Non-change flashes in active block
    stim_f = stim[(stim["is_change"] == 0) & (stim["is_omission"] == 0)
                  & (stim["stimulus_block"] == 0)].copy()
    rng = np.random.default_rng(0)
    N_EVT = min(len(stim_f), 2000)
    events = np.sort(rng.choice(stim_f["t"].dropna().values, N_EVT, replace=False))
    print(f"Flash events: {len(events)}")

    WINDOW = (-0.20, 0.20)
    BIN = 0.002  # 2 ms — fine enough for sub-50ms latencies

    rows = []
    psths = {}
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        t, mu, sem = population_psth(spikes, uids, events, WINDOW, BIN)
        lat = onset_latency(t, mu)
        base = mu[(t >= -0.20) & (t <= -0.05)].mean()
        peak = mu[(t >= 0.0) & (t <= 0.15)].max()
        rows.append(dict(area=area, n_units=len(uids), latency_s=lat,
                          baseline_hz=base, peak_hz=peak, evoked_hz=peak - base))
        psths[area] = dict(time=t, psth=mu, sem=sem, n_units=len(uids))

    df = pd.DataFrame(rows)
    order = [a for a in VISUAL_HIERARCHY + ["MGd", "CA1", "DG", "ProS", "SCig", "MRN"]
             if a in df["area"].values]
    df["area"] = pd.Categorical(df["area"], categories=order, ordered=True)
    df = df.sort_values("area").reset_index(drop=True)
    df.to_csv(REPORTS / "F_latency_hierarchy.csv", index=False)

    print("\n=== Flash-onset latency by area ===")
    print(f"{'area':8s} {'n_u':>4s} {'lat(ms)':>8s} {'base_hz':>8s} {'peak_hz':>8s} {'Piet2025':>10s}")
    piet_ref = {"LGd": 38, "VISp": 44, "VISl": 45, "VISal": 46, "VISpm": 53, "VISam": 53}
    for _, r in df.iterrows():
        ref = piet_ref.get(str(r["area"]), None)
        ref_str = f"{ref} ms" if ref else "-"
        lat_str = f"{r['latency_s']*1000:.1f}" if np.isfinite(r["latency_s"]) else "NaN"
        print(f"{str(r['area']):8s} {r['n_units']:>4d} {lat_str:>8s} {r['baseline_hz']:>8.2f} {r['peak_hz']:>8.2f} {ref_str:>10s}")

    # Hierarchy correlation
    vis_mask = df["area"].isin(VISUAL_HIERARCHY)
    vis_df = df[vis_mask].dropna(subset=["latency_s"]).copy()
    if len(vis_df) >= 3:
        vis_df["rank"] = [VISUAL_HIERARCHY.index(a) for a in vis_df["area"]]
        r, p = stats.spearmanr(vis_df["rank"], vis_df["latency_s"])
        print(f"\nSpearman: visual hierarchy rank vs onset latency: r={r:.3f}, p={p:.3f}")
        print(f"  Piet 2025 predicts r > 0 (latency increases up hierarchy)")

    # === PLOTS ===
    # Panel A: stacked PSTHs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(VISUAL_HIERARCHY)))
    for i, area in enumerate(VISUAL_HIERARCHY):
        if area not in psths:
            continue
        d = psths[area]
        # normalize each PSTH to baseline
        base = d["psth"][(d["time"] >= -0.20) & (d["time"] <= -0.05)].mean()
        pk = d["psth"][(d["time"] >= 0.0) & (d["time"] <= 0.15)].max() - base
        if pk <= 0:
            continue
        norm = (d["psth"] - base) / pk
        ax.plot(d["time"] * 1000, norm, color=cmap[i], lw=1.8, label=f"{area} (n={d['n_units']})")
    ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlim(-50, 150)
    ax.set_xlabel("Time from flash (ms)")
    ax.set_ylabel("Normalized response (0-peak)")
    ax.set_title("Visual hierarchy flash PSTHs (normalized)")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: latency bars vs Piet reference
    ax = axes[1]
    plot_df = df.dropna(subset=["latency_s"]).copy()
    plot_df["lat_ms"] = plot_df["latency_s"] * 1000
    xs = np.arange(len(plot_df))
    ax.bar(xs - 0.2, plot_df["lat_ms"], 0.4, label="This session", color="#444")
    piet_vals = [piet_ref.get(str(a), np.nan) for a in plot_df["area"]]
    ax.bar(xs + 0.2, piet_vals, 0.4, label="Piet 2025 ref", color="#d62728", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_df["area"], rotation=35, ha="right")
    ax.set_ylabel("Flash-onset latency (ms)")
    ax.set_title("Flash-onset latency: this session vs Piet 2025")
    ax.legend()

    fig.suptitle(f"Deliverable F — Flash-onset latency hierarchy, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "F_latency_hierarchy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "F_summary.json").write_text(json.dumps({
        "n_events": int(len(events)),
        "area_latencies_ms": {str(r["area"]): (float(r["latency_s"]) * 1000
                                              if np.isfinite(r["latency_s"]) else None)
                              for _, r in df.iterrows()},
        "piet_2025_reference_ms": piet_ref,
    }, indent=2))
    print(f"Summary: {REPORTS / 'F_summary.json'}")


if __name__ == "__main__":
    main()
