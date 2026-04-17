"""Deliverable B2 (post-critique): change hierarchy with lick/motor controls.

Three variants to address Allen-insider critique that SCig active-dominance and
change amplification may reflect motor/lick preparation, not sensory processing:

1. STRICT: exclude all change trials with response_latency < 250 ms (too-fast
   licks that overlap the 30-200ms measurement window).

2. MISS-ONLY: change PSTH on miss trials only — no lick at all. If SCig (and
   other areas) still show amplification, the signal is sensory/attentional.
   If they collapse, the signal was motor.

3. ADAPTATION-MATCHED: compare change vs N=1 repeats (first presentations after
   a prior change), not all non-change flashes. Controls for adaptation release.

Outputs: B2_change_hierarchy_variants.csv + PNG with 3-panel comparison.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import (ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, load_session)


def population_psth(spike_times_dict, uids, event_times, window, bin_size):
    pre, post = window
    edges = np.arange(pre, post + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    per_unit = []
    for uid in uids:
        st = np.sort(spike_times_dict[uid])
        counts = np.zeros(len(centers))
        for t in event_times:
            lo = np.searchsorted(st, t + pre)
            hi = np.searchsorted(st, t + post)
            rel = st[lo:hi] - t
            if rel.size:
                c, _ = np.histogram(rel, bins=edges)
                counts += c
        per_unit.append(counts / len(event_times) / bin_size)
    M = np.array(per_unit) if per_unit else np.empty((0, len(centers)))
    mean_hz = M.mean(axis=0) if M.size else np.zeros(len(centers))
    sem_hz = M.std(axis=0, ddof=1) / np.sqrt(max(M.shape[0], 1)) if M.shape[0] > 1 else np.zeros(len(centers))
    return centers, mean_hz, sem_hz


def change_ratio_per_area(spikes, area_uids, t_change, t_repeat, window=(-0.25, 0.35), bin_size=0.005):
    """Return dict {area: change/repeat peak ratio} and the PSTHs."""
    results = {}
    for area, uids in area_uids.items():
        if len(uids) < 5:
            continue
        t_c, mu_c, _ = population_psth(spikes, uids, t_change, window, bin_size)
        _, mu_r, _ = population_psth(spikes, uids, t_repeat, window, bin_size)
        base_mask = (t_c >= -0.25) & (t_c <= -0.05)
        evoked_mask = (t_c >= 0.03) & (t_c <= 0.20)
        peak_c = mu_c[evoked_mask].max() - mu_c[base_mask].mean()
        peak_r = mu_r[evoked_mask].max() - mu_r[base_mask].mean()
        results[area] = dict(n_units=len(uids),
                             peak_change=peak_c, peak_repeat=peak_r,
                             ratio=peak_c / peak_r if peak_r > 0 else np.nan,
                             psth_change=mu_c, psth_repeat=mu_r, time=t_c)
    return results


def main() -> None:
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]
    trials = data["trials"]

    # Compute response_latency
    trials = trials.copy()
    trials["response_latency"] = trials["response_time"] - trials["t"]

    area_uids = {
        area: [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
               if str(u) in spikes]
        for area in ALL_TARGET_AREAS
    }
    area_uids = {a: u for a, u in area_uids.items() if len(u) >= 5}

    active_stim = stim[stim["stimulus_block"] == 0].copy()

    # === Variant 1: STRICT — exclude trials with rl < 250ms ===
    legit_changes = trials[
        (trials["go"] == 1) & (trials.get("aborted", 0) != 1)
        & ((trials["response_latency"] >= 0.25) | trials["response_latency"].isna())
    ]
    t_change_legit = legit_changes["t"].dropna().to_numpy()
    print(f"Variant 1 (STRICT): {len(t_change_legit)} change trials (latency>=250ms or miss)")

    # === Variant 2: MISS-ONLY — no lick on these trials ===
    miss_changes = trials[(trials["miss"] == 1) & (trials.get("aborted", 0) != 1)]
    t_change_miss = miss_changes["t"].dropna().to_numpy()
    print(f"Variant 2 (MISS-ONLY): {len(t_change_miss)} miss trials")

    # === Baseline (repeat) — flashes_since_change = 1 (adaptation-matched) ===
    adap_matched = active_stim[
        (active_stim["is_change"] == 0) & (active_stim["is_omission"] == 0)
        & (active_stim["flashes_since_change"] == 1)
    ]
    t_repeat_matched = adap_matched["t"].dropna().to_numpy()
    print(f"Adaptation-matched repeat: {len(t_repeat_matched)} flashes_since_change=1")

    # === Old variant (all non-change) for comparison ===
    all_repeat = active_stim[
        (active_stim["is_change"] == 0) & (active_stim["is_omission"] == 0)
    ]
    t_repeat_all = all_repeat["t"].dropna().to_numpy()
    print(f"All repeats (old baseline): {len(t_repeat_all)}")

    # Match event counts across conditions for fair comparison
    rng = np.random.default_rng(42)
    def sub(arr, n):
        n = min(n, len(arr))
        return np.sort(rng.choice(arr, n, replace=False))

    # Variant 1 comparisons
    n1 = min(len(t_change_legit), len(t_repeat_matched))
    r_strict = change_ratio_per_area(spikes, area_uids, sub(t_change_legit, n1), sub(t_repeat_matched, n1))
    print(f"\n=== Variant 1 (STRICT, n_change={n1}, rl>=250ms OR miss; repeat=fsc1) ===")

    # Variant 2: miss-only change vs adaptation-matched repeat
    n2 = min(len(t_change_miss), len(t_repeat_matched))
    r_miss = change_ratio_per_area(spikes, area_uids, sub(t_change_miss, n2), sub(t_repeat_matched, n2))
    print(f"=== Variant 2 (MISS-ONLY, n_change={n2}, repeat=fsc1) ===")

    # Old variant for reference
    n_old = min(len(t_change_legit), len(t_repeat_all), 200)
    r_old = change_ratio_per_area(spikes, area_uids, sub(t_change_legit, n_old), sub(t_repeat_all, n_old))

    # Consolidate
    rows = []
    for area in area_uids:
        if area not in r_strict:
            continue
        rows.append({
            "area": area,
            "n_units": r_strict[area]["n_units"],
            "ratio_old": r_old.get(area, {}).get("ratio", np.nan),
            "ratio_strict": r_strict[area]["ratio"],
            "ratio_miss_only": r_miss.get(area, {}).get("ratio", np.nan),
            "peak_change_strict_hz": r_strict[area]["peak_change"],
            "peak_change_miss_hz": r_miss.get(area, {}).get("peak_change", np.nan),
        })
    df = pd.DataFrame(rows)
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in df["area"].values]
    df["area"] = pd.Categorical(df["area"], categories=order, ordered=True)
    df = df.sort_values("area").reset_index(drop=True)
    df.to_csv(REPORTS / "B2_change_hierarchy_lick_controlled.csv", index=False)
    print("\n=== Side-by-side ratios ===")
    print(df.to_string(index=False))

    # Critical SCig check
    if "SCig" in df["area"].values:
        sc_row = df[df["area"] == "SCig"].iloc[0]
        print("\n=== CRITICAL SCig sanity check (motor vs sensory) ===")
        print(f"  Old ratio (all change vs all repeat):     {sc_row['ratio_old']:.2f}x")
        print(f"  Strict (rl>=250ms or miss, fsc=1 repeat): {sc_row['ratio_strict']:.2f}x")
        print(f"  MISS-ONLY (no lick, fsc=1 repeat):        {sc_row['ratio_miss_only']:.2f}x")
        if sc_row["ratio_miss_only"] > 1.5:
            print(f"  → SCig change amplification SURVIVES in miss trials — real sensory/attentional signal")
        else:
            print(f"  → SCig change amplification COLLAPSES in miss trials — motor/lick artifact")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    xs = np.arange(len(df))
    for ax, col, title in zip(
        axes, ["ratio_old", "ratio_strict", "ratio_miss_only"],
        ["Old (all change, all repeat)", "STRICT (rl>=250ms, fsc=1)", "MISS-ONLY (no lick, fsc=1)"],
    ):
        colors = []
        for a, v in zip(df["area"], df[col]):
            if a in ("SCig", "MRN"):
                colors.append("#d62728")  # highlight motor areas
            else:
                colors.append("#1f77b4")
        ax.bar(xs, df[col], color=colors)
        ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(df["area"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Change / Repeat peak FR ratio")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Deliverable B2 — Change amplification under lick/motor controls, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "B2_change_hierarchy_variants.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "B2_summary.json").write_text(json.dumps({
        "n_strict": int(n1),
        "n_miss_only": int(n2),
        "n_old": int(n_old),
        "variants": {str(r["area"]): {
            "n_units": int(r["n_units"]),
            "ratio_old": float(r["ratio_old"]) if np.isfinite(r["ratio_old"]) else None,
            "ratio_strict": float(r["ratio_strict"]) if np.isfinite(r["ratio_strict"]) else None,
            "ratio_miss_only": float(r["ratio_miss_only"]) if np.isfinite(r["ratio_miss_only"]) else None,
        } for _, r in df.iterrows()},
    }, indent=2))


if __name__ == "__main__":
    main()
