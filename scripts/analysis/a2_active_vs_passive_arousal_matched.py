"""Deliverable A2 (post-critique): active vs passive flash response, ARO-MATCHED.

Addresses Allen-insider critique: passive block has ~30% quiescent/eyes-closed
time. Blindly pooling all passive flashes may reflect drowsiness rather than
"engaged passive viewing." Piet 2025 and McGinley recommend subsetting to
passive flashes where running + pupil are within active-block quantiles.

Method:
1. Compute per-flash mean pupil + mean running in [-0.5, 0]s pre-flash window.
2. Find the 25-75% IQR of pupil and running from ACTIVE flashes.
3. Keep PASSIVE flashes only if both pupil AND running fall within those IQRs.
4. Keep ACTIVE flashes in same IQR for symmetry (so both sets are "matched state").
5. Recompute per-unit MI using matched flashes.

If visual-cortex passive > active survives after matching → real task effect.
If it collapses or flips → it was an arousal-distribution artifact.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import (ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, load_session)


def mean_pre_stim(df, col, t_centers, window=(-0.5, 0.0)):
    """Mean value of df[col] in (t + window) for each t_center. Vectorized-ish."""
    t_arr = df["t"].values
    v_arr = df[col].values
    valid = np.isfinite(v_arr)
    t_arr, v_arr = t_arr[valid], v_arr[valid]
    out = np.full(len(t_centers), np.nan)
    lo = t_centers + window[0]
    hi = t_centers + window[1]
    idx_lo = np.searchsorted(t_arr, lo)
    idx_hi = np.searchsorted(t_arr, hi)
    for i in range(len(t_centers)):
        if idx_hi[i] > idx_lo[i]:
            out[i] = v_arr[idx_lo[i]:idx_hi[i]].mean()
    return out


def flash_rate(spike_times, event_times, window):
    pre, post = window
    n = 0
    counts = []
    for t in event_times:
        lo, hi = t + pre, t + post
        c = int(np.searchsorted(spike_times, hi) - np.searchsorted(spike_times, lo))
        counts.append(c)
    counts = np.asarray(counts)
    return counts.mean() / (post - pre), counts


def main():
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]
    running, pose = data["running"], data["pose"]

    # Build flash table
    stim_f = stim[(stim["is_change"] == 0) & (stim["is_omission"] == 0)].copy()
    t_active_all = stim_f[stim_f["stimulus_block"] == 0]["t"].dropna().to_numpy()
    t_passive_all = stim_f[stim_f["stimulus_block"] == 5]["t"].dropna().to_numpy()
    print(f"Active flashes (raw): {len(t_active_all)}")
    print(f"Passive flashes (raw): {len(t_passive_all)}")

    # Pre-stim pupil & running
    print("Computing pre-stim pupil and running per flash...")
    pupil_active = mean_pre_stim(pose, "pupil", t_active_all)
    run_active = mean_pre_stim(running, "running", t_active_all)
    pupil_passive = mean_pre_stim(pose, "pupil", t_passive_all)
    run_passive = mean_pre_stim(running, "running", t_passive_all)

    # IQR from active
    pupil_lo = np.nanpercentile(pupil_active, 25)
    pupil_hi = np.nanpercentile(pupil_active, 75)
    run_lo = np.nanpercentile(run_active, 25)
    run_hi = np.nanpercentile(run_active, 75)
    print(f"Active IQR: pupil [{pupil_lo:.0f}, {pupil_hi:.0f}]  running [{run_lo:.1f}, {run_hi:.1f}] cm/s")
    print(f"Passive means: pupil={np.nanmean(pupil_passive):.0f} (active: {np.nanmean(pupil_active):.0f}), "
          f"running={np.nanmean(run_passive):.1f} (active: {np.nanmean(run_active):.1f})")

    mask_active = (
        (pupil_active >= pupil_lo) & (pupil_active <= pupil_hi)
        & (run_active >= run_lo) & (run_active <= run_hi)
    )
    mask_passive = (
        (pupil_passive >= pupil_lo) & (pupil_passive <= pupil_hi)
        & (run_passive >= run_lo) & (run_passive <= run_hi)
    )
    print(f"Matched (both IQRs): active {mask_active.sum()}/{len(t_active_all)} "
          f"({100*mask_active.mean():.1f}%), passive {mask_passive.sum()}/{len(t_passive_all)} "
          f"({100*mask_passive.mean():.1f}%)")

    t_active_matched = t_active_all[mask_active]
    t_passive_matched = t_passive_all[mask_passive]

    rng = np.random.default_rng(0)
    n_match = min(len(t_active_matched), len(t_passive_matched), 2500)
    t_active = np.sort(rng.choice(t_active_matched, n_match, replace=False))
    t_passive = np.sort(rng.choice(t_passive_matched, n_match, replace=False))
    print(f"Final matched sample: {n_match} each")

    WINDOW = (0.03, 0.20)
    rows = []
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = np.sort(spikes[uid])
            r_a, c_a = flash_rate(st, t_active, WINDOW)
            r_p, c_p = flash_rate(st, t_passive, WINDOW)
            denom = r_a + r_p
            mi = (r_a - r_p) / denom if denom > 0 else np.nan
            try:
                _, p = stats.mannwhitneyu(c_a, c_p, alternative="two-sided")
            except ValueError:
                p = np.nan
            rows.append(dict(unit_id=uid, area=area, r_active=r_a, r_passive=r_p, mi=mi, p=p))

    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "A2_unit_active_vs_passive_matched.parquet", index=False)

    # Area-level
    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        mi_mean=("mi", "mean"),
        mi_sem=("mi", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        frac_sig=("p", lambda x: (x < 0.05).mean()),
    ).reset_index()

    # Load original A
    a1 = pd.read_csv(REPORTS / "A_area_active_vs_passive.csv")
    a1 = a1.rename(columns={"mi_mean": "mi_mean_original", "mi_sem": "mi_sem_original"})
    merged = area_agg.merge(a1[["area", "mi_mean_original", "mi_sem_original"]], on="area", how="left")
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in merged["area"].values]
    merged["area"] = pd.Categorical(merged["area"], categories=order, ordered=True)
    merged = merged.sort_values("area").reset_index(drop=True)
    merged.to_csv(REPORTS / "A2_area_matched.csv", index=False)
    print("\n=== Area-level MI: original vs arousal-matched ===")
    print(merged[["area", "n_units", "mi_mean_original", "mi_mean", "frac_sig"]].to_string(index=False))

    # === PLOT: side-by-side ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    xs = np.arange(len(merged))
    for ax, col, col_sem, title in zip(
        axes, ["mi_mean_original", "mi_mean"], ["mi_sem_original", "mi_sem"],
        ["Original (all flashes)", "Arousal-matched (IQR of active)"],
    ):
        colors = ["#1f77b4" if m < 0 else "#d62728" for m in merged[col]]
        ax.bar(xs, merged[col], yerr=merged[col_sem], capsize=3, color=colors)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(merged["area"], rotation=35, ha="right")
        ax.set_ylabel("Modulation index")
        ax.set_title(title)

    fig.suptitle(f"Deliverable A2 — Active vs Passive MI: original vs arousal-matched\n"
                 f"({n_match} matched flashes each, session 1055240613)")
    fig.tight_layout()
    out = REPORTS / "A2_active_vs_passive_matched.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Delta summary
    print("\n=== Δ MI (matched − original) ===")
    merged["delta"] = merged["mi_mean"] - merged["mi_mean_original"]
    for _, r in merged.iterrows():
        arrow = "→"
        if abs(r["delta"]) > 0.05:
            arrow = "⇒" if r["delta"] > 0 else "⇐"
        print(f"  {str(r['area']):8s}  {r['mi_mean_original']:+.3f} {arrow} {r['mi_mean']:+.3f}  (Δ={r['delta']:+.3f})")

    (REPORTS / "A2_summary.json").write_text(json.dumps({
        "n_matched_flashes": int(n_match),
        "active_frac_retained": float(mask_active.mean()),
        "passive_frac_retained": float(mask_passive.mean()),
        "pupil_iqr": [float(pupil_lo), float(pupil_hi)],
        "running_iqr": [float(run_lo), float(run_hi)],
        "area_mi_matched": {str(r["area"]): float(r["mi_mean"]) for _, r in merged.iterrows()},
        "area_mi_original": {str(r["area"]): float(r["mi_mean_original"]) for _, r in merged.iterrows()
                              if np.isfinite(r["mi_mean_original"])},
    }, indent=2))


if __name__ == "__main__":
    main()
