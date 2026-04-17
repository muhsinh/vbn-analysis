"""Deliverable A: active-vs-passive flash response per unit, aggregated by area.

Computes per-unit modulation index MI = (R_active - R_passive) / (R_active + R_passive)
where R_* is the mean post-flash firing rate in [0, 250]ms, on non-change flashes only.
Reports area-level MI distribution and a hierarchy gradient.

Piet 2025 prediction: MI more negative (passive > active) in visual cortex, more positive
(active > passive) in midbrain. Hierarchy cortical < midbrain.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import (ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, bin_unit,
                     load_session)


def flash_rate(spike_times: np.ndarray, event_times: np.ndarray,
               window: tuple[float, float]) -> float:
    """Mean firing rate (Hz) in `window` around events."""
    pre, post = window
    trial_counts = []
    for t in event_times:
        lo, hi = t + pre, t + post
        n = int(np.searchsorted(spike_times, hi) - np.searchsorted(spike_times, lo))
        trial_counts.append(n)
    trial_counts = np.asarray(trial_counts)
    return trial_counts.mean() / (post - pre), trial_counts


def main() -> None:
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]

    # Non-change, non-omission flashes, split by active/passive block
    stim_f = stim[(stim["is_change"] == 0) & (stim["is_omission"] == 0)].copy()
    t_active = stim_f[stim_f["stimulus_block"] == 0]["t"].dropna().to_numpy()
    t_passive = stim_f[stim_f["stimulus_block"] == 5]["t"].dropna().to_numpy()
    print(f"Active flashes: {len(t_active)}, Passive flashes: {len(t_passive)}")

    # Match count for fair per-unit comparison
    rng = np.random.default_rng(0)
    n = min(len(t_active), len(t_passive), 2500)
    t_active = np.sort(rng.choice(t_active, n, replace=False))
    t_passive = np.sort(rng.choice(t_passive, n, replace=False))

    WINDOW = (0.03, 0.20)  # 30-200ms post-flash, skips pre-onset artifact
    BASELINE = (-0.25, -0.05)  # baseline window

    rows = []
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = np.sort(spikes[uid])

            r_act, c_act = flash_rate(st, t_active, WINDOW)
            r_pas, c_pas = flash_rate(st, t_passive, WINDOW)
            r_base_act, _ = flash_rate(st, t_active, BASELINE)
            r_base_pas, _ = flash_rate(st, t_passive, BASELINE)

            # Modulation index: +1 = active-dominant, -1 = passive-dominant
            denom = r_act + r_pas
            mi = (r_act - r_pas) / denom if denom > 0 else np.nan

            # Rank-sum on per-trial spike counts
            try:
                _, p = stats.mannwhitneyu(c_act, c_pas, alternative="two-sided")
            except ValueError:
                p = np.nan

            # Stim-evoked response (above baseline)
            evoked_act = r_act - r_base_act
            evoked_pas = r_pas - r_base_pas

            rows.append(dict(
                unit_id=uid, area=area,
                r_active=r_act, r_passive=r_pas,
                r_base_active=r_base_act, r_base_passive=r_base_pas,
                evoked_active=evoked_act, evoked_passive=evoked_pas,
                mi=mi, p=p, n_active=len(c_act), n_passive=len(c_pas),
            ))

    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "A_unit_active_vs_passive.parquet", index=False)
    print(f"\n{len(df)} unit rows saved")

    # Area-level summary
    summary = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        mi_mean=("mi", "mean"),
        mi_median=("mi", "median"),
        mi_sem=("mi", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        frac_sig=("p", lambda x: (x < 0.05).mean()),
        frac_active_dom=("mi", lambda x: (x > 0).mean()),
        evoked_act_mean=("evoked_active", "mean"),
        evoked_pas_mean=("evoked_passive", "mean"),
    ).reset_index()

    # Sort hierarchy
    order = [a for a in VISUAL_HIERARCHY + ["MGd", "CA1", "DG", "ProS", "SCig", "MRN"]
             if a in summary["area"].values]
    summary["area"] = pd.Categorical(summary["area"], categories=order, ordered=True)
    summary = summary.sort_values("area").reset_index(drop=True)
    summary.to_csv(REPORTS / "A_area_active_vs_passive.csv", index=False)

    print("\n=== Area-level modulation summary ===")
    print(summary.to_string(index=False))

    # === PLOTS ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    for i, area in enumerate(summary["area"]):
        vals = df[df["area"] == area]["mi"].dropna().values
        ax.scatter(np.full_like(vals, i, dtype=float) + rng.uniform(-0.15, 0.15, len(vals)),
                   vals, alpha=0.35, s=12, color="#555555")
    ax.errorbar(
        np.arange(len(summary)),
        summary["mi_mean"],
        yerr=summary["mi_sem"],
        fmt="o", color="#d62728", markersize=8, capsize=5, linewidth=2, zorder=3,
    )
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(summary["area"], rotation=35, ha="right")
    ax.set_ylabel("Modulation index = (active−passive) / (active+passive)")
    ax.set_title("Per-unit active-vs-passive MI by area\n"
                 "Negative = passive-dominant (Piet 2025 cortical pattern)")

    ax = axes[1]
    ax.bar(np.arange(len(summary)), summary["evoked_act_mean"], width=0.35, alpha=0.7,
           label="Active", color="#d62728")
    ax.bar(np.arange(len(summary)) + 0.35, summary["evoked_pas_mean"], width=0.35, alpha=0.7,
           label="Passive", color="#1f77b4")
    ax.set_xticks(np.arange(len(summary)) + 0.175)
    ax.set_xticklabels(summary["area"], rotation=35, ha="right")
    ax.set_ylabel("Evoked FR above baseline (Hz)")
    ax.set_title("Evoked response (post-flash − baseline)")
    ax.legend()

    fig.suptitle(f"Deliverable A — Active vs Passive flash response, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "A_active_vs_passive_hierarchy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Hierarchy correlation: is MI more negative in higher visual?
    vis_summary = summary[summary["area"].isin(VISUAL_HIERARCHY)].copy()
    if len(vis_summary) >= 3:
        vis_summary["hierarchy_rank"] = [VISUAL_HIERARCHY.index(a) for a in vis_summary["area"]]
        r, p = stats.spearmanr(vis_summary["hierarchy_rank"], vis_summary["mi_mean"])
        print(f"\nSpearman: visual hierarchy rank vs area-mean MI: r={r:.3f}, p={p:.3f}")
        print(f"  Piet 2025 predicts r > 0 (task modulation increases up hierarchy)")

    # Summary JSON
    (REPORTS / "A_summary.json").write_text(json.dumps({
        "n_units_total": int(len(df)),
        "n_areas": int(len(summary)),
        "session_wide_mi_mean": float(df["mi"].mean()),
        "session_wide_frac_sig": float((df["p"] < 0.05).mean()),
        "area_mi": {a: float(m) for a, m in zip(summary["area"], summary["mi_mean"])},
    }, indent=2))
    print(f"Summary: {REPORTS / 'A_summary.json'}")


if __name__ == "__main__":
    main()
