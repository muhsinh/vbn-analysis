"""Post-critique summary: side-by-side original vs fixed results for the
findings the expert critique attacked most directly.

Produces a 2x3 panel figure showing:
- (1) Active vs Passive MI: original vs arousal-matched
- (2) Change amplification: original (all repeats) vs strict (adap-matched, no fast-lick)
- (3) Change amp: miss-only (sanity check — does SCig survive without lick?)
- (4) Per-area encoding R²: original (gap=20, no saccade) vs fixed (gap=40, +saccade)
- (5) Variance partitioning: 3-band vs 4-band (+saccade)
- (6) Narrative summary panel with headline deltas
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import REPORTS, VISUAL_HIERARCHY


def main():
    a1 = pd.read_csv(REPORTS / "A_area_active_vs_passive.csv")
    a2 = pd.read_csv(REPORTS / "A2_area_matched.csv")
    b1 = pd.read_csv(REPORTS / "B_change_hierarchy.csv")
    b2 = pd.read_csv(REPORTS / "B2_change_hierarchy_lick_controlled.csv")
    d1 = pd.read_csv(REPORTS / "D_encoding_per_area.csv")
    d2 = pd.read_csv(REPORTS / "D2_encoding_per_area.csv")

    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in d2["area"].values]

    def reorder(df):
        df = df[df["area"].isin(order)].copy()
        df["area"] = pd.Categorical(df["area"], categories=order, ordered=True)
        return df.sort_values("area").reset_index(drop=True)

    a2 = reorder(a2); b2 = reorder(b2); d2 = reorder(d2)
    a1 = reorder(a1); b1 = reorder(b1); d1 = reorder(d1)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    xs = np.arange(len(order))

    # Panel 1: Active-vs-Passive MI, original vs matched
    ax = fig.add_subplot(gs[0, 0])
    # Merge a1 (has original mi_mean) and a2 (has matched mi_mean)
    w = 0.35
    orig = a2["mi_mean_original"].values
    matched = a2["mi_mean"].values
    ax.bar(xs - w/2, orig, w, yerr=a2["mi_sem_original"], capsize=3, label="Original", color="#888")
    ax.bar(xs + w/2, matched, w, yerr=a2["mi_sem"], capsize=3, label="Arousal-matched", color="#d62728")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("MI (active − passive) / (active + passive)")
    ax.set_title("(A) Active vs Passive MI\nOriginal vs arousal-matched", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 2: Change amplification — original vs strict
    ax = fig.add_subplot(gs[0, 1])
    orig_ratio = b2["ratio_old"].values
    strict_ratio = b2["ratio_strict"].values
    miss_ratio = b2["ratio_miss_only"].values
    w = 0.27
    ax.bar(xs - w, orig_ratio, w, label="Old (all change / all repeat)", color="#888")
    ax.bar(xs, strict_ratio, w, label="Strict (rl≥250ms, fsc=1)", color="#1f77b4")
    ax.bar(xs + w, miss_ratio, w, label="MISS-ONLY (no lick)", color="#d62728")
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Change / Repeat peak FR ratio")
    ax.set_title("(B) Change amplification\nold / strict / MISS-ONLY sanity check", fontsize=10)
    ax.legend(fontsize=7)

    # Panel 3: Miss-only focus — highlight SCig survival
    ax = fig.add_subplot(gs[0, 2])
    colors = []
    for a in order:
        if a in ("SCig", "MRN"):
            colors.append("#d62728")
        elif a in ("DG", "CA1"):
            colors.append("#ff7f0e")  # reward-driven -> should drop
        else:
            colors.append("#1f77b4")
    ax.bar(xs, miss_ratio, color=colors)
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Change / Repeat on MISS trials only")
    ax.set_title("(B sanity) MISS-only change amp\n"
                 "red=motor (persists→real); orange=reward (drops→reward-driven)", fontsize=10)

    # Panel 4: Encoding full R² before/after
    ax = fig.add_subplot(gs[1, 0])
    d1_r = d1.set_index("area").reindex(order)
    d2_r = d2.set_index("area").reindex(order)
    w = 0.35
    ax.bar(xs - w/2, d1_r["full_r2_mean"], w, yerr=d1_r["full_r2_sem"],
           capsize=3, color="#888", label="Original (gap=20, 3 bands)")
    ax.bar(xs + w/2, d2_r["full_r2_mean"], w, yerr=d2_r["full_r2_sem"],
           capsize=3, color="#d62728", label="Fixed (gap=40, +saccades +pupil_vel)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Full encoding R²")
    ax.set_title("(D) Behavior → neural R²\nbefore vs after gap_bins fix", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 5: Variance partitioning, 4-band version
    ax = fig.add_subplot(gs[1, 1])
    w = 0.20
    for i, (g, color) in enumerate(zip(
        ["running", "pupil", "face_svd", "saccade"],
        ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"],
    )):
        vals = d2_r[f"{g}_mean"].values
        ax.bar(xs + (i - 1.5) * w, vals, w, label=g, color=color)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Unique R² (drop-group)")
    ax.set_title("(D2) 4-band variance partitioning\n(saccade newly added)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    # Panel 6: Narrative summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = [
        "FINDINGS AFTER CRITIQUE",
        "",
        "✓ SCig active-dominance survives:",
        "   miss-only MI still positive",
        "   full R² doubled (0.013 → 0.029)",
        "   80% units now positive (was 70%)",
        "",
        "✓ Visual passive>active survives",
        "   arousal matching but weaker",
        "   (VISp -24% → -20%, Δ=+0.04)",
        "",
        "✗ DG/CA1 change amp was REWARD-driven:",
        "   DG 1.49× → 0.81× on miss trials",
        "   CA1 1.15× → 0.64× on miss trials",
        "",
        "✗ MGd dropped entirely:",
        "   probe registration artifact",
        "",
        "⚠ Stringer replication still fails:",
        "   but test is a strawman; need",
        "   pop-latent target (not per-unit)",
        "",
        "NEW discoveries:",
        " - VISpm 4.49× change on miss trials",
        " - Saccades explain 2% of SCig variance",
        " - Passive was HYPER-aroused, not drowsy",
    ]
    for i, line in enumerate(txt):
        weight = "bold" if line.startswith(("FINDINGS", "NEW")) else "normal"
        color = "#d62728" if line.startswith(("✗", "⚠")) else ("#2ca02c" if line.startswith("✓") else "k")
        ax.text(0.02, 0.98 - i * 0.036, line, fontsize=9, color=color, weight=weight,
                transform=ax.transAxes, va="top", family="monospace")

    fig.suptitle("Post-critique resolution — VBN session 1055240613 headline findings",
                 fontsize=13)
    out = REPORTS / "Z2_critique_resolution_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

    # Master before/after CSV
    rows = []
    for a in order:
        row = dict(area=a)
        if a in a2["area"].values:
            aa = a2[a2["area"] == a].iloc[0]
            row["mi_original"] = aa["mi_mean_original"]
            row["mi_matched"] = aa["mi_mean"]
        if a in b2["area"].values:
            bb = b2[b2["area"] == a].iloc[0]
            row["change_ratio_old"] = bb["ratio_old"]
            row["change_ratio_strict"] = bb["ratio_strict"]
            row["change_ratio_miss_only"] = bb["ratio_miss_only"]
        if a in d1["area"].values and a in d2["area"].values:
            d1a = d1[d1["area"] == a].iloc[0]
            d2a = d2[d2["area"] == a].iloc[0]
            row["full_r2_old"] = d1a["full_r2_mean"]
            row["full_r2_new"] = d2a["full_r2_mean"]
            row["sacc_unique_r2"] = d2a["saccade_mean"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "Z2_before_after_master.csv", index=False)
    print(f"Master CSV: {REPORTS / 'Z2_before_after_master.csv'}")


if __name__ == "__main__":
    main()
