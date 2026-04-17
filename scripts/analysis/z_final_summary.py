"""Consolidate deliverables A, B, C, D, F into a single multi-panel figure and
a markdown report suitable for pasting into a lab notebook."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import REPORTS


def main() -> None:
    # Load all summaries
    A = json.loads((REPORTS / "A_summary.json").read_text())
    B = json.loads((REPORTS / "B_summary.json").read_text())
    C = json.loads((REPORTS / "C_summary.json").read_text())
    D = json.loads((REPORTS / "D_summary.json").read_text())
    F = json.loads((REPORTS / "F_summary.json").read_text())

    area_A = pd.read_csv(REPORTS / "A_area_active_vs_passive.csv")
    area_B = pd.read_csv(REPORTS / "B_change_hierarchy.csv")
    area_D = pd.read_csv(REPORTS / "D_encoding_per_area.csv")
    area_F = pd.read_csv(REPORTS / "F_latency_hierarchy.csv")

    # Merge on area
    summary = (
        area_A[["area", "n_units", "mi_mean", "mi_sem"]]
        .merge(area_B[["area", "change_amplification", "change_latency_s"]], on="area", how="outer")
        .merge(area_D[["area", "full_r2_mean", "unique_running_mean",
                       "unique_pupil_mean", "unique_face_svd_mean"]], on="area", how="outer")
        .merge(area_F[["area", "latency_s"]].rename(columns={"latency_s": "flash_onset_s"}),
               on="area", how="outer")
    )
    summary = summary[summary["area"].notna() & (summary["area"] != "nan")]
    summary.to_csv(REPORTS / "Z_master_summary.csv", index=False)
    print("=== MASTER AREA SUMMARY ===")
    print(summary.to_string(index=False))

    # === BIG MULTI-PANEL FIGURE ===
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    areas = summary["area"].tolist()
    xs = np.arange(len(areas))

    # Panel A: active-vs-passive MI
    ax = fig.add_subplot(gs[0, 0])
    colors = ["#1f77b4" if m < 0 else "#d62728" for m in summary["mi_mean"]]
    ax.bar(xs, summary["mi_mean"], yerr=summary["mi_sem"], color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Modulation index")
    ax.set_title("(A) Active vs Passive — per-unit MI per area\n"
                 "blue=passive-dominant, red=active-dominant", fontsize=10)

    # Panel B: flash-onset latency (Piet ref overlay)
    ax = fig.add_subplot(gs[0, 1])
    piet_ref_ms = F["piet_2025_reference_ms"]
    lat_ms = summary["flash_onset_s"] * 1000
    piet_vals = [piet_ref_ms.get(a, np.nan) for a in areas]
    ax.bar(xs - 0.2, lat_ms, 0.4, label="This session", color="#444")
    ax.bar(xs + 0.2, piet_vals, 0.4, label="Piet 2025", color="#d62728", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Flash-onset latency (ms)")
    ax.set_title("(F) Flash-onset latency hierarchy", fontsize=10)
    ax.legend(fontsize=8)

    # Panel C: change amplification
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(xs, summary["change_amplification"], color="#888")
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Change / Repeat peak FR ratio")
    ax.set_title("(B) Change amplification\n(>1 = stronger response to change)", fontsize=10)

    # Panel D: full encoding R²
    ax = fig.add_subplot(gs[1, 0])
    colors = ["#2ca02c" if v > 0 else "#bbb" for v in summary["full_r2_mean"]]
    ax.bar(xs, summary["full_r2_mean"], color=colors)
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Full encoding R² (CV)")
    ax.set_title("(D1) Behavior → neural R² (24 covariates)", fontsize=10)

    # Panel E: unique variance by band
    ax = fig.add_subplot(gs[1, 1])
    w = 0.27
    ax.bar(xs - w, summary["unique_running_mean"], w, label="Running", color="#d62728")
    ax.bar(xs, summary["unique_pupil_mean"], w, label="Pupil", color="#1f77b4")
    ax.bar(xs + w, summary["unique_face_svd_mean"], w, label="Face SVD", color="#2ca02c")
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Unique R²")
    ax.set_title("(D2) Variance partitioning (drop-group)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # Panel F: Deliverable C arousal → hit/miss
    ax = fig.add_subplot(gs[1, 2])
    tert_run_rates = [C["running_tertile"]["hit_rate_low"],
                      C["running_tertile"]["hit_rate_mid"],
                      C["running_tertile"]["hit_rate_high"]]
    tert_pupil_rates = [C["pupil_tertile"]["hit_rate_low"],
                        C["pupil_tertile"]["hit_rate_mid"],
                        C["pupil_tertile"]["hit_rate_high"]]
    x3 = np.arange(3)
    ax.bar(x3 - 0.2, tert_run_rates, 0.4, label="Running", color="#d62728")
    ax.bar(x3 + 0.2, tert_pupil_rates, 0.4, label="Pupil", color="#1f77b4")
    ax.set_xticks(x3)
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1.05)
    p_run = C["running_tertile"]["p"]
    p_pup = C["pupil_tertile"]["p"]
    ax.set_title(f"(C) Pre-stim arousal → hit\nrunning p={p_run:.2f}, pupil p={p_pup:.2f}", fontsize=10)
    ax.legend(fontsize=8)

    # Panel G: relationship between active-vs-passive MI and running encoding
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(summary["mi_mean"], summary["unique_running_mean"],
               s=80, c="#d62728", edgecolor="k", zorder=3)
    for i, a in enumerate(areas):
        ax.text(summary["mi_mean"].iloc[i] + 0.01, summary["unique_running_mean"].iloc[i],
                a, fontsize=8)
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    ax.set_xlabel("State MI (active−passive)")
    ax.set_ylabel("Unique R² (running)")
    from scipy.stats import pearsonr
    mask = np.isfinite(summary["mi_mean"]) & np.isfinite(summary["unique_running_mean"])
    if mask.sum() >= 3:
        r, p = pearsonr(summary["mi_mean"][mask], summary["unique_running_mean"][mask])
        ax.set_title(f"State sensitivity vs running encoding\nr={r:.3f}, p={p:.3f}", fontsize=10)

    # Panel H: change latency vs flash latency
    ax = fig.add_subplot(gs[2, 1])
    x_lat = summary["flash_onset_s"] * 1000
    y_lat = summary["change_latency_s"] * 1000
    ax.scatter(x_lat, y_lat, s=80, c="#1f77b4", edgecolor="k", zorder=3)
    for i, a in enumerate(areas):
        if np.isfinite(x_lat.iloc[i]) and np.isfinite(y_lat.iloc[i]):
            ax.text(x_lat.iloc[i] + 1, y_lat.iloc[i], a, fontsize=8)
    lim_lo = np.nanmin([x_lat.min(), y_lat.min()]) - 5
    lim_hi = np.nanmax([x_lat.max(), y_lat.max()]) + 5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="gray", alpha=0.4)
    ax.set_xlabel("Flash-onset latency (ms)")
    ax.set_ylabel("Change-signal latency (ms)")
    ax.set_title("(F vs B) Onset vs change latency\ndotted = identity", fontsize=10)

    # Panel I: unit R² distribution per area (violin-style)
    ax = fig.add_subplot(gs[2, 2])
    unit_df = pd.read_parquet(REPORTS / "D_encoding_unit_r2.parquet")
    unit_df = unit_df[unit_df["area"].isin(areas)]
    parts = ax.violinplot(
        [unit_df[unit_df["area"] == a]["full_r2"].dropna().values for a in areas],
        positions=xs, showmedians=True, widths=0.7,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#888")
        pc.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(areas, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Per-unit full R²")
    ax.set_title("(D3) Per-unit full R² distribution", fontsize=10)
    ax.set_ylim(-0.25, 0.25)

    fig.suptitle("VBN session 1055240613 — integrated analysis summary", fontsize=13)
    out = REPORTS / "Z_final_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
