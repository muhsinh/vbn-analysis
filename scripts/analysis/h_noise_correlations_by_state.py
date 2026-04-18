"""Deliverable H (new, unpublished territory): noise correlations by state.

For each brain area and for each area-pair, compute the mean noise correlation
(pairwise Pearson correlation of residual firing, after subtracting flash PSTH
template) during ACTIVE block vs PASSIVE block.

Scientific question (flagged by literature agent as unpublished at the VBN
6-area × hippocampus × thalamus × midbrain scale): does behavioral state
reorganize the neural population's correlation structure? Population-geometry
theory (Averbeck, Kohn, Abbott) predicts yes.

Implementation:
1. Bin spike counts at 25 ms across active block (t=27-3630s) and passive
   block (t=5184-8787s).
2. Per area, subtract the stimulus-locked PSTH template for ALL flashes in
   each block (residualization).
3. Compute pairwise correlation matrix within each area × state.
4. Compare mean within-area correlation active vs passive.
5. Optionally: cross-area correlations for the hierarchy pairs.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import (ALL_TARGET_AREAS, BIN_SIZE, REPORTS, VISUAL_HIERARCHY,
                     bin_unit, load_session, time_grid_for_block)


def residualize_flashes(Y, grid, flash_times, window=(-0.25, 0.5), bin_size=BIN_SIZE):
    """Subtract flash-locked PSTH template per unit. Returns residual."""
    n_bins, n_units = Y.shape
    pre_bins = int(-window[0] / bin_size)
    post_bins = int(window[1] / bin_size)
    n_tpl = pre_bins + post_bins
    tpl = np.zeros((n_units, n_tpl), dtype=np.float32)
    n_valid = 0
    for ft in flash_times:
        idx = int((ft - grid[0]) / bin_size)
        lo, hi = idx - pre_bins, idx + post_bins
        if lo < 0 or hi > n_bins:
            continue
        tpl += Y[lo:hi, :].T
        n_valid += 1
    tpl /= max(n_valid, 1)

    Y_res = Y.copy()
    for ft in flash_times:
        idx = int((ft - grid[0]) / bin_size)
        lo, hi = idx - pre_bins, idx + post_bins
        if lo < 0 or hi > n_bins:
            continue
        Y_res[lo:hi, :] -= tpl.T
    return Y_res


def compute_nc_matrix(Y):
    """Pairwise correlation matrix (n_units, n_units)."""
    # Remove units with zero variance
    sd = Y.std(axis=0)
    valid = sd > 0
    Y = Y[:, valid]
    if Y.shape[1] < 2:
        return np.array([[]])
    # Fast correlation via standardization
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    C = (Y.T @ Y) / Y.shape[0]
    np.fill_diagonal(C, np.nan)  # exclude self-pairs
    return C


def main():
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]

    # Two blocks
    active_start, active_end = 27.6, 3630.0
    passive_start, passive_end = 5184.1, 8786.5
    grid_act = time_grid_for_block(active_start, active_end, BIN_SIZE)
    grid_pas = time_grid_for_block(passive_start, passive_end, BIN_SIZE)
    print(f"Active: {len(grid_act)} bins | Passive: {len(grid_pas)} bins")

    flashes_act = stim[(stim["stimulus_block"] == 0) & (stim["is_omission"] == 0)]["t"].dropna().values
    flashes_pas = stim[(stim["stimulus_block"] == 5) & (stim["is_omission"] == 0)]["t"].dropna().values
    print(f"Flashes act: {len(flashes_act)} | pas: {len(flashes_pas)}")

    skip_areas = {"MGd", "MGv", "MGm"}
    target = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam",
              "CA1", "DG", "ProS", "SCig", "MRN"]

    area_units = {}
    for area in target:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes and area not in skip_areas]
        if len(uids) < 8:
            continue
        area_units[area] = uids
    print(f"Areas with ≥8 units: {list(area_units.keys())}")

    # Within-area noise correlations by state
    rows = []
    area_nc_pairs = {}  # for cross-area plots
    for area, uids in area_units.items():
        # Build active and passive spike count matrices
        Y_act = np.zeros((len(grid_act), len(uids)), dtype=np.float32)
        Y_pas = np.zeros((len(grid_pas), len(uids)), dtype=np.float32)
        for i, uid in enumerate(uids):
            st = spikes[uid]
            st_act = st[(st >= active_start) & (st < active_end)]
            st_pas = st[(st >= passive_start) & (st < passive_end)]
            Y_act[:, i] = bin_unit(st_act, grid_act, BIN_SIZE)
            Y_pas[:, i] = bin_unit(st_pas, grid_pas, BIN_SIZE)

        # Residualize flash-locked variance per block
        Y_act_r = residualize_flashes(Y_act, grid_act, flashes_act)
        Y_pas_r = residualize_flashes(Y_pas, grid_pas, flashes_pas)

        # Compute within-area NC
        C_act = compute_nc_matrix(Y_act_r)
        C_pas = compute_nc_matrix(Y_pas_r)

        # Upper triangle means
        iu = np.triu_indices(C_act.shape[0], k=1)
        nc_act = C_act[iu]
        nc_pas = C_pas[iu]

        # Save pair-level values for cross-state scatter
        area_nc_pairs[area] = dict(act=nc_act, pas=nc_pas)

        # Summary stats
        t_stat, p_val = stats.wilcoxon(nc_act[np.isfinite(nc_act) & np.isfinite(nc_pas)],
                                        nc_pas[np.isfinite(nc_act) & np.isfinite(nc_pas)],
                                        nan_policy="omit") if (nc_act.size > 10) else (np.nan, np.nan)

        rows.append(dict(
            area=area, n_units=len(uids), n_pairs=len(nc_act),
            mean_nc_active=float(np.nanmean(nc_act)),
            mean_nc_passive=float(np.nanmean(nc_pas)),
            delta_nc=float(np.nanmean(nc_act) - np.nanmean(nc_pas)),
            median_nc_active=float(np.nanmedian(nc_act)),
            median_nc_passive=float(np.nanmedian(nc_pas)),
            wilcoxon_stat=float(t_stat) if np.isfinite(t_stat) else None,
            wilcoxon_p=float(p_val) if np.isfinite(p_val) else None,
        ))
        print(f"  {area:6s} n_u={len(uids):3d} n_pairs={len(nc_act):5d}  "
              f"NC_act={np.nanmean(nc_act):+.4f} NC_pas={np.nanmean(nc_pas):+.4f} "
              f"Δ={np.nanmean(nc_act)-np.nanmean(nc_pas):+.4f} p={p_val:.4f}")

    df = pd.DataFrame(rows)
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in df["area"].values]
    df["area"] = pd.Categorical(df["area"], categories=order, ordered=True)
    df = df.sort_values("area").reset_index(drop=True)
    df.to_csv(REPORTS / "H_noise_correlations_by_state.csv", index=False)

    print("\n=== Within-area noise correlations by state ===")
    print(df[["area", "n_units", "mean_nc_active", "mean_nc_passive",
              "delta_nc", "wilcoxon_p"]].to_string(index=False))

    # === PLOTS ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(len(df))
    ax = axes[0]
    w = 0.35
    ax.bar(xs - w/2, df["mean_nc_active"], w, label="Active", color="#d62728")
    ax.bar(xs + w/2, df["mean_nc_passive"], w, label="Passive", color="#1f77b4")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(df["area"], rotation=35, ha="right")
    ax.set_ylabel("Mean within-area noise correlation")
    ax.set_title("(H) Mean noise correlation by state, per area")
    ax.legend()
    # mark significance
    for i, p in enumerate(df["wilcoxon_p"]):
        if p is not None and p < 0.001:
            ax.text(i, max(df["mean_nc_active"].iloc[i], df["mean_nc_passive"].iloc[i]) + 0.005,
                    "***", ha="center", fontsize=10)

    ax = axes[1]
    ax.bar(xs, df["delta_nc"], color=["#d62728" if v > 0 else "#1f77b4" for v in df["delta_nc"]])
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(df["area"], rotation=35, ha="right")
    ax.set_ylabel("Δ NC (active − passive)")
    ax.set_title("(H) State-driven restructuring of noise correlations\n"
                 "positive = more correlated during active; negative = more during passive")

    fig.suptitle(f"Deliverable H — Noise correlations by state, session 1055240613\n"
                 f"flash-locked variance residualized before correlation")
    fig.tight_layout()
    out = REPORTS / "H_noise_correlations_by_state.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "H_summary.json").write_text(json.dumps({
        "method": "pairwise Pearson on flash-residualized 25ms binned counts, per area, per block",
        "active_block": [active_start, active_end],
        "passive_block": [passive_start, passive_end],
        "areas": {str(r["area"]): {
            "n_units": int(r["n_units"]),
            "n_pairs": int(r["n_pairs"]),
            "mean_nc_active": float(r["mean_nc_active"]),
            "mean_nc_passive": float(r["mean_nc_passive"]),
            "delta_nc": float(r["delta_nc"]),
            "wilcoxon_p": float(r["wilcoxon_p"]) if r["wilcoxon_p"] is not None else None,
        } for _, r in df.iterrows()},
    }, indent=2))


if __name__ == "__main__":
    main()
