"""Deliverable G (post-critique): fair Stringer 2019 replication.

Stringer 2019 (Science) claimed face SVD explains ~⅓ of V1 variance. My original
D1/D2 analysis gave face_svd ~0 unique variance. The behavior-neural expert
critic flagged this as a strawman: Stringer used population-latent targets (PCs
of neural ensemble), not per-unit spike counts at 25 ms bins. And she used
spontaneous activity, not a flash-locked task.

This script does the FAIR test:
  1. Per area, compute top-K (=10) principal components of the unit × time
     spike-count matrix across the active block.
  2. Residualize out flash-locked variance (subtract PSTH template).
  3. Run CCA / multi-output ridge from 24 behavioral covariates → top-K PCs.
  4. Report fraction of population variance predictable from behavior.

Expected result if Stringer replicates: cumulative canonical correlation
magnitude ~0.5+ and predicted variance ≥ 10% of residual population variance.
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from _shared import (ALL_TARGET_AREAS, BIN_SIZE, REPORTS, VISUAL_HIERARCHY,
                     bin_unit, interp_to_grid, load_session, time_grid_for_block)


def residualize_flash_psth(Y, grid, flash_times, window=(-0.25, 0.5), bin_size=BIN_SIZE):
    """Subtract out the flash-locked PSTH template from each unit, leaving
    residual activity that is (roughly) not flash-driven. This is what
    Stringer calls 'spontaneous-like' variance."""
    n_bins, n_units = Y.shape
    # Compute per-unit mean PSTH across flashes
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

    # Subtract template at every flash
    Y_res = Y.copy()
    for ft in flash_times:
        idx = int((ft - grid[0]) / bin_size)
        lo, hi = idx - pre_bins, idx + post_bins
        if lo < 0 or hi > n_bins:
            continue
        Y_res[lo:hi, :] -= tpl.T
    return Y_res


def main():
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]
    running, pose = data["running"], data["pose"]
    eye = pd.read_parquet(REPORTS.parent / "eye" / "session_1055240613_eye_features.parquet")

    t_start, t_end = 27.6, 3630.0
    grid = time_grid_for_block(t_start, t_end, BIN_SIZE)

    # Flash times in active block
    stim_act = stim[(stim["stimulus_block"] == 0) & (stim["is_omission"] == 0)]
    flash_times = stim_act["t"].dropna().to_numpy()
    print(f"Flash times in active block: {len(flash_times)}")

    # Build behavior covariates (same 25 as D2 but pooled)
    feat_cols = ["running", "pupil", "pupil_vel", "pupil_x", "pupil_y"] + \
                [f"face_svd_{i}" for i in range(20)]
    X_list = []
    for c in feat_cols:
        if c == "running":
            X_list.append(interp_to_grid(running, c, grid))
        elif c == "pupil_vel":
            X_list.append(interp_to_grid(eye, c, grid))
        else:
            X_list.append(interp_to_grid(pose, c, grid))
    X = np.column_stack(X_list).astype(np.float32)
    X = np.nan_to_num(X)
    mu, sd = X.mean(axis=0), X.std(axis=0); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    print(f"Behavior X: {X.shape}")

    # Define behavior bands for later variance partitioning
    band_ranges = {
        "running": (0, 1),
        "pupil": (1, 5),    # pupil, pupil_vel, pupil_x, pupil_y
        "face_svd": (5, 25),
    }

    target_areas = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam",
                    "CA1", "DG", "ProS", "SCig", "MRN"]
    K_PCS = 10  # top K components per area

    results = []
    area_pc_data = {}  # for plotting

    for area in target_areas:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 10:  # need enough units for a population
            continue
        print(f"\n[{area}] n_units = {len(uids)}")

        # Build per-area spike count matrix (n_bins, n_units)
        Y = np.zeros((len(grid), len(uids)), dtype=np.float32)
        for i, uid in enumerate(uids):
            st = spikes[uid]; st = st[(st >= t_start) & (st < t_end)]
            c = bin_unit(st, grid, BIN_SIZE)
            Y[:, i] = c
        # Z-score per unit
        Y_mu = Y.mean(axis=0, keepdims=True)
        Y_sd = Y.std(axis=0, keepdims=True); Y_sd[Y_sd == 0] = 1.0
        Y = (Y - Y_mu) / Y_sd

        # Residualize flash-locked variance
        Y_res = residualize_flash_psth(Y, grid, flash_times, window=(-0.25, 0.5))

        # Population PCA on residual
        pca = PCA(n_components=min(K_PCS, Y.shape[1] - 1))
        PC = pca.fit_transform(Y_res)  # (n_bins, K)
        var_explained = pca.explained_variance_ratio_
        print(f"  top-{len(var_explained)} PCs explain {var_explained.sum()*100:.1f}% of residual variance")

        # Fit ridge behavior → PC (per PC). Forward-chain CV.
        n = X.shape[0]
        split = int(0.8 * n)
        gap = 40
        Xtr, Xte = X[:split - gap], X[split:]
        PCtr, PCte = PC[:split - gap], PC[split:]

        r2_full = []
        r2_per_band = {b: [] for b in band_ranges}
        for k in range(PC.shape[1]):
            # Full model
            m = Ridge(alpha=100.0)
            m.fit(Xtr, PCtr[:, k])
            y_pred = m.predict(Xte)
            r2_full.append(r2_score(PCte[:, k], y_pred))
            # Drop-one-band for unique R²
            for band, (lo, hi) in band_ranges.items():
                keep = [i for i in range(X.shape[1]) if not (lo <= i < hi)]
                m_red = Ridge(alpha=100.0)
                m_red.fit(Xtr[:, keep], PCtr[:, k])
                y_pred_red = m_red.predict(Xte[:, keep])
                r2_red = r2_score(PCte[:, k], y_pred_red)
                r2_per_band[band].append(r2_full[k] - r2_red)

        # Weight PCs by their explained variance for an aggregate score
        w = var_explained / var_explained.sum()
        weighted_r2_full = float(np.sum(w * np.asarray(r2_full)))
        weighted_r2_band = {b: float(np.sum(w * np.asarray(r2_per_band[b])))
                            for b in band_ranges}

        # CCA for overall variance-shared signal
        n_comp = min(5, X.shape[1], PC.shape[1])
        try:
            cca = CCA(n_components=n_comp)
            cca.fit(Xtr, PCtr)
            Xc_te, Yc_te = cca.transform(Xte, PCte)
            # Canonical correlations on test data
            cca_r = [float(np.corrcoef(Xc_te[:, i], Yc_te[:, i])[0, 1]) for i in range(n_comp)]
        except Exception as e:
            print(f"  CCA failed: {e}")
            cca_r = [np.nan] * n_comp

        results.append(dict(
            area=area, n_units=len(uids),
            weighted_r2_full=weighted_r2_full,
            pct_pop_var_captured=float(var_explained.sum()),
            weighted_r2_running=weighted_r2_band["running"],
            weighted_r2_pupil=weighted_r2_band["pupil"],
            weighted_r2_face_svd=weighted_r2_band["face_svd"],
            cca_r_top3=cca_r[:3],
            per_pc_r2=r2_full,
            per_pc_var_explained=var_explained.tolist(),
        ))
        area_pc_data[area] = dict(var_explained=var_explained, r2_full=r2_full, cca_r=cca_r)

    df = pd.DataFrame(results)
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in df["area"].values]
    df["area"] = pd.Categorical(df["area"], categories=order, ordered=True)
    df = df.sort_values("area").reset_index(drop=True)

    print("\n=== Population-latent Stringer test ===")
    print(df[["area", "n_units", "weighted_r2_full", "weighted_r2_running",
              "weighted_r2_pupil", "weighted_r2_face_svd"]].to_string(index=False))

    df_to_save = df.copy()
    df_to_save["cca_r_top3"] = df_to_save["cca_r_top3"].astype(str)
    df_to_save["per_pc_r2"] = df_to_save["per_pc_r2"].astype(str)
    df_to_save["per_pc_var_explained"] = df_to_save["per_pc_var_explained"].astype(str)
    df_to_save.to_csv(REPORTS / "G_stringer_fair_test.csv", index=False)

    # === PLOTS ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(len(df))
    ax = axes[0]
    w = 0.23
    for i, (band, color) in enumerate(zip(
        ["weighted_r2_running", "weighted_r2_pupil", "weighted_r2_face_svd", "weighted_r2_full"],
        ["#d62728", "#1f77b4", "#2ca02c", "#444"],
    )):
        label = band.replace("weighted_r2_", "").replace("_", " ").replace("full", "TOTAL")
        ax.bar(xs + (i - 1.5) * w, df[band], w, label=label, color=color,
               alpha=0.7 if band != "weighted_r2_full" else 1.0)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(df["area"], rotation=35, ha="right")
    ax.set_ylabel("Weighted R² (PC-variance-weighted)")
    ax.set_title("(G) Fair Stringer test: behavior → top-10 PC residual activity\nvariance partitioning per band")
    ax.legend(fontsize=8)

    ax = axes[1]
    for area, d in area_pc_data.items():
        cca_r = d["cca_r"]
        cca_r_pos = [max(0, abs(r)) if np.isfinite(r) else 0 for r in cca_r]
        ax.plot(np.arange(1, len(cca_r_pos) + 1), cca_r_pos, "-o", label=area, alpha=0.7)
    ax.set_xlabel("Canonical component #")
    ax.set_ylabel("|Canonical correlation| (held-out)")
    ax.set_title("(G) CCA between behavior and PC residual — held-out correlations\n"
                 "(Stringer 2019: V1 typically shows r > 0.3 for top 2-3 components)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.axhline(0.3, color="#d62728", ls="--", alpha=0.4, label="Stringer-V1 reference")

    fig.suptitle(f"Deliverable G — fair Stringer 2019 test, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "G_stringer_fair_test.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "G_summary.json").write_text(json.dumps({
        "method": "top-10 PCs per area × flash-residualized × 25 cov ridge + CCA",
        "n_pcs": K_PCS,
        "results": [
            {
                "area": str(r["area"]),
                "n_units": int(r["n_units"]),
                "weighted_total_r2": float(r["weighted_r2_full"]),
                "weighted_unique_running": float(r["weighted_r2_running"]),
                "weighted_unique_pupil": float(r["weighted_r2_pupil"]),
                "weighted_unique_face_svd": float(r["weighted_r2_face_svd"]),
                "cca_top3": [float(v) if np.isfinite(v) else None for v in r["cca_r_top3"]],
            }
            for _, r in df.iterrows()
        ],
    }, indent=2))
    print(f"Summary: {REPORTS / 'G_summary.json'}")


if __name__ == "__main__":
    main()
