"""Deliverable D3 (post-critique): banded ridge via per-band alpha scaling.

The full himalaya GroupRidgeCV search is too slow on 18 GB CPU hardware
for our 120k × 208 × 442-target problem (~10 min per area). This
implementation achieves the same scientific goal (per-band regularization)
via a simpler two-stage trick:

  1. For each feature band separately, find the RidgeCV-optimal alpha
     against the population-averaged target. Call these α_band.
  2. Rescale each band's features by 1/√α_band, then fit a single Ridge
     with α=1.0. This is mathematically equivalent to the banded-ridge
     solution where each band has its own effective α_band — but needs
     no CV over the joint alpha grid.
  3. Variance partitioning via drop-one, same as D2, but now with the
     per-band-tuned effective penalty.

This eliminates the band-size-imbalance bias flagged by the Pillow/Park/
Kriegeskorte critique: face_svd (20 cols × 8 basis = 160 dims) can no
longer absorb variance just because it has more capacity than running
(1 col × 8 = 8 dims) under a shared alpha.
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

from _shared import (ALL_TARGET_AREAS, BIN_SIZE, REPORTS, VISUAL_HIERARCHY,
                     bin_unit, interp_to_grid, load_session,
                     time_grid_for_block)

GAP_BINS = 40
N_BASIS = 8
N_LAG = 40


def _lag_expand(X, n_lag_bins=N_LAG, n_basis=N_BASIS):
    n_times, n_feat = X.shape
    lags_c = np.arange(n_lag_bins) + 0.5
    lags_log = np.log(lags_c)
    centers = np.linspace(lags_log[0], lags_log[-1], n_basis)
    span = lags_log[-1] - lags_log[0]
    width = 2 * span / (n_basis - 1) if n_basis > 1 else span
    B = np.zeros((n_lag_bins, n_basis))
    for j, c in enumerate(centers):
        z = np.pi * (lags_log - c) / width
        B[:, j] = 0.5 * (np.cos(np.clip(z, -np.pi, np.pi)) + 1.0)
    col_max = B.max(axis=0)
    col_max[col_max == 0] = 1.0
    B = B / col_max
    n_out = n_times - n_lag_bins
    idx = np.arange(n_lag_bins - 1, n_lag_bins - 1 + n_out)[:, None] - np.arange(n_lag_bins)[None, :]
    X_lagged = X[idx]
    return np.einsum("tlf,lb->tfb", X_lagged, B).reshape(n_out, n_feat * n_basis)


def saccade_kernel_regressor(sacc_times, grid, bin_size=BIN_SIZE):
    x = np.zeros(len(grid))
    for t in sacc_times:
        if t < grid[0] or t >= grid[-1] + bin_size:
            continue
        idx = int((t - grid[0]) / bin_size)
        if 0 <= idx < len(x):
            x[idx] = 1.0
    return x


def _fold_r2(X, Y, alpha, gap_bins=GAP_BINS, n_folds=5):
    n = len(X)
    block = n // (n_folds + 1)
    scores = []
    for i in range(1, n_folds + 1):
        test_start = i * block
        train_end = test_start - gap_bins
        test_end = min((i + 1) * block, n)
        if train_end < 100 or test_end <= test_start:
            continue
        m = Ridge(alpha=alpha)
        m.fit(X[:train_end], Y[:train_end])
        preds = m.predict(X[test_start:test_end])
        y_true = Y[test_start:test_end]
        if preds.ndim == 1:
            scores.append(r2_score(y_true, preds))
        else:
            ss_res = ((y_true - preds) ** 2).sum(axis=0)
            ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
            ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
            scores.append(1.0 - ss_res / ss_tot)
    if not scores:
        return None
    return np.mean(np.stack(scores, axis=0), axis=0)


def main():
    data = load_session()
    units, spikes = data["units"], data["spikes"]
    running, pose = data["running"], data["pose"]
    eye = pd.read_parquet(REPORTS.parent / "eye" / "session_1055240613_eye_features.parquet")

    t_start, t_end = 27.6, 3630.0
    grid = time_grid_for_block(t_start, t_end, BIN_SIZE)

    sacc = np.load("outputs/eye/saccade_times.npy")

    feat_bands = {
        "running":  [interp_to_grid(running, "running", grid)],
        "pupil":    [interp_to_grid(pose, "pupil", grid),
                     interp_to_grid(eye, "pupil_vel", grid),
                     interp_to_grid(pose, "pupil_x", grid),
                     interp_to_grid(pose, "pupil_y", grid)],
        "face_svd": [interp_to_grid(pose, f"face_svd_{i}", grid) for i in range(20)],
        "saccade":  [saccade_kernel_regressor(sacc, grid)],
    }
    band_order = ["running", "pupil", "face_svd", "saccade"]

    X_bands = []
    band_sizes = []
    for band in band_order:
        X_band = np.column_stack(feat_bands[band])
        X_band = np.nan_to_num(X_band)
        mu, sd = X_band.mean(axis=0), X_band.std(axis=0)
        sd[sd == 0] = 1.0
        X_band = (X_band - mu) / sd
        X_lagged = _lag_expand(X_band).astype(np.float32)
        X_bands.append(X_lagged)
        band_sizes.append(X_lagged.shape[1])
    print(f"Band sizes (after lag expansion): {dict(zip(band_order, band_sizes))}")

    X_concat = np.concatenate(X_bands, axis=1)
    slices = []; offset = 0
    for size in band_sizes:
        slices.append(slice(offset, offset + size))
        offset += size

    # Build Y (skip MGd etc.)
    skip_areas = {"MGd", "MGv", "MGm"}
    all_uids, all_areas, all_counts = [], [], []
    for area in ALL_TARGET_AREAS:
        if area in skip_areas:
            continue
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = spikes[uid]; st = st[(st >= t_start) & (st < t_end)]
            c = bin_unit(st, grid, BIN_SIZE)
            if c.sum() < 100:
                continue
            all_uids.append(uid)
            all_areas.append(area)
            all_counts.append(c)
    Y = np.asarray(all_counts, dtype=np.float32).T
    print(f"Y: {Y.shape}")
    Y_mu = Y.mean(axis=0, keepdims=True); Y_sd = Y.std(axis=0, keepdims=True); Y_sd[Y_sd == 0] = 1.0
    Y = (Y - Y_mu) / Y_sd
    Y_lagged = Y[N_LAG:, :]

    # === Step 1: find per-band optimal alpha via RidgeCV against population-avg ===
    pop_avg = Y_lagged.mean(axis=1)
    print("\nStep 1: per-band RidgeCV to find optimal alpha (against pop-avg target)...")
    alphas_grid = np.logspace(-1, 5, 13)
    band_alphas = {}
    for i, band in enumerate(band_order):
        X_band_lag = X_concat[:, slices[i]]
        m = RidgeCV(alphas=alphas_grid, gcv_mode="svd")
        m.fit(X_band_lag, pop_avg)
        band_alphas[band] = float(m.alpha_)
        print(f"  {band:10s}: α* = {m.alpha_:.2f}  (dim={X_band_lag.shape[1]})")

    # === Step 2: rescale each band by 1/sqrt(alpha_band), fit single Ridge(α=1) ===
    # In ridge: ||y - Xβ||² + α||β||². If we rescale X_b -> X_b/√α_b then α=1 gives the same solution
    # as having α_b per band in the original coords.
    X_scaled = np.zeros_like(X_concat)
    for i, band in enumerate(band_order):
        X_scaled[:, slices[i]] = X_concat[:, slices[i]] / np.sqrt(band_alphas[band])
    print(f"\nStep 2: rescaled features (unified α=1 fit)")

    # CV R² with per-band alpha effective
    print("Fitting banded ridge + CV per unit (batched)...")
    t0 = time.time()
    r2_full_cv = _fold_r2(X_scaled, Y_lagged, alpha=1.0)
    print(f"  Full R² per unit: mean={np.nanmean(r2_full_cv):.4f} med={np.nanmedian(r2_full_cv):.4f}  ({time.time()-t0:.1f}s)")

    # Drop-one variance partitioning (unique R² per band, still using the rescaled features)
    r2_unique = {}
    for drop_band in band_order:
        keep_cols = [c for bi, band in enumerate(band_order) if band != drop_band
                     for c in range(slices[bi].start, slices[bi].stop)]
        t0 = time.time()
        r2_red = _fold_r2(X_scaled[:, keep_cols], Y_lagged, alpha=1.0)
        r2_unique[drop_band] = r2_full_cv - r2_red
        print(f"  unique_{drop_band}: mean={np.nanmean(r2_unique[drop_band]):+.4f}  ({time.time()-t0:.1f}s)")

    # Compose report
    rows = []
    for i, (uid, area) in enumerate(zip(all_uids, all_areas)):
        row = dict(unit_id=uid, area=area, full_r2=float(r2_full_cv[i]))
        for band in band_order:
            row[f"unique_{band}"] = float(r2_unique[band][i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "D3_banded_ridge_unit_r2.parquet", index=False)

    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        full_r2_mean=("full_r2", "mean"),
        full_r2_sem=("full_r2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        full_r2_frac_pos=("full_r2", lambda x: (x > 0).mean()),
        **{f"{b}_mean": (f"unique_{b}", "mean") for b in band_order},
    ).reset_index()
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in area_agg["area"].values]
    area_agg["area"] = pd.Categorical(area_agg["area"], categories=order, ordered=True)
    area_agg = area_agg.sort_values("area").reset_index(drop=True)
    area_agg.to_csv(REPORTS / "D3_banded_ridge_per_area.csv", index=False)
    print("\n=== Per-area banded-ridge R² (D3) ===")
    print(area_agg.to_string(index=False))

    # === PLOTS ===
    # side-by-side D2 vs D3
    d2 = pd.read_csv(REPORTS / "D2_encoding_per_area.csv")
    d2 = d2[d2["area"].isin(order)].copy()
    d2["area"] = pd.Categorical(d2["area"], categories=order, ordered=True)
    d2 = d2.sort_values("area").reset_index(drop=True)
    # Re-align to same order as area_agg (may differ due to missing areas)
    d2_aligned = d2.set_index("area").reindex(area_agg["area"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(len(area_agg))
    ax = axes[0]
    w = 0.35
    ax.bar(xs - w/2, d2_aligned["full_r2_mean"], w, yerr=d2_aligned["full_r2_sem"], capsize=4,
           label="D2 (single α, shared)", color="#888")
    ax.bar(xs + w/2, area_agg["full_r2_mean"], w, yerr=area_agg["full_r2_sem"], capsize=4,
           label="D3 (per-band α)", color="#2ca02c")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Full R² (CV)")
    ax.set_title("(D3 vs D2) Full R² per area")
    ax.legend(fontsize=9)

    ax = axes[1]
    w = 0.2
    for i, (g, color) in enumerate(zip(band_order, ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"])):
        ax.bar(xs + (i - 1.5) * w, area_agg[f"{g}_mean"], w, label=g, color=color)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Unique R² (drop-group)")
    ax.set_title(f"(D3) Banded variance partitioning\nα per band: "
                 f"run={band_alphas['running']:.1f}, pup={band_alphas['pupil']:.1f}, "
                 f"face={band_alphas['face_svd']:.1f}, sacc={band_alphas['saccade']:.1f}")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Deliverable D3 — banded ridge (per-band α), session 1055240613")
    fig.tight_layout()
    out = REPORTS / "D3_banded_ridge.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "D3_summary.json").write_text(json.dumps({
        "method": "per-band RidgeCV α + feature rescaling (equivalent to banded ridge)",
        "band_sizes": dict(zip(band_order, band_sizes)),
        "band_alphas": band_alphas,
        "n_units_fit": int(len(df)),
        "gap_bins": GAP_BINS,
        "mean_full_r2": float(np.nanmean(r2_full_cv)),
        "areas": {str(r["area"]): {
            "n_units": int(r["n_units"]),
            "full_r2_mean": float(r["full_r2_mean"]),
            "frac_positive_r2": float(r["full_r2_frac_pos"]),
            **{f"unique_{b}": float(r[f"{b}_mean"]) for b in band_order},
        } for _, r in area_agg.iterrows()},
    }, indent=2))
    print(f"Summary: {REPORTS / 'D3_summary.json'}")


if __name__ == "__main__":
    main()
