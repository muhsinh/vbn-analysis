"""Deliverable D2 (post-critique): per-area encoding with full behavioral covariates.

Adds to D1:
- pupil_vel (phasic arousal, Reimer/McGinley — was computed in eye_features but
  never fed to the encoder; this was a coded bug)
- saccade events (head-fixed mice saccade ~1/s; saccadic suppression modulates
  V1 30-50% for 100-200ms — McFarland 2015)
- gap_bins = 40 (was 20, inside raised-cosine kernel support — real bug)

Feature bands now:
- running (1 col)
- pupil (5 cols: area, vel, x, y, blink_indicator)
- face_svd (20 cols)
- saccades (1 col, event-based, raised-cosine around t=0)

Variance partitioning as before but with 4 bands. Keep in mind: band-size
imbalance is still a known methodology weakness (banded ridge would be the
proper fix; that's a separate upgrade).
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from _shared import (ALL_TARGET_AREAS, BIN_SIZE, REPORTS, VISUAL_HIERARCHY,
                     bin_unit, interp_to_grid, load_session,
                     time_grid_for_block)

GAP_BINS = 40  # >= n_lag_bins=40 (previous value 20 was inside kernel support)
N_BASIS = 8
N_LAG = 40
ALPHA = 100.0


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
    if n_out <= 0:
        return np.empty((0, n_feat * n_basis))
    idx = np.arange(n_lag_bins - 1, n_lag_bins - 1 + n_out)[:, None] - np.arange(n_lag_bins)[None, :]
    X_lagged = X[idx]
    return np.einsum("tlf,lb->tfb", X_lagged, B).reshape(n_out, n_feat * n_basis)


def _forward_chain_folds(n, n_folds=5, gap_bins=GAP_BINS):
    block = n // (n_folds + 1)
    for i in range(1, n_folds + 1):
        test_start = i * block
        train_end = test_start - gap_bins
        test_end = min((i + 1) * block, n)
        if train_end < 100 or test_end <= test_start:
            continue
        yield train_end, test_start, test_end


def batched_cv_r2(X, Y, alpha=ALPHA, n_folds=5, gap_bins=GAP_BINS):
    n_targets = Y.shape[1]
    fold_r2 = []
    for train_end, test_start, test_end in _forward_chain_folds(len(X), n_folds, gap_bins):
        m = Ridge(alpha=alpha, solver="auto")
        m.fit(X[:train_end], Y[:train_end])
        preds = m.predict(X[test_start:test_end])
        y_true = Y[test_start:test_end]
        ss_res = ((y_true - preds) ** 2).sum(axis=0)
        ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
        fold_r2.append(1.0 - ss_res / ss_tot)
    if not fold_r2:
        return np.full(n_targets, np.nan)
    return np.mean(np.stack(fold_r2, axis=0), axis=0)


def saccade_kernel_regressor(sacc_times, grid, bin_size=BIN_SIZE, pre=0.2, post=0.3):
    """Return a 1D array with 1 at bins containing a saccade onset, else 0.
    Raised-cosine basis is applied by _lag_expand elsewhere."""
    x = np.zeros(len(grid))
    for t in sacc_times:
        if t < grid[0] or t >= grid[-1] + bin_size:
            continue
        idx = int((t - grid[0]) / bin_size)
        if 0 <= idx < len(x):
            x[idx] = 1.0
    return x


def main():
    data = load_session()
    units, spikes = data["units"], data["spikes"]
    running, pose = data["running"], data["pose"]
    eye = pd.read_parquet(REPORTS.parent / "eye" / "session_1055240613_eye_features.parquet")

    t_start, t_end = 27.6, 3630.0
    grid = time_grid_for_block(t_start, t_end, BIN_SIZE)
    print(f"Active block grid: {len(grid)} bins, gap_bins={GAP_BINS}")

    # Load saccade times
    sacc = np.load("outputs/eye/saccade_times.npy")
    print(f"Saccades: total {len(sacc)}, in active block: {((sacc>=t_start)&(sacc<t_end)).sum()}")

    # Build covariate matrix (27 features total)
    feat_cols = {
        "running":  [interp_to_grid(running, "running", grid)],
        "pupil":    [interp_to_grid(pose, "pupil", grid),
                     interp_to_grid(eye, "pupil_vel", grid),
                     interp_to_grid(pose, "pupil_x", grid),
                     interp_to_grid(pose, "pupil_y", grid)],
        "face_svd": [interp_to_grid(pose, f"face_svd_{i}", grid) for i in range(20)],
        "saccade":  [saccade_kernel_regressor(sacc, grid)],
    }
    # Flatten in consistent order
    feat_order = ["running", "pupil", "face_svd", "saccade"]
    X_list = []
    feat_groups = {}
    col = 0
    for g in feat_order:
        cols = list(range(col, col + len(feat_cols[g])))
        feat_groups[g] = cols
        col += len(feat_cols[g])
        X_list.extend(feat_cols[g])
    X = np.column_stack(X_list).astype(np.float32)
    X = np.nan_to_num(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    print(f"X: {X.shape}  groups: { {g: len(c) for g, c in feat_groups.items()} }")

    print("Lag-expanding X...")
    X_lagged = _lag_expand(X).astype(np.float32)
    print(f"  X_lagged: {X_lagged.shape}  {X_lagged.nbytes/1e6:.0f} MB")

    # Precompute reduced design matrices (drop one group at a time)
    def cols_for_groups(groups):
        cols = []
        for g in groups:
            for c in feat_groups[g]:
                cols.extend(range(c * N_BASIS, (c + 1) * N_BASIS))
        return cols

    X_reduced = {}
    for drop in feat_order:
        keep = [g for g in feat_order if g != drop]
        X_reduced[drop] = X_lagged[:, cols_for_groups(keep)]

    # Build Y matrix
    print("\nBuilding spike count matrix (dropping MGd, MGv, MGm — probe registration issue)...")
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
    print(f"  Y: {Y.shape}")
    Y_mu = Y.mean(axis=0, keepdims=True)
    Y_sd = Y.std(axis=0, keepdims=True)
    Y_sd[Y_sd == 0] = 1.0
    Y = (Y - Y_mu) / Y_sd
    Y_lagged = Y[N_LAG:, :]

    print("\nFitting full model (batched)...")
    t0 = time.time()
    r2_full = batched_cv_r2(X_lagged, Y_lagged)
    print(f"  full R² per unit: mean={np.nanmean(r2_full):.4f}  med={np.nanmedian(r2_full):.4f}  max={np.nanmax(r2_full):.4f}  ({time.time()-t0:.1f}s)")

    r2_unique = {}
    for group, X_red in X_reduced.items():
        t0 = time.time()
        r2_red = batched_cv_r2(X_red, Y_lagged)
        r2_unique[group] = r2_full - r2_red
        print(f"  unique_{group}: mean={np.nanmean(r2_unique[group]):+.4f}  ({time.time()-t0:.1f}s)")

    rows = []
    for i, (uid, area) in enumerate(zip(all_uids, all_areas)):
        row = dict(unit_id=uid, area=area, full_r2=float(r2_full[i]))
        for g in feat_order:
            row[f"unique_{g}"] = float(r2_unique[g][i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "D2_encoding_unit_r2.parquet", index=False)

    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        full_r2_mean=("full_r2", "mean"),
        full_r2_sem=("full_r2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        full_r2_frac_pos=("full_r2", lambda x: (x > 0).mean()),
        **{f"{g}_mean": (f"unique_{g}", "mean") for g in feat_order},
    ).reset_index()
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in area_agg["area"].values]
    area_agg["area"] = pd.Categorical(area_agg["area"], categories=order, ordered=True)
    area_agg = area_agg.sort_values("area").reset_index(drop=True)
    area_agg.to_csv(REPORTS / "D2_encoding_per_area.csv", index=False)
    print("\n=== Per-area encoding R² (with saccades + pupil_vel) ===")
    print(area_agg.to_string(index=False))

    # === PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(len(area_agg))
    ax = axes[0]
    colors = ["#2ca02c" if v > 0 else "#bbb" for v in area_agg["full_r2_mean"]]
    ax.bar(xs, area_agg["full_r2_mean"], yerr=area_agg["full_r2_sem"], capsize=4, color=colors)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Full encoding R² (CV)")
    ax.set_title("Full R² (27 covariates: +pupil_vel +saccades)")
    for i, (n, v) in enumerate(zip(area_agg["n_units"], area_agg["full_r2_mean"])):
        ax.text(i, v + area_agg["full_r2_sem"].iloc[i] * 1.5, f"n={n}", ha="center", fontsize=8)

    ax = axes[1]
    w = 0.2
    for i, g in enumerate(feat_order):
        ax.bar(xs + (i - 1.5) * w, area_agg[f"{g}_mean"], w, label=g,
               color=["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"][i])
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Unique R² (drop-group)")
    ax.set_title("Variance partitioning: running / pupil(5) / face_svd(20) / saccade(1)")
    ax.legend()

    fig.suptitle(f"Deliverable D2 — Encoding with pupil_vel + saccades, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "D2_encoding_per_area.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "D2_summary.json").write_text(json.dumps({
        "feat_groups": {g: len(c) for g, c in feat_groups.items()},
        "n_units_fit": int(len(df)),
        "gap_bins": GAP_BINS,
        "session_mean_full_r2": float(np.nanmean(r2_full)),
        "areas": {str(r["area"]): {
            "n_units": int(r["n_units"]),
            "full_r2_mean": float(r["full_r2_mean"]),
            "frac_positive_r2": float(r["full_r2_frac_pos"]),
            **{f"unique_{g}": float(r[f"{g}_mean"]) for g in feat_order},
        } for _, r in area_agg.iterrows()},
    }, indent=2))
    print(f"Summary: {REPORTS / 'D2_summary.json'}")


if __name__ == "__main__":
    main()
