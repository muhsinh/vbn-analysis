"""Deliverable D: per-area behavior -> neural encoding R² + variance partitioning.

Batched across units for speed: all units in an area fit simultaneously as
multi-target Ridge. Variance partitioning by dropping feature groups (running,
pupil, face_svd). Active block only.

Output: per-unit R² parquet + per-area summary CSV + figure.
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from _shared import (ALL_TARGET_AREAS, BIN_SIZE, REPORTS, VISUAL_HIERARCHY,
                     bin_unit, interp_to_grid, load_session,
                     time_grid_for_block)


def _lag_expand(X, n_lag_bins=40, n_basis=8):
    """Raised cosine basis, feature-major output (f0b0..f0b{K-1}, f1b0...)."""
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


def _forward_chain_folds(n: int, n_folds: int = 5, gap_bins: int = 40):
    """Yield (train_end, test_start, test_end) tuples for forward-chain CV."""
    block = n // (n_folds + 1)
    for i in range(1, n_folds + 1):
        test_start = i * block
        train_end = test_start - gap_bins
        test_end = min((i + 1) * block, n)
        if train_end < 100 or test_end <= test_start:
            continue
        yield train_end, test_start, test_end


def batched_cv_r2(X: np.ndarray, Y: np.ndarray, alpha: float = 100.0,
                  n_folds: int = 5, gap_bins: int = 40) -> np.ndarray:
    """Forward-chain CV for multi-target ridge. Returns mean R² per target (shape [n_targets])."""
    n_targets = Y.shape[1]
    fold_r2 = []
    for train_end, test_start, test_end in _forward_chain_folds(len(X), n_folds, gap_bins):
        m = Ridge(alpha=alpha, solver="auto")
        m.fit(X[:train_end], Y[:train_end])
        preds = m.predict(X[test_start:test_end])
        # Per-target R²
        y_true = Y[test_start:test_end]
        ss_res = ((y_true - preds) ** 2).sum(axis=0)
        ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
        fold_r2.append(1.0 - ss_res / ss_tot)
    if not fold_r2:
        return np.full(n_targets, np.nan)
    return np.mean(np.stack(fold_r2, axis=0), axis=0)


def main() -> None:
    data = load_session()
    units, spikes = data["units"], data["spikes"]
    running, pose = data["running"], data["pose"]

    t_start, t_end = 27.6, 3630.0
    grid = time_grid_for_block(t_start, t_end, BIN_SIZE)
    print(f"Active block grid: {len(grid)} bins of {BIN_SIZE}s = {(t_end-t_start)/60:.1f} min")

    feat_cols = ["running", "pupil", "pupil_x", "pupil_y"] + [f"face_svd_{i}" for i in range(20)]
    X_list = [interp_to_grid(running, c, grid) if c == "running"
              else interp_to_grid(pose, c, grid) for c in feat_cols]
    X = np.column_stack(X_list).astype(np.float32)
    X = np.nan_to_num(X)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    feat_groups = {
        "running": [0],
        "pupil": [1, 2, 3],
        "face_svd": list(range(4, 24)),
    }

    N_BASIS, N_LAG = 8, 40
    print("Lag-expanding X...")
    t0 = time.time()
    X_lagged = _lag_expand(X, n_lag_bins=N_LAG, n_basis=N_BASIS).astype(np.float32)
    print(f"  X_lagged: {X_lagged.shape} {X_lagged.nbytes/1e6:.0f} MB  ({time.time()-t0:.1f}s)")

    # Build all reduced design matrices once
    def cols_for_groups(groups):
        cols = []
        for g in groups:
            for c in feat_groups[g]:
                cols.extend(range(c * N_BASIS, (c + 1) * N_BASIS))
        return cols

    X_full_cols = cols_for_groups(["running", "pupil", "face_svd"])
    X_reduced = {
        "running": X_lagged[:, cols_for_groups(["pupil", "face_svd"])],
        "pupil": X_lagged[:, cols_for_groups(["running", "face_svd"])],
        "face_svd": X_lagged[:, cols_for_groups(["running", "pupil"])],
    }

    ALPHA = 100.0  # middle of [1, 10000] — validated on pilot as near-optimal
    # Build Y matrix: all units concatenated
    print("\nBuilding spike count matrix...")
    all_uids = []
    all_areas = []
    all_counts = []
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = spikes[uid]
            st = st[(st >= t_start) & (st < t_end)]
            c = bin_unit(st, grid, BIN_SIZE)
            if c.sum() < 100:
                continue
            all_uids.append(uid)
            all_areas.append(area)
            all_counts.append(c)
    Y = np.asarray(all_counts, dtype=np.float32).T  # (n_bins, n_units)
    print(f"  Y: {Y.shape} = ({len(all_uids)} units, {len(grid)} bins)")
    # Z-score per unit
    Y_mu = Y.mean(axis=0, keepdims=True)
    Y_sd = Y.std(axis=0, keepdims=True)
    Y_sd[Y_sd == 0] = 1.0
    Y = (Y - Y_mu) / Y_sd
    # Align to X_lagged (drop first N_LAG bins)
    Y_lagged = Y[N_LAG:, :]

    print("\nFitting full model (batched)...")
    t0 = time.time()
    r2_full = batched_cv_r2(X_lagged, Y_lagged, alpha=ALPHA)
    print(f"  full R² per unit computed ({time.time()-t0:.1f}s)")
    print(f"  mean full R²: {np.nanmean(r2_full):.4f}  (med: {np.nanmedian(r2_full):.4f}, max: {np.nanmax(r2_full):.4f})")

    r2_unique = {}
    for group, X_red in X_reduced.items():
        t0 = time.time()
        r2_red = batched_cv_r2(X_red, Y_lagged, alpha=ALPHA)
        r2_unique[group] = r2_full - r2_red
        print(f"  unique_{group}: mean={np.nanmean(r2_unique[group]):.4f}  ({time.time()-t0:.1f}s)")

    # Assemble DataFrame
    rows = []
    for i, (uid, area) in enumerate(zip(all_uids, all_areas)):
        rows.append(dict(
            unit_id=uid, area=area,
            full_r2=float(r2_full[i]),
            unique_running=float(r2_unique["running"][i]),
            unique_pupil=float(r2_unique["pupil"][i]),
            unique_face_svd=float(r2_unique["face_svd"][i]),
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "D_encoding_unit_r2.parquet", index=False)

    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        full_r2_mean=("full_r2", "mean"),
        full_r2_sem=("full_r2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        full_r2_frac_pos=("full_r2", lambda x: (x > 0).mean()),
        unique_running_mean=("unique_running", "mean"),
        unique_pupil_mean=("unique_pupil", "mean"),
        unique_face_svd_mean=("unique_face_svd", "mean"),
    ).reset_index()
    order = [a for a in VISUAL_HIERARCHY + ["MGd", "CA1", "DG", "ProS", "SCig", "MRN"]
             if a in area_agg["area"].values]
    area_agg["area"] = pd.Categorical(area_agg["area"], categories=order, ordered=True)
    area_agg = area_agg.sort_values("area").reset_index(drop=True)
    area_agg.to_csv(REPORTS / "D_encoding_per_area.csv", index=False)
    print("\n=== Per-area encoding R² ===")
    print(area_agg.to_string(index=False))

    # === PLOTS ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.bar(range(len(area_agg)), area_agg["full_r2_mean"],
           yerr=area_agg["full_r2_sem"], capsize=4, color="#444")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(range(len(area_agg)))
    ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Mean R² (CV)")
    ax.set_title("Full encoding R² per area\n(24 behavioral covariates, active block)")
    for i, (n, v) in enumerate(zip(area_agg["n_units"], area_agg["full_r2_mean"])):
        ax.text(i, v + area_agg["full_r2_sem"].iloc[i] * 1.5, f"n={n}",
                ha="center", fontsize=8)

    ax = axes[1]
    xs = np.arange(len(area_agg))
    w = 0.27
    ax.bar(xs - w, area_agg["unique_running_mean"], w, label="Running",
           color="#d62728")
    ax.bar(xs, area_agg["unique_pupil_mean"], w, label="Pupil (3 cols)", color="#1f77b4")
    ax.bar(xs + w, area_agg["unique_face_svd_mean"], w, label="Face SVD (20 cols)",
           color="#2ca02c")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(area_agg["area"], rotation=35, ha="right")
    ax.set_ylabel("Unique R² (drop-group)")
    ax.set_title("Variance partitioning — unique contribution per feature band\n"
                 "(Stringer test: does face_svd add variance beyond running+pupil?)")
    ax.legend()

    fig.suptitle(f"Deliverable D — Per-area encoding, active block, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "D_encoding_per_area.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "D_summary.json").write_text(json.dumps({
        "n_units_fit": int(len(df)),
        "alpha": ALPHA,
        "n_basis": N_BASIS,
        "session_mean_full_r2": float(np.nanmean(r2_full)),
        "session_frac_units_positive_r2": float((r2_full > 0).mean()),
        "areas": {str(r["area"]): {
            "n_units": int(r["n_units"]),
            "full_r2_mean": float(r["full_r2_mean"]),
            "frac_positive_r2": float(r["full_r2_frac_pos"]),
            "unique_running": float(r["unique_running_mean"]),
            "unique_pupil": float(r["unique_pupil_mean"]),
            "unique_face_svd": float(r["unique_face_svd_mean"]),
        } for _, r in area_agg.iterrows()},
    }, indent=2))
    print(f"Summary: {REPORTS / 'D_summary.json'}")


if __name__ == "__main__":
    main()
