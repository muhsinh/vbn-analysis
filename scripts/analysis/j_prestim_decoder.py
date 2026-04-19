"""Deliverable J — pre-stim population state → hit/miss.

Scout recommendation #1 (highest value/impact). The Steinmetz 2019 test done
right: can pre-stimulus population activity predict trial outcome *after*
regressing out arousal covariates (pupil, running)?

For each change trial, extract pre-stim (-500, 0) ms spike counts per unit in
each target area. Build logistic regression: P(hit) ~ neural_state. Residualize
first by regressing out pupil + running in that same window. Report held-out
AUC per area per session, plus cross-session aggregate.

Null/baseline: AUC from behavior-alone (pupil+running), AUC from random shuffle.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _multi_session import load_session_bundle

SESSIONS = [1055240613, 1067588044, 1115086689]
CROSS_DIR = Path("outputs/cross_session")
REPORT_DIR = Path("reports/cross_session")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PRESTIM_WIN = (-0.5, 0.0)  # seconds before change
BEHAV_WIN = (-0.5, 0.0)

TARGET_AREAS = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam",
                "CA1", "DG", "ProS", "SCig", "MRN"]


def canonical_area(a: str) -> str:
    if not isinstance(a, str):
        return str(a)
    for prefix in TARGET_AREAS + ["CA3", "POST", "SUB", "MGd", "MGv", "MGm"]:
        if a == prefix or a.startswith(prefix + "-") or a.startswith(prefix + "l") or a.startswith(prefix + "o"):
            return prefix
    return a


def spike_count(st: np.ndarray, t: float, window=PRESTIM_WIN) -> int:
    lo, hi = t + window[0], t + window[1]
    return int(np.searchsorted(st, hi) - np.searchsorted(st, lo))


def mean_in_window(df: pd.DataFrame, col: str, t_centers: np.ndarray, window=BEHAV_WIN):
    if df is None or col not in df.columns:
        return np.full(len(t_centers), np.nan)
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


def cv_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    if len(np.unique(y)) < 2 or (y == 1).sum() < 3 or (y == 0).sum() < 3:
        return np.nan
    # Stratified k-fold; cap n_splits by minority class
    minority = min((y == 1).sum(), (y == 0).sum())
    n = min(n_splits, minority, 5)
    if n < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        m = LogisticRegressionCV(cv=3, Cs=np.logspace(-2, 2, 5),
                                  max_iter=500, class_weight="balanced")
        try:
            m.fit(X[tr], y[tr])
            preds = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], preds))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else np.nan


def residualize_neural(X: np.ndarray, behav: np.ndarray) -> np.ndarray:
    """Regress out behav covariates from each neural column."""
    if behav.size == 0 or behav.shape[1] == 0:
        return X
    m = LinearRegression()
    # Handle NaN in behav: impute column means
    behav_imp = behav.copy()
    col_mean = np.nanmean(behav_imp, axis=0)
    for j in range(behav_imp.shape[1]):
        nan_mask = ~np.isfinite(behav_imp[:, j])
        behav_imp[nan_mask, j] = col_mean[j] if np.isfinite(col_mean[j]) else 0.0
    m.fit(behav_imp, X)
    residual = X - m.predict(behav_imp)
    return residual


def fit_session(session_id: int) -> pd.DataFrame:
    bundle = load_session_bundle(session_id)
    units = bundle["units"]
    units["canonical"] = units["ecephys_structure_acronym"].map(canonical_area)
    spikes = bundle["spikes"]
    trials = bundle["trials"].copy()
    running = bundle["running"]
    eye = bundle["eye"]

    # Change trials only (go + not aborted + not auto_rewarded)
    mask = (
        (trials.get("go", 0) == 1)
        & (trials.get("aborted", 0) != 1)
        & (trials.get("auto_rewarded", 0) != 1)
        & trials["t"].notna()
    )
    trials = trials[mask].copy()
    y = trials.get("hit", 0).astype(int).values
    t_events = trials["t"].values

    if (y == 1).sum() < 5 or (y == 0).sum() < 5:
        print(f"  [{session_id}] insufficient hit/miss split: hit={int(y.sum())}, miss={int((1-y).sum())}")
        return pd.DataFrame()

    # Build behavior covariates (pre-stim means)
    behav_cols = []
    if eye is not None:
        for c in ["pupil", "pupil_vel"]:
            if c in eye.columns:
                behav_cols.append(mean_in_window(eye, c, t_events))
    if running is not None:
        behav_cols.append(mean_in_window(running, "running", t_events))
    behav = np.column_stack(behav_cols) if behav_cols else np.empty((len(t_events), 0))
    # Normalize
    for j in range(behav.shape[1]):
        col = behav[:, j]
        valid = np.isfinite(col)
        if valid.sum() > 2:
            col[valid] = (col[valid] - col[valid].mean()) / (col[valid].std() + 1e-9)
        behav[:, j] = col

    # Behavior-only AUC baseline
    behav_imp = behav.copy()
    col_mean = np.nanmean(behav_imp, axis=0)
    for j in range(behav_imp.shape[1]):
        nan_mask = ~np.isfinite(behav_imp[:, j])
        behav_imp[nan_mask, j] = col_mean[j] if np.isfinite(col_mean[j]) else 0.0
    auc_behav = cv_auc(behav_imp, y) if behav_imp.shape[1] > 0 else np.nan

    # Per area decoder
    rows = [{"session_id": session_id, "area": "BEHAVIOR_ONLY", "n_units": behav.shape[1],
             "n_trials": len(y), "auc_raw": auc_behav, "auc_residualized": auc_behav}]
    for area in TARGET_AREAS:
        uids = [str(u) for u in units[units["canonical"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 10:
            continue
        # Build neural matrix: (n_trials, n_units) spike counts
        X = np.zeros((len(t_events), len(uids)))
        for j, uid in enumerate(uids):
            st = np.sort(spikes[uid])
            X[:, j] = [spike_count(st, t) for t in t_events]
        # Normalize
        X_sd = X.std(axis=0); X_sd[X_sd == 0] = 1.0
        X_z = (X - X.mean(axis=0)) / X_sd

        # AUC: raw + residualized
        auc_raw = cv_auc(X_z, y)
        X_res = residualize_neural(X_z, behav)
        auc_res = cv_auc(X_res, y)

        rows.append(dict(
            session_id=session_id, area=area, n_units=len(uids),
            n_trials=len(y),
            n_hits=int(y.sum()), n_misses=int((1 - y).sum()),
            auc_raw=auc_raw, auc_residualized=auc_res,
        ))
    df = pd.DataFrame(rows)
    print(f"  [{session_id}] decoders done, {len(df)-1} areas (n_hits={int(y.sum())}, n_misses={int((1-y).sum())})")
    return df


def main():
    all_rows = []
    for sid in SESSIONS:
        print(f"\n[{sid}] fitting pre-stim decoders...")
        all_rows.append(fit_session(sid))
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(REPORT_DIR / "J_prestim_decoder.csv", index=False)
    print("\n=== Per-area × session AUC ===")
    print(full.round(3).to_string(index=False))

    # Cross-session mean
    cross = full.groupby("area").agg(
        n_sessions=("session_id", "nunique"),
        n_units_mean=("n_units", "mean"),
        auc_raw_mean=("auc_raw", "mean"),
        auc_raw_sem=("auc_raw", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        auc_residualized_mean=("auc_residualized", "mean"),
        auc_residualized_sem=("auc_residualized", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
    ).reset_index()
    cross.to_csv(REPORT_DIR / "J_prestim_decoder_cross_session.csv", index=False)
    print("\n=== Cross-session mean AUC ===")
    print(cross.round(3).to_string(index=False))

    # === PLOT ===
    plot_df = cross[cross["area"] != "BEHAVIOR_ONLY"].copy()
    behav_bar = cross[cross["area"] == "BEHAVIOR_ONLY"]["auc_raw_mean"].values
    behav_auc = float(behav_bar[0]) if len(behav_bar) else 0.5

    order = [a for a in TARGET_AREAS if a in plot_df["area"].values]
    plot_df["area"] = pd.Categorical(plot_df["area"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("area").reset_index(drop=True)
    xs = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.38
    ax.bar(xs - w/2, plot_df["auc_raw_mean"], w, yerr=plot_df["auc_raw_sem"],
           capsize=3, label="Raw (neural only)", color="#1f77b4")
    ax.bar(xs + w/2, plot_df["auc_residualized_mean"], w, yerr=plot_df["auc_residualized_sem"],
           capsize=3, label="Residualized (neural after removing pupil+running)", color="#d62728")
    ax.axhline(0.5, color="k", lw=0.5, ls="--", alpha=0.5, label="chance")
    ax.axhline(behav_auc, color="#888", lw=1, ls=":", alpha=0.7, label=f"behavior-only AUC = {behav_auc:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(plot_df["area"], rotation=35, ha="right")
    ax.set_ylabel("Hit vs Miss decoder AUC (cross-session mean ± SEM)")
    ax.set_title(f"Deliverable J — Pre-stim [{PRESTIM_WIN[0]:.1f}, {PRESTIM_WIN[1]:.1f}]s neural state → hit/miss (N={len(SESSIONS)})")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    out = REPORT_DIR / "J_prestim_decoder.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORT_DIR / "J_prestim_decoder.json").write_text(json.dumps({
        "method": "5-fold stratified logistic regression CV, class-weight=balanced",
        "prestim_window_s": list(PRESTIM_WIN),
        "behav_covariates": ["pupil", "pupil_vel", "running"],
        "behavior_only_AUC_mean": behav_auc,
        "cross_session_area_auc": {
            str(r["area"]): {
                "n_sessions": int(r["n_sessions"]),
                "auc_raw": float(r["auc_raw_mean"]),
                "auc_residualized": float(r["auc_residualized_mean"]),
            } for _, r in cross.iterrows()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
