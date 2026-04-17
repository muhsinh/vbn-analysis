"""Deliverable C: pre-stimulus arousal predicts hit vs miss on change trials.

For each change trial (n=180), compute mean running + pupil in [-1, 0]s window.
Bin trials into arousal tertiles (pupil), quartiles and plot hit rate vs bin.
Steinmetz 2019 / Reimer 2014 prediction: arousal predicts trial outcome.

Also control: do pre-stim neural population dynamics predict outcome (Steinmetz)?
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import REPORTS, load_session


def mean_in_window(df: pd.DataFrame, col: str, t_center: float,
                   window: tuple[float, float]) -> float:
    lo, hi = t_center + window[0], t_center + window[1]
    mask = (df["t"] >= lo) & (df["t"] <= hi)
    v = df.loc[mask, col]
    return float(v.mean()) if len(v) else np.nan


def logistic_univariate(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple univariate logistic regression for odds ratio + Wald p."""
    from sklearn.linear_model import LogisticRegression
    x_z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    mask = np.isfinite(x_z) & np.isfinite(y)
    x_z, y = x_z[mask], y[mask].astype(int)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return dict(coef=np.nan, odds_ratio=np.nan, auroc=np.nan, n=len(y))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(x_z.reshape(-1, 1), y)
    from sklearn.metrics import roc_auc_score
    probs = lr.predict_proba(x_z.reshape(-1, 1))[:, 1]
    auroc = roc_auc_score(y, probs)
    return dict(coef=float(lr.coef_[0, 0]), odds_ratio=float(np.exp(lr.coef_[0, 0])),
                auroc=float(auroc), n=int(len(y)))


def main() -> None:
    data = load_session()
    trials, running, pose = data["trials"], data["running"], data["pose"]

    # Only change trials (go), not aborted, not auto-rewarded
    go = trials[
        (trials["go"] == 1)
        & (trials.get("aborted", 0) != 1)
        & (trials.get("auto_rewarded", 0) != 1)
        & trials["t"].notna()
    ].copy()
    print(f"Change (go) trials: {len(go)}")
    print(f"  hits: {int(go['hit'].sum())}, misses: {int(go['miss'].sum())}")

    # Pre-stim window: -1 to 0 s
    W = (-1.0, 0.0)
    go["pre_running"] = [mean_in_window(running, "running", t, W) for t in go["t"]]
    go["pre_pupil"] = [mean_in_window(pose, "pupil", t, W) for t in go["t"]]
    go["pre_face_movement"] = [mean_in_window(pose, "face_svd_0", t, W) for t in go["t"]]
    # Face SVD 0 is mean motion energy

    go = go.dropna(subset=["pre_running", "pre_pupil", "hit", "miss"])
    print(f"After NaN drop: {len(go)} trials")

    # Binary outcome: hit=1, miss=0
    y = go["hit"].astype(int).values

    # === Univariate logistic regressions ===
    print("\n=== Univariate pre-stim -> hit ===")
    results = {}
    for col in ["pre_running", "pre_pupil", "pre_face_movement"]:
        x = go[col].values
        r = logistic_univariate(x, y)
        results[col] = r
        print(f"  {col:22s}: OR={r['odds_ratio']:.3f} per SD, AUROC={r['auroc']:.3f}, n={r['n']}")

    # === Multivariate logistic regression ===
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = go[["pre_running", "pre_pupil", "pre_face_movement"]].values
    X_z = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(X_z, y)
    probs = lr.predict_proba(X_z)[:, 1]
    auroc_full = roc_auc_score(y, probs)
    print(f"\nMultivariate (all 3 covariates) AUROC: {auroc_full:.3f}")

    # === Tertile binning by pupil (classic Reimer arousal proxy) ===
    go["pupil_tertile"] = pd.qcut(go["pre_pupil"], 3, labels=["low", "mid", "high"])
    tert = go.groupby("pupil_tertile", observed=True).agg(
        n_trials=("hit", "count"),
        hit_rate=("hit", "mean"),
        mean_pupil=("pre_pupil", "mean"),
        mean_running=("pre_running", "mean"),
    ).reset_index()
    print("\n=== Hit rate by pre-stim pupil tertile ===")
    print(tert.to_string(index=False))
    # Chi-square test for trend
    from scipy.stats import chi2_contingency
    table = pd.crosstab(go["pupil_tertile"], go["hit"], dropna=True).values
    chi2, p_chi2, _, _ = chi2_contingency(table)
    print(f"  Chi² test across tertiles: χ² = {chi2:.2f}, p = {p_chi2:.3f}")

    # Running tertiles
    go["running_tertile"] = pd.qcut(go["pre_running"], 3, labels=["low", "mid", "high"])
    tert_run = go.groupby("running_tertile", observed=True).agg(
        n_trials=("hit", "count"),
        hit_rate=("hit", "mean"),
        mean_running=("pre_running", "mean"),
    ).reset_index()
    print("\n=== Hit rate by pre-stim running tertile ===")
    print(tert_run.to_string(index=False))
    table_run = pd.crosstab(go["running_tertile"], go["hit"], dropna=True).values
    chi2_r, p_chi2_r, _, _ = chi2_contingency(table_run)
    print(f"  Chi² test across tertiles: χ² = {chi2_r:.2f}, p = {p_chi2_r:.3f}")

    # === PLOT ===
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # Row 1: distributions of pre-stim signals per outcome
    for ax, col, label in zip(axes[0], ["pre_running", "pre_pupil", "pre_face_movement"],
                              ["Running (cm/s)", "Pupil (au)", "Face SVD PC0"]):
        hit_vals = go[go["hit"] == 1][col].values
        miss_vals = go[go["hit"] == 0][col].values
        ax.hist([hit_vals, miss_vals], bins=20, label=[f"hit (n={len(hit_vals)})",
                                                        f"miss (n={len(miss_vals)})"],
                color=["#2ca02c", "#d62728"], alpha=0.65, density=True)
        ax.set_xlabel(f"Pre-stim {label}")
        ax.set_ylabel("Density")
        r = results[col]
        ax.set_title(f"{col}\nOR={r['odds_ratio']:.2f}/SD, AUROC={r['auroc']:.3f}")
        ax.legend(fontsize=8)

    # Row 2: tertile hit rates
    ax = axes[1, 0]
    ax.bar(range(len(tert_run)), tert_run["hit_rate"], color=["#bbbbbb", "#888888", "#444444"])
    ax.set_xticks(range(len(tert_run)))
    ax.set_xticklabels(tert_run["running_tertile"])
    ax.set_ylabel("Hit rate")
    ax.set_title(f"Hit rate by running tertile\nχ²={chi2_r:.1f}, p={p_chi2_r:.3f}")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(tert_run["hit_rate"]):
        ax.text(i, v + 0.02, f"{v:.2f}\n(n={tert_run['n_trials'].iloc[i]})",
                ha="center", fontsize=9)

    ax = axes[1, 1]
    ax.bar(range(len(tert)), tert["hit_rate"], color=["#bbbbbb", "#888888", "#444444"])
    ax.set_xticks(range(len(tert)))
    ax.set_xticklabels(tert["pupil_tertile"])
    ax.set_ylabel("Hit rate")
    ax.set_title(f"Hit rate by pupil tertile\nχ²={chi2:.1f}, p={p_chi2:.3f}")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(tert["hit_rate"]):
        ax.text(i, v + 0.02, f"{v:.2f}\n(n={tert['n_trials'].iloc[i]})",
                ha="center", fontsize=9)

    # Multivariate AUROC curve
    from sklearn.metrics import roc_curve
    ax = axes[1, 2]
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, lw=2, label=f"Multivariate AUROC={auroc_full:.3f}")
    for col in ["pre_running", "pre_pupil", "pre_face_movement"]:
        r = results[col]
        ax.plot([], [], " ", label=f"{col}: AUROC={r['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Arousal -> hit, ROC")
    ax.legend(fontsize=8)

    fig.suptitle(f"Deliverable C — Pre-stim arousal predicts hit/miss, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "C_arousal_hit_miss.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Save summary
    summary = dict(
        n_change_trials=int(len(go)),
        n_hits=int(go["hit"].sum()),
        n_misses=int(go["miss"].sum()),
        univariate={k: {kk: (float(vv) if isinstance(vv, (int, float)) and np.isfinite(vv) else vv)
                        for kk, vv in v.items()} for k, v in results.items()},
        multivariate_auroc=float(auroc_full),
        running_tertile={
            "hit_rate_low": float(tert_run.iloc[0]["hit_rate"]),
            "hit_rate_mid": float(tert_run.iloc[1]["hit_rate"]),
            "hit_rate_high": float(tert_run.iloc[2]["hit_rate"]),
            "chi2": float(chi2_r), "p": float(p_chi2_r),
        },
        pupil_tertile={
            "hit_rate_low": float(tert.iloc[0]["hit_rate"]),
            "hit_rate_mid": float(tert.iloc[1]["hit_rate"]),
            "hit_rate_high": float(tert.iloc[2]["hit_rate"]),
            "chi2": float(chi2), "p": float(p_chi2),
        },
    )
    (REPORTS / "C_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {REPORTS / 'C_summary.json'}")


if __name__ == "__main__":
    main()
