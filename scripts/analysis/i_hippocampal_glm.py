"""Deliverable I — per-unit GLM decomposition of hippocampal responses.

Scout-1 recommendation #1. Directly tests the cross-session finding that DG +
CA1 change signals collapse on miss trials.

For each hippocampal-area unit in each session, fit a 3-regressor GLM on
trial-evoked firing rate:

    evoked_FR[trial] ~ β0 + β_change·is_change[trial]
                         + β_lick·licked[trial]
                         + β_reward·got_reward[trial]

where:
    is_change    = 1 for go trials (change flash), 0 for catch trials
    licked       = 1 for hit + false alarm, 0 for miss + correct reject
    got_reward   = 1 for hit only (Allen VBN gives reward only on hits)

If the hippocampal change signal is actually change-driven: β_change > 0, others ~0
If it's reward-driven (our claim):                            β_reward > 0, others ~0
If it's pure motor/lick:                                      β_lick > 0, others ~0

Evoked_FR computed in [0.05, 0.30]s post-change-or-sham-change.

Output: per-unit β vector + area aggregates + cross-session convergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _multi_session import load_session_bundle

HIPPOCAMPAL_AREAS = ["CA1", "DG", "ProS"]
WINDOW = (0.05, 0.30)  # 50-300ms post-flash
BASELINE_WIN = (-0.25, -0.05)
SESSIONS = [1055240613, 1067588044, 1115086689]
CROSS_DIR = Path("outputs/cross_session")
REPORT_DIR = Path("reports/cross_session")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def canonical_area(a: str) -> str:
    if not isinstance(a, str):
        return str(a)
    for prefix in ["CA1", "CA3", "DG", "ProS", "POST", "SUB", "LGd", "VISp", "VISl",
                   "VISal", "VISrl", "VISpm", "VISam", "SCig", "SCiw", "MRN"]:
        if a == prefix or a.startswith(prefix + "-") or a.startswith(prefix + "l") or a.startswith(prefix + "o"):
            return prefix
    return a


def evoked_rate(spike_times: np.ndarray, t: float, window=WINDOW) -> float:
    pre, post = window
    lo, hi = t + pre, t + post
    n = int(np.searchsorted(spike_times, hi) - np.searchsorted(spike_times, lo))
    return n / (post - pre)


def baseline_rate(spike_times: np.ndarray, t: float, window=BASELINE_WIN) -> float:
    pre, post = window
    lo, hi = t + pre, t + post
    n = int(np.searchsorted(spike_times, hi) - np.searchsorted(spike_times, lo))
    return n / (post - pre)


def fit_session(session_id: int) -> pd.DataFrame:
    bundle = load_session_bundle(session_id)
    units = bundle["units"]
    units["canonical"] = units["ecephys_structure_acronym"].map(canonical_area)
    spikes = bundle["spikes"]
    trials = bundle["trials"].copy()

    # Derive regressors
    trials["is_change"] = trials.get("go", 0).astype(int)
    trials["licked"] = (trials.get("hit", 0) + trials.get("false_alarm", 0)).astype(int)
    trials["got_reward"] = trials.get("hit", 0).astype(int)

    # Drop aborted + auto-rewarded, require t column
    mask = (trials.get("aborted", 0) != 1) & (trials.get("auto_rewarded", 0) != 1) & trials["t"].notna()
    trials = trials[mask].copy()
    # For catch trials, "t" might be the sham-change time — use it regardless
    print(f"  n trials after filter: {len(trials)} "
          f"(hit={int(trials['got_reward'].sum())}, "
          f"miss={int((trials['is_change'] * (1 - trials['licked'])).sum())}, "
          f"FA={int(((1 - trials['is_change']) * trials['licked']).sum())}, "
          f"CR={int(((1 - trials['is_change']) * (1 - trials['licked'])).sum())})")

    X = trials[["is_change", "licked", "got_reward"]].values.astype(float)
    X_aug = np.column_stack([np.ones(len(X)), X])  # [intercept, change, lick, reward]

    from numpy.linalg import lstsq

    rows = []
    for area in HIPPOCAMPAL_AREAS:
        uids = [str(u) for u in units[units["canonical"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = np.sort(spikes[uid])
            # Evoked FR minus baseline, per trial
            y = np.array([evoked_rate(st, t) - baseline_rate(st, t) for t in trials["t"].values])
            if y.std() < 1e-6:
                continue
            # Fit least-squares (linear GLM)
            beta, residuals, rank, _ = lstsq(X_aug, y, rcond=None)
            y_hat = X_aug @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # Per-coefficient t-statistic
            dof = max(len(y) - X_aug.shape[1], 1)
            mse = ss_res / dof
            try:
                cov = mse * np.linalg.inv(X_aug.T @ X_aug)
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                se = np.full(X_aug.shape[1], np.nan)
            t_stats = beta / se
            rows.append(dict(
                session_id=session_id, area=area, unit_id=uid,
                n_trials=int(len(y)),
                n_spikes=int(len(st)),
                beta_intercept=float(beta[0]),
                beta_change=float(beta[1]),
                beta_lick=float(beta[2]),
                beta_reward=float(beta[3]),
                t_change=float(t_stats[1]),
                t_lick=float(t_stats[2]),
                t_reward=float(t_stats[3]),
                r2=float(r2),
            ))
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for sid in SESSIONS:
        print(f"\n[{sid}] fitting GLM...")
        df = fit_session(sid)
        all_rows.append(df)
        print(f"  {len(df)} units fit across {df['area'].nunique() if len(df) else 0} areas")
    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(REPORT_DIR / "I_hippocampal_glm_units.parquet", index=False)

    # Per-area, per-session aggregation
    print("\n=== β distribution per area × session ===")
    agg = full.groupby(["area", "session_id"]).agg(
        n_units=("unit_id", "count"),
        beta_change_mean=("beta_change", "mean"),
        beta_lick_mean=("beta_lick", "mean"),
        beta_reward_mean=("beta_reward", "mean"),
        beta_change_median=("beta_change", "median"),
        beta_lick_median=("beta_lick", "median"),
        beta_reward_median=("beta_reward", "median"),
        frac_sig_change=("t_change", lambda x: (x.abs() > 1.96).mean()),
        frac_sig_lick=("t_lick", lambda x: (x.abs() > 1.96).mean()),
        frac_sig_reward=("t_reward", lambda x: (x.abs() > 1.96).mean()),
        frac_beta_reward_dominant=(
            "unit_id",
            lambda _: 0.0  # fill later using proper group access
        ),
    ).reset_index()

    # Compute frac_beta_reward_dominant properly
    def dom_frac(group):
        # Fraction of units where |β_reward| > |β_change| AND |β_reward| > |β_lick|
        reward_wins = (
            (group["beta_reward"].abs() > group["beta_change"].abs())
            & (group["beta_reward"].abs() > group["beta_lick"].abs())
        )
        return reward_wins.mean()

    def sign_frac(group):
        return (group["beta_reward"] > 0).mean()

    reward_dom = full.groupby(["area", "session_id"]).apply(dom_frac, include_groups=False).rename("frac_reward_dominant")
    reward_pos = full.groupby(["area", "session_id"]).apply(sign_frac, include_groups=False).rename("frac_beta_reward_positive")
    agg = agg.drop(columns=["frac_beta_reward_dominant"]).merge(reward_dom, on=["area", "session_id"]).merge(reward_pos, on=["area", "session_id"])

    agg.to_csv(REPORT_DIR / "I_hippocampal_glm_area_session.csv", index=False)
    print(agg[["area", "session_id", "n_units", "beta_change_mean", "beta_lick_mean",
               "beta_reward_mean", "frac_reward_dominant", "frac_beta_reward_positive"]].to_string(index=False))

    # Cross-session aggregate
    print("\n=== Cross-session β means (averaged over sessions) ===")
    cross = full.groupby("area").agg(
        n_units=("unit_id", "count"),
        beta_change=("beta_change", "mean"),
        beta_lick=("beta_lick", "mean"),
        beta_reward=("beta_reward", "mean"),
        frac_reward_dominant=("unit_id", lambda _: np.nan),  # fix below
    ).reset_index()

    def cross_dom(group):
        reward_wins = (
            (group["beta_reward"].abs() > group["beta_change"].abs())
            & (group["beta_reward"].abs() > group["beta_lick"].abs())
        )
        return reward_wins.mean()

    dom_cross = full.groupby("area").apply(cross_dom, include_groups=False).rename("frac_reward_dominant")
    cross = cross.drop(columns=["frac_reward_dominant"]).merge(dom_cross, on="area")
    cross.to_csv(REPORT_DIR / "I_hippocampal_glm_cross_session.csv", index=False)
    print(cross.round(4).to_string(index=False))

    # === PLOTS ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax_row, area in zip(axes, HIPPOCAMPAL_AREAS):
        sub = full[full["area"] == area]
        if sub.empty:
            for ax in ax_row:
                ax.text(0.5, 0.5, f"no {area} units", ha="center", va="center", transform=ax.transAxes)
            continue
        # Panel 1: β distribution
        ax = ax_row[0]
        ax.hist([sub["beta_change"], sub["beta_lick"], sub["beta_reward"]],
                bins=25, label=["β_change", "β_lick", "β_reward"],
                color=["#1f77b4", "#ff7f0e", "#d62728"], alpha=0.7)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("β coefficient (Hz / regressor)")
        ax.set_ylabel("Unit count")
        ax.set_title(f"{area} — β distribution (n={len(sub)} units, N=3 sessions)")
        ax.legend(fontsize=8)

        # Panel 2: β_reward vs β_change scatter, session-colored
        ax = ax_row[1]
        for sid, color in zip(SESSIONS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            s = sub[sub["session_id"] == sid]
            if not s.empty:
                ax.scatter(s["beta_change"], s["beta_reward"], s=20, alpha=0.6,
                           label=f"session {sid}", color=color)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4); ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        lims = [-abs(sub[["beta_change", "beta_reward"]].values).max(),
                 abs(sub[["beta_change", "beta_reward"]].values).max()]
        ax.plot(lims, lims, ls=":", color="gray", alpha=0.5, label="y=x")
        ax.set_xlabel("β_change"); ax.set_ylabel("β_reward")
        ax.set_title(f"{area} — reward vs change, per unit")
        ax.legend(fontsize=7)

        # Panel 3: dominance bar
        ax = ax_row[2]
        dom = full[full["area"] == area].groupby("session_id").apply(cross_dom, include_groups=False)
        ax.bar(range(len(dom)), dom.values, color="#d62728")
        ax.set_xticks(range(len(dom))); ax.set_xticklabels([str(s) for s in dom.index], rotation=15)
        ax.set_ylabel("Fraction of units where β_reward dominates")
        ax.set_title(f"{area} — reward-dominant fraction per session")
        ax.set_ylim(0, 1)
        ax.axhline(1/3, color="k", lw=0.5, ls="--", alpha=0.5, label="chance (3 regressors)")
        ax.legend(fontsize=8)

    fig.suptitle(f"Deliverable I — hippocampal GLM: change vs lick vs reward (N={len(SESSIONS)} sessions)", fontsize=12)
    fig.tight_layout()
    out = REPORT_DIR / "I_hippocampal_glm.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # JSON summary
    summary = {
        "method": "3-regressor OLS per unit (intercept + is_change + licked + got_reward)",
        "window": list(WINDOW), "baseline": list(BASELINE_WIN),
        "cross_session_area": {
            str(r["area"]): {
                "n_units": int(r["n_units"]),
                "beta_change": float(r["beta_change"]),
                "beta_lick": float(r["beta_lick"]),
                "beta_reward": float(r["beta_reward"]),
                "frac_reward_dominant": float(r["frac_reward_dominant"]),
            } for _, r in cross.iterrows()
        },
    }
    (REPORT_DIR / "I_hippocampal_glm.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
