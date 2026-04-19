"""Deliverable K — adaptation trajectory geometry across the hierarchy.

Scout recommendation #2. Per area, per epoch (active/passive):
  - For each flash number N ∈ {1..8} within same-image runs, compute the
    population response vector (n_units-D) = per-unit baseline-corrected
    evoked firing rate averaged across all flash-N events.
  - Reference: flash-1 response vector (v1).
  - For each N, quantify:
      (a) magnitude ratio |v_N| / |v_1|  → gain-scaling index (1.0 = no change)
      (b) angle between v_N and v_1 (degrees) → rotational index (0° = pure gain)

If adaptation is pure gain-scaling: magnitude decays with N, angle stays ~0°.
If adaptation rotates the representation: angle increases with N.

Hypothesis from scout: rotation should increase with hierarchy depth
(VISpm/VISam rotate more than VISp). Compare active vs passive.
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

SESSIONS = [1055240613, 1067588044, 1115086689]
CROSS_DIR = Path("outputs/cross_session")
REPORT_DIR = Path("reports/cross_session")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_HIERARCHY = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam"]
TARGET_AREAS = VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
WINDOW = (0.03, 0.20)
BASELINE_WIN = (-0.25, -0.05)
FLASH_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8]


def canonical_area(a: str) -> str:
    if not isinstance(a, str):
        return str(a)
    for prefix in TARGET_AREAS + ["CA3", "POST", "SUB", "MGd", "MGv", "MGm"]:
        if a == prefix or a.startswith(prefix + "-") or a.startswith(prefix + "l") or a.startswith(prefix + "o"):
            return prefix
    return a


def population_response_vector(
    spikes: dict, uids: list[str], event_times: np.ndarray,
    window=WINDOW, baseline_win=BASELINE_WIN,
) -> np.ndarray:
    """Return a per-unit mean evoked FR vector (baseline-subtracted), length = n_units."""
    n_units = len(uids)
    if len(event_times) == 0:
        return np.full(n_units, np.nan)
    pre, post = window
    bpre, bpost = baseline_win
    out = np.zeros(n_units)
    for i, uid in enumerate(uids):
        st = np.sort(spikes[uid])
        evoked = 0.0
        baseline = 0.0
        for t in event_times:
            lo_e, hi_e = t + pre, t + post
            lo_b, hi_b = t + bpre, t + bpost
            evoked += (np.searchsorted(st, hi_e) - np.searchsorted(st, lo_e)) / (post - pre)
            baseline += (np.searchsorted(st, hi_b) - np.searchsorted(st, lo_b)) / (bpost - bpre)
        out[i] = (evoked - baseline) / len(event_times)
    return out


def fit_session(session_id: int):
    bundle = load_session_bundle(session_id)
    units = bundle["units"]
    units["canonical"] = units["ecephys_structure_acronym"].map(canonical_area)
    spikes = bundle["spikes"]
    stim = bundle["stim"]

    if "flashes_since_change" not in stim.columns:
        print(f"  [{session_id}] no flashes_since_change column; skipping")
        return pd.DataFrame()

    rows = []
    for epoch_name, epoch_block in [("active", bundle["active_block"]),
                                      ("passive", bundle["passive_block"])]:
        epoch_stim = stim[
            (stim["stimulus_block"] == epoch_block)
            & (stim["is_change"] == 0)
            & (stim["is_omission"] == 0)
            & (stim["flashes_since_change"].isin(FLASH_NUMBERS))
        ]
        for area in TARGET_AREAS:
            uids = [str(u) for u in units[units["canonical"] == area]["id"].tolist()
                    if str(u) in spikes]
            if len(uids) < 10:
                continue

            # Reference: flash-1 response
            t_ref = epoch_stim[epoch_stim["flashes_since_change"] == 1]["t"].dropna().to_numpy()
            if len(t_ref) < 10:
                continue
            v1 = population_response_vector(spikes, uids, t_ref)
            v1_norm = np.linalg.norm(v1)
            if v1_norm < 1e-6:
                continue

            for fn in FLASH_NUMBERS:
                t_fn = epoch_stim[epoch_stim["flashes_since_change"] == fn]["t"].dropna().to_numpy()
                if len(t_fn) < 5:
                    continue
                v = population_response_vector(spikes, uids, t_fn)
                v_norm = np.linalg.norm(v)
                mag_ratio = v_norm / v1_norm
                cos = float(np.dot(v1, v) / (v1_norm * v_norm)) if v_norm > 1e-9 else np.nan
                cos = np.clip(cos, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos))) if np.isfinite(cos) else np.nan
                rows.append(dict(
                    session_id=session_id,
                    area=area,
                    epoch=epoch_name,
                    flash_n=fn,
                    n_units=len(uids),
                    n_events=int(len(t_fn)),
                    mag_ratio=mag_ratio,
                    angle_deg=angle_deg,
                ))
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for sid in SESSIONS:
        print(f"\n[{sid}] computing adaptation geometry...")
        all_rows.append(fit_session(sid))
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(REPORT_DIR / "K_adaptation_geometry.csv", index=False)
    print(f"\nTotal rows: {len(full)} ({full['area'].nunique()} areas × {full['session_id'].nunique()} sessions × {full['epoch'].nunique()} epochs)")

    # Cross-session mean per area × epoch × flash_n
    agg = full.groupby(["area", "epoch", "flash_n"]).agg(
        mag_ratio_mean=("mag_ratio", "mean"),
        mag_ratio_sem=("mag_ratio", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        angle_deg_mean=("angle_deg", "mean"),
        angle_deg_sem=("angle_deg", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        n_sessions=("session_id", "nunique"),
    ).reset_index()
    agg.to_csv(REPORT_DIR / "K_adaptation_geometry_cross_session.csv", index=False)

    # === PLOTS ===
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in full["area"].unique()]

    # Panel 1: magnitude ratio vs flash_n, per area, active+passive overlaid
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey="row")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(order)))
    area_color = {a: cmap[i] for i, a in enumerate(order)}

    for row_i, (metric, ylabel, ylim) in enumerate([
        ("mag_ratio", "|v_N| / |v_1|", (0.0, 1.5)),
        ("angle_deg", "angle (deg)", (0, 90)),
    ]):
        for col_i, epoch_name in enumerate(["active", "passive"]):
            ax = axes[row_i, col_i]
            for area in order:
                sub = agg[(agg["area"] == area) & (agg["epoch"] == epoch_name)]
                if sub.empty:
                    continue
                sub = sub.sort_values("flash_n")
                ax.errorbar(sub["flash_n"], sub[f"{metric}_mean"],
                             yerr=sub[f"{metric}_sem"], color=area_color[area],
                             marker="o", label=area, alpha=0.8, lw=1.5)
            if metric == "mag_ratio":
                ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.4)
            else:
                ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
            ax.set_xlabel("flash N (since last change)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{epoch_name.upper()}")
            ax.set_ylim(ylim)
            if col_i == 0:
                ax.legend(fontsize=7, ncol=2, loc="best")

        # Panel: active vs passive delta at flash 8 (final adaptation)
        ax = axes[row_i, 2]
        last = agg[agg["flash_n"] == FLASH_NUMBERS[-1]]
        pivot = last.pivot_table(index="area", columns="epoch", values=f"{metric}_mean")
        pivot = pivot.reindex([a for a in order if a in pivot.index])
        xs = np.arange(len(pivot))
        w = 0.38
        if "active" in pivot.columns:
            ax.bar(xs - w/2, pivot["active"], w, label="active", color="#d62728", alpha=0.8)
        if "passive" in pivot.columns:
            ax.bar(xs + w/2, pivot["passive"], w, label="passive", color="#1f77b4", alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"Flash {FLASH_NUMBERS[-1]} {metric}")
        if metric == "mag_ratio":
            ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.legend(fontsize=8)

        # Panel: active - passive (state effect)
        ax = axes[row_i, 3]
        if "active" in pivot.columns and "passive" in pivot.columns:
            delta = pivot["active"] - pivot["passive"]
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in delta]
            ax.bar(xs, delta, color=colors)
            ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
            ax.set_title(f"Δ{metric} (active − passive)")

    fig.suptitle(f"Deliverable K — adaptation trajectory geometry (N={len(SESSIONS)} sessions)", fontsize=12)
    fig.tight_layout()
    out = REPORT_DIR / "K_adaptation_geometry.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Hierarchy-gradient test: does angle at flash 8 increase with hierarchy depth?
    print("\n=== Hierarchy gradient test (angle at flash 8, active) ===")
    last_act = agg[(agg["flash_n"] == FLASH_NUMBERS[-1]) & (agg["epoch"] == "active")]
    vis = last_act[last_act["area"].isin(VISUAL_HIERARCHY)].copy()
    vis["rank"] = [VISUAL_HIERARCHY.index(a) for a in vis["area"]]
    print(vis[["area", "rank", "angle_deg_mean"]].round(2).to_string(index=False))
    if len(vis) >= 4:
        from scipy import stats
        r, p = stats.spearmanr(vis["rank"], vis["angle_deg_mean"])
        print(f"  Spearman r = {r:.3f}, p = {p:.3f}")
        print(f"  Scout prediction: r > 0 (angle increases up hierarchy → representational rotation)")

    (REPORT_DIR / "K_adaptation_geometry.json").write_text(json.dumps({
        "method": "per-area population response vector per flash_n; magnitude ratio + angle vs flash-1",
        "n_sessions": len(SESSIONS),
        "cross_session_flash8": {
            str(r["area"]): {
                "epoch": str(r["epoch"]),
                "mag_ratio": float(r["mag_ratio_mean"]),
                "angle_deg": float(r["angle_deg_mean"]),
            } for _, r in agg[agg["flash_n"] == FLASH_NUMBERS[-1]].iterrows()
        },
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
