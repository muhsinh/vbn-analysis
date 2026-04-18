"""Deliverable F2 (post-critique): parameter-free latency via ZETA test.

Replaces the baseline-rate-biased 4σ threshold used in F with the ZETA test
(Montijn 2021, eLife 71969) — parameter-free, operates directly on spike times
without binning assumptions.

For each unit: ZETA gives (p, latency). Aggregate per-area latency as the
median across significantly responsive units (ZETA p < 0.05).

Expected fix: LGd anomaly. In F, LGd showed 47 ms onset — later than V1 at
31 ms — which is anatomically wrong (LGd is V1's thalamic input). The 4σ
method has a baseline-rate bias (higher baseline → higher absolute threshold
→ later crossings). ZETA is scale-invariant.
"""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared import ALL_TARGET_AREAS, REPORTS, VISUAL_HIERARCHY, load_session

# Suppress ZETA's verbose warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def zeta_latency(spike_times: np.ndarray, event_times: np.ndarray,
                 window: float = 0.25) -> tuple[float, float, float]:
    """Return (zeta_p, mean_latency_s, peak_latency_s) for a single unit.

    Uses zetapy.zetatest. The `dblUseMaxDur` is the analysis window post-event.
    """
    from zetapy import zetatest
    if len(spike_times) < 5 or len(event_times) < 5:
        return (np.nan, np.nan, np.nan)
    try:
        p, dZETA, _ = zetatest(
            spike_times.astype(np.float64),
            event_times.astype(np.float64),
            dblUseMaxDur=float(window),
            intResampNum=100,
            boolReturnRate=False,
            boolParallel=False,
        )
        p = float(p) if p is not None else np.nan
        lat_onset = np.nan
        lat_peak = np.nan
        if isinstance(dZETA, dict):
            # dblLatencyZETA = first peak-deviation time (ZETA-latency; best for onset)
            # dblLatencyInvZETA = time of inverse peak (suppression or rebound)
            lat_onset = float(dZETA.get("dblLatencyZETA", np.nan) or np.nan)
            lat_peak = float(dZETA.get("dblLatencyInvZETA", np.nan) or np.nan)
        return (p, lat_onset, lat_peak)
    except Exception:
        return (np.nan, np.nan, np.nan)


def main() -> None:
    data = load_session()
    units, spikes, stim = data["units"], data["spikes"], data["stim"]

    # Non-change flashes, active block
    stim_f = stim[(stim["is_change"] == 0) & (stim["is_omission"] == 0)
                  & (stim["stimulus_block"] == 0)].copy()
    rng = np.random.default_rng(0)
    N_EVT = min(len(stim_f), 1000)  # ZETA is slower per unit; sample events
    events = np.sort(rng.choice(stim_f["t"].dropna().values, N_EVT, replace=False))
    print(f"Flash events for ZETA: {len(events)}")

    # Per-unit ZETA
    rows = []
    total_units = sum(
        1 for a in ALL_TARGET_AREAS
        for u in units[units["ecephys_structure_acronym"] == a]["id"]
        if str(u) in spikes
    )
    print(f"Running ZETA on {total_units} units across {len(ALL_TARGET_AREAS)} areas...")
    done = 0
    for area in ALL_TARGET_AREAS:
        uids = [str(u) for u in units[units["ecephys_structure_acronym"] == area]["id"].tolist()
                if str(u) in spikes]
        if len(uids) < 5:
            continue
        for uid in uids:
            st = np.sort(spikes[uid])
            p, onset, peak = zeta_latency(st, events, window=0.25)
            rows.append(dict(unit_id=uid, area=area, zeta_p=p,
                             onset_latency_s=onset, peak_latency_s=peak,
                             n_spikes=int(len(st))))
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{total_units}")

    df = pd.DataFrame(rows)
    df.to_parquet(REPORTS / "F2_unit_zeta.parquet", index=False)
    print(f"\n{len(df)} units ZETA-tested. {(df['zeta_p'] < 0.05).sum()} significantly responsive.")

    # Per-area aggregation
    sig = df[df["zeta_p"] < 0.05].copy()
    area_agg = sig.groupby("area").agg(
        n_units=("unit_id", "count"),
        n_sig=("zeta_p", lambda x: (x < 0.05).sum()),
        onset_median_s=("onset_latency_s", "median"),
        onset_iqr_low=("onset_latency_s", lambda x: x.quantile(0.25)),
        onset_iqr_high=("onset_latency_s", lambda x: x.quantile(0.75)),
        peak_median_s=("peak_latency_s", "median"),
    ).reset_index()
    # Include all areas even if sparse
    all_areas = df.groupby("area")["unit_id"].count().reset_index(name="n_units_total")
    area_agg = area_agg.merge(all_areas, on="area", how="outer")
    order = [a for a in VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
             if a in area_agg["area"].values]
    area_agg["area"] = pd.Categorical(area_agg["area"], categories=order, ordered=True)
    area_agg = area_agg.sort_values("area").reset_index(drop=True)
    area_agg.to_csv(REPORTS / "F2_latency_zeta_per_area.csv", index=False)

    print("\n=== ZETA onset latency per area ===")
    print(f"{'area':8s} {'n_u':>4s} {'n_sig':>5s} {'onset_med_ms':>12s} {'IQR_low_ms':>12s} {'IQR_high_ms':>12s}")
    piet_ref = {"LGd": 38, "VISp": 44, "VISl": 45, "VISal": 46, "VISpm": 53, "VISam": 53}
    for _, r in area_agg.iterrows():
        area = str(r["area"])
        n_tot = int(r.get("n_units_total", 0)) if not pd.isna(r.get("n_units_total")) else 0
        n_sig = int(r["n_sig"]) if not pd.isna(r["n_sig"]) else 0
        onset_ms = r["onset_median_s"] * 1000 if np.isfinite(r["onset_median_s"]) else np.nan
        lo_ms = r["onset_iqr_low"] * 1000 if np.isfinite(r["onset_iqr_low"]) else np.nan
        hi_ms = r["onset_iqr_high"] * 1000 if np.isfinite(r["onset_iqr_high"]) else np.nan
        ref = piet_ref.get(area, "")
        print(f"{area:8s} {n_tot:>4d} {n_sig:>5d} "
              f"{f'{onset_ms:.1f}' if np.isfinite(onset_ms) else 'NaN':>12s} "
              f"{f'{lo_ms:.1f}' if np.isfinite(lo_ms) else 'NaN':>12s} "
              f"{f'{hi_ms:.1f}' if np.isfinite(hi_ms) else 'NaN':>12s}  Piet={ref}")

    # === PLOTS ===
    # 1. Per-area onset latency bar + individual unit points
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    xs = np.arange(len(area_agg))
    medians = area_agg["onset_median_s"] * 1000
    yerr = np.stack([
        (area_agg["onset_median_s"] - area_agg["onset_iqr_low"]) * 1000,
        (area_agg["onset_iqr_high"] - area_agg["onset_median_s"]) * 1000,
    ])
    ax.bar(xs, medians, color="#444", alpha=0.7)
    ax.errorbar(xs, medians, yerr=yerr, fmt="none", ecolor="k", capsize=4)
    # Overlay individual unit points
    for i, area in enumerate(area_agg["area"]):
        u_vals = sig[sig["area"] == area]["onset_latency_s"].dropna().values * 1000
        if len(u_vals):
            ax.scatter(np.full_like(u_vals, i, dtype=float) + rng.uniform(-0.15, 0.15, len(u_vals)),
                       u_vals, s=10, alpha=0.25, color="gray")
    # Piet overlay
    piet_ref_ms = [piet_ref.get(a, np.nan) for a in area_agg["area"]]
    ax.scatter(xs, piet_ref_ms, marker="_", s=400, color="#d62728", zorder=5,
               label="Piet 2025 reference")
    ax.set_xticks(xs)
    ax.set_xticklabels(area_agg["area"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("ZETA onset latency (ms)")
    ax.set_title(f"(F2) ZETA parameter-free onset latency per area\n"
                 f"bars=median, whiskers=IQR, dots=individual units, red=Piet 2025")
    ax.legend()
    ax.set_ylim(0, 150)

    # 2. Compare F (4σ) vs F2 (ZETA) on the same areas
    ax = axes[1]
    f1 = pd.read_csv(REPORTS / "F_latency_hierarchy.csv")
    f1["area"] = pd.Categorical(f1["area"], categories=order, ordered=True)
    f1 = f1.sort_values("area").reset_index(drop=True)
    f2_lat_ms = area_agg["onset_median_s"].values * 1000
    f1_lat_ms = f1["latency_s"].values * 1000 if "latency_s" in f1 else np.full(len(f1), np.nan)
    w = 0.35
    ax.bar(xs - w/2, f1_lat_ms[:len(xs)], w, label="F: 4σ threshold", color="#888")
    ax.bar(xs + w/2, f2_lat_ms, w, label="F2: ZETA (parameter-free)", color="#2ca02c")
    piet_vals = [piet_ref.get(a, np.nan) for a in area_agg["area"]]
    ax.scatter(xs, piet_vals, marker="_", s=400, color="#d62728", zorder=5, label="Piet 2025")
    ax.set_xticks(xs)
    ax.set_xticklabels(area_agg["area"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Onset latency (ms)")
    ax.set_title("4σ vs ZETA: does the LGd anomaly resolve?")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 150)

    fig.suptitle(f"Deliverable F2 — ZETA latencies, session 1055240613")
    fig.tight_layout()
    out = REPORTS / "F2_zeta_latency_hierarchy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    (REPORTS / "F2_summary.json").write_text(json.dumps({
        "method": "ZETA test (Montijn 2021), parameter-free",
        "n_events": int(len(events)),
        "n_units_total": int(len(df)),
        "n_units_sig_zeta_p<0.05": int((df["zeta_p"] < 0.05).sum()),
        "area_onset_ms": {
            str(r["area"]): (float(r["onset_median_s"]) * 1000
                             if np.isfinite(r["onset_median_s"]) else None)
            for _, r in area_agg.iterrows()
        },
        "piet_2025_reference_ms": piet_ref,
    }, indent=2))


if __name__ == "__main__":
    main()
