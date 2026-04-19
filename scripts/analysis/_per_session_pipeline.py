"""Per-session replication pipeline. Runs A2/B2/D2/H for any session_id.

Designed to be called with session_id parameter from the cross-session notebook.
Outputs go to outputs/cross_session/<session_id>/.
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from _multi_session import BIN_SIZE, load_session_bundle

# Visual hierarchy (plus hippocampal + midbrain areas)
VISUAL_HIERARCHY = ["LGd", "VISp", "VISl", "VISal", "VISrl", "VISpm", "VISam"]
TARGET_AREAS = VISUAL_HIERARCHY + ["CA1", "DG", "ProS", "SCig", "MRN"]
SKIP_AREAS = {"MGd", "MGv", "MGm"}  # known off-target on 1055240613, harmless to skip elsewhere


def _canonicalize_area(a: str) -> str:
    """Collapse subregion labels (e.g. LGd-sh, LGd-co → LGd; DG-mo → DG)."""
    if not isinstance(a, str):
        return str(a)
    for prefix in ["LGd", "DG", "SCig", "SCiw", "VISp", "VISl", "VISam", "VISpm",
                    "VISal", "VISrl", "MGd", "MGv", "MGm", "MRN", "CA1", "CA3", "ProS",
                    "POST", "SUB", "MB"]:
        if a == prefix:
            return prefix
        if a.startswith(prefix + "-") or a.startswith(prefix + "l") or a.startswith(prefix + "o"):
            return prefix
    return a


def _area_units_map(units: pd.DataFrame, spikes: dict) -> dict:
    units = units.copy()
    units["canonical"] = units["ecephys_structure_acronym"].map(_canonicalize_area)
    m = {}
    for area in TARGET_AREAS:
        ids = [str(u) for u in units[units["canonical"] == area]["id"].tolist()
               if str(u) in spikes]
        if len(ids) >= 5:
            m[area] = ids
    return m


def _flash_rate(spike_times, event_times, window):
    pre, post = window
    counts = []
    for t in event_times:
        lo, hi = t + pre, t + post
        n = int(np.searchsorted(spike_times, hi) - np.searchsorted(spike_times, lo))
        counts.append(n)
    counts = np.asarray(counts)
    return counts.mean() / (post - pre), counts


def _mean_pre_stim(df, col, t_centers, window=(-0.5, 0.0)):
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


def _time_grid(t_start, t_end, bin_size=BIN_SIZE):
    return np.arange(t_start, t_end, bin_size)


def _bin_unit(spike_times, grid, bin_size=BIN_SIZE):
    edges = np.append(grid, grid[-1] + bin_size)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts


def _interp_to_grid(df, col, grid):
    m = df[["t", col]].dropna().sort_values("t")
    return np.interp(grid, m["t"].values, m[col].values)


# =============================================================================
# Deliverable A2: active vs passive MI, arousal-matched
# =============================================================================
def run_A2(bundle):
    units, spikes, stim = bundle["units"], bundle["spikes"], bundle["stim"]
    running, eye = bundle["running"], bundle["eye"]
    ab, pb = bundle["active_block"], bundle["passive_block"]

    stim_f = stim[(stim["is_change"] == 0) & (stim["is_omission"] == 0)].copy()
    t_active_all = stim_f[stim_f["stimulus_block"] == ab]["t"].dropna().to_numpy()
    t_passive_all = stim_f[stim_f["stimulus_block"] == pb]["t"].dropna().to_numpy()

    pupil_active = _mean_pre_stim(eye, "pupil", t_active_all) if eye is not None else np.full_like(t_active_all, np.nan)
    run_active = _mean_pre_stim(running, "running", t_active_all) if running is not None else np.full_like(t_active_all, np.nan)
    pupil_passive = _mean_pre_stim(eye, "pupil", t_passive_all) if eye is not None else np.full_like(t_passive_all, np.nan)
    run_passive = _mean_pre_stim(running, "running", t_passive_all) if running is not None else np.full_like(t_passive_all, np.nan)

    # Active-block IQR
    pupil_lo, pupil_hi = np.nanpercentile(pupil_active, [25, 75])
    run_lo, run_hi = np.nanpercentile(run_active, [25, 75])

    mask_active = (
        (pupil_active >= pupil_lo) & (pupil_active <= pupil_hi)
        & (run_active >= run_lo) & (run_active <= run_hi)
    )
    mask_passive = (
        (pupil_passive >= pupil_lo) & (pupil_passive <= pupil_hi)
        & (run_passive >= run_lo) & (run_passive <= run_hi)
    )

    rng = np.random.default_rng(0)
    n = min(mask_active.sum(), mask_passive.sum(), 2000)
    if n < 20:
        # Fall back to unmatched sampling
        n = min(len(t_active_all), len(t_passive_all), 2000)
        t_active = np.sort(rng.choice(t_active_all, n, replace=False))
        t_passive = np.sort(rng.choice(t_passive_all, n, replace=False))
        matched = False
    else:
        t_active = np.sort(rng.choice(t_active_all[mask_active], n, replace=False))
        t_passive = np.sort(rng.choice(t_passive_all[mask_passive], n, replace=False))
        matched = True

    WINDOW = (0.03, 0.20)
    area_map = _area_units_map(units, spikes)
    rows = []
    for area, uids in area_map.items():
        for uid in uids:
            st = np.sort(spikes[uid])
            r_a, c_a = _flash_rate(st, t_active, WINDOW)
            r_p, c_p = _flash_rate(st, t_passive, WINDOW)
            denom = r_a + r_p
            mi = (r_a - r_p) / denom if denom > 0 else np.nan
            try:
                _, p = stats.mannwhitneyu(c_a, c_p, alternative="two-sided")
            except ValueError:
                p = np.nan
            rows.append(dict(unit_id=uid, area=area, r_active=r_a, r_passive=r_p, mi=mi, p=p))
    df = pd.DataFrame(rows)

    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        mi_mean=("mi", "mean"),
        mi_sem=("mi", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        frac_sig=("p", lambda x: (x < 0.05).mean()),
    ).reset_index()
    return dict(unit_df=df, area_df=area_agg, matched=matched,
                n_matched=int(n), pupil_iqr=(pupil_lo, pupil_hi),
                running_iqr=(run_lo, run_hi))


# =============================================================================
# Deliverable B2: change amplification, miss-only, adaptation-matched
# =============================================================================
def run_B2(bundle):
    units, spikes, stim, trials = bundle["units"], bundle["spikes"], bundle["stim"], bundle["trials"]
    ab = bundle["active_block"]

    if trials is None or len(trials) == 0:
        return None
    trials = trials.copy()
    if "response_time" in trials.columns and "t" in trials.columns:
        trials["response_latency"] = trials["response_time"] - trials["t"]
    else:
        trials["response_latency"] = np.nan

    area_map = _area_units_map(units, spikes)

    def _psth(uids, events, window, bin_size=0.005):
        pre, post = window
        edges = np.arange(pre, post + bin_size, bin_size)
        centers = (edges[:-1] + edges[1:]) / 2
        per_unit = []
        for uid in uids:
            st = np.sort(spikes[uid])
            counts = np.zeros(len(centers))
            for t in events:
                lo, hi = np.searchsorted(st, t + pre), np.searchsorted(st, t + post)
                rel = st[lo:hi] - t
                if rel.size:
                    c, _ = np.histogram(rel, bins=edges)
                    counts += c
            per_unit.append(counts / max(len(events), 1) / bin_size)
        M = np.array(per_unit) if per_unit else np.empty((0, len(centers)))
        return centers, M.mean(axis=0) if M.size else np.zeros(len(centers))

    # Define windows
    active_stim = stim[stim["stimulus_block"] == ab]
    legit_changes = trials[
        (trials.get("go", 0) == 1) & (trials.get("aborted", 0) != 1)
        & ((trials["response_latency"] >= 0.25) | trials["response_latency"].isna())
    ]
    miss_changes = trials[(trials.get("miss", 0) == 1) & (trials.get("aborted", 0) != 1)]

    if "flashes_since_change" in active_stim.columns:
        adap_matched = active_stim[
            (active_stim["is_change"] == 0)
            & (active_stim["is_omission"] == 0)
            & (active_stim["flashes_since_change"] == 1)
        ]
    else:
        adap_matched = active_stim[
            (active_stim["is_change"] == 0) & (active_stim["is_omission"] == 0)
        ]
    all_repeat = active_stim[(active_stim["is_change"] == 0) & (active_stim["is_omission"] == 0)]

    rng = np.random.default_rng(42)
    def sub(arr, n):
        n = min(n, len(arr)); return np.sort(rng.choice(arr, n, replace=False))

    t_change_legit = legit_changes["t"].dropna().to_numpy()
    t_change_miss = miss_changes["t"].dropna().to_numpy()
    t_repeat_matched = adap_matched["t"].dropna().to_numpy()
    t_repeat_all = all_repeat["t"].dropna().to_numpy()

    n1 = min(len(t_change_legit), len(t_repeat_matched))
    n2 = min(len(t_change_miss), len(t_repeat_matched))

    results = []
    for area, uids in area_map.items():
        if len(uids) < 5:
            continue
        # Old (all change, all repeat)
        t_ch_o = sub(t_change_legit, min(len(t_change_legit), len(t_repeat_all), 200))
        t_rp_o = sub(t_repeat_all, min(len(t_change_legit), len(t_repeat_all), 200))
        t_c, mu_c_o = _psth(uids, t_ch_o, (-0.25, 0.35))
        _, mu_r_o = _psth(uids, t_rp_o, (-0.25, 0.35))
        # Strict
        ratio_strict = ratio_miss = np.nan
        peak_change_miss = np.nan
        if n1 >= 5 and len(t_repeat_matched) >= 5:
            t_ch_s = sub(t_change_legit, n1); t_rp_s = sub(t_repeat_matched, n1)
            _, mu_c_s = _psth(uids, t_ch_s, (-0.25, 0.35))
            _, mu_r_s = _psth(uids, t_rp_s, (-0.25, 0.35))
        else:
            mu_c_s = mu_r_s = np.zeros_like(mu_c_o)
        if n2 >= 5 and len(t_repeat_matched) >= 5:
            t_ch_m = sub(t_change_miss, n2); t_rp_m = sub(t_repeat_matched, n2)
            _, mu_c_m = _psth(uids, t_ch_m, (-0.25, 0.35))
            _, mu_r_m = _psth(uids, t_rp_m, (-0.25, 0.35))
        else:
            mu_c_m = mu_r_m = np.zeros_like(mu_c_o)

        base_mask = (t_c >= -0.25) & (t_c <= -0.05)
        evoked_mask = (t_c >= 0.03) & (t_c <= 0.20)

        def ratio(mu_c, mu_r):
            pc = mu_c[evoked_mask].max() - mu_c[base_mask].mean()
            pr = mu_r[evoked_mask].max() - mu_r[base_mask].mean()
            return pc / pr if pr > 0 else np.nan, pc

        r_old, _ = ratio(mu_c_o, mu_r_o)
        r_strict, _ = ratio(mu_c_s, mu_r_s)
        r_miss, pc_miss = ratio(mu_c_m, mu_r_m)

        results.append(dict(area=area, n_units=len(uids),
                           ratio_old=r_old, ratio_strict=r_strict,
                           ratio_miss=r_miss, peak_change_miss_hz=pc_miss))

    return dict(
        area_df=pd.DataFrame(results),
        n_change_strict=int(n1), n_miss=int(n2),
        n_change_trials=len(legit_changes), n_miss_trials=len(miss_changes),
    )


# =============================================================================
# Deliverable D2: encoding R² per area (running + pupil features only)
# =============================================================================
def run_D2(bundle):
    units, spikes = bundle["units"], bundle["spikes"]
    running, eye = bundle["running"], bundle["eye"]
    ab_start, ab_end = bundle["active_range"]

    if running is None or eye is None:
        return None

    grid = _time_grid(ab_start, ab_end)
    feat_cols = {
        "running": [_interp_to_grid(running, "running", grid)],
        "pupil": [_interp_to_grid(eye, "pupil", grid),
                  _interp_to_grid(eye, "pupil_vel", grid),
                  _interp_to_grid(eye, "pupil_x", grid),
                  _interp_to_grid(eye, "pupil_y", grid)],
    }
    band_order = ["running", "pupil"]
    X_list, slices, offset = [], [], 0
    for band in band_order:
        for f in feat_cols[band]:
            X_list.append(f)
        slices.append(slice(offset, offset + len(feat_cols[band])))
        offset += len(feat_cols[band])
    X = np.column_stack(X_list).astype(np.float32)
    X = np.nan_to_num(X)
    mu, sd = X.mean(axis=0), X.std(axis=0); sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # Lag expand (8-basis raised cosine, 40 lag bins)
    N_BASIS, N_LAG = 8, 40
    lags_c = np.arange(N_LAG) + 0.5
    lags_log = np.log(lags_c)
    centers = np.linspace(lags_log[0], lags_log[-1], N_BASIS)
    span = lags_log[-1] - lags_log[0]
    width = 2 * span / (N_BASIS - 1)
    B = np.zeros((N_LAG, N_BASIS))
    for j, c in enumerate(centers):
        z = np.pi * (lags_log - c) / width
        B[:, j] = 0.5 * (np.cos(np.clip(z, -np.pi, np.pi)) + 1.0)
    col_max = B.max(axis=0); col_max[col_max == 0] = 1.0
    B = B / col_max
    n_out = len(X) - N_LAG
    idx = np.arange(N_LAG - 1, N_LAG - 1 + n_out)[:, None] - np.arange(N_LAG)[None, :]
    X_lagged = np.einsum("tlf,lb->tfb", X[idx], B).reshape(n_out, X.shape[1] * N_BASIS).astype(np.float32)

    # Build Y per unit (active block only)
    area_map = _area_units_map(units, spikes)
    all_uids, all_areas, counts_list = [], [], []
    for area, uids in area_map.items():
        for uid in uids:
            st = spikes[uid]; st = st[(st >= ab_start) & (st < ab_end)]
            c = _bin_unit(st, grid)
            if c.sum() < 100:
                continue
            all_uids.append(uid); all_areas.append(area); counts_list.append(c)
    if not counts_list:
        return None
    Y = np.asarray(counts_list, dtype=np.float32).T
    Y_mu = Y.mean(axis=0, keepdims=True); Y_sd = Y.std(axis=0, keepdims=True); Y_sd[Y_sd == 0] = 1.0
    Y = (Y - Y_mu) / Y_sd
    Y_lag = Y[N_LAG:, :]

    # Forward-chain CV with gap = N_LAG
    gap_bins = N_LAG
    def _folds(n, n_folds=5):
        block = n // (n_folds + 1)
        for i in range(1, n_folds + 1):
            ts, te = i * block, min((i + 1) * block, n)
            tre = ts - gap_bins
            if tre < 100 or te <= ts:
                continue
            yield tre, ts, te

    def _cv(X, Y, alpha=100.0):
        fold_r2 = []
        for tre, ts, te in _folds(len(X)):
            m = Ridge(alpha=alpha, solver="auto")
            m.fit(X[:tre], Y[:tre])
            preds = m.predict(X[ts:te])
            yt = Y[ts:te]
            ss_res = ((yt - preds) ** 2).sum(axis=0)
            ss_tot = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
            ss_tot = np.where(ss_tot == 0, 1.0, ss_tot)
            fold_r2.append(1.0 - ss_res / ss_tot)
        return np.mean(np.stack(fold_r2, axis=0), axis=0) if fold_r2 else np.full(Y.shape[1], np.nan)

    r2_full = _cv(X_lagged, Y_lag)
    # Drop-one per band
    r2_unique = {}
    for drop in band_order:
        keep_cols = []
        for bi, band in enumerate(band_order):
            if band == drop:
                continue
            sl = slices[bi]
            for c in range(sl.start, sl.stop):
                keep_cols.extend(range(c * N_BASIS, (c + 1) * N_BASIS))
        r2_red = _cv(X_lagged[:, keep_cols], Y_lag)
        r2_unique[drop] = r2_full - r2_red

    rows = [dict(unit_id=uid, area=area, full_r2=float(r2_full[i]),
                 unique_running=float(r2_unique["running"][i]),
                 unique_pupil=float(r2_unique["pupil"][i]))
            for i, (uid, area) in enumerate(zip(all_uids, all_areas))]
    df = pd.DataFrame(rows)
    area_agg = df.groupby("area").agg(
        n_units=("unit_id", "count"),
        full_r2_mean=("full_r2", "mean"),
        full_r2_sem=("full_r2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        full_r2_frac_pos=("full_r2", lambda x: (x > 0).mean()),
        running_mean=("unique_running", "mean"),
        pupil_mean=("unique_pupil", "mean"),
    ).reset_index()
    return dict(unit_df=df, area_df=area_agg)


# =============================================================================
# Deliverable H: noise correlations by state
# =============================================================================
def run_H(bundle):
    units, spikes, stim = bundle["units"], bundle["spikes"], bundle["stim"]
    ab_start, ab_end = bundle["active_range"]
    pb_start, pb_end = bundle["passive_range"]

    grid_act = _time_grid(ab_start, ab_end)
    grid_pas = _time_grid(pb_start, pb_end)

    flashes_act = stim[(stim["stimulus_block"] == bundle["active_block"])
                        & (stim["is_omission"] == 0)]["t"].dropna().values
    flashes_pas = stim[(stim["stimulus_block"] == bundle["passive_block"])
                        & (stim["is_omission"] == 0)]["t"].dropna().values

    def _residualize(Y, grid, flash_times, window=(-0.25, 0.5), bin_size=BIN_SIZE):
        n_bins, n_units = Y.shape
        pre_bins = int(-window[0] / bin_size); post_bins = int(window[1] / bin_size)
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

    def _nc(Y):
        sd = Y.std(axis=0); valid = sd > 0; Y = Y[:, valid]
        if Y.shape[1] < 2:
            return np.array([[]])
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        C = (Y.T @ Y) / Y.shape[0]
        np.fill_diagonal(C, np.nan)
        return C

    area_map = _area_units_map(units, spikes)
    rows = []
    for area, uids in area_map.items():
        if len(uids) < 8:
            continue
        Y_act = np.zeros((len(grid_act), len(uids)), dtype=np.float32)
        Y_pas = np.zeros((len(grid_pas), len(uids)), dtype=np.float32)
        for i, uid in enumerate(uids):
            st = spikes[uid]
            Y_act[:, i] = _bin_unit(st[(st >= ab_start) & (st < ab_end)], grid_act)
            Y_pas[:, i] = _bin_unit(st[(st >= pb_start) & (st < pb_end)], grid_pas)
        Y_act_r = _residualize(Y_act, grid_act, flashes_act)
        Y_pas_r = _residualize(Y_pas, grid_pas, flashes_pas)
        C_act = _nc(Y_act_r); C_pas = _nc(Y_pas_r)
        iu = np.triu_indices(C_act.shape[0], k=1)
        nc_a = C_act[iu]; nc_p = C_pas[iu]
        valid = np.isfinite(nc_a) & np.isfinite(nc_p)
        if valid.sum() < 10:
            continue
        try:
            _, p = stats.wilcoxon(nc_a[valid], nc_p[valid])
        except ValueError:
            p = np.nan
        rows.append(dict(area=area, n_units=len(uids), n_pairs=int(len(nc_a)),
                          mean_nc_active=float(np.nanmean(nc_a)),
                          mean_nc_passive=float(np.nanmean(nc_p)),
                          delta_nc=float(np.nanmean(nc_a) - np.nanmean(nc_p)),
                          wilcoxon_p=float(p) if np.isfinite(p) else None))
    return dict(area_df=pd.DataFrame(rows))


# =============================================================================
# Driver
# =============================================================================
def run_all(session_id: int, save: bool = True) -> dict:
    t_total = time.time()
    print(f"\n{'='*70}\nSESSION {session_id}\n{'='*70}")
    bundle = load_session_bundle(session_id)
    out = bundle["out_dir"]
    results = {}

    for name, fn in [("A2", run_A2), ("B2", run_B2), ("D2", run_D2), ("H", run_H)]:
        print(f"[{session_id}] running {name}...")
        t0 = time.time()
        try:
            r = fn(bundle)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            r = None
        print(f"  {name} done in {time.time()-t0:.1f}s")
        results[name] = r
        if save and r is not None:
            if "area_df" in r and isinstance(r["area_df"], pd.DataFrame):
                r["area_df"].to_csv(out / f"{name}_area.csv", index=False)
            if "unit_df" in r and isinstance(r["unit_df"], pd.DataFrame):
                r["unit_df"].to_parquet(out / f"{name}_unit.parquet", index=False)
            meta = {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str, list, tuple, dict))
                    and not isinstance(v, pd.DataFrame)}
            (out / f"{name}_meta.json").write_text(json.dumps(meta, default=str, indent=2))

    results["session_id"] = session_id
    results["n_units_total"] = len(bundle["units"])
    results["active_range"] = bundle["active_range"]
    results["passive_range"] = bundle["passive_range"]
    print(f"[{session_id}] TOTAL: {time.time()-t_total:.1f}s")
    return results


if __name__ == "__main__":
    import sys
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 1055240613
    run_all(sid)
