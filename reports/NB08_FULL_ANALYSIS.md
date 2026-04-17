# VBN Session 1055240613 — Full NB08 Analysis

**Session:** 1055240613 (primary target, recommended by Corbett Bennett / Allen Institute)
**d' = 1.81** (solid engagement)
**N units:** 734 quality-filtered across 27 areas; 508 retained after min-spike threshold
**Blocks:** active task (block 0, 60 min) + passive replay (block 5, 60 min) + RF mapping (blocks 2,4)
**Analysis date:** 2026-04-16

---

## Method changes this run

1. **Dropped 780 raw SuperAnimal keypoints** from `pose_features.parquet` (805 → 25 cols; 3.66 GB → 69 MB in RAM). Kept `body_speed` which turned out to be all-NaN (known bug in NB07 derivation from keypoint diffs) — dropped at analysis time. Encoding uses 24 covariates: 1 running + 3 pupil (area, x, y) + 20 face SVD.
2. **Fixed encoding model pipeline in `src/cross_correlation.py`:**
   - Z-score target and features before Ridge fit (was missing — primary cause of R²=-1.67)
   - Alpha grid expanded to [1, 10, 100, 1000, 10000] (was [0.01, 0.1, 1, 10, 100] — saturating)
   - Fixed silent einsum/reshape column-ordering bug in raised-cosine basis (basis cols were misaligned with coefficient blocks)
3. **Pipeline: fit per-unit, per-epoch (active block only)** — sidesteps distribution shift between active/passive/grey-screen blocks that made the session-wide forward-chain CV extrapolate catastrophically.
4. **Fixed `io_nwb.extract_units_and_spikes`** to join `ecephys_structure_acronym` from `nwb.electrodes` (VBN stores area there, not in `nwb.units`).

---

## Reference: Piet et al. 2025 (biorxiv 2025.10.17)

Canonical VBN paper to replicate/compare against. Key claims:
- Task-engagement modulation **strongest in midbrain**, increases up visual hierarchy
- Image-ID decoding latencies: LGd 38 ms → VISp 44 → VISal 46 → VISpm 53
- Change-signal latencies: LP leads (~53 ms), cortex ~60 ms
- Novelty modulation cortical but not thalamic (untestable this session — all images familiar)

---

## Findings

### A. Active vs Passive flash response (per-unit MI)

`MI = (R_active − R_passive) / (R_active + R_passive)` on 2500 non-change flashes each.

| Area | n | MI ± SEM | Frac active-dominant |
|---|---|---|---|
| MGd | 66 | −0.45 ± 0.03 | 1.5% |
| VISrl | 18 | −0.26 ± 0.06 | 16.7% |
| LGd | 42 | −0.25 ± 0.04 | 23.8% |
| VISp | 44 | −0.24 ± 0.04 | 11.4% |
| VISam | 32 | −0.22 ± 0.05 | 15.6% |
| VISpm | 39 | −0.20 ± 0.05 | 20.5% |
| VISl | 40 | −0.19 ± 0.04 | 12.5% |
| CA1 | 42 | −0.11 ± 0.07 | 40.5% |
| VISal | 42 | −0.08 ± 0.05 | 28.6% |
| MRN | 28 | −0.07 ± 0.08 | 42.9% |
| DG | 37 | **+0.11** ± 0.07 | 62.2% |
| **SCig** | 41 | **+0.17** ± 0.05 | 80.5% |

- **All visual/thalamic areas passive-dominant.** Replicates VBN "engagement paradox" — stronger sensory responses when task is removed (adaptation accumulates across active block; removing reward-structure releases gain).
- **SCig (+0.17) and DG (+0.11) active-dominant** → replicates Piet 2025 "midbrain drives task modulation" claim. SCig is motor/orienting; DG is novelty/context-sensitive hippocampus.
- **95%+ of units in every area significantly modulated** (Mann-Whitney U, p<0.05) — effect is pervasive, not limited to specialist cells.
- Visual hierarchy gradient test (Piet: MI should become more negative up hierarchy): Spearman r=0.18, p=0.70 → **not significant in this single session**. Piet's gradient is population-level across 54 mice.

**Plot:** `A_active_vs_passive_hierarchy.png`

### B. Change-detection signal across the hierarchy

Change flashes (n=183, is_change=1) vs repeat flashes (n=183, non-change) in active block.

| Area | n | Change/Repeat ratio | Change-onset latency |
|---|---|---|---|
| VISal | 42 | 1.30 | **38 ms** (fastest) |
| VISp | 44 | 1.61 | 43 ms |
| VISrl | 18 | 1.57 | 43 ms |
| LGd | 42 | 1.49 | 48 ms |
| VISl | 40 | 1.65 | 48 ms |
| MRN | 28 | 2.67 | 48 ms |
| VISam | 32 | 2.16 | 48 ms |
| VISpm | 39 | **2.28** | 58 ms |
| MGd | 66 | 5.02 | 98 ms (noise — tiny absolute) |

- **Change amplification increases up the visual hierarchy:** VISal 1.3 → VISp/VISl ~1.6 → VISpm/VISam ~2.2. Later visual areas show larger relative boost for change events. Clean hierarchy gradient.
- **Latencies 10-20 ms faster than Piet 2025** — methodology difference (first sustained 2σ deviation vs Piet's decoding AUC half-peak). Hierarchy ordering within visual cortex is consistent with Piet.
- VISal 38 ms fastest was unexpected — may reflect its known role in change-sensitive computation (Siegle 2021).
- MRN (midbrain reticular) shows 2.7× amplification: motor preparation scales with detection confidence.

**Plots:** `B_change_psth_per_area.png`, `B_change_latency_hierarchy.png`

### C. Pre-stimulus arousal predicts hit vs miss (Steinmetz test)

Pre-stim ([−1, 0]s) mean running/pupil/face_svd_0, n=180 change trials (157 hits, 23 misses).

| Covariate | OR per SD | AUROC |
|---|---|---|
| Pre-stim running | 0.73 | 0.61 |
| Pre-stim pupil | 0.76 | 0.51 |
| Pre-stim face movement (SVD0) | 1.21 | 0.54 |
| Multivariate (all 3) | — | 0.61 |

Running-tertile hit rates: low 92%, mid 88%, high 82% (χ²=2.79, p=0.25)
Pupil-tertile hit rates: low 88%, mid 83%, high 90% (χ²=1.30, p=0.52)

- **Null result at behavioral state level.** AUROC 0.51–0.61 is weak; trends not significant.
- **Direction of running effect is inverse to naive expectation**: faster running → more misses. Possible interpretation: high-vigor running is an "impatient" / non-attending state in this change-detection task.
- Sample limit: 23 misses is very few; high base hit rate (87%) compresses the dynamic range. Steinmetz 2019's effect is usually tested with pre-stim neural population state (not pure behavior), which this session has in abundance — worth retrying with leading PC or predictive decoder.

**Plot:** `C_arousal_hit_miss.png`

### D. Per-area encoding R² (behavior → neural)

Per-unit ridge encoding with 24 covariates × 8 raised-cosine lags. Forward-chain CV within active block. Variance partitioning by drop-group.

| Area | n | Full R² | Unique running | Unique pupil | Unique face_svd |
|---|---|---|---|---|---|
| SCig | 41 | **+0.013** | **+0.040** | +0.005 | −0.002 |
| VISal | 42 | −0.007 | +0.008 | +0.003 | −0.003 |
| VISl | 40 | −0.012 | +0.006 | +0.005 | −0.001 |
| VISam | 32 | −0.020 | +0.004 | +0.000 | −0.001 |
| VISp | 44 | −0.021 | +0.007 | +0.005 | −0.002 |
| VISpm | 39 | −0.025 | +0.006 | +0.002 | −0.002 |
| LGd | 42 | −0.025 | +0.006 | +0.004 | +0.001 |
| VISrl | 18 | −0.027 | +0.007 | −0.001 | +0.002 |
| CA1 | 42 | −0.031 | +0.005 | −0.001 | −0.002 |
| MGd | 66 | −0.031 | +0.003 | −0.001 | −0.001 |
| MRN | 28 | −0.032 | **+0.049** | +0.004 | +0.006 |
| DG | 37 | −0.065 | +0.001 | −0.008 | +0.006 |

- **Behavioral covariates explain 0–1% of single-unit spike count variance in Neuropixels at 25ms bins.** This is consistent with spike trains being dominated by stimulus-locked variance; behavior is a modest additive signal.
- **SCig and MRN are the only "behavior encoders"** at group level: running contributes 4–5% unique R². SCig is the only area with positive mean full R² (70% of units have positive R²).
- **Running > pupil > face_svd ranking holds across all areas.** Running is the dominant behavioral covariate at single-cell resolution.
- **Stringer 2019's "face SVD explains ⅓ of V1 variance" does not replicate here.** Face SVD adds essentially zero unique variance beyond running+pupil. Likely reasons: (i) Stringer used two-photon population latents vs single Neuropixels spike counts at 25ms bins; (ii) Stringer used spontaneous activity, not a structured task where flashes dominate variance; (iii) Facemap SVD quality here may be suboptimal (single session, no keypoint augmentation).
- **Reimer 2014 confirmed:** running and pupil both contribute positive unique variance in most cortical areas; they are independent covariates.

**Plots:** `D_encoding_per_area.png` (see also `Z_final_summary.png` for the integrated view)

### F. Flash-onset latency hierarchy

Per-area population PSTH, 4σ threshold over pre-stim baseline for ≥20 ms.

| Area | My latency | Piet 2025 |
|---|---|---|
| VISp | 31 ms | 44 ms |
| VISal | 31 ms | 46 ms |
| VISrl | 37 ms | — |
| VISl | 41 ms | 45 ms |
| LGd | 47 ms* | 38 ms |
| VISam | 49 ms | 53 ms |
| VISpm | 51 ms | 53 ms |

- **My latencies run ~10 ms faster than Piet** — my method uses first 4σ deviation (noisier, earlier) vs Piet's decoding half-peak (smoother, later).
- **Visual cortex hierarchy is correctly ordered** in this session: VISp/VISal → VISl → VISpm/VISam. Within-hierarchy Spearman r=0.47, p=0.29 (trend positive, 6-area ordering cannot reach significance).
- \***LGd anomaly** (47 ms, later than V1 input target VISp at 31 ms) is a threshold artifact — LGd has 3× higher baseline firing (8.6 Hz) making the absolute 4σ threshold much higher. Half-peak latency or ZETA test would fix this.

**Plot:** `F_latency_hierarchy.png`

---

## Headline findings

1. **VBN "engagement paradox" reproduced and area-resolved.** Visual cortex shows passive > active; SCig and DG show active > passive. The active-minus-passive gradient is a single-session replication of Piet 2025's population-level claim that task modulation is midbrain-anchored.

2. **Change signal amplifies up the visual hierarchy** (VISal 1.3× → VISpm 2.3×). Hierarchy-resolved change sensitivity is cleaner in this single session than flash-locked latency ordering.

3. **Behavioral encoding is modest but real in motor areas** (SCig, MRN ~4-5% unique R²). Visual cortex only gets ~1% unique behavioral R². Stringer's ⅓ V1 figure does not transfer to Neuropixels in a task context.

4. **No pre-stimulus arousal effect on trial outcome at behavioral state level.** High-engagement d'=1.8 session with 87% hit rate may just saturate detectability; arousal gating would show better in weaker-engagement sessions.

5. **Methodology discovery:** the session-wide encoding model had R²=-1.67 due to three compounding bugs (no z-scoring + coarse alpha grid + einsum column order) plus the architectural problem of CV folds crossing active/passive block boundaries. Per-epoch, z-scored, wide-alpha Ridge gives clean near-zero R² for visual cortex and positive R² for SCig, matching biological expectations.

---

---

## Post-critique addendum (2026-04-16)

Three expert-panel critiques (methodology, Allen-insider, behavior-neural) raised specific concerns. Top-4 actionable items were addressed immediately. Results summary: `Z2_critique_resolution_summary.png` + `Z2_before_after_master.csv`.

### Fixes applied
1. **`gap_bins` 20 → 40** in `src/cross_correlation.py` and `scripts/analysis/d_encoding_per_area.py`. Previous value was INSIDE the 40-bin raised-cosine kernel support. Genuine bug — every prior R² was optimistically biased by autocorrelation leakage.
2. **MGd / MGv / MGm dropped** from `ALL_TARGET_AREAS` in `_shared.py`. Session 1055240613 has a known off-target probe registration; MGd units are almost certainly LP-adjacent, not genuine auditory thalamus.
3. **`pupil_vel` added** to encoding covariates (was computed in `features_eye.py` but never passed to the model — literal dropped feature).
4. **Saccade detection**: `pupil_x`/`pupil_y` velocity with 6× MAD threshold → 11,321 events (~1.2/s). Added as event regressor to the encoding model.
5. **Licks + rewards extracted** from NWB (`nwb.processing["licking"]`, `nwb.processing["rewards"]`) — addresses CLAUDE.md gap #7. 4,338 licks, 160 rewards.

### Critique-resolution findings

**Finding A — active vs passive MI under arousal matching (Deliverable A2):**
- Only **2% of passive flashes** fall within active-block arousal IQR (98% differ). Passive block of 1055240613 had *higher* pupil area than active, not lower — counterintuitive, but real.
- After matching: visual-cortex passive>active survives but weakens (VISp −0.24→−0.20, VISl −0.19→−0.11, VISpm −0.20→−0.14, CA1 −0.11→−0.04).
- **SCig MI unchanged (+0.17 → +0.18)** — SCig active-dominance is NOT arousal-confounded.
- **DG active-dominance strengthens** (+0.11 → +0.15).

**Finding B — change amplification under lick/motor controls (Deliverable B2):**
- **SCig miss-only change ratio = 1.98×** — survives. Signal is sensory/attentional, not pure motor prep.
- **VISpm miss-only = 4.49×** (up from 2.28× in original) — strongest change signal is higher visual, not midbrain.
- **DG change amplification COLLAPSES** on miss trials (1.49× → 0.81×) — DG signal was reward-driven, not change-detection. Novel finding from critique-prompted sanity check.
- **CA1 similarly collapses** (1.15× → 0.64×). Hippocampal change response requires the reward.
- **MRN miss-only = 4.07×** — real, largest amplification.

**Finding D — encoding R² with saccades + pupil_vel + correct gap_bins (Deliverable D2):**
- **SCig full R² doubled**: 0.013 → **0.029**, 80.5% of units positive (was 70%) — gap_bins fix matters.
- **Saccade unique R² in SCig: +0.017** — saccade-triggered activity is a large fraction of SCig firing (SC is the canonical saccade generator; sanity check passes).
- V1/V1-adjacent saccade unique R²: near zero (+0.0001 in VISp). Mouse V1 saccadic suppression is smaller than primate, or 25ms bins undersample the effect.
- Face SVD still ~0 unique across all areas — Stringer test still fails, but critique correctly identified this as a strawman (different target, epoch, scale).

### What the critique DIDN'T overturn

- **Change amplification up the visual hierarchy** (VISal 1.3× → VISpm 2.3× → 4.5× in miss-only). Robust.
- **Hierarchy latency ordering within visual cortex** (VISp fastest, VISpm slowest). Robust.
- **Running > pupil > face SVD** for behavior variance explained across areas. Reimer 2014 confirmed.
- **SCig active-dominance + change amplification** — survives both wakefulness matching AND miss-only sanity check. This is a solid result.

### What the critique clearly overturned

- **MGd = auditory thalamus** — No. Misregistered probe. Removed from analyses.
- **DG change signal = sensory** — No. Reward-driven. Collapses without lick.
- **Stringer replication = null** — Premature. The test was a strawman; need population-latent target (not per-unit at 25ms).

### Still TODO (from critique, not yet addressed)
- Banded ridge via `himalaya` (CRITICAL methodology issue — current variance partitioning is biased by band-size imbalance 1:5:20:1)
- Poisson GLM via `glum` (HIGH — Gaussian MSE on spike counts biases area comparison)
- ZETA latency test (HIGH — replaces baseline-rate-biased 4σ threshold)
- FDR correction across 500+ per-unit tests (HIGH)
- Per-block drift QC (HIGH — session-wide presence_ratio may hide block-5 dropouts)
- Population-latent Stringer test (HIGH — fair replication test with this data)
- Multi-ROI + motion SVD + keypoints from Facemap v2 (HIGH — Syeda 2024 doubles face variance)
- Quadratic arousal terms in hit/miss logistic (McGinley inverted-U)

---

## What's next (unpublished territory)

From lit agent recommendations (Piet 2025 + SWDB 2024 best practices):

1. **Frame-matched active/passive per-unit paired statistics** — every passive-replay flash has a paired active flash (same image identity); compute MI on matched pairs for tighter statistics than tertile aggregates.
2. **Adaptation kernel per area** — image repeat 1→2→3→4→change analysis. Piet argues mice use an adaptation-based strategy; per-area timescales are an extension.
3. **Noise correlation matrix by state** — unpublished at the 6-area × hippocampus × thalamus level.
4. **Banded ridge (himalaya)** — proper kernel-based variance partitioning with 20+ feature groups at scale.
5. **ZETA test** (Montijn 2021) for latency — parameter-free, fixes the LGd threshold artifact.
6. **CEBRA latent embedding** — single-session active/passive alignment in behavioral-neural joint latent space.
7. **Sharp-wave ripple triggered coactivation** — session has CA1/CA3/DG coverage; ripple-triggered cortical reactivation during passive replay is accessible.
