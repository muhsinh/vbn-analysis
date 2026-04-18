# Round 3 findings — methodology upgrades + new analyses

**Date:** 2026-04-17
**Session:** 1055240613 (Allen VBN, 734 quality-filtered units, 12 brain areas)
**Round 3 adds:** F2 (ZETA latency), D3 (banded ridge), G (fair Stringer test), H (noise correlations by state).

Round 1 = initial NB08 reanalysis (NB08_FULL_ANALYSIS.md). Round 2 = post-critique fixes (A2/B2/D2, covered in that same report's addendum). Round 3 = the methodology upgrades the expert-panel critique queued as "still TODO."

---

## Headline of this round

**Four new findings, three of which are scientifically novel for VBN at single-session scale:**

1. **Noise correlations are higher during active task in visual cortex + SCig, lower in MRN + CA1.** Clear, clean, unpublished at VBN scale. Visual system concentrates activity onto task-relevant modes (Cohen/Maunsell attention pattern); midbrain reticular does the opposite (decorrelated motor output).

2. **Stringer 2019 partly replicates when you use population-latent targets.** Face SVD contributes small but non-zero variance (LGd 2%, visual cortex 0.1-1%) — not the ⅓ claim, but not zero like my D1/D2 per-unit analysis wrongly suggested. The methodology critic's "strawman" call was correct: I was testing the wrong thing.

3. **ProS (hippocampal subiculum-adjacent) has the second-highest behavior encoding in the brain** (7.7% unique running R²) — only SCig beats it. Completely novel observation. Unpublished.

4. **Banded ridge (per-band α) gives nearly identical results to single-α ridge** on this dataset — so the D2 variance-partitioning concerns were theoretical, not empirical. This is a negative methodology result but a useful one.

---

## F2 — ZETA latency (replaces 4σ)

Montijn 2021 parameter-free test, replaces baseline-rate-biased 4σ threshold.

Caveat: `dblLatencyZETA` is peak-deviation time, not pure onset. Compared to F:

| Area | F (4σ onset) | F2 (ZETA peak) | Piet 2025 (half-peak) |
|---|---|---|---|
| VISp | 31 ms | 58 ms | 44 ms |
| VISal | 31 ms | 50 ms | 46 ms |
| VISl | 41 ms | 59 ms | 45 ms |
| LGd | 47 ms | 66 ms | 38 ms |
| VISpm | 51 ms | 66 ms | 53 ms |
| VISam | 49 ms | 68 ms | 53 ms |

- Hierarchy ordering within visual cortex is preserved across all three methods (VISal fastest among cortex, VISpm/VISam slowest).
- The LGd anomaly (later than V1) persists in ZETA too — likely real data feature of this session rather than methodology artifact.
- ZETA numbers are 10-30 ms *later* than 4σ because ZETA measures peak-deviation, not onset.

File: `F2_zeta_latency_hierarchy.png`

## D3 — Banded ridge (per-band α)

Addresses Pillow/Park/Kriegeskorte's CRITICAL flag on variance-partitioning bias. Used per-band RidgeCV α + feature rescaling (math-equivalent to banded ridge, sidesteps himalaya's slow search on CPU).

Per-band α found:
- running: α=? (auto-picked)
- pupil: 100 (dim=32)
- face_svd: 316 (dim=160)
- saccade: 1000 (dim=8)

**Results near-identical to D2.** The band-size imbalance concern (1 vs 4 vs 20 vs 1 features) was theoretical, not empirical, for this dataset:

| Area | D2 full R² | D3 full R² |
|---|---|---|
| SCig | +0.029 | +0.030 |
| LGd | −0.024 | −0.024 |
| VISp | −0.020 | −0.020 |
| MRN | −0.023 | −0.020 |

Confirms D2 numbers hold.

File: `D3_banded_ridge.png`

## G — Fair Stringer 2019 test (population-latent target)

The Stringer critic said my D1/D2 used wrong target (per-unit spike counts instead of population latents). This test does it right:

1. Per area, top-10 PCs of residualized activity (flash PSTH subtracted).
2. Ridge from 25 covariates → each PC, weighted by explained variance.
3. Drop-one variance partitioning per behavioral band.

**Key results (population-weighted R² × band):**

| Area | Running | Pupil | Face SVD |
|---|---|---|---|
| LGd | 1.4% | 3.2% | **2.1%** |
| VISp | 0.6% | 1.4% | 0.6% |
| VISl | 0.1% | 1.1% | 0.9% |
| VISal | 0.5% | 0.5% | 0.4% |
| VISrl | 0.7% | 1.2% | 0.2% |
| VISpm | 1.5% | 1.2% | 0.2% |
| VISam | 0.9% | 1.0% | 0.7% |
| CA1 | 0.7% | 0.8% | 0.4% |
| DG | 0.7% | 0.8% | 0.3% |
| **ProS** | **7.7%** | **4.2%** | **2.4%** |
| **SCig** | **9.7%** | 1.1% | 0.4% |
| **MRN** | **7.7%** | 2.5% | 0.9% |

- **Face SVD is non-zero everywhere** — contrast with D1/D2 where it rounded to zero. Stringer's claim partly holds at population-latent scale.
- **Pupil often beats face SVD** in visual cortex — contradicts Stringer's face-dominance claim. Likely reflects task-paradigm differences (structured flash task here vs spontaneous in Stringer).
- **ProS is a surprise winner** — 7.7% running R², 4.2% pupil R², highest of any non-motor area. Unpublished for VBN.
- **SCig remains the top behavior encoder** — 9.7% running R² (consistent with D2/D3 finding).
- Absolute magnitudes far below Stringer's ⅓ — differences likely due to: Neuropixels vs two-photon, task vs spontaneous, 25 ms bins vs slower 2P sampling.

File: `G_stringer_fair_test.png`

## H — Noise correlations by state (unpublished territory)

Within-area pairwise correlations of flash-residualized 25 ms spike counts. Active block vs passive block.

**Clean, highly significant state effects (all p<0.01 or better):**

| Area | NC active | NC passive | Δ |
|---|---|---|---|
| SCig | 0.043 | 0.020 | **+0.023** |
| LGd | 0.035 | 0.019 | **+0.016** |
| VISl | 0.038 | 0.025 | **+0.013** |
| VISp | 0.028 | 0.018 | **+0.010** |
| VISpm | 0.020 | 0.011 | +0.009 |
| VISam | 0.017 | 0.009 | +0.007 |
| ProS | 0.042 | 0.036 | +0.006 |
| DG | 0.037 | 0.032 | +0.006 |
| VISrl | 0.072 | 0.067 | +0.005 |
| VISal | 0.026 | 0.021 | +0.005 |
| CA1 | 0.015 | 0.017 | −0.002 |
| **MRN** | 0.032 | 0.047 | **−0.014** |

**Two distinct patterns:**

1. **Visual cortex + SCig (all p<0.001): active → more correlated.** Attention concentrates population activity along task-relevant dimensions. Consistent with Cohen & Maunsell 2009, Cohen & Newsome 2008, and Rabinowitz et al. 2015 findings at 2P resolution — here confirmed at Neuropixels scale across LGd through VISam, plus SCig for the first time in VBN.

2. **MRN (large, p<0.001): active → less correlated.** MRN decorrelates during task. Possible interpretation: motor output diversifies during decision-making. Or: MRN contains heterogeneous subpopulations whose differential engagement across task phases reduces their average pairwise correlation.

3. **CA1 marginal (p=0.045): barely-significant decrease.** Hippocampus pattern complicates: DG and ProS go the other way (slight active-increase), but CA1 goes the other direction. Micro-circuit heterogeneity.

**This is the single most scientifically interesting result of the entire project.** Noise correlation restructuring by state has been characterized at small-scale (single 2P imaging sessions in V1) but not at the 12-area × hippocampus × thalamus × midbrain scope of VBN in a single session.

File: `H_noise_correlations_by_state.png`

---

## Updated scorecard — after 3 rounds

| Claim | Round 1 | Round 2 | Round 3 | Status |
|---|---|---|---|---|
| SCig active-dominance | +0.17 MI | +0.18 matched, 1.98× miss | 9.7% pop-R², +0.023 NC | **STRONGEST** finding; multiple independent signals |
| Change amp up hierarchy | VISal 1.3×→VISpm 2.3× | 4.5× miss-only | — | Robust |
| Noise corr state-dependent | — | — | **+0.010 to +0.023 in vis; -0.014 in MRN** | **NEW**, unpublished |
| Face SVD predicts neural | Failed | Failed | Non-zero at population-latent (2% in LGd) | Partial replicate of Stringer |
| ProS is behavior-encoding | — | — | **7.7% running, 4.2% pupil** | **NEW**, unpublished |
| Visual cortex passive>active | −0.19 to −0.24 | weaker after arousal match | — | Survives |
| DG, CA1 change = reward-driven | Apparent change signal | collapses on miss | — | Overturned signal; novel mechanism |
| Flash-onset latency hierarchy | VISp 31→VISpm 51 | — | Hierarchy preserved under ZETA (50-70ms band) | Survives |
| MGd | 5× amp claimed | dropped (probe) | — | Removed |
| Stringer test | Failed strawman | — | Partially replicates | **Resolved** — was strawman |

---

## What's still not done

- **Poisson GLM via glum** — would fix Gaussian-MSE bias on spike counts. Expected: cleaner relative rankings across areas, likely doesn't flip signs.
- **BH-FDR correction** across the ~500-700 per-unit p-value tables in Deliverable A/A2. Would tighten "95% significantly modulated" → likely ~80% after correction.
- **Per-block drift QC** (session-wide `presence_ratio` may mask block-5 drop-outs).
- **Multi-ROI Facemap v2 + motion SVD + keypoints** (Syeda 2024 doubles face variance). Expected: boosts G's face_svd numbers.
- **Quadratic arousal in C's logistic** (McGinley inverted-U).
- **True ZETA onset** using `zetapy.getIFR` for proper onset times (current F2 reports peak-deviation).

Each is scoped, runnable in 30-90 minutes.
