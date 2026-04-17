# Catching an R² = −1.67 bug in a 700-neuron encoding model, then having three expert agents tear the analysis apart

**Author:** Abdul Muhsin Hameed ([@shnazzers](https://x.com/shnazzers), [abdulh.xyz](https://abdulh.xyz))
**Session date:** April 2026
**Environment:** Claude Code (Opus 4.7, 1M context) on M3 Pro / 18 GB RAM
**Dataset:** Allen Institute Visual Behavior Neuropixels, session 1055240613 — 734 quality-filtered neurons across 12 brain areas (visual cortex, thalamus, hippocampus, midbrain), 60 min active task + 60 min passive replay, change-detection paradigm with 180 go trials / 24 catch trials, d' = 1.81.
**Repo:** [github.com/muhsinh/vbn-analysis](https://github.com/muhsinh/vbn-analysis)

---

## What this session is

A single multi-hour Claude Code session in which I:

1. Diagnosed why Jupyter notebook NB08 had consumed 60 GB of RAM and crashed (silent pipeline bug: 780 unused SuperAnimal keypoint columns were being fed into a lag-expanded ridge regression design matrix).
2. Hit a second, deeper bug: the encoding model was returning R² = **−1.67** on cross-validation — worse than predicting the mean. The permutation test confirmed it (z = −3.46 *below* the null distribution).
3. Fired **two subagents in parallel** — a literature survey of current VBN-era papers, and an independent code audit of the encoding pipeline. Both came back with specific, mergeable findings.
4. Applied the code fixes, ran five deliverables end-to-end, produced headline results, then **fired three more subagents in parallel** — each embodying a composite of real researchers in one of three subfields (computational-neuroscience methodology; Allen Institute VBN insider; behavior-neural integration).
5. Synthesized their critiques, found that **two of my headline findings were compromised** (one motor-confounded, one probe-registration artifact) and one was a strawman test. Applied another round of top-4 fixes. Re-ran everything. Documented what survived vs. what didn't.

The session demonstrates something I believe most people aren't doing with coding agents yet: **using them not as typists, but as adversarial peer reviewers**. Every claim in the final report survived an agent-panel critique before being published.

---

## Artifacts produced this session

```
outputs/reports/
├── NB08_FULL_ANALYSIS.md         ← Full findings + method changes + caveats
├── Z_final_summary.png            ← 9-panel integrated figure (round 1)
├── Z2_critique_resolution_summary.png  ← 6-panel before/after figure (round 2)
├── Z2_before_after_master.csv     ← Consolidated delta table
├── A_active_vs_passive_hierarchy.png   + CSV, JSON
├── A2_active_vs_passive_matched.png    + CSV, JSON (arousal-matched version)
├── B_change_psth_per_area.png          + CSV
├── B2_change_hierarchy_variants.png    + CSV (lick-controlled + miss-only version)
├── C_arousal_hit_miss.png              + CSV
├── D_encoding_per_area.png             + CSV (original)
├── D2_encoding_per_area.png            + CSV (with saccades + pupil_vel + gap-fix)
├── F_latency_hierarchy.png             + CSV
```

~15 PNG figures, ~10 CSV/JSON summaries, ~3000 LOC of analysis scripts (`scripts/analysis/*.py`), all reproducible from the cached artifacts.

---

## Act 1 — The 60 GB OOM

Opened the session on "my laptop died running NB08." Suspected memory; verified by reading the cached data shapes.

```
pose_features.parquet: shape=(575523, 805)  mem=3660.3 MB
```

805 columns. The pipeline was feeding 780 raw SuperAnimal-quadruped keypoint columns into a lag-expanded ridge regression (8 raised-cosine basis × 780 features = 6,240 columns in the design matrix; at 383k rows × float64, that's a ~20 GB design matrix, copied across CV folds and variance-partitioning reductions → 60 GB peak).

Thought about it properly against Stringer 2019, Syeda 2024, Musall 2019, and Siegle 2021: **no published VBN-era study motivates raw-keypoint inclusion in an encoding model for a head-fixed mouse.** SuperAnimal-quadruped was trained on freely-moving animals; on a head-fixed preparation, head keypoints (nose/ear/eye) are near-zero variance (pure noise), and remaining limb/tail signal is redundant with the NWB wheel encoder (locomotion) + the bottom-up face.mp4 SVD (grooming / forelimb reach).

Dropped the 780 raw keypoint columns at the NB07 merge step. Kept only the derived `body_speed` scalar. Result:

```
AFTER:  shape=(575523, 25)  mem=69.1 MB     # 53× reduction
```

That fixes the OOM. But running NB08 on the slim data revealed the *actual* bug.

---

## Act 2 — The R² = −1.67 bug

The encoding model (behavior → neural activity, 2 covariates: running + pupil) was returning catastrophically negative R²:

```
enc.full_r2:     -1.67
enc.unique_r2:   {"running": -0.47, "pupil": -0.20}
enc.shared_r2:   -1.00
enc.perm_p_value: 1.00
enc.perm_z_score: -3.46        # real model is 3.46σ WORSE than random shuffles
```

A correctly-specified model cannot produce R² < 0 under permutation by chance. Unique R² must be ≥ 0 under partition (dropping a feature can only worsen R², or leave it unchanged). Both constraints were violated simultaneously.

I fired two subagents:

- **Agent A** (`code-reviewer`): read `src/cross_correlation.py`, `src/modeling.py`, `src/neural_events.py`, the notebook cell, and diagnose the bug. Hypotheses I pre-specified: (1) distribution shift across active/passive/grey-screen blocks in forward-chain CV; (2) raised-cosine basis not unit-L2 normalized; (3) variance-partitioning math wrong; (4) missing z-scoring on target + features; (5) pop-summed target diluting per-unit signal.
- **Agent B** (`general-purpose`): literature survey of VBN-era papers (Piet 2025, Siegle 2021, Musall 2019, Stringer 2019, Syeda 2024, McGinley 2015) and identification of novel questions a single session can answer.

Both returned. Agent A's diagnosis (verbatim, slightly condensed):

> **All 5 hypotheses CONFIRMED, plus 3 additional bugs:**
>
> - `cross_correlation.py:260` has a silent einsum/reshape column-ordering bug: `np.einsum("tlf,lb->fbt", X_lagged, B).reshape(n_out, n_feat * n_basis, order="F")` produces columns in the wrong order. Combined with the coefficient-block extraction at line 518, coefficients are being read from the wrong columns of the ridge solution. Any kernel visualization output by this pipeline is silently corrupted.
> - RidgeCV alpha grid `[0.01, 0.1, 1.0, 10.0, 100.0]` saturates on a 383k × 16 design matrix. Expand to `[1, 10, 100, 1000, 10000]`.
> - No z-scoring of target (pop-summed spike counts) or features (running in cm/s, pupil in pixel²). Isotropic L2 penalty cannot operate correctly on incommensurate scales.

Three targeted fixes. Applied them in minutes.

Agent B's literature report identified **Piet et al. 2025** ([bioRxiv 2025.10.17.683190](https://www.biorxiv.org/content/10.1101/2025.10.17.683190v1)) as the canonical VBN paper to replicate — with specific published values to compare against:

- Image-ID decoding latencies: LGd 38 ms → VISp/VISl/VISal 44–46 ms → VISpm/VISam ~53 ms
- Change-signal latencies: LP leads at ~53 ms, cortex ~60 ms
- Task-engagement modulation **strongest in midbrain**, increases up the visual hierarchy
- Mice solve the task via an adaptation-based strategy, not image-comparison

That's my replication target. Also identified what's **not** been published at single-session resolution: frame-matched active/passive paired statistics, per-unit change amplification by area × outcome, adaptation kernel per area, and hippocampus/thalamus noise correlations by state.

---

## Act 3 — Five deliverables

Ran five analyses end-to-end. Each produces a figure + CSV + JSON. Summaries:

**A. Active vs Passive per-unit modulation index (MI = (R_act − R_pas)/(R_act + R_pas)).** 508 units across 12 areas, 2,500 matched non-change flashes per condition.
- All visual/thalamic areas passive-dominant: MGd −0.45, VISrl −0.26, LGd −0.25, VISp −0.24, VISam −0.22.
- **SCig (superior colliculus, midbrain) +0.17** and **DG (hippocampal dentate gyrus) +0.11** active-dominant.
- 95%+ of units per area significantly modulated (Mann-Whitney U).

**B. Change-detection signal across the hierarchy.** Change flashes (n=183) vs repeat flashes in the active block.
- Change amplification scales up the visual hierarchy: **VISal 1.30× → VISp 1.61× → VISpm 2.28×**.
- Onset latencies correctly ordered within visual cortex: VISp 43 ms → VISpm 58 ms.
- My latencies run ~10 ms faster than Piet 2025's half-peak method — expected from measuring first 2σ deviation vs. AUC half-peak.

**C. Pre-stim arousal → hit/miss.** 180 change trials (157 hits / 23 misses). Univariate logistic regression on pre-stim running, pupil, and face movement.
- AUROC 0.51–0.61 (weak). Running was the best predictor (AUROC 0.61) but with inverse direction: fast running → more misses.
- Interpretation: highly-engaged mouse at 87% hit rate saturates the behavioral dynamic range.

**D. Per-area encoding R² with variance partitioning (24 covariates: running + 3 pupil + 20 face SVD).** Active block, per-unit batched ridge. 508 units.
- Mean full R² slightly negative across cortical/thalamic areas — expected for Neuropixels single-unit firing at 25 ms bins in a flash-dominated task.
- **SCig is the only area with positive group R²**: +0.013, 70% of units positive.
- Running contributes most unique variance everywhere (Niell & Stryker confirmed). Face SVD adds ~0 beyond running+pupil — apparent Stringer 2019 non-replication.

**F. Flash-onset latency hierarchy.** 4σ-over-baseline method, 2,000 non-change flashes. Visual cortex ordering correct: VISp 31 → VISl 41 → VISpm 51 ms. LGd anomaly (47 ms, later than V1) = baseline-rate threshold artifact.

At this point I had a working analysis, five figures, and a markdown report. Ready to ship.

---

## Act 4 — The peer review

Before shipping, I fired three more subagents *in parallel*, each prompted to channel a composite of real working researchers in a specific sub-field. Each got the full analysis output + scripts. Each was asked to tear it apart.

- **Agent C** — methodology critic (composite of Jonathan Pillow, Il Memming Park, Nikolaus Kriegeskorte, Eero Simoncelli)
- **Agent D** — VBN insider (Joshua Siegle, Corbett Bennett, Shawn Olsen, Marina Garrett, the Piet 2025 authors)
- **Agent E** — behavior-neural expert (McGinley, Stringer, Musall, Churchland, Cardin, Harris, Reimer/Tolias)

Convergence matrix from their responses (ranked by severity):

| Issue | Meth | Allen | Behav | Severity |
|---|---|---|---|---|
| Use ZETA for latency (retire 4σ) | ✓ | ✓ | ✓ | **UNANIMOUS** |
| Banded ridge (`himalaya`) — drop-one is biased with 1:3:20 unequal bands | ✓ | — | ✓ | CRITICAL |
| `gap_bins = 20` inside raised-cosine 40-bin kernel support (literal bug) | ✓ | — | — | HIGH |
| SCig active-dominance may be **lick motor contamination** | — | ✓ | — | CRITICAL |
| Passive block ~30% quiescent — no wakefulness filter | — | ✓ | — | CRITICAL |
| MGd (66 units) is **probe-registration artifact**, not auditory thalamus | — | ✓ | — | HIGH |
| `pupil_vel` computed in `features_eye.py:46-54` but **never fed to the encoder** | — | — | ✓ | CRITICAL |
| No saccade detection → V1 unique variance underestimated | — | — | ✓ | CRITICAL |
| Stringer test is strawman: per-unit spike count ≠ population latent | — | — | ✓ | HIGH |
| Gaussian MSE on Poisson spike counts biases area comparison | ✓ | — | — | HIGH |
| No FDR correction across 500+ per-unit p-values | ✓ | — | — | HIGH |

Two of my six headline findings were in doubt:

- **"SCig active-dominance replicates Piet 2025's midbrain claim"** — Agent D said this might be motor/lick contamination, because the analysis window [30, 200] ms overlaps with earliest lick latency (250 ms for fast responders). Proposed test: re-run on miss trials only (no lick ever).
- **"MGd shows 5× change amplification"** — Agent D flagged this as a misregistered probe. Session 1055240613 is known to have an off-target probe that registers LP-adjacent units as "MGd." Drop entirely.

---

## Act 5 — Re-running with fixes

Applied top-4 fixes from the critique synthesis:

1. **Gap_bins 20 → 40** everywhere in `src/cross_correlation.py` and the analysis scripts (the previous value was inside the raised-cosine kernel support — genuine bug).
2. **Dropped MGd/MGv/MGm** from `ALL_TARGET_AREAS`.
3. **Added `pupil_vel`** to the encoding covariates.
4. **Detected 11,321 saccade events** from pupil_x/pupil_y velocity (6× MAD threshold, 50 ms min-inter-onset-interval → ~1.2/s for a head-fixed mouse, within McFarland 2015's reported range) and added as an event regressor.

Also extracted lick times (4,338 events) and rewards (160 events) from the NWB file — addressed a long-standing CLAUDE.md gap.

Re-ran three deliverables under the new conditions:

**A2. Active vs Passive, arousal-matched.** Filter passive flashes to those where both running AND pupil fall within the active-block 25–75% IQR.
- **Only 2% of passive flashes (88/4437) fall in the matched arousal range.** Confirms the wakefulness-filter concern: most of the passive block is *not* at active-block arousal levels.
- Passive block mouse was **hyper-aroused** (mean pupil 3009 vs active 2288), not drowsy — counterintuitive but real.
- Visual-cortex passive>active **survives matching but weakens**: VISp −0.24 → −0.20, VISl −0.19 → −0.11, VISpm −0.20 → −0.14. Real task effect, plus an arousal-distribution component.
- **SCig MI unchanged (+0.17 → +0.18)** — SCig active-dominance is **not** arousal-confounded.
- **DG active-dominance strengthens** (+0.11 → +0.15).

**B2. Change amplification under lick/motor controls.** Variants: (1) exclude change trials with response latency < 250 ms, (2) miss-only (no lick possible), (3) adaptation-matched repeat baseline (flashes_since_change = 1).
- **SCig miss-only ratio = 1.98×.** SCig change amplification *survives* the miss-only sanity check. Signal is sensory/attentional, **not motor prep.**
- **VISpm miss-only = 4.49×** (up from 2.28× in the original analysis) — strongest change signal in the hierarchy when properly measured.
- **DG change amplification COLLAPSES on miss trials: 1.49× → 0.81×.** **Novel finding emergent from critique-prompted sanity check: DG "change detection" is reward-driven, not sensory.**
- **CA1 similarly collapses** (1.15× → 0.64×).

**D2. Encoding with saccades + pupil_vel + correct gap_bins.**
- **SCig full R² DOUBLED** (0.013 → 0.029; 80% of units now positive, up from 70%). Gap_bins fix matters a lot.
- **SCig saccade unique R² = 0.017** — saccade-triggered activity predicts SCig firing. Sanity check on saccade detector: SC is the canonical saccade generator, so this is expected and validates the detector.
- V1 saccade unique R² near zero — mouse V1 saccadic suppression is smaller than primate, or 25 ms bins undersample the effect.

---

## Final scorecard — what survived critique

| Claim | Original | Post-critique | Verdict |
|---|---|---|---|
| SCig active-dominance | +0.17 MI | +0.18 arousal-matched, **+1.98× miss-only** | **SURVIVES** — real task signal |
| SCig full encoding R² | +0.013 | **+0.029** (2×) | **STRONGER** after gap_bins fix |
| Visual cortex passive > active | −0.19 to −0.24 | −0.11 to −0.20 | **SURVIVES** — partly arousal-confounded |
| Change amp scales up hierarchy | VISal 1.3× → VISpm 2.3× | VISal 2.0× → VISpm **4.5×** on miss | **STRONGER** with adaptation-matched baseline |
| DG change signal | +0.11 MI, 1.49× | **0.81× on miss trials** | **OVERTURNED** — reward-driven |
| CA1 change signal | 1.15× | **0.64× on miss trials** | **OVERTURNED** — reward-driven |
| MGd 5× change amplification | 5.02× | — | **REMOVED** — misregistered probe |
| Stringer replication fails | face_svd R² ~0 | face_svd R² ~0 | **INCONCLUSIVE** — test was strawman |
| Flash-onset latency hierarchy | VISp 31 → VISpm 51 ms | unchanged | **SURVIVES** — caveat on LGd anomaly |

Three of nine claims in the final headline list came out of this process *stronger* than they started. Two were overturned. One was removed entirely. One was flagged as an unfair test and requires a rewrite to evaluate.

---

## Still TODO (queued for next session)

Honest list of remaining critique items I haven't addressed:

1. `himalaya.BandedRidgeCV` with per-band alphas — only clean fix for 1:5:20:1 band-size imbalance in variance partitioning
2. Poisson GLM via `glum` — replaces Gaussian-MSE bias on spike counts
3. ZETA test (`zetapy`) for latency — retires the baseline-rate-biased 4σ threshold
4. BH-FDR across 500+ per-unit tests
5. Per-block drift QC (per-block presence_ratio check, not session-wide)
6. Population-latent Stringer test (top 10 PCs per area × CCA against 24 covariates) — fair version of the Stringer replication
7. Multi-ROI Facemap v2 + motion SVD + face keypoints (Syeda 2024 doubles face variance)
8. Quadratic arousal terms in hit/miss logistic (McGinley inverted-U)

Each is a specific, scoped fix with a concrete expected delta.

---

## What I think is worth noting

A coding agent is a tool for putting **tighter feedback loops around your own reasoning**. The most valuable moves in this session weren't any single code edit — they were:

- Spawning three critic agents in parallel with specific expert framings, and taking their findings seriously enough to *throw out two of my headline results*.
- Letting the literature agent tell me that Piet 2025 existed (I had not known about it), which gave me specific numbers to try to replicate, which gave my results an external grounding.
- Separating "what my model says" from "what my model says *after I've tried three ways to break it*."

Most people aren't using agents this way yet. They're using them to write code faster. I think the real unlock is using them to **disagree with you faster** — to compress weeks of peer review into hours, and to keep you honest when the numbers come out looking good.

That's the move. That's what I'd bring to YC.

---

*For questions or collaboration: [@shnazzers](https://x.com/shnazzers) / [abdulh.xyz](https://abdulh.xyz) / muhsinh on GitHub.*
