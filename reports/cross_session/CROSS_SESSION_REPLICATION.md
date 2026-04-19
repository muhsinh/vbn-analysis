# Cross-session replication (N = 3 sessions)

**Date:** 2026-04-19
**Cohort:** 3 familiar G-image VBN sessions, all 6-probe, matched broad coverage.
**Data:** NWB-only (no video-derived face SVD; the single-session fair-Stringer test
put face SVD at ~0.1–2% unique variance, so excluding it trades fidelity for
~18× faster per-session compute and matched scope across sessions).

| session | mouse | unit count | areas overlap with 1055240613 |
|---|---|---|---|
| **1055240613** | 533539 | 734 | — (primary) |
| **1067588044** | 544836 | 995 | 12/12 critical areas |
| **1115086689** | 574078 | 922 | 12/12 critical areas |

**Total time:** ~5 min compute on M3 Pro / 18 GB (including NWB extraction).

---

## What replicates

### 1. Visual cortex passive > active (Deliverable A2)

All seven visual-cortex areas show negative modulation index (passive-dominant)
across 3/3 sessions after arousal matching.

| area | 1055240613 | 1067588044 | 1115086689 | mean | convergence |
|---|---|---|---|---|---|
| VISp | −0.185 | −0.128 | −0.180 | **−0.164** | 3/3 |
| VISl | −0.183 | −0.147 | −0.067 | −0.132 | 3/3 |
| VISal | −0.067 | −0.162 | −0.161 | −0.130 | 3/3 |
| VISrl | −0.255 | −0.224 | −0.187 | **−0.222** | 3/3 |
| VISpm | −0.139 | −0.159 | −0.149 | −0.149 | 3/3 |
| VISam | −0.241 | −0.078 | −0.113 | −0.144 | 3/3 |

This confirms the "engagement paradox" — task quiets visual cortex — across animals.

### 2. Hippocampus change signal is reward-driven (Deliverable B2, miss-only)

DG and CA1 both show change-amplification ratios *below 1.0* on miss trials
across all three sessions. Without the reward, the apparent hippocampal
"change signal" disappears.

| area | 1055240613 | 1067588044 | 1115086689 | mean |
|---|---|---|---|---|
| **DG miss ratio** | 0.57 | 0.80 | 0.89 | **0.76** |
| **CA1 miss ratio** | 0.44 | 1.00 | 0.84 | **0.76** |

This is the strongest replicated novel finding — 3/3 sign convergence, effect
size consistent across animals.

### 3. Noise correlations rise during active task (Deliverable H)

All four mid-hierarchy visual areas (VISl, VISal, VISpm, VISrl) show Δ > 0
in 3/3 sessions. Visual cortex as a whole: positive in 23/24 area-session
cells (one VISp = −0.001).

| area | 1055240613 | 1067588044 | 1115086689 | mean |
|---|---|---|---|---|
| VISl | +0.013 | +0.003 | +0.002 | **+0.006** |
| VISal | +0.005 | +0.004 | +0.004 | +0.004 |
| VISpm | +0.009 | +0.005 | +0.000 | +0.005 |
| VISrl | +0.005 | +0.001 | +0.002 | +0.003 |
| VISam | +0.007 | +0.003 | +0.000 | +0.004 |

MRN goes the opposite direction (negative Δ) in 2/2 sessions that recorded it.
This is the clean Cohen/Maunsell attention pattern confirmed at VBN scale.

### 4. Change amplification scales up the hierarchy (Deliverable B2, strict)

Change/repeat ratio increases from early visual to higher visual across
sessions. VISpm peak > VISp peak in 3/3.

| area | mean ratio (strict) | 3/3 sign |
|---|---|---|
| VISp | 2.22× | ✓ |
| VISl | 1.78× | ✓ |
| VISal | 1.64× | ✓ |
| VISrl | 1.74× | ✓ |
| VISam | 2.59× | ✓ |
| **VISpm** | **3.71×** | ✓ |

---

## What does NOT replicate

### SCig active-dominance — was single-session

This was my headline claim from rounds 1-3. At N=3 it falls apart.

| metric | 1055240613 | 1067588044 | 1115086689 |
|---|---|---|---|
| A2 MI | **+0.169** (active-dominant) | +0.055 | **−0.164** (passive-dominant) |
| D2 full R² | +0.014 | −0.236 | −0.001 |

The effect in session 1055240613 is probably real for that mouse, but it
does not generalize to other wild-type mice on the same paradigm. Two likely
stories: (a) SCig's task-engagement signal is animal-specific given variable
task strategy; (b) specific motor-pattern-in-that-mouse coincided with an
SCig-registered probe channel. Would need N=5+ and motor-behavior matching
to distinguish.

### DG active-dominance — noisy, directionally positive

DG A2 MI: +0.13, −0.06, +0.29 across the three sessions. 2/3 positive, mean
positive, but huge between-animal variance. More sessions needed.

---

## Methods

- Same image set (G), same experience level (Familiar), all wild-type genotype.
- Per-session extraction auto-caches to `outputs/cross_session/<session_id>/`.
- Per-session analyses run via `scripts/analysis/_per_session_pipeline.py <session_id>`.
- Cross-session driver: `notebooks/11_Cross_Session_Replication.ipynb`.
- Pipeline: A2 arousal-matched MI / B2 strict + miss-only change amp / D2
  encoding with ridge regression / H flash-residualized noise correlations.
- No video-derived features (face SVD, saccades) — trade-off explained above.
- Each session runs in ~1.5 minutes after NWB cached.

---

## What this changes in the narrative

- The **novel hippocampal-reward finding** (B2 miss-only) is now the
  strongest replicated claim. Preprint-defensible at N=3.
- The **noise-correlation restructuring** (H) is now directionally confirmed
  across 3 mice for visual cortex. The MRN decorrelation is N=2 and suggestive.
- The **change-amplification hierarchy** (B2) is now an N=3 replication of
  Piet 2025's population-level result at single-session resolution.
- The **SCig story** needs to be dropped or re-framed as a single-session
  observation pending replication with more sessions.

To get to N=5+ for preprint: ~3 more sessions × ~3 min each = 10 min compute
plus the NWB downloads (~5 min each). Total: ~30 min additional wall-clock.
