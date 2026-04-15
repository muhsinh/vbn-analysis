# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintenance Rule

**Update this file at the end of every response** where code, config, or architectural decisions changed. Add new findings to the relevant section or to `## Recent Changes`. Keep entries concise — this is a reference, not a journal. Skip updates for pure explanation responses with no changes.

## Project Overview

Notebook-driven neuroscience pipeline that correlates behavioral changes with neural activity in the Allen Institute Visual Behavior Neuropixels (VBN) dataset. Code lives in `src/` (17 modules), executed through 10 Jupyter notebooks (00–09) in four phases.

## Environment Setup

```bash
conda env create -f environment/environment.yml
conda activate vbn-analysis
```

Python 3.10 (pinned). Optional: `conda install -c sleap sleap` for pose estimation.

## Common Commands

```bash
# Build docs locally
mkdocs build --strict
mkdocs serve                    # preview at localhost:8000

# Run pipeline in mock mode (no data needed)
MOCK_MODE=true jupyter lab notebooks/09_End_to_End_Run_and_QC_Checklist.ipynb

# Run a single notebook from CLI
jupyter nbconvert --execute --to notebook notebooks/00_Setup_and_Configuration.ipynb
```

No test suite exists. No linter configured. QC is done via artifact inspection in notebooks.

## Architecture

**Execution flow:** Notebook 00 (config) → 01–08 sequentially, or Notebook 09 (orchestrator that runs all phases). Outputs cached in `outputs/` (git-ignored).

**Module dependency chain:**
- `config.py` → foundational, imported by everything
- `io_sessions.py` → orchestrates `io_nwb.py`, `io_video.py`, `io_s3.py` for per-session data
- `features_task.py`, `features_eye.py`, `features_pose.py` → consume data from `io_*` modules
- `neural_events.py`, `cross_correlation.py`, `modeling.py` → Phase 4 analyses
- `timebase.py` → enforces canonical `nwb_seconds` clock on all artifacts
- `viz.py` → plotting for all stages; `qc.py` → validation checks; `reports.py` → artifact registry

**Access modes:** `ACCESS_MODE=sdk` (auto-downloads via AllenSDK) or `ACCESS_MODE=manual` (local paths from `sessions.csv`). `MOCK_MODE=true` generates synthetic data for testing.

## Key Conventions

- All timestamps throughout the pipeline use **NWB seconds** (`timebase.CANONICAL_TIMEBASE = "nwb_seconds"`). Every parquet artifact embeds this in schema metadata plus a `.meta.json` sidecar.
- Modules are imported directly (not packaged): `from config import get_config`. Notebooks add `src/` to `sys.path` in their setup cell.
- `sessions.csv` is the session manifest (155 session IDs). Column schema: `session_id, nwb_path, video_dir, notes`.
- Output directory structure: `outputs/{neural,behavior,eye,video,pose,fusion,models,reports}/`.

## Documentation

MkDocs Material site. GitHub Actions deploys on push to `main` when `docs/` or `mkdocs.yml` change. Custom CSS in `docs/stylesheets/extra.css`, template override in `overrides/main.html`, JS in `docs/javascripts/extra.js`. Dark/light mode follows system preference.

## Known Issues / Workarounds

- **`io_nwb.py:62` FIXED (2026-04-14)**: `resolve_nwb_path` now returns `nwb_path or nwb_path_override` so the sessions.csv path is used when the SDK attribute is missing. NB00 also sets `os.environ.setdefault("ACCESS_MODE", "manual")` so notebooks default to manual mode.
- **Notebook ROOT setup**: Fixed 2026-04-13 — notebooks 00–04, 07–09 now use conditional `if not (ROOT / "src").exists(): ROOT = ROOT.parent`. Works from repo root or inside `notebooks/`.

## Local Data (as of 2026-04-14)

Sessions with real NWB data:
- `1043752325` → NWB path populated in sessions.csv. File was truncated (203 MB, expected ~3 GB) — re-download in progress via `scripts/redownload_nwb.py`.
- `1055240613` → NWB path populated in sessions.csv. File was truncated (130 MB, expected ~2.99 GB) — re-download in progress. Videos not yet downloaded.

`sessions.csv` has NWB paths populated for both sessions. All others fall back to mock.

**To re-download truncated NWB files:**
```bash
conda activate vbn-analysis
python scripts/redownload_nwb.py
```

## Recommended Run Command

```bash
conda activate vbn-analysis
export ACCESS_MODE=manual
cd /Users/muh/projects/vbn-analysis
jupyter lab
```

## Pose Estimation Tools Under Consideration

Beyond SLEAP (currently installed, ~150 frames labeled):
- **Facemap** (`pip install facemap`) — eye.mp4: use pupil tracker (zero labels, outputs area/position/blink). face.mp4: use SVD motion energy (zero labels, angle-agnostic); keypoint model needs ~20 frames fine-tune (bottom-up angle, pretrained model was side-face).
- **SuperAnimal (DLC)** (`pip install deeplabcut`) — side.mp4 only. Use `superanimal_quadruped` (NOT topviewmouse — that's overhead only). Head-fix hardware may confuse head keypoints; body/limb/tail reliable.
- **Lightning Pose** — semi-supervised training using existing 150 SLEAP labels + unlabeled video. 2–4x fewer labels needed vs supervised-only.
- **Keypoint-MoSeq** (`pip install keypoint-moseq`) — behavioral syllable segmentation from pose output. State-of-the-art (Nature Methods 2024). Auto-discovers syllable count, handles tracking noise.
- **DO NOT USE**: DANNCE (requires overlapping multi-camera geometry), MotionMapper (wrong for mice).

## Neural-Behavior Analysis Tools Under Consideration

- **CEBRA** (`pip install cebra`) — contrastive neural-behavioral embedding. Handles multi-session alignment across 155 sessions. Demonstrated on Allen Institute Neuropixels data. Add to `src/cross_correlation.py` or new `src/embeddings.py`.
- **A-SOiD** — active learning behavioral classifier. 85% less training data than supervised methods.
- **Ridge regression** — add `RidgeCV` to `src/modeling.py` as baseline decoder alongside XGBoost.

## Known Pipeline Gaps (Critical)

Discovered 2026-04-13 via NWB deep dive:
1. ~~**No unit quality filtering**~~ — **FIXED 2026-04-14** in `io_nwb.extract_units_and_spikes()`.
2. **No brain area grouping** — `ecephys_structure_acronym` never used. All area-specific effects average out. HIGH PRIORITY.
3. ~~**Running speed not extracted**~~ — **FIXED 2026-04-14** via `io_sessions.SessionBundle.load_running_speed()` + NB03.
4. **Stimulus presentations not extracted** — `nwb.intervals["stimulus_presentations"]`. Needed for flash-locked PETHs, active vs. passive epoch split, image decoding. HIGH PRIORITY (see Literature section).
5. ~~**Trial features gutted**~~ — **FIXED 2026-04-14**: `extract_trials()` now keeps hit/miss/FA/CR, response_latency, image identity, change_time_no_display_delay, go/catch.
6. **Eye tracking incomplete** — no blink filtering (`likely_blink`), pupil position/corneal reflection unused.
7. **Licking/rewards never extracted** — `nwb.processing["licking"]`, `nwb.processing["rewards"]`.
8. **Active vs. passive replay epoch never split** — the VBN dataset has both in every session; the passive epoch is the highest-value unused signal in the entire dataset (see Literature).
9. **No session-level engagement filter** — many sessions have d-prime < 0.5 (mice at chance). Including them contaminates all analyses. Add d-prime filter per session.
10. **Pre-stimulus behavioral state not used as trial-level covariate** — hit/miss PETHs confounded by arousal at time of trial. Need mean running + pupil in [-1, 0]s window per trial.

## Literature Context and Research Gaps

### Landmark findings
- **Niell & Stryker 2010** — locomotion doubles V1 firing rates; effect is cortical, not thalamic.
- **Reimer et al. 2014/2016** — running and pupil are *separate* signals; model as independent covariates.
- **Stringer et al. 2019 (Science)** — facial movements predict ~⅓ of V1 variance, far more than running+pupil alone. Motivates face.mp4 SVD.
- **Musall et al. 2019 (Nature Neuroscience)** — uninstructed movements dominate single-trial variance. Hit/miss PETHs are confounded by movement unless regressed out.
- **Steinmetz et al. 2019 (Nature)** — pre-stimulus population activity predicts trial-by-trial engagement.
- **Siegle et al. 2021 (Nature)** — passive-viewing baseline for visual hierarchy (V1→LM→AL→PM→AM). This is the comparison target.
- **Syeda et al. 2024 (Nature Neuroscience / Facemap paper)** — face keypoints + SVD doubles explained variance vs SVD alone.

### What the VBN dataset uniquely enables (unpublished territory)
1. **Active vs. passive replay in same session** — every session has both; nobody has analyzed behavioral state modulation across this contrast at scale.
2. **Image familiarity × behavioral state** — familiar/novel counterbalanced within-session; interaction with arousal unaddressed.
3. **6-area noise correlation matrix by state** — inter-area noise correlations at scale are unpublished.
4. **Pre-stimulus state × hit/miss/FA/CR** — arousal tertile split within outcome cleanly separates perception from arousal.

### Recommended analytical additions (priority order)
1. ~~Extract `stimulus_presentations` epoch labels~~ — **DONE 2026-04-14**
2. ~~Area stratification by `ecephys_structure_acronym`~~ — **DONE 2026-04-14**
3. Pre-stimulus running + pupil as trial-level covariates in PETH analysis
4. Session d-prime filter (drop d-prime < 0.5)
5. Face SVD via Facemap (zero labels, top 10-20 PCs as encoding covariates)
6. Replace RidgeCV with banded ridge (`pip install himalaya`) for unbiased variance partitioning
7. Noise correlation matrix by area and behavioral state
8. CEBRA for multi-session latent embedding

## Correct AllenSDK 2.16.0 Session API

```python
# session = cache.get_ecephys_session(ecephys_session_id=...)
session.get_units()              # DataFrame with all quality metrics + brain area
session.spike_times              # dict: {unit_id: np.array}
session.trials                   # all trial columns including hit/miss/FA/CR
session.eye_tracking             # pupil + corneal + blink mask
session.stimulus_presentations   # per-flash table
session.running_speed            # velocity (cm/s) timeseries
session.licks                    # lick timestamps
session.rewards                  # reward times + volume
session.channels                 # electrode metadata + CCF coordinates
session.get_lfp(probe_id)        # xarray, 1250Hz (separate probe NWB file)
```

Note: `session.nwb_path` does NOT exist — this is the bug in `io_nwb.py:62`.

## Recent Changes

| Date | Change |
|------|--------|
| 2026-04-13 | Fixed notebook ROOT setup cells (00–04, 07–09) |
| 2026-04-13 | Populated sessions.csv NWB paths for sessions 1043752325, 1055240613 |
| 2026-04-13 | Added this maintenance convention |
| 2026-04-13 | Deep research: pose tools (Facemap, SuperAnimal, LP), NN architectures (CEBRA, Keypoint-MoSeq, A-SOiD), full NWB schema + pipeline gaps documented |
| 2026-04-13 | Confirmed session 1055240613 as primary target (recommended by Corbett Bennett, Allen Institute). Videos not yet downloaded locally. facemap + deeplabcut not yet installed. |
| 2026-04-13 | Camera identification confirmed from screenshots: eye.mp4=macro IR eye closeup, face.mp4=bottom-up face view, side.mp4=full body side view head-fixed on wheel |
| 2026-04-13 | Multi-expert analysis: running speed already in NWB (encoder, cm/s) — do NOT use side camera for this. Face.mp4 > side.mp4 for neural prediction (Stringer: face explains 10-25% V1 variance orthogonal to pupil+running). Priority: fix encoding GLM + extract running/licks/rewards from NWB first, then eye.mp4 Facemap, then face.mp4. Side camera / SuperAnimal is lowest priority. |
| 2026-04-13 | Notebook 08 updated: cell 4 adds BEHAVIOR_COLS/GAP_BINS/N_PERMUTATIONS config; cell 6 merges running+pupil+pose into combined behavior_df, uses new behavior_cols API, handles both single and multi-covariate dict structures in prints. |
| 2026-04-13 | Fixed cross_correlation.py + modeling.py per Pillow critique: (1) fit_encoding_model now forward-chain CV only (no future leakage), gap_bins=20 default; (2) raised_cosine_basis + _add_raised_cosine_lags added (±1s, log-spaced, 8 basis fns); (3) permutation_test() added — circular shifts null distribution, wired into compute_neural_behavior_alignment; (4) fit_multi_covariate_encoding_model() added — RidgeCV full model + variance partitioning per covariate; (5) compute_neural_behavior_alignment now accepts behavior_cols list for multi-covariate analysis; (6) time_blocked_splits gets gap_bins parameter. |
| 2026-04-14 | Both NWB files confirmed truncated (130MB/203MB instead of ~3GB). Added scripts/redownload_nwb.py to force-delete and re-download via AllenSDK. |
| 2026-04-14 | Fixed NB03 cell 6: SESSION_IDS pinned to [1055240613]; added running speed extraction + print after trials. |
| 2026-04-14 | Fixed NB08 cell 6: running_df was referenced but never defined (would crash immediately) — added load inside loop from cache or bundle.load_running_speed(). Fixed PETH grouping to use go/catch > hit/miss > trial_type. Fixed selectivity (cell 10) to same column priority. Fixed summary (cell 11) to handle full_r2 (multi-covariate) and mean_r2 (single). |
| 2026-04-14 | Fixed NB09: SESSION_IDS pinned to [1055240613] (was [:1] = 1043752325); added running/n_units/n_trials to QC loop. |
| 2026-04-14 | Added NWB + video download cells to NB01 (cells 10-11): AllenSDK NWB download with truncation detection; boto3 unsigned S3 video download for eye+side cameras. No credentials needed. |
| 2026-04-14 | Literature review complete. Key gaps identified: active vs. passive epoch split (highest priority), area stratification, pre-stimulus behavioral state as trial covariate, banded ridge regression (himalaya), face SVD via Facemap, session d-prime filter. See Literature section below. |
| 2026-04-14 | Session 1055240613 NWB re-downloaded successfully: 2.78 GB. Session 1043752325 timed out — use aws s3 cp fallback. |
| 2026-04-14 | Fixed resolve_nwb_path: `return nwb_path or nwb_path_override` — falls back to sessions.csv path when SDK attribute is None (root cause of NB04 "No eye data" bug). NB00 now sets ACCESS_MODE=manual via os.environ.setdefault. |
| 2026-04-14 | Area stratification implemented: extract_units_and_spikes() now preserves ecephys_structure_acronym; compute_alignment_by_area() in cross_correlation.py runs encoding model per area. |
| 2026-04-14 | extract_stimulus_presentations() added to io_nwb.py — per-flash table with image_name, is_change, is_omission, active (epoch flag). SessionBundle.load_stimulus_presentations() added. |
| 2026-04-14 | NB08 new cell 10: per-area encoding R2 + active/passive epoch split. Prints Delta R2 (active - passive) — the key VBN comparison. NB03 updated to extract + cache stimulus presentations. |
| 2026-04-14 | NB06 rewritten: Facemap pupil tracking (eye.mp4, zero labels), Facemap SVD motion energy (face.mp4, zero labels), SuperAnimal-quadruped (side.mp4, zero labels). Graceful skip if video missing. |
| 2026-04-14 | NB07 rewritten: aligns Facemap + SuperAnimal frame indices to NWB seconds via frame_times (linear fallback), blink filtering + interpolation, body_speed from keypoint velocity, merges all pose signals to single pose_features.parquet. |
