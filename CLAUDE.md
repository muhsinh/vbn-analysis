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

- **`io_nwb.py:62` broken in SDK mode**: `BehaviorEcephysSession` has no `.nwb_path` attribute — the `hasattr` guard silently returns `None`. Workaround: use `ACCESS_MODE=manual` and populate `sessions.csv`.
- **Notebook ROOT setup**: Fixed 2026-04-13 — notebooks 00–04, 07–09 now use conditional `if not (ROOT / "src").exists(): ROOT = ROOT.parent`. Works from repo root or inside `notebooks/`.

## Local Data (as of 2026-04-13)

Sessions with real NWB data downloaded locally:
- `1043752325` → `data/allensdk_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions/1043752325/ecephys_session_1043752325.nwb` + video metadata in `data/raw/visual-behavior-neuropixels/1043752325/behavior_videos/`
- `1055240613` → `data/allensdk_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions/1055240613/ecephys_session_1055240613.nwb`

`sessions.csv` has NWB paths populated for these two sessions. All others fall back to mock.

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
1. **No unit quality filtering** — noise units included. Fix: filter `quality == "good"` in `extract_units_and_spikes()`.
2. **No brain area grouping** — `ecephys_structure_acronym` never used. All area-specific effects average out.
3. **Running speed not extracted** — in `nwb.processing["running"]["running_speed"]`. Must be covariate in all neural analyses.
4. **Stimulus presentations not extracted** — `nwb.intervals["stimulus_presentations"]`. Needed for flash-locked PETHs, image decoding, omission responses.
5. **Trial features gutted** — `features_task.py` keeps 4 of 20+ columns. Missing: hit/miss/FA/CR, response_latency, image identity, change_time_no_display_delay, lick_times.
6. **Eye tracking incomplete** — no blink filtering (`likely_blink`), pupil position/corneal reflection unused.
7. **Licking/rewards never extracted** — `nwb.processing["licking"]`, `nwb.processing["rewards"]`.

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
