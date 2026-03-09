# VBN Analysis Suite (Allen Institute Visual Behavior Neuropixels)

A reproducible, notebook-driven analysis suite for the Allen Institute Visual Behavior Neuropixels dataset. This project correlates **behavioral changes with neural activity changes** using correct timebase alignment across neural data, behavior, eye tracking, video, and pose estimation.

## What This Does

Given Neuropixels recordings of mice performing a visual change-detection task, this pipeline:

1. **Extracts** neural spike data, behavioral events, eye tracking, and video from NWB files
2. **Estimates pose** from video using SLEAP (automated inference from trained models)
3. **Engineers features** from pose (velocity, acceleration, body geometry, behavioral motifs)
4. **Correlates** behavior with neural activity through:
   - Peri-event time histograms (PETHs)
   - Time-lagged cross-correlation
   - Sliding-window correlation (when during a session is coupling strongest?)
   - Encoding models (behavior → neural)
   - Decoding models (neural → behavior)
   - Granger causality (which leads: brain or behavior?)
   - Unit selectivity screening (which neurons differentiate trial types?)

## Pipeline Phases

1. **Phase 1: Setup + discovery** (Notebooks 00–01)
2. **Phase 2: Signal extraction** (Notebooks 02–04)
3. **Phase 3: Video alignment + pose** (Notebooks 05–07)
4. **Phase 4: Neural-behavior correlation + QC** (Notebooks 08–09)

## Quickstart

### 1) Create an environment

**Recommended:** Use conda-forge to avoid source builds for scientific packages.

```bash
conda env create -f environment/environment.yml
conda activate vbn-analysis
```

If you use pip directly and it tries to compile NumPy/Pandas, prefer conda-forge or adjust pins to install wheels.

### 2) Configure sessions

- `sessions.csv` should contain columns: `session_id,nwb_path,video_dir,notes`.
- If `sessions.csv` is missing but `sessions.txt` exists, the pipeline will generate a template `sessions.csv`.

### 3) Run setup + end-to-end

- Start with `notebooks/00_Setup_and_Configuration.ipynb`.
- Then run the main entry-point: `notebooks/09_End_to_End_Run_and_QC_Checklist.ipynb`.

Configuration knobs live in Notebook 00 (or environment variables):
- `ACCESS_MODE` = `sdk` or `manual`
- `POSE_TOOL` = `sleap` or `dlc`
- `BIN_SIZE_S` = bin width for fusion (seconds)
- `MOCK_MODE` = `true` to force synthetic data
- `VIDEO_SOURCE` = `auto` (default), `local`, or `s3`
- `VIDEO_CACHE_DIR` = local cache for S3 downloads
- `VIDEO_BUCKET` = S3 bucket name (default `allen-brain-observatory`)
- `VIDEO_BASE_PATH` = S3 base prefix (default `visual-behavior-neuropixels/raw-data`)
- `VIDEO_CAMERAS` = comma-separated cameras (default `eye,face,side`)

## Access Modes

- **sdk** (default): Uses AllenSDK to download/cache NWB and discover video assets when available.
- **manual**: Uses `nwb_path`/`video_dir` from `sessions.csv` without downloading.

## Video Access (Download-First)

Video assets are fetched from the public S3 bucket by default. The pipeline writes two flat, canonical artifacts:
- `outputs/video/video_assets.parquet` (one row per `session_id` + `camera`)
- `outputs/video/frame_times.parquet` (frame index + timestamps per `session_id` + `camera`)

Set `VIDEO_SOURCE=local` to disable S3 downloads and rely solely on `sessions.csv` `video_dir` or existing cache.

## Automated Pose Estimation

The pipeline supports automated SLEAP inference so you don't need to manually label every frame:

### Workflow

1. **Label a small set of frames** (~100–200 is enough to start) using the SLEAP GUI
2. **Train a SLEAP model**: `sleap-train <config.json> <labels.slp>`
3. **Place the model** in `pose_projects/` or `data/sleap_models/`
4. **Run Notebook 07** — it auto-discovers the model and runs batch inference on all cached videos

### Auto-Discovery

Notebook 07 automatically finds pose data in priority order:
1. Existing `.parquet` predictions from prior runs
2. SLEAP CSV exports (from manual labeling + inference)
3. `.slp` prediction files (from automated inference)

No manual path pasting required.

### Active Learning

After running inference, the pipeline suggests which frames to label next based on prediction confidence. This maximizes model improvement per labeled frame — critical when you have limited labeling time.

## Pose Feature Extraction

From raw keypoint coordinates, the pipeline derives:
- **Per-keypoint velocity and acceleration**
- **Body length** (proxy for stretch/posture)
- **Head angle and angular velocity**
- **Inter-keypoint distances** (adjacent pairs)
- **Overall pose speed** (mean across keypoints)
- **Stillness detection** (binary, percentile-based threshold)
- **Behavioral motifs** (k-means or HMM clustering of features)

Low-confidence keypoint detections are filtered before feature extraction.

## Neural-Behavior Correlation (Notebook 08)

The core analysis answers: **do changes in behavior align with changes in neural activity?**

| Analysis | What it tells you |
|---|---|
| **PETHs** | How firing rates change around behavioral events (stimulus, lick, reward) |
| **Cross-correlation** | Time lag between neural and behavioral signals (who leads?) |
| **Sliding-window correlation** | When during the session is neural-behavior coupling strongest? |
| **Encoding model** | Can behavior predict neural firing? (behavior → neural, Ridge/Poisson) |
| **Decoding model** | Can neural activity predict behavior? (neural → behavior, Ridge) |
| **Granger causality** | Does neural activity *cause* behavior changes, or vice versa? |
| **Unit selectivity** | Which neurons differentiate trial types (d-prime + Mann-Whitney U) |

All analyses use time-blocked cross-validation and support lagged features.

## Timebase Guarantee

All exported artifacts use a single canonical timebase: **`nwb_seconds`**.
- When applicable, tables include a `t` column in NWB seconds.
- Each artifact includes metadata (or sidecar JSON) with:
  - `timebase="nwb_seconds"`
  - provenance: `session_id`, `code_version`, `created_at`, `alignment_method`

## Artifact Registry

Run any notebook and then check:
- `outputs/reports/artifact_registry.parquet`

The registry includes: `step`, `artifact_path`, `exists`, `last_modified`, `session_id`, `notes`.

## Notebook Map

| Goal | Notebook |
|---|---|
| Setup and environment checks | `00_Setup_and_Configuration.ipynb` |
| Session inventory & modality flags | `01_Session_Discovery_and_Metadata.ipynb` |
| Neural spikes/units | `02_Neural_Data_Spikes_and_Events.ipynb` |
| Behavior/task alignment | `03_Behavior_and_Task_Alignment.ipynb` |
| Eye tracking QC/features | `04_Eye_Tracking_QC_and_Features.ipynb` |
| Video discovery + frame timebase | `05_Video_IO_and_Frame_Timebase.ipynb` |
| Pose estimation setup (SLEAP/DLC) | `06_Pose_Estimation_Setup_SLEAP_or_DLC.ipynb` |
| Pose features + motifs + auto-inference | `07_Pose_to_Motifs_Feature_Engineering.ipynb` |
| Neural-behavior correlation + modeling | `08_Neural_Behavior_Fusion_and_Modeling.ipynb` |
| End-to-end pipeline + QC | `09_End_to_End_Run_and_QC_Checklist.ipynb` |

## Source Modules (`src/`)

| Module | Purpose |
|---|---|
| `config.py` | Configuration, paths, environment variables |
| `io_sessions.py` | Session discovery and bundle orchestration |
| `io_nwb.py` | NWB file I/O and data extraction |
| `io_s3.py` | Unsigned S3 downloads for video assets |
| `io_video.py` | Video asset discovery and frame time alignment |
| `features_task.py` | Task/behavior feature extraction |
| `features_eye.py` | Eye tracking feature extraction |
| `features_pose.py` | Pose feature extraction (velocity, angles, distances, confidence filtering) |
| `pose_inference.py` | Automated SLEAP inference, model discovery, active learning |
| `neural_events.py` | PETHs, population analysis, trial-averaged rates, selectivity screening |
| `cross_correlation.py` | Cross-correlation, encoding/decoding models, Granger causality |
| `motifs.py` | Behavioral motif discovery (k-means, HMM) |
| `modeling.py` | Fusion table building, XGBoost modeling |
| `timebase.py` | Timebase utilities and artifact writing |
| `qc.py` | Quality control (monotonicity, dropped frames, FPS) |
| `reports.py` | Artifact registry, logging, notebook header parsing |
| `viz.py` | Visualization (rasters, PETHs, cross-correlation, behavior summaries) |

## Notes on Dependencies

- AllenSDK is pinned to a known working version.
- SLEAP is required for automated pose inference (`conda install -c sleap sleap` or `pip install sleap`).
- scikit-learn and scipy are required for correlation analyses.
- NumPy/Pandas are resolved by the environment manager unless conflicts arise.
- Prefer conda-forge to avoid source builds.

## Outputs

Artifacts are written under `outputs/`:
- `neural/` — unit tables and spike times
- `behavior/` — trials, events, task features
- `eye/` — eye tracking features
- `video/` — video assets and frame times
- `pose/` — pose predictions, features, motifs
- `fusion/` — time-aligned multi-modal feature matrices
- `models/` — trained models, metrics, alignment reports, selectivity screens
- `reports/` — config snapshots, run summaries, artifact registry, QC checklists

SLEAP labeling exports are written to:
- `outputs/labeling/sleap/{session_id}/{camera}/labels.csv`
- `outputs/labeling/sleap/{session_id}/{camera}/frames/*.png`

`outputs/` is ignored by git.
