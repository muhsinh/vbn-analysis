# Notebook Map

Detailed reference for all 10 notebooks in the VBN Analysis Suite pipeline. For each notebook: purpose, inputs, outputs, configuration options, and key operations.

---

## Dependency Graph

```
00_Setup_and_Configuration
    |
01_Session_Discovery_and_Metadata
    |
    +---> 02_Neural_Data_Spikes_and_Events
    |         |
    +---> 03_Behavior_and_Task_Alignment
    |         |
    +---> 04_Eye_Tracking_QC_and_Features
    |         |
    +---> 05_Video_IO_and_Frame_Timebase
              |
          06_Pose_Estimation_Setup_SLEAP_or_DLC
              |
          07_Pose_to_Motifs_Feature_Engineering
              |
          08_Neural_Behavior_Fusion_and_Modeling  <-- requires 02, 03, 04, 07
              |
          09_End_to_End_Run_and_QC_Checklist      <-- runs all of the above
```

!!! info "Parallel Branches"
    Notebooks 02, 03, 04, and 05 are **independent of each other** and can be run in any order (or in parallel). They all depend on Notebook 01.

    Notebooks 06 and 07 depend on 05. Notebook 08 depends on 02, 03 (optionally 04), and 07.

---

## Notebook 00: Setup and Configuration

**Purpose**: Set configuration parameters, validate the environment, and write a config snapshot.

### Inputs

| Input | Source | Required |
|---|---|---|
| Environment variables | Shell / `.env` | No (defaults exist) |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Config snapshot | JSON | `outputs/reports/config_snapshot.json` |

### Configuration Options

| Variable | Default | Description |
|---|---|---|
| `ACCESS_MODE` | `"sdk"` | `"sdk"` or `"manual"` |
| `POSE_TOOL` | `"sleap"` | `"sleap"` or `"dlc"` |
| `BIN_SIZE_S` | `0.025` | Bin width for fusion (seconds) |
| `MOCK_MODE` | `false` | Use synthetic data |
| `VIDEO_SOURCE` | `"auto"` | `"auto"`, `"local"`, or `"s3"` |
| `VIDEO_CACHE_DIR` | `data/raw/visual-behavior-neuropixels` | Where to cache S3 downloads |
| `VIDEO_BUCKET` | `"allen-brain-observatory"` | S3 bucket name |
| `VIDEO_BASE_PATH` | `"visual-behavior-neuropixels/raw-data"` | S3 prefix |
| `VIDEO_CAMERAS` | `"eye,face,side"` | Which cameras to process |

### Key Operations

1. Imports all `src/` modules and validates they load correctly
2. Reads environment variables and creates a `Config` object
3. Creates output directory structure (`outputs/neural/`, `outputs/behavior/`, etc.)
4. Writes `config_snapshot.json` for reproducibility
5. Prints a summary of the active configuration

---

## Notebook 01: Session Discovery and Metadata

**Purpose**: Load the session inventory, resolve NWB file paths, and inspect which data modalities are available for each session.

### Inputs

| Input | Source | Required |
|---|---|---|
| `sessions.csv` or `sessions.txt` | Project root | Yes |
| NWB files (for modality inspection) | Local disk or AllenSDK cache | Only if `inspect_modalities=True` |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Normalized `sessions.csv` | CSV | `sessions.csv` |
| Session bundles (in memory) | Python objects | Not persisted |

### Key Operations

1. Loads `sessions.csv` (or generates from `sessions.txt`)
2. Normalizes columns (adds missing columns if needed)
3. For each session:
    - Resolves NWB path (SDK download or manual path)
    - Inspects available modalities (spikes, trials, eye, behavior, stimulus)
4. Creates `SessionBundle` objects for downstream notebooks
5. Prints a session summary table

### Configuration

| Option | Effect |
|---|---|
| `ACCESS_MODE="sdk"` | Downloads NWB files via AllenSDK |
| `ACCESS_MODE="manual"` | Uses paths from `sessions.csv` |
| `MOCK_MODE=true` | Creates mock bundles without real data |

---

## Notebook 02: Neural Data -- Spikes and Events

**Purpose**: Extract spike-sorted units and spike times from NWB files. Save as parquet + NPZ.

### Inputs

| Input | Source | Required |
|---|---|---|
| NWB file | Resolved in Notebook 01 | Yes |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Unit table | Parquet | `outputs/neural/session_{id}_units.parquet` |
| Spike times | NPZ | `outputs/neural/session_{id}_spike_times.npz` |
| Unit table sidecar | JSON | `outputs/neural/session_{id}_units.parquet.meta.json` |
| Spike times sidecar | JSON | `outputs/neural/session_{id}_spike_times.npz.meta.json` |

### Key Operations

1. Opens NWB file via `open_nwb_handle()`
2. Extracts units table and spike times via `extract_units_and_spikes()`
3. Saves to disk via `save_units_and_spikes()`
4. Visualizes spike raster and firing rate distribution

### Key Cells

- **Unit quality filtering**: Applies quality metrics thresholds (SNR, ISI violations)
- **Raster plot**: `viz.plot_raster(spike_times)`
- **Firing rate histogram**: `viz.plot_firing_rate_summary(spike_times)`

---

## Notebook 03: Behavior and Task Alignment

**Purpose**: Extract trial information and behavioral events from NWB. Derive task features.

### Inputs

| Input | Source | Required |
|---|---|---|
| NWB file | Resolved in Notebook 01 | Yes |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Trials table | Parquet | `outputs/behavior/session_{id}_trials.parquet` |
| Behavior events | Parquet | `outputs/behavior/session_{id}_events.parquet` |

### Key Operations

1. Extracts trials and behavioral events from NWB
2. Renames columns (e.g., `start_time` -> `t_start`)
3. Derives task features (trial type distribution, response patterns)
4. Saves to disk with timebase metadata
5. Visualizes trial distribution and timeline

### Key Cells

- **Trial summary**: `viz.plot_behavior_summary(trials)`
- **Task features**: `features_task.derive_task_features(trials, events)`

---

## Notebook 04: Eye Tracking QC and Features

**Purpose**: Extract eye tracking data, compute derived features (pupil z-score, velocity), and run QC checks.

### Inputs

| Input | Source | Required |
|---|---|---|
| NWB file | Resolved in Notebook 01 | Yes |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Eye features | Parquet | `outputs/eye/session_{id}_eye_features.parquet` |

### Key Operations

1. Extracts eye tracking from NWB `processing/eye_tracking`
2. Derives features: `pupil` (raw), `pupil_z` (z-scored), `pupil_vel` (velocity)
3. Runs QC: missing data fraction, signal range
4. Saves to disk with timebase metadata
5. Visualizes eye signal over time

### Key Cells

- **Eye QC**: `viz.plot_eye_qc(eye_df)`
- **QC summary**: `qc.eye_qc_summary(eye_df)`

!!! note "Eye Tracking Availability"
    Not all sessions have eye tracking data. If absent, this notebook logs a warning and the `SessionBundle` records `eye_unavailable` in its QC flags.

---

## Notebook 05: Video IO and Frame Timebase

**Purpose**: Discover video assets (local or S3), download if needed, extract frame timestamps, and build the video asset registry.

### Inputs

| Input | Source | Required |
|---|---|---|
| Session bundles | From Notebook 01 | Yes |
| Video files | Local disk or S3 | Depends on VIDEO_SOURCE |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Video asset registry | Parquet | `outputs/video/video_assets.parquet` |
| Frame times | Parquet | `outputs/video/frame_times.parquet` |

### Key Operations

1. For each session and camera:
    - Checks for local video files
    - Downloads from S3 if `VIDEO_SOURCE != "local"` and local files are missing
    - Loads timestamp `.npy` files
    - Computes frame metrics (n_frames, fps, time range)
    - Runs QC (monotonicity, dropped frames)
2. Builds/updates the flat video asset registry
3. Builds/updates the frame times table
4. Visualizes frame-to-frame timing and gap analysis

### Key Cells

- **Video alignment plot**: `viz.plot_video_alignment(frame_times)`
- **Preview clip**: `io_video.create_preview_clip(video_path, output_path)`

### Configuration

| Option | Effect |
|---|---|
| `VIDEO_SOURCE="auto"` | Downloads missing videos from S3 |
| `VIDEO_SOURCE="local"` | Uses only local files, no downloads |
| `VIDEO_CAMERAS="side"` | Processes only the side camera |

---

## Notebook 06: Pose Estimation Setup (SLEAP or DLC)

**Purpose**: Scaffold pose estimation projects, export frames for labeling, and prepare for model training.

### Inputs

| Input | Source | Required |
|---|---|---|
| Video files | From Notebook 05 | Yes |
| Frame times | From Notebook 05 | Yes |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Labeling frames | PNG | `outputs/labeling/sleap/{id}/{camera}/frames/*.png` |
| Labels manifest | CSV | `outputs/labeling/sleap/{id}/{camera}/labels.csv` |
| Labeling video | MP4 | `outputs/labeling/sleap/{id}/{camera}/labeling.mp4` |
| Project scaffold | Directory | `pose_projects/session_{id}/sleap/` |

### Key Operations

1. Scaffolds a pose project directory
2. Samples frame indices across the session (default: 50 frames)
3. Exports sampled frames as PNG images for labeling
4. Optionally exports a labeling video clip
5. Writes a `labels.csv` mapping frame indices to timestamps

### Key Cells

- **Frame sampling**: `features_pose.sample_frame_indices(frame_times, n_samples=150)`
- **Frame export**: `features_pose.export_labeling_frames(...)`
- **Video export**: `features_pose.export_labeling_video(...)`

!!! tip "After Running Notebook 06"
    Open the exported images or video in the SLEAP GUI to label keypoints. Save your labels as a `.slp` file in `pose_projects/`. Then proceed to training and Notebook 07.

---

## Notebook 07: Pose to Motifs -- Feature Engineering

**Purpose**: Load pose predictions (from SLEAP inference, CSV exports, or existing parquet), extract behavioral features, and discover behavioral motifs.

### Inputs

| Input | Source | Required |
|---|---|---|
| Pose predictions | Parquet, CSV, or SLP files | Yes |
| Frame times | From Notebook 05 | Yes (for timestamp alignment) |
| Trained SLEAP model | `pose_projects/` or `data/sleap_models/` | For auto-inference |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Pose predictions | Parquet | `outputs/pose/session_{id}_pose_predictions.parquet` |
| SLEAP predictions | SLP | `outputs/pose/predictions/session_{id}_{camera}.predictions.slp` |
| Pose features | Parquet | `outputs/pose/session_{id}_pose_features.parquet` |
| Motifs | Parquet | `outputs/pose/session_{id}_motifs.parquet` |

### Key Operations

1. **Auto-discover pose data** (priority: parquet > CSV > SLP)
2. **If no predictions exist**: auto-discover SLEAP model and run batch inference
3. **Convert** SLP/CSV to standardized parquet format
4. **Filter** low-confidence keypoints
5. **Extract features**: velocity, acceleration, body length, head angle, stillness
6. **Discover motifs**: k-means or HMM clustering on feature space
7. **Active learning**: suggest frames for labeling based on confidence

### Key Cells

- **Batch inference**: `pose_inference.run_batch_inference()`
- **Feature extraction**: `features_pose.derive_pose_features(pose_df)`
- **Motif discovery**: `motifs.motifs_kmeans(features, n_clusters=8)`
- **Active learning**: `pose_inference.suggest_frames_to_label(slp_path)`
- **Motif visualization**: `viz.plot_motif_transition(motifs)`

### Configuration

| Option | Effect |
|---|---|
| `POSE_TOOL="sleap"` | Uses SLEAP for inference |
| `POSE_TOOL="dlc"` | Uses DeepLabCut (DLC) |

---

## Notebook 08: Neural-Behavior Fusion and Modeling

**Purpose**: Align neural and behavioral data on a common time grid, run correlation analyses, fit encoding/decoding models, and screen units for task selectivity.

### Inputs

| Input | Source | Required |
|---|---|---|
| Spike times | From Notebook 02 | Yes |
| Trials | From Notebook 03 | Yes |
| Eye features | From Notebook 04 | Optional |
| Pose features + motifs | From Notebook 07 | Yes (for pose-neural correlation) |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Fusion table | Parquet | `outputs/fusion/session_{id}_fusion.parquet` |
| Model metrics | JSON | `outputs/models/session_{id}_metrics.json` |
| Alignment report | JSON | `outputs/models/session_{id}_alignment.json` |
| Selectivity screen | Parquet | `outputs/models/session_{id}_selectivity.parquet` |

### Key Operations

1. **Build fusion table**: Bin spikes and behavior onto a common time grid
2. **Cross-correlation**: Population rate vs behavior signal
3. **Per-unit cross-correlation**: Distribution of lags across units
4. **Sliding-window correlation**: Time-varying coupling strength
5. **Encoding model**: Behavior features -> neural firing rate (Ridge/Poisson)
6. **Decoding model**: Neural population -> behavior (Ridge)
7. **Granger causality**: Both directions (neural->behavior and behavior->neural)
8. **Trial-averaged PETHs**: Grouped by trial type and reward
9. **Unit selectivity screening**: d-prime and Mann-Whitney U for each unit
10. **XGBoost model**: Full fusion table modeling (if configured)

### Key Cells

- **Full analysis**: `cross_correlation.compute_neural_behavior_alignment(...)`
- **Visualization**: `viz.plot_crosscorrelation()`, `viz.plot_sliding_correlation()`, `viz.plot_encoding_decoding()`, `viz.plot_granger_summary()`
- **Selectivity**: `neural_events.screen_selective_units(...)`
- **XGBoost**: `modeling.fit_and_evaluate(X, y, "xgboost", "count", categorical_cols)`

### Configuration

| Option | Effect |
|---|---|
| `BIN_SIZE_S` | Controls time resolution of all analyses |
| `MODEL_NAME` | `"xgboost"` for the fusion model |

---

## Notebook 09: End-to-End Run and QC Checklist

**Purpose**: Run the entire pipeline for all sessions and generate a QC summary.

### Inputs

| Input | Source | Required |
|---|---|---|
| `sessions.csv` | Project root | Yes |
| All upstream artifacts | From Notebooks 00--08 | Generated during run |

### Outputs

| Artifact | Format | Path |
|---|---|---|
| Artifact registry | Parquet | `outputs/reports/artifact_registry.parquet` |
| Run summary | Parquet | `outputs/reports/run_summary.parquet` |
| QC checklist | JSON | `outputs/reports/qc_checklist.json` |
| All upstream artifacts | Various | All `outputs/` subdirectories |

### Key Operations

1. Loads session list
2. For each session, runs the full pipeline (Notebooks 01--08 logic)
3. Catches and logs errors per session (does not stop on failure)
4. Builds the artifact registry
5. Writes the run summary
6. Prints a QC checklist

### Key Cells

- **Pipeline loop**: Iterates over sessions, calling `SessionBundle` methods
- **Registry**: `reports.write_artifact_registry()`
- **Summary**: `reports.write_run_summary(summary_df)`
- **Validation**: `reports.validate_artifact_schema()` for each critical artifact

---

## How to Run a Subset of the Pipeline

### Run Only Neural Analysis (No Video/Pose)

Run notebooks 00, 01, 02, 03, 04 only. Skip 05--07. Notebook 08 can run without pose data but will lack pose-neural correlations.

### Run Only Pose Estimation (No Neural)

Run notebooks 00, 01, 05, 06, 07. Skip 02--04 and 08.

### Run a Single Notebook After Config Change

If you change a configuration parameter (e.g., `BIN_SIZE_S`):

1. Re-run Notebook 00 to update the config snapshot
2. Re-run only the affected notebooks:

| Changed Parameter | Re-run |
|---|---|
| `ACCESS_MODE` | 01 + all downstream |
| `BIN_SIZE_S` | 08 only |
| `POSE_TOOL` | 06, 07, 08 |
| `VIDEO_SOURCE` | 05 + downstream |
| `VIDEO_CAMERAS` | 05, 06, 07, 08 |
| `MOCK_MODE` | All notebooks |
| `MODEL_NAME` | 08 only |

---

## How to Add a New Session

1. Add the session ID to `sessions.csv`:

    ```csv
    session_id,nwb_path,video_dir,notes
    1098595957,,,existing session
    9999999999,/path/to/new.nwb,/path/to/videos,new session
    ```

2. Re-run Notebook 01 to discover the new session
3. Run Notebooks 02--08 for the new session (Notebook 09 will process it automatically)

!!! tip "SDK Mode Auto-Download"
    In SDK mode, you only need to add the session ID. The AllenSDK will download the NWB file automatically:

    ```csv
    session_id,nwb_path,video_dir,notes
    9999999999,,,auto-download via SDK
    ```

---

## How to Re-Run a Single Notebook

1. Open the notebook in Jupyter
2. **Restart the kernel** (important to clear stale imports)
3. Run all cells from top to bottom
4. Check the outputs for the expected artifacts

!!! warning "Always Restart the Kernel"
    If you have made changes to `src/` modules, stale cached imports will cause inconsistencies. Always restart the kernel before a full re-run.

Alternatively, use the autoreload extension for iterative development:

```python
%load_ext autoreload
%autoreload 2
```

---

## Pipeline Execution Times (Estimates)

| Notebook | Typical Time | Bottleneck |
|---|---|---|
| 00 | < 5 seconds | Config setup |
| 01 | 10--60 seconds | NWB resolution (SDK may download) |
| 02 | 30--120 seconds | NWB reading (file I/O) |
| 03 | 10--30 seconds | NWB reading |
| 04 | 10--30 seconds | NWB reading |
| 05 | 1--30 minutes | S3 downloads (if needed) |
| 06 | 1--5 minutes | Frame export (video I/O) |
| 07 | 5--60 minutes | SLEAP inference (GPU-dependent) |
| 08 | 1--10 minutes | Model fitting (CPU) |
| 09 | Sum of above | Processes all sessions |
