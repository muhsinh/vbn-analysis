# VBN Analysis Suite

**A reproducible, notebook-driven analysis pipeline for the Allen Institute Visual Behavior Neuropixels dataset.**

---

## The Core Question

> **Do changes in behavior align with changes in neural activity?**

The VBN Analysis Suite takes Neuropixels recordings of mice performing a visual change-detection task and builds a complete, time-aligned bridge between neural firing and behavioral signals --- from raw NWB files all the way to Granger causality tests that reveal which leads: brain or behavior.

Every artifact shares a single canonical timebase (`nwb_seconds`), so you never have to worry about clock drift, frame-rate mismatches, or misaligned modalities.

---

## Pipeline at a Glance

```
Phase 1                Phase 2                Phase 3                Phase 4
Setup & Discovery      Signal Extraction      Video & Pose           Neural-Behavior
                                                                     Correlation
+-----------------+    +-----------------+    +-----------------+    +-----------------+
| NB 00  Config   | -> | NB 02  Spikes   | -> | NB 05  Video IO | -> | NB 08  Fusion   |
| NB 01  Sessions |    | NB 03  Behavior |    | NB 06  Pose     |    |   & Modeling     |
|                 |    | NB 04  Eye QC   |    |   Setup (SLEAP) |    | NB 09  End-to-  |
|                 |    |                 |    | NB 07  Pose     |    |   End & QC       |
|                 |    |                 |    |   Features      |    |                 |
+-----------------+    +-----------------+    +-----------------+    +-----------------+
```

| Phase | Notebooks | What happens |
|-------|-----------|-------------|
| **1 -- Setup & Discovery** | `00`, `01` | Environment validation, session inventory, modality flags |
| **2 -- Signal Extraction** | `02`, `03`, `04` | Neural spikes/units, behavioral trials/events, eye tracking QC |
| **3 -- Video & Pose** | `05`, `06`, `07` | Video asset download, frame timebase, SLEAP pose estimation, feature engineering |
| **4 -- Correlation & QC** | `08`, `09` | Neural-behavior fusion, 6 correlation analyses, end-to-end QC checklist |

---

## Key Features

### Automated SLEAP Inference

Label a small set of frames (~100--200), train a model, and the pipeline runs batch inference on every cached video automatically. Active learning suggests which frames to label next for maximum model improvement.

### Rich Pose Feature Engineering

From raw keypoints the pipeline derives:

- Per-keypoint **velocity** and **acceleration**
- **Body length** (stretch/posture proxy)
- **Head angle** and angular velocity
- **Inter-keypoint distances** (adjacent pairs)
- **Pose speed** (mean across keypoints)
- **Stillness detection** (percentile-based threshold)
- **Behavioral motifs** (k-means or HMM clustering)

### Six Correlation Analyses

| Analysis | Question it answers |
|----------|-------------------|
| **Peri-Event Time Histograms (PETHs)** | How do firing rates change around behavioral events? |
| **Time-Lagged Cross-Correlation** | What is the temporal lag between neural and behavioral signals? |
| **Sliding-Window Correlation** | *When* during the session is neural-behavior coupling strongest? |
| **Encoding Model** (behavior --> neural) | Can behavior predict neural firing? |
| **Decoding Model** (neural --> behavior) | Can neural activity predict behavior? |
| **Granger Causality** | Does neural activity *cause* behavior changes, or vice versa? |

All analyses use **time-blocked cross-validation** and support **lagged features**.

### Timebase Guarantee

Every exported artifact carries:

- A `t` column in **NWB seconds**
- A sidecar `.meta.json` with `timebase`, `session_id`, `code_version`, `created_at`, and `alignment_method`

No clock drift. No frame-rate mismatches. One timebase to rule them all.

---

## Documentation Map

| Section | What you will find |
|---------|-------------------|
| [Getting Started](getting-started/index.md) | Installation, quickstart, configuration reference |
| [Installation](getting-started/installation.md) | Conda/pip setup, SLEAP, AllenSDK, system requirements |
| [Quickstart](getting-started/quickstart.md) | Run your first session end-to-end in 15 minutes |
| [Configuration](getting-started/configuration.md) | Every environment variable, Config fields, provenance tracking |

---

## What You Will Need

!!! info "Prerequisites"

    Before you begin, make sure you have the following:

    **Data**

    - Access to the [Allen Institute Visual Behavior Neuropixels dataset](https://portal.brain-map.org/) (SDK mode downloads automatically, or provide local NWB files in manual mode)
    - Approximately **50--150 GB** of free disk space per session (NWB + video + pose outputs)

    **Compute**

    - **CPU**: Any modern multi-core processor (4+ cores recommended)
    - **RAM**: 16 GB minimum, 32 GB recommended (NWB files are large)
    - **GPU**: Optional but recommended for SLEAP inference (CUDA-capable NVIDIA GPU)
    - **OS**: macOS, Linux, or Windows (WSL2 recommended on Windows)

    **Software**

    - Python 3.10
    - Conda (Miniconda or Mambaforge recommended)
    - Git
    - JupyterLab or Jupyter Notebook

!!! warning "Disk Space"

    A single VBN session with all three camera videos can consume 30--50 GB. If you plan to analyze multiple sessions, ensure you have several hundred GB available or configure `VIDEO_SOURCE=local` to skip S3 downloads.

---

## Source Modules

The `src/` directory contains the Python library that powers the notebooks:

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration singleton, paths, environment variable loading |
| `io_sessions.py` | Session discovery, `SessionBundle` orchestration, caching |
| `io_nwb.py` | NWB file I/O, spike/trial/eye extraction |
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
| `timebase.py` | Timebase utilities, artifact writing with provenance |
| `qc.py` | Quality control (monotonicity, dropped frames, FPS) |
| `reports.py` | Artifact registry, logging, notebook header parsing |
| `viz.py` | Visualization (rasters, PETHs, cross-correlation, behavior summaries) |

---

## Output Structure

All artifacts are written under the `outputs/` directory (git-ignored):

```
outputs/
  neural/          # Unit tables, spike times (.parquet, .npz)
  behavior/        # Trials, events, task features
  eye/             # Eye tracking features
  video/           # Video assets, frame times
  pose/            # Pose predictions, features, motifs
  fusion/          # Time-aligned multi-modal feature matrices
  models/          # Trained models, metrics, selectivity screens
  reports/         # Config snapshots, run summaries, artifact registry, QC
    logs/          # Per-session log files
  cache/           # Intermediate computation cache (joblib)
  labeling/        # SLEAP labeling exports (frames + labels.csv)
```

---

<div style="text-align: center; margin-top: 2em;">
<strong>Ready to get started?</strong><br>
Head to the <a href="getting-started/installation.md">Installation Guide</a> or jump straight to the <a href="getting-started/quickstart.md">Quickstart</a>.
</div>
