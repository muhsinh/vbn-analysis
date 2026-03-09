---
hero: true
hide:
  - navigation
  - toc
---

# VBN Analysis Suite

<div class="vbn-pipeline vbn-animate" markdown>
<div class="vbn-pipeline-node phase-1" data-href="pipeline/phase1-setup/" markdown>
<div class="vbn-pipeline-icon">1</div>
<div class="vbn-pipeline-label">Setup & Discovery</div>
<div class="vbn-pipeline-notebooks">NB 00, 01</div>
<div class="vbn-pipeline-summary">Environment config, session inventory, modality flags</div>
</div>
<div class="vbn-pipeline-node phase-2" data-href="pipeline/phase2-signals/" markdown>
<div class="vbn-pipeline-icon">2</div>
<div class="vbn-pipeline-label">Signal Extraction</div>
<div class="vbn-pipeline-notebooks">NB 02, 03, 04</div>
<div class="vbn-pipeline-summary">Neural spikes, behavioral trials, eye tracking QC</div>
</div>
<div class="vbn-pipeline-node phase-3" data-href="pipeline/phase3-video-pose/" markdown>
<div class="vbn-pipeline-icon">3</div>
<div class="vbn-pipeline-label">Video & Pose</div>
<div class="vbn-pipeline-notebooks">NB 05, 06, 07</div>
<div class="vbn-pipeline-summary">S3 video download, SLEAP pose estimation, feature engineering</div>
</div>
<div class="vbn-pipeline-node phase-4" data-href="pipeline/phase4-correlation/" markdown>
<div class="vbn-pipeline-icon">4</div>
<div class="vbn-pipeline-label">Correlation & QC</div>
<div class="vbn-pipeline-notebooks">NB 08, 09</div>
<div class="vbn-pipeline-summary">Neural-behavior fusion, 6 analyses, end-to-end QC</div>
</div>
</div>

---

## What It Does

<div class="vbn-features vbn-animate" markdown>
<div class="vbn-feature-card" markdown>
<span class="vbn-feature-icon">&#x1f9ec;</span>

### Automated SLEAP Inference

Label ~100-200 frames, train a model, and the pipeline runs batch inference on every cached video. Active learning suggests which frames to label next.

</div>
<div class="vbn-feature-card" markdown>
<span class="vbn-feature-icon">&#x1f4ca;</span>

### Rich Pose Features

From raw keypoints: velocity, acceleration, body length, head angle, inter-keypoint distances, pose speed, stillness detection, and behavioral motifs.

</div>
<div class="vbn-feature-card" markdown>
<span class="vbn-feature-icon">&#x1f50d;</span>

### Six Correlation Analyses

PETHs, time-lagged cross-correlation, sliding-window correlation, encoding models, decoding models, and Granger causality. All time-blocked cross-validated.

</div>
<div class="vbn-feature-card" markdown>
<span class="vbn-feature-icon">&#x23f1;&#xfe0f;</span>

### One Canonical Timebase

Every artifact uses NWB seconds. No clock drift, no frame-rate mismatches. Spike times, trials, eye tracking, video frames, and pose all share one clock.

</div>
</div>

---

## Six Analyses, One Question

> **Do changes in behavior align with changes in neural activity?**

<div class="vbn-analysis-grid vbn-animate" markdown>
<div class="vbn-analysis-card" markdown>

#### Peri-Event Time Histograms

How do firing rates change around behavioral events?

</div>
<div class="vbn-analysis-card" markdown>

#### Time-Lagged Cross-Correlation

What is the temporal lag between neural and behavioral signals?

</div>
<div class="vbn-analysis-card" markdown>

#### Sliding-Window Correlation

When during the session is neural-behavior coupling strongest?

</div>
<div class="vbn-analysis-card" markdown>

#### Encoding Model

Can behavior predict neural firing? (behavior &rarr; neural)

</div>
<div class="vbn-analysis-card" markdown>

#### Decoding Model

Can neural activity predict behavior? (neural &rarr; behavior)

</div>
<div class="vbn-analysis-card" markdown>

#### Granger Causality

Does neural activity *cause* behavior changes, or vice versa?

</div>
</div>

---

## Navigate the Docs

<div class="vbn-nav-cards vbn-animate" markdown>
<a href="getting-started/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Getting Started</div>
<div class="vbn-nav-card-desc">Installation, quickstart, configuration reference</div>
</a>
<a href="pipeline/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Pipeline</div>
<div class="vbn-nav-card-desc">Phase-by-phase walkthrough with code snippets</div>
</a>
<a href="modules/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Modules</div>
<div class="vbn-nav-card-desc">API reference for all 14 source modules</div>
</a>
<a href="guides/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Guides</div>
<div class="vbn-nav-card-desc">SLEAP workflow, correlation interpretation, troubleshooting</div>
</a>
</div>

---

## Requirements at a Glance

<div class="vbn-specs vbn-animate" markdown>
<div class="vbn-spec" markdown>
<div class="vbn-spec-label">Python</div>
<div class="vbn-spec-value">3.10</div>
</div>
<div class="vbn-spec" markdown>
<div class="vbn-spec-label">RAM</div>
<div class="vbn-spec-value">16 GB min / 32 GB recommended</div>
</div>
<div class="vbn-spec" markdown>
<div class="vbn-spec-label">Disk</div>
<div class="vbn-spec-value">50-150 GB per session</div>
</div>
<div class="vbn-spec" markdown>
<div class="vbn-spec-label">GPU</div>
<div class="vbn-spec-value">Optional (CUDA for SLEAP)</div>
</div>
</div>

??? info "Full prerequisites list"

    **Data**

    - Access to the [Allen Institute Visual Behavior Neuropixels dataset](https://portal.brain-map.org/) (SDK mode downloads automatically, or provide local NWB files in manual mode)
    - Approximately **50-150 GB** of free disk space per session (NWB + video + pose outputs)

    **Compute**

    - **CPU**: Any modern multi-core processor (4+ cores recommended)
    - **RAM**: 16 GB minimum, 32 GB recommended (NWB files are large)
    - **GPU**: Optional but recommended for SLEAP inference (CUDA-capable NVIDIA GPU)
    - **OS**: macOS, Linux, or Windows (WSL2 recommended on Windows)

    **Software**

    - Python 3.10 with Conda (Miniconda or Mambaforge recommended)
    - Git and JupyterLab

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
