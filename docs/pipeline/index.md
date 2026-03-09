# Pipeline Overview

The VBN Analysis Suite processes Allen Institute Visual Behavior Neuropixels (VBN) data through four sequential phases, each building on the outputs of the previous one. Every artifact is written with a canonical **NWB-seconds timebase** and machine-readable provenance metadata, so downstream consumers never have to guess which clock a timestamp refers to.

---

## Data Flow

<div class="vbn-flow vbn-animate" markdown>
<div class="vbn-flow-step phase-1" markdown>
<div class="vbn-flow-dot">1</div>
<div class="vbn-flow-body" markdown>
<div class="vbn-flow-title">Setup & Discovery</div>
<div class="vbn-flow-meta">NB 00: Config | NB 01: Sessions</div>

Environment validation, dependency checks, session inventory from SDK or local files.

<div class="vbn-flow-artifacts">
<span class="vbn-flow-artifact">sessions.csv</span>
<span class="vbn-flow-artifact">SessionBundle</span>
<span class="vbn-flow-artifact">config_snapshot.json</span>
</div>
</div>
</div>

<div class="vbn-flow-step phase-2" markdown>
<div class="vbn-flow-dot">2</div>
<div class="vbn-flow-body" markdown>
<div class="vbn-flow-title">Signal Extraction</div>
<div class="vbn-flow-meta">NB 02: Neural | NB 03: Behavior | NB 04: Eye</div>

Extract neural spikes and unit tables, behavioral trials and events, eye tracking features with QC.

<div class="vbn-flow-artifacts">
<span class="vbn-flow-artifact">units.parquet</span>
<span class="vbn-flow-artifact">spike_times.npz</span>
<span class="vbn-flow-artifact">trials.parquet</span>
<span class="vbn-flow-artifact">events.parquet</span>
<span class="vbn-flow-artifact">eye_features.parquet</span>
</div>
</div>
</div>

<div class="vbn-flow-step phase-3" markdown>
<div class="vbn-flow-dot">3</div>
<div class="vbn-flow-body" markdown>
<div class="vbn-flow-title">Video & Pose</div>
<div class="vbn-flow-meta">NB 05: Video I/O | NB 06: Pose Setup | NB 07: Features</div>

Download video assets from S3, align frame timestamps, run SLEAP pose estimation, derive pose features.

<div class="vbn-flow-artifacts">
<span class="vbn-flow-artifact">video_assets.parquet</span>
<span class="vbn-flow-artifact">frame_times.parquet</span>
<span class="vbn-flow-artifact">pose_predictions.parquet</span>
<span class="vbn-flow-artifact">pose_features.parquet</span>
</div>
</div>
</div>

<div class="vbn-flow-step phase-4" markdown>
<div class="vbn-flow-dot">4</div>
<div class="vbn-flow-body" markdown>
<div class="vbn-flow-title">Neural-Behavior Correlation</div>
<div class="vbn-flow-meta">NB 08: Fusion & Modeling | NB 09: QC Checklist</div>

Build time-aligned fusion table, run all 6 correlation analyses, generate QC report.

<div class="vbn-flow-artifacts">
<span class="vbn-flow-artifact">fusion_table.parquet</span>
<span class="vbn-flow-artifact">PETHs</span>
<span class="vbn-flow-artifact">cross-correlation</span>
<span class="vbn-flow-artifact">encoding/decoding</span>
<span class="vbn-flow-artifact">Granger causality</span>
<span class="vbn-flow-artifact">QC report</span>
</div>
</div>
</div>
</div>

---

## Phase Summary

| Phase | Notebooks | Input | Output | Key Module(s) |
|-------|-----------|-------|--------|----------------|
| **Phase 1: Setup & Discovery** | 00, 01 | Environment, `sessions.csv` or `sessions.txt` | `SessionBundle` objects, config snapshot | `config`, `io_sessions` |
| **Phase 2: Signal Extraction** | 02, 03, 04 | NWB files (real or mock) | `units.parquet`, `spike_times.npz`, `trials.parquet`, `events.parquet`, `eye_features.parquet` | `io_nwb`, `features_task`, `features_eye` |
| **Phase 3: Video & Pose** | 05, 06, 07 | S3 video assets, SLEAP/DLC models | `video_assets.parquet`, `frame_times.parquet`, `pose_predictions.parquet`, `pose_features.parquet` | `io_video`, `features_pose`, `pose_inference` |
| **Phase 4: Neural-Behavior Correlation** | 08, 09 | All Phase 2 & 3 outputs | Fusion table, PETHs, cross-correlation, encoding/decoding models, Granger causality, QC report | `neural_events`, `cross_correlation`, `modeling` |

---

## Canonical Timebase

Every timestamped artifact in the pipeline uses **NWB seconds**, the same clock that the Neuropixels hardware synchronizes to inside each NWB file. This means:

- Spike times, trial events, eye tracking samples, video frame timestamps, and pose predictions all share a single reference clock.
- Every parquet file carries a `timebase` field in its Parquet schema metadata **and** in a sidecar `.meta.json` file.
- You never need to convert between clocks when joining tables.

```json title="Example sidecar: session_1234567890_units.parquet.meta.json"
{
  "timebase": "nwb_seconds",
  "provenance": {
    "session_id": 1234567890,
    "code_version": "9e576ad",
    "created_at": "2026-03-09T12:00:00+00:00",
    "alignment_method": "nwb"
  }
}
```

---

## Artifact Caching

The pipeline uses a two-tier caching strategy:

1. **Output artifacts** (`outputs/neural/`, `outputs/behavior/`, etc.) are deterministic parquet files keyed by session ID. If the file exists on disk, it is loaded directly without re-extracting from NWB.
2. **Step cache** (`outputs/cache/session_<id>/`) uses `joblib` serialization keyed by an MD5 hash of the step parameters. This covers expensive intermediate computations.

```python title="How caching works under the hood"
def cache_step(session_id, step, params, compute_fn):
    path = _cache_path(session_id, step, params)  # (1)!
    if path.exists():
        return joblib.load(path)                   # (2)!
    result = compute_fn()
    joblib.dump(result, path)                      # (3)!
    return result
```

1. Path is `outputs/cache/session_<id>/<step>_<md5>.joblib`
2. Cache hit: returns instantly
3. Cache miss: compute, then persist for next time

---

## Next Steps

<div class="vbn-nav-cards" markdown>
<a href="phase1-setup/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Phase 1: Setup & Discovery</div>
<div class="vbn-nav-card-desc">Configuration, session discovery, the SessionBundle dataclass</div>
</a>
<a href="phase2-signals/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Phase 2: Signal Extraction</div>
<div class="vbn-nav-card-desc">Neural, behavioral, and eye-tracking data extraction</div>
</a>
<a href="phase3-video-pose/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Phase 3: Video & Pose</div>
<div class="vbn-nav-card-desc">Video assets, SLEAP inference, pose feature engineering</div>
</a>
<a href="phase4-correlation/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Phase 4: Neural-Behavior Correlation</div>
<div class="vbn-nav-card-desc">PETHs, cross-correlation, encoding/decoding, Granger causality</div>
</a>
</div>
