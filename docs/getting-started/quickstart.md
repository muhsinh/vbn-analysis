# Quickstart

This guide walks you through running the VBN Analysis Suite end-to-end on a single session. You can use **mock mode** (no data required) to verify the pipeline, or **real data** for actual analysis.

---

## Step 1: Set Up `sessions.csv`

The pipeline reads session information from `sessions.csv` in the project root. Create it with these columns:

```csv
session_id,nwb_path,video_dir,notes
1064644573,/path/to/session_1064644573.nwb,/path/to/video_dir,First test session
```

| Column | Required | Description |
|--------|----------|-------------|
| `session_id` | Yes | Integer session ID from the Allen Institute dataset |
| `nwb_path` | SDK mode: optional; Manual mode: required | Path to the `.nwb` file on disk |
| `video_dir` | Optional | Local directory containing behavior videos |
| `notes` | Optional | Free-text notes for your records |

!!! tip "SDK mode auto-downloads"

    In `ACCESS_MODE=sdk` (the default), you only need the `session_id`. The pipeline uses the AllenSDK to download and cache the NWB file automatically. Leave `nwb_path` and `video_dir` empty:

    ```csv
    session_id,nwb_path,video_dir,notes
    1064644573,,,SDK auto-download
    ```

!!! info "Fallback from sessions.txt"

    If `sessions.csv` does not exist but a `sessions.txt` file does (one session ID per line), the pipeline generates a template `sessions.csv` automatically:

    ```text
    1064644573
    1065437523
    1066148193
    ```

---

## Step 2: Run Notebook 00 -- Configuration

Open the setup notebook to validate your environment and write the configuration snapshot:

```bash
jupyter lab notebooks/00_Setup_and_Configuration.ipynb
```

This notebook:

1. Loads environment variables and builds the `Config` object
2. Creates the output directory structure
3. Writes `outputs/reports/config_snapshot.json`
4. Validates that required packages are importable

!!! note "Setting environment variables"

    You can set configuration before launching Jupyter:

    ```bash
    export ACCESS_MODE=sdk
    export MOCK_MODE=false
    export POSE_TOOL=sleap
    jupyter lab
    ```

    Or use a `.env` file (requires `python-dotenv`). See [Configuration](configuration.md) for the full reference.

---

## Step 3: Run in Mock Mode (No Data Required)

Mock mode generates synthetic data for every modality, letting you test the full pipeline without downloading anything:

```bash
export MOCK_MODE=true
jupyter lab notebooks/09_End_to_End_Run_and_QC_Checklist.ipynb
```

This runs all pipeline phases with fake data:

- Synthetic spike trains
- Simulated behavioral trials and events
- Random eye tracking signals
- Generated pose keypoints with timestamps
- Placeholder video assets

Mock mode is useful for:

- Verifying your installation
- Testing code changes without waiting for data downloads
- CI/CD pipelines
- Demonstrating the pipeline to colleagues

!!! warning "Mock data is not real"

    The synthetic data has plausible statistical properties but does **not** reflect actual neural or behavioral dynamics. Do not draw scientific conclusions from mock-mode outputs.

---

## Step 4: Run the Full Pipeline on One Session

For a real analysis, ensure `MOCK_MODE=false` (the default) and run Notebook 09, which orchestrates all pipeline phases:

```bash
export MOCK_MODE=false
jupyter lab notebooks/09_End_to_End_Run_and_QC_Checklist.ipynb
```

Notebook 09 calls the individual phase notebooks in sequence:

```
00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
```

Alternatively, run each notebook individually for more control:

| Order | Notebook | Phase | Approx. time |
|-------|----------|-------|---------------|
| 1 | `00_Setup_and_Configuration` | Setup | < 1 min |
| 2 | `01_Session_Discovery_and_Metadata` | Setup | 1-5 min (SDK download) |
| 3 | `02_Neural_Data_Spikes_and_Events` | Extraction | 2-10 min |
| 4 | `03_Behavior_and_Task_Alignment` | Extraction | 1-3 min |
| 5 | `04_Eye_Tracking_QC_and_Features` | Extraction | 1-3 min |
| 6 | `05_Video_IO_and_Frame_Timebase` | Video | 5-30 min (S3 download) |
| 7 | `06_Pose_Estimation_Setup_SLEAP_or_DLC` | Pose | 2-5 min |
| 8 | `07_Pose_to_Motifs_Feature_Engineering` | Pose | 5-60 min (SLEAP inference) |
| 9 | `08_Neural_Behavior_Fusion_and_Modeling` | Correlation | 5-15 min |

---

## Step 5: Expected Outputs

After a successful run, the `outputs/` directory contains:

```
outputs/
  neural/
    session_1064644573_units.parquet        # Unit metadata table
    session_1064644573_spike_times.npz      # Spike times per unit
  behavior/
    session_1064644573_trials.parquet       # Trial table (type, timing, outcome)
    session_1064644573_events.parquet       # Behavioral events (licks, rewards)
  eye/
    session_1064644573_eye_features.parquet # Pupil diameter, gaze, blinks
  video/
    video_assets.parquet                    # Video file inventory
    frame_times.parquet                     # Frame timestamps (NWB seconds)
  pose/
    session_1064644573_pose_predictions.parquet  # Raw keypoint coordinates
    session_1064644573_pose_features.parquet     # Derived features
  fusion/
    session_1064644573_fusion.parquet       # Time-aligned multi-modal matrix
  models/
    session_1064644573_alignment_report.json # Full correlation results
  reports/
    config_snapshot.json                    # Configuration at run time
    artifact_registry.parquet              # Inventory of all outputs
    run_summary.parquet                    # Per-session status
    logs/
      session_1064644573.log              # Detailed run log
```

Every `.parquet` file has a companion `.parquet.meta.json` sidecar:

```json
{
  "timebase": "nwb_seconds",
  "provenance": {
    "session_id": 1064644573,
    "code_version": "9e576ad...",
    "created_at": "2026-03-09T12:34:56.789000+00:00",
    "alignment_method": "nwb"
  }
}
```

---

## Using `src/` Modules Directly from Python

You do not have to use the notebooks. The `src/` modules are a standalone Python library that you can import in scripts, REPL sessions, or your own analysis code.

### Setup: add `src/` to your path

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
```

### Load configuration

```python
from config import get_config, write_config_snapshot

cfg = get_config()
print(f"Access mode:  {cfg.access_mode}")
print(f"Bin size:     {cfg.bin_size_s} s")
print(f"Outputs dir:  {cfg.outputs_dir}")
print(f"Mock mode:    {cfg.mock_mode}")

# Write a snapshot for reproducibility
snapshot_path = write_config_snapshot()
print(f"Config saved to {snapshot_path}")
```

### Load a session bundle

```python
from io_sessions import load_sessions_csv, get_session_bundle

# Load the session inventory
sessions_df = load_sessions_csv()
print(sessions_df)

# Get a fully-resolved session bundle
session_id = 1064644573
bundle = get_session_bundle(session_id, sessions_df)

print(f"NWB path:   {bundle.nwb_path}")
print(f"Modalities: {bundle.modalities_present}")
```

### Extract neural data

```python
# Load (or extract + cache) spike data
units, spike_times = bundle.load_spikes()

print(f"Units:      {len(units)} neurons")
print(f"Spike dict: {len(spike_times)} entries")

# Inspect a single unit
unit_id = list(spike_times.keys())[0]
st = spike_times[unit_id]
print(f"Unit {unit_id}: {len(st)} spikes, range [{st.min():.1f}, {st.max():.1f}] s")
```

### Extract behavior and eye tracking

```python
# Behavior
trials, events = bundle.load_trials_and_events()
print(f"Trials: {len(trials)} rows")
print(f"Events: {len(events)} rows")

# Eye tracking
eye = bundle.load_eye_features()
if eye is not None:
    print(f"Eye features: {list(eye.columns)}")
```

### Run cross-correlation analysis

```python
from cross_correlation import compute_neural_behavior_alignment

# Assuming you have pose features loaded
import pandas as pd
pose_features = pd.read_parquet(
    cfg.outputs_dir / "pose" / f"session_{session_id}_pose_features.parquet"
)

results = compute_neural_behavior_alignment(
    spike_times_dict=spike_times,
    behavior_df=pose_features,
    trials=trials,
    bin_size=cfg.bin_size_s,
    behavior_col="pose_speed",
    max_lag_bins=40,
)

print(f"Peak lag:  {results['peak_lag_s']:.3f} s")
print(f"Peak corr: {results['peak_corr']:.3f}")
print(f"Encoding R2:  {results['encoding']['mean_r2']:.3f}")
print(f"Decoding R2:  {results['decoding']['mean_r2']:.3f}")
print(f"Granger N->B: p={results['granger_neural_to_behavior']['p_value']:.4f}")
print(f"Granger B->N: p={results['granger_behavior_to_neural']['p_value']:.4f}")
```

### Compute PETHs for a single unit

```python
from neural_events import compute_peth
import numpy as np

# Align spikes to stimulus onset times
stimulus_times = trials["t"].dropna().to_numpy()
unit_id = list(spike_times.keys())[0]

peth = compute_peth(
    spike_times[unit_id],
    stimulus_times,
    window=(-0.5, 1.0),
    bin_size=0.01,
)

print(f"PETH: {peth['n_trials']} trials, peak rate {peth['mean_rate'].max():.1f} Hz")
```

### Build a fusion table

```python
from modeling import build_fusion_table

# Load motifs (or pose features with motif_id column)
motifs = pd.read_parquet(
    cfg.outputs_dir / "pose" / f"session_{session_id}_pose_features.parquet"
)

fusion = build_fusion_table(
    spike_times=spike_times,
    motifs=motifs,
    bin_size_s=cfg.bin_size_s,
)

print(f"Fusion table: {fusion.shape[0]} time bins x {fusion.shape[1]} columns")
print(f"Columns: {list(fusion.columns[:10])}...")
```

### Visualize results

```python
from viz import plot_peth, plot_crosscorrelation, plot_behavior_summary

# PETH plot
plot_peth(peth, unit_id=unit_id)

# Cross-correlation plot
plot_crosscorrelation(results["crosscorrelation"], bin_size=cfg.bin_size_s)

# Behavior summary
plot_behavior_summary(trials)
```

---

## Next Steps

- Read the [Configuration Reference](configuration.md) to learn about every tunable knob
- Explore individual notebooks for detailed per-phase documentation
- Check the [Pipeline Phases](../index.md) overview for the big picture
