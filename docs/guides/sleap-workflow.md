# SLEAP Workflow Guide

This guide covers the complete pipeline for pose estimation using SLEAP: from labeling your first frames, through training a model, to running batch inference across all sessions and extracting behavioral features.

---

## Overview

The pose estimation pipeline has five phases:

```
Label frames --> Train model --> Run inference --> Extract features --> Correlate with neural
  (GUI)         (CLI/API)       (Notebook 07)    (Notebook 07)        (Notebook 08)
```

Each phase feeds into the next. The pipeline is designed so you can start with a small number of labeled frames (~150), get initial results, and improve iteratively through an **active learning loop**.

---

## Phase 1: Labeling Frames

### How Many Frames Do You Need?

!!! tip "The ~150 Frame Reality"
    For single-animal, single-camera VBN videos, **100--200 labeled frames** are enough to train a usable SLEAP model. You do not need thousands.

    The key is frame **diversity**, not quantity. Prioritize frames that cover:

    - Different body postures (crouching, stretching, turning)
    - Different lighting conditions (if any)
    - Edge cases (mouse near wall, partially occluded, grooming)
    - Different time points throughout the session (early, middle, late)

### Exporting Frames for Labeling

Notebook 06 exports frames from your cached videos and prepares them for SLEAP labeling. The pipeline writes:

- `outputs/labeling/sleap/{session_id}/{camera}/labels.csv` -- frame metadata
- `outputs/labeling/sleap/{session_id}/{camera}/frames/*.png` -- individual frame images

The export function automatically samples frames spread across the entire session:

```python
from features_pose import sample_frame_indices, export_labeling_frames
from io_video import load_frame_times

# Sample 150 frames evenly distributed across the video
frame_times = load_frame_times(session_id=1098595957, camera="side")
frame_indices = sample_frame_indices(frame_times, n_samples=150)

# Export as PNG images for labeling
export_labeling_frames(
    video_path=Path("data/raw/.../side.mp4"),
    frame_indices=frame_indices,
    output_dir=Path("outputs/labeling/sleap/1098595957/side"),
    frame_times=frame_times,
    session_id=1098595957,
    camera="side",
)
```

!!! info "Video Export Alternative"
    If you prefer to label directly from a video file (faster in the SLEAP GUI), use `export_labeling_video()` instead. This writes a short `.mp4` containing only your sampled frames:

    ```python
    from features_pose import export_labeling_video

    export_labeling_video(
        video_path=Path("data/raw/.../side.mp4"),
        frame_indices=frame_indices,
        output_dir=Path("outputs/labeling/sleap/1098595957/side"),
        frame_times=frame_times,
        session_id=1098595957,
        camera="side",
        label_fps=30.0,
        write_pngs=True,  # also export individual PNGs
    )
    ```

### Labeling Strategy

1. **Open the SLEAP GUI**: `sleap-label`
2. **Create a new project** and import your exported video or images
3. **Define your skeleton** with the keypoints relevant to your analysis:
    - For a basic mouse skeleton: `nose`, `left_ear`, `right_ear`, `neck`, `body_center`, `tail_base`, `tail_tip`
    - Fewer keypoints = faster training and higher accuracy
4. **Label consistently**:
    - Always label the same keypoints on the same body parts
    - Mark a point as "not visible" rather than guessing its position
    - Use the SLEAP "Find next suggestion" feature to be directed to diverse frames
5. **Save frequently** as a `.slp` file in your `pose_projects/` directory

!!! warning "Labeling Consistency Matters More Than Quantity"
    Inconsistent labels (labeling the left ear as the right ear, or placing the nose keypoint at different locations on the snout) will hurt model quality far more than having fewer frames. If in doubt, skip the frame rather than guessing.

### Which Frames to Pick

If you are manually selecting frames (rather than using SLEAP's suggestion tool), prioritize:

| Priority | Frame Type | Why |
|---|---|---|
| 1 | Mouse in different body orientations | Covers rotation invariance |
| 2 | Frames near stimulus onset | These are the most scientifically important |
| 3 | Frames where the mouse is moving fast | Fast motion causes blur; the model needs examples |
| 4 | Frames where the mouse is at edges of the frame | Partial visibility |
| 5 | Frames from different time points in the session | Handles lighting/drift changes |

---

## Phase 2: Training a Model

### Using the SLEAP CLI

Once you have ~150 labeled frames saved in a `.slp` file, train a model:

```bash
# Basic single-animal bottom-up model
sleap-train \
    --default_single \
    --epochs 100 \
    --batch_size 4 \
    --run_path pose_projects/trained_models/my_first_model \
    pose_projects/my_labels.slp
```

!!! tip "Training Configuration"
    For VBN side-camera videos, a single-animal bottom-up approach works best. Key config options:

    | Parameter | Recommended Value | Notes |
    |---|---|---|
    | `--default_single` | (flag) | Use single-animal model architecture |
    | `--epochs` | 100--200 | More epochs = longer training but usually better |
    | `--batch_size` | 4 | Reduce to 2 if GPU runs out of memory |
    | `--run_path` | `pose_projects/trained_models/...` | Auto-discoverable location |

### Using the Python API

The `pose_inference` module provides a wrapper around the SLEAP training CLI:

```python
from pose_inference import train_sleap_model
from pathlib import Path

model_dir = train_sleap_model(
    labels_path=Path("pose_projects/my_labels.slp"),
    config_path=None,       # use default single-animal config
    output_dir=None,        # auto: pose_projects/trained_models/<labels_stem>
    epochs=100,
    batch_size=4,
)
print(f"Model saved to: {model_dir}")
```

### Using a Custom Training Config

For more control, create a JSON config file and pass it:

```bash
sleap-train my_config.json pose_projects/my_labels.slp \
    --run_path pose_projects/trained_models/custom_model
```

See the [SLEAP documentation](https://sleap.ai/guides/choosing-models.html) for config format details.

### What to Expect During Training

- Training 100 epochs on ~150 frames typically takes **15--45 minutes** on a modern GPU
- Watch the validation loss -- it should decrease steadily for the first 50--80 epochs
- If validation loss plateaus or increases, the model is overfitting (try more labeled frames or data augmentation)
- The trained model directory will contain:
    - `training_config.json` -- the full config used
    - `best_model.h5` -- model weights at lowest validation loss
    - `training_log.csv` -- loss history

!!! danger "GPU Memory"
    If you see `ResourceExhaustedError` or `OOM` errors during training:

    1. Reduce `batch_size` to 2 or 1
    2. Close other GPU-consuming applications
    3. Reduce input resolution in the SLEAP config
    4. If on a laptop, consider using Google Colab with a GPU runtime

---

## Phase 3: Where to Place Trained Models

The pipeline auto-discovers SLEAP models by searching these directories (in order):

1. `pose_projects/` -- your local project directory
2. `outputs/models/` -- pipeline output area
3. `data/sleap_models/` -- shared/downloaded models

A model is identified by any of:

| Pattern | Description |
|---|---|
| Directory containing `training_config.json` | Standard SLEAP trained model |
| `.zip` or `.pkg.slp` file | Exported SLEAP model package |
| `.h5` or `.keras` file | Bare weight files |

```python
from pose_inference import discover_sleap_models

models = discover_sleap_models()
for m in models:
    print(f"  {m['type']:10s}  {m['path']}")
```

Example output:

```
  directory   pose_projects/trained_models/my_first_model
  package     data/sleap_models/vbn_side_v2.pkg.slp
```

!!! note "Model Priority"
    When multiple models are found, `run_batch_inference()` uses the **first** model returned by `discover_sleap_models()`. To use a specific model, pass `model_paths` explicitly.

---

## Phase 4: Running Batch Inference

### From Notebook 07

Notebook 07 handles this automatically. In the relevant cell, it will:

1. Auto-discover your trained model
2. Find all locally-cached videos
3. Run inference on each video
4. Convert `.slp` predictions to `.parquet`
5. Extract pose features

### Programmatic Batch Inference

```python
from pose_inference import run_batch_inference

results = run_batch_inference(
    session_ids=None,       # None = all cached sessions
    cameras=["side"],       # which cameras to process
    model_paths=None,       # None = auto-discover
    batch_size=4,           # reduce if GPU OOM
    skip_existing=True,     # skip videos with existing predictions
)

print(results[["session_id", "camera", "n_frames", "status"]])
```

Output:

```
   session_id camera  n_frames         status
0  1098595957   side    612345        SUCCESS
1  1099234091   side    608712        SUCCESS
2  1100123456   side         0  NO_LOCAL_VIDEO
```

### Running Inference on a Single Video

```python
from pose_inference import run_sleap_inference

slp_path = run_sleap_inference(
    video_path=Path("data/raw/.../side.mp4"),
    model_paths=["pose_projects/trained_models/my_first_model"],
    output_path=Path("outputs/pose/predictions/session_1098595957_side.predictions.slp"),
    batch_size=4,
    peak_threshold=0.2,
)
```

### Converting Predictions to Parquet

After inference, convert the `.slp` file to a standard parquet format:

```python
from pose_inference import slp_to_parquet

n_frames = slp_to_parquet(
    slp_path=Path("outputs/pose/predictions/session_1098595957_side.predictions.slp"),
    session_id=1098595957,
    camera="side",
    output_path=Path("outputs/pose/session_1098595957_pose_predictions.parquet"),
    confidence_threshold=0.0,  # filter low-confidence predictions
)
print(f"Converted {n_frames} frames")
```

The resulting parquet file contains:

| Column | Type | Description |
|---|---|---|
| `session_id` | int | Session identifier |
| `camera` | str | Camera name (side, face, eye) |
| `frame_idx` | int | Frame number in the video |
| `instance` | int | Instance index (0 for single-animal) |
| `t` | float | Timestamp in NWB seconds |
| `{keypoint}_x` | float | X coordinate of keypoint |
| `{keypoint}_y` | float | Y coordinate of keypoint |
| `{keypoint}_score` | float | Confidence score (0--1) |
| `instance_score` | float | Overall instance confidence |

---

## Phase 5: Active Learning Loop

After your first inference run, you can dramatically improve model quality by **strategically labeling the frames the model is least confident about**.

### The Loop

```
Initial labels (150) --> Train --> Infer --> Suggest uncertain frames
                          ^                         |
                          |                         v
                     Retrain <-- Label suggested frames (20-50)
```

### Getting Suggestions

```python
from pose_inference import suggest_frames_to_label

# Find frames where the model is least confident
frames_to_label = suggest_frames_to_label(
    slp_path=Path("outputs/pose/predictions/session_1098595957_side.predictions.slp"),
    n_suggestions=20,
    strategy="spread",  # "low_confidence" or "spread"
)

print(f"Suggested frame indices: {frames_to_label}")
# Example: [142, 5891, 12340, 28901, 45672, ...]
```

!!! tip "Strategy Choice"
    - **`low_confidence`**: Returns the N frames with lowest mean keypoint confidence. Best for quickly fixing the worst predictions.
    - **`spread`**: Selects low-confidence frames spread across the entire video. Best for improving generalization and handling temporal variation.

### Active Learning Workflow

1. Run inference with your current model
2. Call `suggest_frames_to_label()` to get 20--50 frame suggestions
3. Open the predictions `.slp` file in the SLEAP GUI
4. Navigate to the suggested frames and correct/add labels
5. Save the updated `.slp` file
6. Retrain with the augmented label set
7. Re-run inference
8. Repeat until confidence scores plateau

!!! info "Diminishing Returns"
    Typically, 2--3 active learning rounds (adding ~50 frames each time) bring you to ~95% of achievable accuracy. Beyond ~300--400 total labeled frames, improvements become marginal for single-animal tracking.

---

## Format Conversion

### SLEAP CSV to Parquet

If you have existing SLEAP CSV exports (from manual labeling or GUI export):

```python
from features_pose import export_pose_predictions_from_sleap_csv
from io_video import load_frame_times

frame_times = load_frame_times(session_id=1098595957, camera="side")

parquet_path = export_pose_predictions_from_sleap_csv(
    csv_path=Path("outputs/pose/my_sleap_export.csv"),
    session_id=1098595957,
    camera="side",
    frame_times=frame_times,
    output_path=Path("outputs/pose/session_1098595957_pose_predictions.parquet"),
)
```

!!! note "CSV Column Naming"
    SLEAP CSV exports use dots in column names (e.g., `nose.x`). The converter automatically normalizes these to underscores (`nose_x`) for parquet compatibility.

### Auto-Discovery of SLEAP CSVs

The pipeline can find existing CSV exports automatically:

```python
from pose_inference import auto_discover_sleap_csvs

csvs = auto_discover_sleap_csvs(session_id=1098595957)
for c in csvs:
    print(f"  {c['camera']}  {c['path']}")
```

It searches `outputs/pose/` and `outputs/labeling/` recursively and infers `session_id` and `camera` from filenames and directory structure.

### Data Discovery Priority

Notebook 07 loads pose data in this priority order:

1. **Existing `.parquet`** predictions from prior runs (fastest)
2. **SLEAP CSV exports** from manual labeling or GUI export
3. **`.slp` prediction files** from automated inference

This means once you have a parquet file, it will be used automatically without re-processing.

---

## Feature Extraction from Pose

After predictions are in parquet format, the `features_pose` module derives behavioral features:

```python
from features_pose import derive_pose_features, filter_by_confidence
import pandas as pd

# Load predictions
pose_df = pd.read_parquet("outputs/pose/session_1098595957_pose_predictions.parquet")

# Filter low-confidence detections
pose_df = filter_by_confidence(pose_df, threshold=0.3, method="nan")

# Extract features
features = derive_pose_features(pose_df, confidence_threshold=0.3)
print(features.columns.tolist())
```

Features computed:

| Feature | Column(s) | Description |
|---|---|---|
| Overall body speed | `pose_speed`, `pose_speed_std` | Mean and std of all keypoint speeds |
| Per-keypoint velocity | `{kp}_vel` | Speed of each keypoint (px/s) |
| Per-keypoint acceleration | `{kp}_accel` | Rate of velocity change |
| Body length | `body_length` | Distance between first and last keypoint |
| Head angle | `head_angle` | Angle of head direction (radians) |
| Head angular velocity | `head_angular_vel` | Rate of head turning |
| Inter-keypoint distances | `dist_{kp1}_{kp2}` | Distance between adjacent keypoints |
| Stillness | `is_still` | Binary flag: pose_speed below 10th percentile |

### Confidence Filtering

!!! warning "Always Filter Before Feature Extraction"
    Low-confidence keypoint detections produce noisy, unreliable features. The `filter_by_confidence()` function offers two modes:

    - **`method="nan"`** (default): Replaces low-confidence x/y coordinates with NaN. Preserves all frames but features may be NaN where keypoints are uncertain.
    - **`method="drop"`**: Drops entire rows where mean keypoint confidence is below threshold. Gives cleaner data but fewer frames.

    A threshold of **0.3** is a good starting point. Increase to **0.5** for stricter filtering.

---

## Common SLEAP Issues

### GPU Memory (OOM)

**Symptom**: `ResourceExhaustedError`, `CUDA out of memory`, or training/inference hangs.

**Fix**:

```python
# Reduce batch size for inference
run_batch_inference(batch_size=2)  # or even 1

# For training
sleap-train --batch_size 2 --epochs 100 ...
```

### Video Codec Not Supported

**Symptom**: SLEAP GUI shows black frames or `Could not load video`.

**Fix**: Re-encode the video with H.264:

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a copy output.mp4
```

### SLEAP Import Error

**Symptom**: `ImportError: No module named 'sleap'`

**Fix**:

```bash
# Conda (recommended for GPU support)
conda install -c conda-forge -c nvidia -c sleap sleap

# Pip (CPU only is more reliable)
pip install sleap
```

!!! danger "SLEAP + TensorFlow Version Conflicts"
    SLEAP pins specific TensorFlow versions. Installing other packages that require different TensorFlow versions will cause conflicts. Use a **dedicated conda environment** for SLEAP work:

    ```bash
    conda create -n sleap-env -c conda-forge -c nvidia -c sleap sleap
    conda activate sleap-env
    pip install -e .  # install your project in the SLEAP env
    ```

### SLP File Format Version Mismatch

**Symptom**: `Error reading .slp file` or `Unsupported format version`.

**Fix**: Update SLEAP to the latest version, or re-export your labels from the SLEAP GUI.

### Model Not Found by Auto-Discovery

**Symptom**: `FileNotFoundError: No SLEAP models found`.

**Fix**: Ensure your model directory contains `training_config.json` and is placed in one of:

- `pose_projects/`
- `outputs/models/`
- `data/sleap_models/`

Verify with:

```python
from pose_inference import discover_sleap_models
print(discover_sleap_models())
# Should return at least one entry
```
