# VBN Analysis

Local-first analysis tools for the Allen Institute Visual Behavior Neuropixels (VBN) dataset.

## Features

- **Local data management**: Download and cache VBN data locally (default: `~/data/vbn_cache`)
- **Video discovery**: Search for and preview behavior video files
- **Eye tracking visualization**: Render eye tracking data as videos when raw videos aren't available
- **Frame export**: Extract frames and timestamps for pose labeling
- **Pose estimation scaffolding**: Standardized interfaces for SLEAP and DeepLabCut

## Quick Start

### 1. Create Environment

```bash
cd ~/projects/vbn-analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Package

```bash
pip install -e ".[dev]"
```

### 3. Create Data Directories

```bash
mkdir -p ~/data/vbn_cache ~/data/vbn_outputs
```

### 4. Download Metadata

```bash
python scripts/download_metadata.py
```

This downloads ~160MB of metadata CSV files.

### 5. Download a Session

```bash
python scripts/download_session.py 1055240613
```

This downloads the NWB file (~2-4GB) for the specified session.

### 6. Check for Video Files

```bash
python scripts/video_manifest.py 1055240613
```

Note: The public VBN dataset includes processed eye tracking data but may not include raw video files.

### 7. Generate Preview

```bash
python scripts/preview_video.py 1055240613 --duration 15
```

### 8. Export Frames for Labeling

```bash
python scripts/export_frames.py 1055240613 --n-frames 100
python scripts/sample_label_frames.py 1055240613 --n-samples 50 --strategy behavior-change
```

### 9. Explore in Notebook

```bash
jupyter lab notebooks/01_preview_first_session.ipynb
```

## Project Structure

```
vbn-analysis/
├── src/vbn/              # Main package
│   ├── config.py         # Configuration loading
│   ├── cache.py          # AllenSDK cache wrapper
│   ├── io.py             # Session loading, data access
│   ├── video.py          # Video discovery and preview
│   ├── frames.py         # Frame extraction
│   └── pose/             # Pose estimation module
│       ├── schema.py     # Standardized pose format
│       ├── sleap.py      # SLEAP inference
│       └── dlc.py        # DeepLabCut inference
├── scripts/              # CLI tools
│   ├── download_metadata.py
│   ├── download_session.py
│   ├── video_manifest.py
│   ├── preview_video.py
│   ├── export_frames.py
│   ├── sample_label_frames.py
│   ├── train_sleap.py
│   ├── train_dlc.py
│   ├── run_sleap_inference.py
│   └── run_dlc_inference.py
├── notebooks/            # Jupyter notebooks
├── configs/              # Configuration files
└── pyproject.toml        # Package configuration
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Available settings:

```bash
VBN_CACHE_DIR=$HOME/data/vbn_cache      # Where to store downloaded data
VBN_OUTPUTS_DIR=$HOME/data/vbn_outputs  # Where to save outputs
VBN_VIDEO_DIRS=$HOME/data/vbn_videos    # Optional: where raw MP4s live (use ":" to separate multiple dirs)
VBN_VIDEO_CAMERA=body                  # Optional default: body|eye|face|any
VBN_VIDEO_STAGE=symlink                # Optional default: symlink|copy|none
VBN_LOG_LEVEL=INFO                      # Logging level
```

### Default Paths

- **Cache directory**: `~/data/vbn_cache`
- **Outputs directory**: `~/data/vbn_outputs`
- **Video search roots**: `VBN_VIDEO_DIRS` (optional; recommended if you have raw MP4s)

These can be overridden via environment variables or command-line arguments.

When a raw video is selected for a session (e.g., for preview, frame export, or pose inference), it is staged into `~/data/vbn_outputs/<session_id>/videos/` by default (via symlink) and recorded in `videos.json` in that folder.

## Pose Estimation Workflow

### 1. Export and Label Frames

```bash
# Export frames from eye tracking data
python scripts/export_frames.py 1055240613 --n-frames 100

# Sample diverse frames for labeling
python scripts/sample_label_frames.py 1055240613 --n-samples 50 --strategy behavior-change
```

### 2. Label Frames

Use SLEAP or DeepLabCut GUI to label the frames in `~/data/vbn_outputs/1055240613/labeling/`.

### 3. Train Model

**SLEAP:**
```bash
python scripts/train_sleap.py path/to/project.slp --epochs 100 --batch-size 4
```

**DeepLabCut:**
```bash
python scripts/train_dlc.py path/to/config.yaml --max-iters 50000
```

### 4. Run Inference

**SLEAP:**
```bash
# Either provide the video explicitly...
python scripts/run_sleap_inference.py video.mp4 models/sleap/model --session-id 1055240613

# ...or let the project resolve it from VBN_VIDEO_DIRS using the session id
python scripts/run_sleap_inference.py models/sleap/model --session-id 1055240613
```

**DeepLabCut:**
```bash
# Either provide the video explicitly...
python scripts/run_dlc_inference.py video.mp4 dlc_project/config.yaml --session-id 1055240613

# ...or let the project resolve it from VBN_VIDEO_DIRS using the session id
python scripts/run_dlc_inference.py dlc_project/config.yaml --session-id 1055240613
```

### 5. Standard Output Format

All pose outputs use a standardized schema:

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | VBN session identifier |
| frame_idx | int | Frame index in video |
| timestamp_sec | float | Time in seconds |
| node | str | Keypoint name (e.g., "nose") |
| x | float | X coordinate (pixels) |
| y | float | Y coordinate (pixels) |
| score | float | Confidence (0-1) |

## Data Notes

### VBN Public Dataset

The Visual Behavior Neuropixels dataset includes:
- **153 NWB files** (~524GB total)
- **Electrophysiology**: Spike times, LFP, unit metrics
- **Behavior**: Running speed, licks, rewards
- **Eye tracking**: Pupil position, area, blinks (processed data)
- **Stimuli**: Natural images, gabors, flashes

**Important**: The public dataset includes processed eye tracking signals but may not include raw behavior video files (MP4/AVI). This package handles this gracefully by visualizing the available eye tracking time series.

### Session 1055240613

This session is recommended for getting started because it has:
- Good eye tracking data quality
- High behavioral variability
- Complete neural recordings

## API Usage

```python
from vbn import (
    get_cache, load_session, get_eye_tracking,
    discover_videos, extract_frames_from_eye_tracking
)
from vbn.pose import PoseOutput, load_pose_outputs

# Load session
cache = get_cache()
session = load_session(cache, 1055240613)

# Get eye tracking data
eye_df = get_eye_tracking(session)
print(f"Eye tracking: {len(eye_df)} samples")

# Check for videos
videos = discover_videos(session_id=1055240613)

# Load pose outputs
pose_df = load_pose_outputs("pose_sleap.csv")
```

## Troubleshooting

### "No video files found"

The VBN public dataset may not include raw behavior videos. Use eye tracking data instead:

```bash
python scripts/preview_video.py 1055240613 --source eye-tracking
```

### Preparing a batch of videos (next N)

If you have a list of session IDs (e.g. `sessions.txt` with one id per line), you can automatically download + stage the next N videos:

```bash
python scripts/prepare_next_videos.py --sessions-file sessions.txt --n 5 --camera body --stage symlink
```

This will stage each selected video into `~/data/vbn_outputs/<session_id>/videos/` and write `videos.json` there.

### "Session not found"

Ensure metadata is downloaded first:

```bash
python scripts/download_metadata.py
```

### "Out of disk space"

Each session NWB file is 2-4GB. Ensure you have sufficient space in your cache directory.

### "allensdk not found"

Install the package with all dependencies:

```bash
pip install -e ".[dev]"
```

## Dependencies

- Python >= 3.10
- allensdk >= 2.16.0
- opencv-python >= 4.8.0
- pandas, numpy, matplotlib
- h5py, pyyaml, python-dotenv

Optional (for pose estimation):
- sleap >= 1.3.0 (`pip install -e ".[sleap]"`)
- deeplabcut >= 2.3.0 (`pip install -e ".[dlc]"`)

## License

MIT License
