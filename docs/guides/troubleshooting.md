# Troubleshooting

A comprehensive guide to diagnosing and fixing common issues with the VBN Analysis Suite. Each entry follows the format: **Symptom** -- **Cause** -- **Fix**.

---

## Installation Issues

### NumPy Compile From Source

**Symptom**: `pip install` takes a very long time and shows compiler output for NumPy, or fails with a C/Fortran compiler error.

**Cause**: pip is building NumPy from source because no pre-built wheel is available for your platform/Python version combination.

**Fix**: Use conda-forge, which always provides pre-built binaries:

```bash
conda install -c conda-forge numpy pandas scipy scikit-learn
```

If you must use pip, ensure you are on a supported Python version (3.9-3.12) and platform:

```bash
python -m pip install --only-binary=:all: numpy pandas
```

---

### SLEAP + TensorFlow Version Conflicts

**Symptom**: Installing SLEAP breaks other packages, or you see errors like `ImportError: cannot import name 'XXX' from 'tensorflow'`.

**Cause**: SLEAP pins specific TensorFlow versions (e.g., 2.11.x). Other packages in your environment may require different versions.

**Fix**: Create a dedicated conda environment for SLEAP:

```bash
conda create -n sleap-env python=3.9
conda activate sleap-env
conda install -c conda-forge -c nvidia -c sleap sleap
pip install -e /path/to/vbn-analysis
```

!!! warning "Do Not Mix SLEAP and Other TF-Dependent Packages"
    If you need packages that require different TensorFlow versions (e.g., certain deep learning libraries), maintain separate environments and switch between them.

---

### conda vs pip Conflicts

**Symptom**: `PackagesNotFoundError`, `UnsatisfiableError`, or broken environment after mixing conda and pip installs.

**Cause**: conda and pip track dependencies independently. Installing the same package with both tools can create inconsistencies.

**Fix**: Use the provided environment file, which handles the dependency resolution:

```bash
conda env create -f environment/environment.yml
conda activate vbn-analysis
```

If your environment is already broken:

```bash
conda deactivate
conda env remove -n vbn-analysis
conda env create -f environment/environment.yml
```

---

## Import Errors

### Module Not Found When Running Notebooks

**Symptom**: `ModuleNotFoundError: No module named 'config'` or similar when running a notebook cell.

**Cause**: The `src/` directory is not on Python's `sys.path`. Notebooks run from the `notebooks/` directory, which does not automatically include `src/`.

**Fix**: Notebook 00 sets up the path. Make sure you run Notebook 00 before any other notebook. If running a notebook standalone, add this to the first cell:

```python
import sys
from pathlib import Path

# Add src/ to the path
src_dir = str(Path("../src").resolve())
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
```

---

### AllenSDK Import Error

**Symptom**: `ImportError: No module named 'allensdk'`

**Cause**: AllenSDK is not installed.

**Fix**:

```bash
pip install allensdk
```

If you do not want to install the SDK, switch to manual mode:

```python
ACCESS_MODE = "manual"
```

---

### pynwb Import Error

**Symptom**: `ImportError: No module named 'pynwb'`

**Cause**: pynwb is not installed but is required to read NWB files.

**Fix**:

```bash
pip install pynwb
# or
conda install -c conda-forge pynwb
```

If you are developing without real NWB files, enable mock mode:

```bash
export MOCK_MODE=true
```

---

### OpenCV Import Error

**Symptom**: `ImportError: No module named 'cv2'` when exporting labeling frames or working with video.

**Cause**: OpenCV is not installed.

**Fix**:

```bash
# Recommended (conda-forge provides pre-built binaries)
conda install -c conda-forge opencv

# Alternative (pip)
pip install opencv-python
```

After installing, restart your Jupyter kernel.

---

## NWB Loading Failures

### File Not Found

**Symptom**: `FileNotFoundError: NWB file not found at ...`

**Cause**: The NWB path in `sessions.csv` is incorrect, or the file has not been downloaded.

**Fix**:

1. Check your `sessions.csv`:

    ```python
    from io_sessions import load_sessions_csv
    sessions = load_sessions_csv()
    print(sessions[["session_id", "nwb_path"]])
    ```

2. If using SDK mode, ensure the AllenSDK cache directory exists:

    ```python
    from config import get_config
    cfg = get_config()
    print(f"Cache dir: {cfg.data_dir / 'allensdk_cache'}")
    ```

3. If using manual mode, verify the file exists:

    ```bash
    ls -la /path/to/your/session.nwb
    ```

---

### AllenSDK Cache Issues

**Symptom**: `KeyError`, `ValueError`, or `FileNotFoundError` from within AllenSDK code.

**Cause**: The AllenSDK cache manifest is corrupted or out of date.

**Fix**: Clear the cache and re-download:

```bash
rm -rf data/allensdk_cache/manifest.json
```

Then re-run Notebook 01.

---

### HDF5 Version Mismatch

**Symptom**: `ValueError: Unable to open file` or `RuntimeError: HDF5 error`.

**Cause**: The HDF5 library version does not match the version used to write the NWB file.

**Fix**:

```bash
conda install -c conda-forge h5py hdf5
```

If the issue persists, check the h5py/HDF5 versions:

```python
import h5py
print(f"h5py version: {h5py.__version__}")
print(f"HDF5 version: {h5py.version.hdf5_version}")
```

The VBN NWB files require HDF5 >= 1.10.

---

### NWB File Corruption

**Symptom**: `OSError: Unable to open file` with HDF5 truncation error.

**Cause**: The NWB file download was interrupted.

**Fix**: Delete the corrupted file and re-download:

```bash
rm data/allensdk_cache/session_1098595957.nwb
# Re-run Notebook 01 to re-download
```

---

## Video Issues

### ffmpeg Not Found

**Symptom**: Video preview clips are not generated, or `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`.

**Cause**: ffmpeg is not installed or not on your PATH.

**Fix**:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

!!! note "ffmpeg Is Optional"
    The pipeline falls back to OpenCV for video operations when ffmpeg is not available. Install ffmpeg for better performance and codec support.

---

### Codec Not Supported

**Symptom**: `cv2.error: ... could not find codec parameters` or black frames when reading video.

**Cause**: The video uses a codec that OpenCV does not support on your platform.

**Fix**: Re-encode with H.264 (universally supported):

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a copy output.mp4
```

---

### S3 Download Timeout

**Symptom**: `botocore.exceptions.ReadTimeoutError` or download hangs indefinitely.

**Cause**: Slow network connection or S3 throttling.

**Fix**:

1. Retry the download (transient network issues):

    ```python
    from io_video import build_video_assets
    assets = build_video_assets(session_id=1098595957)
    ```

2. Download manually with the AWS CLI:

    ```bash
    aws s3 cp --no-sign-request \
        s3://allen-brain-observatory/visual-behavior-neuropixels/raw-data/1098595957/behavior_videos/side.mp4 \
        data/raw/visual-behavior-neuropixels/1098595957/behavior_videos/side.mp4
    ```

3. If S3 is consistently slow, download during off-peak hours or from a cloud instance in `us-west-2` (same region as the bucket).

---

### S3 Access Denied

**Symptom**: `botocore.exceptions.ClientError: An error occurred (403) when calling the GetObject operation: Access Denied`

**Cause**: The S3 request is not using unsigned access.

**Fix**: The pipeline uses unsigned (anonymous) access by default. If you see this error, ensure boto3 is configured correctly:

```python
from io_s3 import _client
client = _client()  # should use UNSIGNED config
```

If you have AWS credentials configured that override unsigned access, unset them:

```bash
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
```

---

### boto3 Not Installed

**Symptom**: `ImportError: boto3 is required for S3 downloads`

**Cause**: boto3 is not installed.

**Fix**:

```bash
pip install boto3
```

If you do not need S3 downloads, set:

```bash
export VIDEO_SOURCE="local"
```

---

## SLEAP Issues

### GPU Out of Memory (OOM)

**Symptom**: `ResourceExhaustedError`, `CUDA out of memory`, or process killed during training/inference.

**Cause**: The GPU does not have enough memory for the requested batch size.

**Fix**:

```python
# Reduce batch size
from pose_inference import run_batch_inference
results = run_batch_inference(batch_size=1)
```

For training:

```bash
sleap-train --batch_size 1 ...
```

If batch_size=1 still fails, your GPU may be too small. Options:

- Use a machine with a larger GPU
- Use Google Colab (free T4 GPU)
- Run on CPU (slow but works): `CUDA_VISIBLE_DEVICES="" sleap-train ...`

---

### Model Not Found

**Symptom**: `FileNotFoundError: No SLEAP models found`

**Cause**: No trained model exists in the auto-discovery directories.

**Fix**:

1. Check the expected directories:

    ```python
    from config import get_config
    cfg = get_config()
    print(f"Pose projects: {cfg.pose_projects_dir}")
    print(f"Models: {cfg.outputs_dir / 'models'}")
    print(f"SLEAP models: {cfg.data_dir / 'sleap_models'}")
    ```

2. Verify your model has a `training_config.json`:

    ```bash
    find pose_projects/ -name "training_config.json"
    ```

3. If using a packaged model (`.zip` or `.pkg.slp`), place it in one of the directories above.

---

### SLP File Format Version

**Symptom**: `Error reading .slp file` or `Unsupported format version`.

**Cause**: The `.slp` file was created with a different version of SLEAP.

**Fix**: Update SLEAP to the latest version:

```bash
pip install --upgrade sleap
```

If you cannot update, re-export labels from the SLEAP GUI using your current version.

---

## Plotting Issues

### Matplotlib Backend Error

**Symptom**: `RuntimeError: Tcl_AsyncDelete: async handler deleted by the wrong thread` or figures not displaying.

**Cause**: Wrong matplotlib backend for your environment.

**Fix**: Set the backend explicitly at the top of your notebook:

```python
import matplotlib
matplotlib.use("Agg")      # non-interactive (for scripts)
# or
matplotlib.use("nbAgg")    # for Jupyter notebooks
# or
%matplotlib inline          # Jupyter magic (most common)
```

---

### Figures Not Showing in Jupyter

**Symptom**: Plot cells execute without error but no figure appears.

**Cause**: Missing `%matplotlib inline` magic or the figure is created but not shown.

**Fix**: Add this to the first cell of your notebook:

```python
%matplotlib inline
```

Or call `plt.show()` explicitly after each plot:

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
```

---

### Figures Too Small or Unreadable

**Symptom**: Plots are tiny or text is overlapping.

**Fix**: Set global defaults:

```python
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
```

---

## Memory Issues

### Large NWB Files Exhaust RAM

**Symptom**: Kernel crashes or `MemoryError` when opening NWB files.

**Cause**: Some NWB files are 3-5 GB and loading them fully into memory can exceed available RAM.

**Fix**:

1. The pipeline opens NWB files in read mode (`"r"`) and extracts data incrementally. Avoid loading the entire file at once.

2. Process one session at a time rather than loading multiple sessions:

    ```python
    for session_id in session_ids:
        bundle = get_session_bundle(session_id)
        units, spikes = bundle.load_spikes()
        # Process and save results
        del units, spikes  # free memory
    ```

3. Increase available memory or use a machine with more RAM.

---

### Too Many Sessions at Once

**Symptom**: Kernel crashes during Notebook 09 (end-to-end) with many sessions.

**Cause**: Loading all sessions simultaneously exceeds memory.

**Fix**: Process sessions in batches by editing the session list in Notebook 09:

```python
# Instead of processing all sessions:
# SESSION_IDS = sessions["session_id"].tolist()

# Process in batches:
all_ids = sessions["session_id"].tolist()
BATCH_SIZE = 5
for i in range(0, len(all_ids), BATCH_SIZE):
    SESSION_IDS = all_ids[i:i+BATCH_SIZE]
    # ... run pipeline ...
```

---

### Spike Times Dictionary Is Large

**Symptom**: Memory usage spikes when loading spike times for sessions with many units (500+).

**Fix**: Filter to units of interest before analysis:

```python
import numpy as np

# Keep only units with sufficient firing rate
min_rate = 0.5  # Hz
filtered_spikes = {}
for uid, times in spike_times.items():
    duration = times.max() - times.min()
    if duration > 0 and len(times) / duration >= min_rate:
        filtered_spikes[uid] = times

print(f"Kept {len(filtered_spikes)}/{len(spike_times)} units")
```

---

## Mock Mode Issues

### Mock Mode Not Activating

**Symptom**: The pipeline tries to load real NWB files even with `MOCK_MODE=true`.

**Cause**: The environment variable is not set correctly, or `get_config()` was called before the variable was set.

**Fix**:

1. Set the variable **before** importing any pipeline modules:

    ```python
    import os
    os.environ["MOCK_MODE"] = "true"

    # Now import
    from config import get_config
    cfg = get_config()
    print(f"Mock mode: {cfg.mock_mode}")  # should be True
    ```

2. If the config was already initialized, reset it:

    ```python
    import config
    config._CONFIG = None  # force re-initialization
    os.environ["MOCK_MODE"] = "true"
    cfg = config.get_config()
    ```

---

### Mock Data Too Simple

**Symptom**: Analyses return trivial results (all zeros, NaN, or constant values) in mock mode.

**Cause**: The mock NWB object has only 3 units with very few spikes and 2 trials. This is expected.

**Fix**: Mock mode is for testing pipeline execution, not for meaningful analysis. For realistic synthetic data, create a richer mock:

```python
import numpy as np

# Generate realistic synthetic spike trains
spike_times = {
    f"unit_{i}": np.sort(np.random.uniform(0, 3600, size=np.random.randint(100, 5000)))
    for i in range(50)
}
```

---

## Parquet Metadata Issues

### Missing Sidecar .meta.json

**Symptom**: `validate_artifact_schema()` returns `False` even though the parquet file exists.

**Cause**: The `.meta.json` sidecar file is missing or was deleted.

**Fix**: Re-run the notebook that produced the artifact. The `write_parquet_with_timebase()` function always writes both the parquet file and its sidecar.

To regenerate a sidecar manually:

```python
import json
from config import make_provenance

meta = {
    "timebase": "nwb_seconds",
    "provenance": make_provenance(session_id=1098595957, alignment_method="nwb"),
}

with open("outputs/neural/session_1098595957_units.parquet.meta.json", "w") as f:
    json.dump(meta, f, indent=2)
```

---

### Parquet Schema Mismatch

**Symptom**: `ValueError: Missing required columns ['t', 'frame_idx'] for artifact ...`

**Cause**: The DataFrame being written does not contain the expected columns. This usually means an upstream extraction step failed silently.

**Fix**:

1. Check the source data:

    ```python
    import pandas as pd
    df = pd.read_parquet("outputs/pose/session_1098595957_pose_predictions.parquet")
    print(df.columns.tolist())
    ```

2. If columns are missing, re-run the upstream notebook that produces this artifact.

3. If the issue persists, check whether the SLEAP export or NWB extraction returned empty data.

---

### Timebase Metadata Says Wrong Value

**Symptom**: Sidecar JSON has `"timebase": "unknown"` or a different value.

**Cause**: The artifact was written by code that did not pass the `timebase` parameter.

**Fix**: This should not happen with the standard pipeline. If it does, re-run the producing notebook or manually fix the sidecar:

```python
import json

with open("outputs/path/to/artifact.parquet.meta.json", "r+") as f:
    meta = json.load(f)
    meta["timebase"] = "nwb_seconds"
    f.seek(0)
    json.dump(meta, f, indent=2)
    f.truncate()
```

!!! danger "Only Fix Metadata If You Are Sure"
    Only set `timebase` to `"nwb_seconds"` if you have verified that the `t` column in the parquet file is actually in NWB seconds. Incorrect metadata is worse than missing metadata.

---

## Jupyter Kernel Issues

### Kernel Restarts Mid-Execution

**Symptom**: Kernel dies without an error message, often during NWB loading or video processing.

**Cause**: Out of memory (OOM kill by the OS).

**Fix**:

1. Monitor memory usage: add `!free -h` (Linux) or check Activity Monitor (macOS)
2. Process sessions one at a time
3. Close other memory-intensive applications
4. Increase swap space

---

### Stale Module Imports

**Symptom**: Changes to `src/` files are not reflected when re-running notebook cells.

**Cause**: Python caches imported modules. Re-running an import cell does not reload the module.

**Fix**: Use `importlib.reload()`:

```python
import importlib
import config
importlib.reload(config)
```

Or restart the kernel and re-run from the top.

For automatic reloading during development:

```python
%load_ext autoreload
%autoreload 2
```

---

## Getting Help

If none of the above solutions work:

1. Check the session log: `outputs/reports/logs/session_{id}.log`
2. Check the artifact registry: `outputs/reports/artifact_registry.parquet`
3. Run with verbose logging:

    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)
    ```

4. Open an issue with:
    - The exact error message and full traceback
    - Your Python/conda environment info (`conda list` or `pip freeze`)
    - Your OS and Python version
    - The notebook and cell number where the error occurs
