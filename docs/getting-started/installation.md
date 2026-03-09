# Installation

This guide walks you through setting up the VBN Analysis Suite from scratch. The recommended path uses **conda-forge** to avoid compiling scientific packages from source.

---

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Python** | 3.10 | 3.10 (pinned in `environment.yml`) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 50 GB free | 500+ GB (multiple sessions with video) |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Not required | CUDA-capable NVIDIA GPU (for SLEAP inference) |
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10 (WSL2) | macOS or Linux |

!!! warning "Disk space is the most common bottleneck"

    A single VBN session with all three camera videos (eye, face, side) can consume 30-50 GB. Plan accordingly if you intend to analyze multiple sessions. You can reduce disk usage by setting `VIDEO_SOURCE=local` to skip S3 video downloads.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/vbn-analysis.git
cd vbn-analysis
```

---

## Step 2: Create the Conda Environment

The project ships an `environment.yml` that pins Python 3.10 and pulls all scientific packages from `conda-forge`.

```bash
conda env create -f environment/environment.yml
conda activate vbn-analysis
```

This installs:

| Layer | Packages |
|-------|----------|
| **conda-forge** | `python=3.10`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `h5py`, `opencv`, `pyarrow`, `pyyaml`, `tqdm`, `joblib`, `ffmpeg`, `boto3` |
| **pip** | `allensdk==2.16.0`, `pynwb>=2.5.0`, `xgboost>=1.7.0`, `hmmlearn>=0.3.0`, `python-dotenv>=1.0.0`, `ipykernel`, `jupyterlab` |

!!! tip "Use mamba for faster solves"

    If conda takes a long time to solve the environment, use [mamba](https://mamba.readthedocs.io/) as a drop-in replacement:

    ```bash
    mamba env create -f environment/environment.yml
    ```

---

## Step 3: Pip Fallback (Alternative)

If you cannot use conda, create a virtual environment and install manually:

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Core scientific stack (prefer wheels)
pip install numpy pandas scipy scikit-learn matplotlib h5py pyarrow pyyaml tqdm joblib boto3

# OpenCV
pip install opencv-python

# NWB and Allen SDK
pip install pynwb>=2.5.0
pip install allensdk==2.16.0

# Modeling
pip install xgboost>=1.7.0 hmmlearn>=0.3.0

# Development
pip install python-dotenv>=1.0.0 ipykernel jupyterlab
```

!!! danger "NumPy compile-from-source trap"

    If pip tries to **compile NumPy from source** (you will see a long build with C compiler output), it means no pre-built wheel is available for your platform. Solutions:

    1. **Use conda-forge** (strongly recommended). It always has wheels.
    2. Pin a NumPy version that has wheels for your platform: `pip install numpy==1.24.4`
    3. Install `openblas` or `mkl` system libraries first, then retry.

    The same issue can affect `scipy` and `pandas`.

---

## Step 4: Install SLEAP

SLEAP is required for automated pose estimation (Notebooks 06-07). It has its own dependency tree, so install it carefully.

=== "conda (recommended)"

    ```bash
    conda install -c conda-forge -c nvidia -c sleap sleap
    ```

    This pulls SLEAP with CUDA support if an NVIDIA GPU is detected.

=== "pip"

    ```bash
    pip install sleap
    ```

    !!! note "CUDA on pip"

        The pip installation does **not** automatically configure CUDA. If you need GPU-accelerated inference, install the appropriate `cudatoolkit` and `cudnn` versions manually, or use the conda path above.

=== "CPU-only"

    If you do not have a GPU:

    ```bash
    pip install sleap
    # SLEAP will fall back to CPU inference (slower but functional)
    ```

!!! info "SLEAP is optional for initial exploration"

    You can run Notebooks 00-05 without SLEAP. The pose estimation notebooks (06-07) will fail gracefully if SLEAP is not installed, and Notebook 08 can work with pre-existing pose predictions in `.parquet` format.

---

## Step 5: AllenSDK Notes

The Allen SDK (`allensdk==2.16.0`) is pinned to a known working version. It provides:

- Automatic NWB file download and caching
- Session metadata queries
- Video asset discovery (in SDK access mode)

```python
# Verify AllenSDK is working
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
    cache_dir="data/allen_cache"
)
sessions = cache.get_ecephys_session_table()
print(f"Found {len(sessions)} sessions")
```

!!! warning "AllenSDK version sensitivity"

    The AllenSDK API changes between versions. Do **not** upgrade to a newer version without testing. If you encounter import errors or API changes, pin to `allensdk==2.16.0`.

!!! note "Manual access mode"

    If you already have NWB files on disk and do not want to download anything, set `ACCESS_MODE=manual` and the AllenSDK download machinery is bypassed entirely. See [Configuration](configuration.md) for details.

---

## Step 6: Register the Jupyter Kernel

Make the conda environment available as a Jupyter kernel:

```bash
python -m ipykernel install --user --name vbn-analysis --display-name "VBN Analysis"
```

Then select **"VBN Analysis"** from the kernel picker in JupyterLab.

---

## Step 7: Verify the Installation

Run this verification script to confirm everything is in place:

```python
#!/usr/bin/env python
"""Verify VBN Analysis Suite installation."""
import sys
from pathlib import Path

def check(name, import_fn):
    try:
        mod = import_fn()
        version = getattr(mod, "__version__", "ok")
        print(f"  [PASS] {name} ({version})")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("=== VBN Analysis Installation Check ===\n")

# Add src/ to path
src = Path(__file__).resolve().parent / "src"
if src.exists():
    sys.path.insert(0, str(src))
else:
    # Try relative to cwd
    src = Path("src")
    sys.path.insert(0, str(src.resolve()))

results = []

print("Core packages:")
results.append(check("numpy", lambda: __import__("numpy")))
results.append(check("pandas", lambda: __import__("pandas")))
results.append(check("scipy", lambda: __import__("scipy")))
results.append(check("sklearn", lambda: __import__("sklearn")))
results.append(check("matplotlib", lambda: __import__("matplotlib")))
results.append(check("h5py", lambda: __import__("h5py")))
results.append(check("pyarrow", lambda: __import__("pyarrow")))
results.append(check("cv2 (OpenCV)", lambda: __import__("cv2")))
results.append(check("boto3", lambda: __import__("boto3")))
results.append(check("joblib", lambda: __import__("joblib")))
results.append(check("tqdm", lambda: __import__("tqdm")))

print("\nNWB / Allen SDK:")
results.append(check("pynwb", lambda: __import__("pynwb")))
results.append(check("allensdk", lambda: __import__("allensdk")))

print("\nModeling:")
results.append(check("xgboost", lambda: __import__("xgboost")))
results.append(check("hmmlearn", lambda: __import__("hmmlearn")))

print("\nPose estimation:")
results.append(check("sleap", lambda: __import__("sleap")))

print("\nVBN src modules:")
results.append(check("config", lambda: __import__("config")))
results.append(check("timebase", lambda: __import__("timebase")))
results.append(check("io_sessions", lambda: __import__("io_sessions")))

passed = sum(results)
total = len(results)
print(f"\n{'='*40}")
print(f"Result: {passed}/{total} checks passed")

if passed < total:
    print("\nSome packages are missing. The pipeline may still work")
    print("for notebooks that don't require the missing packages.")
```

Save this as `verify_install.py` in the project root and run:

```bash
python verify_install.py
```

!!! tip "Expected output"

    All core packages and VBN src modules should pass. SLEAP may fail if not installed. That's fine for Notebooks 00-05.

---

## Common Installation Issues

### NumPy / SciPy compiling from source

**Symptom**: `pip install` runs for 10+ minutes with C compiler output.

**Fix**: Use conda-forge, which always has pre-built wheels:

```bash
conda install -c conda-forge numpy scipy pandas
```

### CUDA not detected by SLEAP

**Symptom**: SLEAP installed but inference runs on CPU.

**Fix**: Verify CUDA toolkit and cuDNN are installed and visible:

```bash
nvidia-smi          # Should show your GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the GPU list is empty, install the matching CUDA toolkit:

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
```

### AllenSDK import errors

**Symptom**: `ImportError` when importing `allensdk`.

**Fix**: The AllenSDK has specific version requirements for its dependencies. Force-reinstall the pinned version:

```bash
pip install --force-reinstall allensdk==2.16.0
```

### OpenCV codec issues

**Symptom**: `cv2.VideoCapture` opens but `cap.read()` returns `(False, None)`.

**Fix**: Install OpenCV with FFmpeg support via conda-forge:

```bash
conda install -c conda-forge opencv
# OR
pip install opencv-python-headless
```

Also verify `ffmpeg` is on your PATH:

```bash
ffmpeg -version
```

### `ModuleNotFoundError: No module named 'config'`

**Symptom**: Importing VBN src modules fails outside notebooks.

**Fix**: The `src/` directory must be on your Python path. The notebooks handle this automatically, but for standalone scripts:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from config import get_config  # Now works
```

Or set the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## Next Steps

Your environment is ready. Head to the [Quickstart](quickstart.md) to run your first analysis.
