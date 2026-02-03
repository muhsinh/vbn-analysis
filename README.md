# VBN Analysis Suite (Allen Institute Visual Behavior Neuropixels)

A reproducible, notebook-driven analysis suite for the Allen Institute Visual Behavior Neuropixels dataset. This project prioritizes **correct timebase alignment** across neural data, behavior, eye tracking, video, and pose.

## Pipeline Phases
1. **Phase 1: Setup + discovery** (Notebooks 00–01)
2. **Phase 2: Signal extraction** (Notebooks 02–04)
3. **Phase 3: Video alignment + pose** (Notebooks 05–07)
4. **Phase 4: Fusion + modeling + batch QC** (Notebooks 08–09)

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

## Access Modes
- **sdk** (default): Uses AllenSDK to download/cache NWB and discover video assets when available.
- **manual**: Uses `nwb_path`/`video_dir` from `sessions.csv` without downloading.

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

## Notebook Map (“If you want X, run notebook Y”)
- **Setup and environment checks** → `00_Setup_and_Configuration.ipynb`
- **Session inventory & modality flags** → `01_Session_Discovery_and_Metadata.ipynb`
- **Neural spikes/units** → `02_Neural_Data_Spikes_and_Events.ipynb`
- **Behavior/task alignment** → `03_Behavior_and_Task_Alignment.ipynb`
- **Eye tracking QC/features** → `04_Eye_Tracking_QC_and_Features.ipynb`
- **Video discovery + frame timebase** → `05_Video_IO_and_Frame_Timebase.ipynb`
- **Pose estimation scaffolding** → `06_Pose_Estimation_Setup_SLEAP_or_DLC.ipynb`
- **Pose → motifs** → `07_Pose_to_Motifs_Feature_Engineering.ipynb`
- **Fusion + modeling** → `08_Neural_Behavior_Fusion_and_Modeling.ipynb`
- **End-to-end pipeline + QC** → `09_End_to_End_Run_and_QC_Checklist.ipynb`

## Notes on Dependencies
- AllenSDK is pinned to a known working version.
- NumPy/Pandas are resolved by the environment manager unless conflicts arise.
- Prefer conda-forge to avoid source builds.

## Outputs
Artifacts are written under `outputs/`:
- `neural/`, `behavior/`, `eye/`, `video/`, `pose/`, `fusion/`, `models/`, `reports/`

`outputs/` is ignored by git.
