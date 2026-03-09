# Module API Reference

Complete API documentation for every public module in the VBN Analysis Suite.
All modules live under `src/` and are imported by the notebook pipeline.

## Module Overview

| Module | File | Purpose | Key Functions |
|--------|------|---------|---------------|
| [config](config.md) | `config.py` | Global configuration, paths, and provenance tracking | `Config`, `get_config()`, `write_config_snapshot()`, `make_provenance()`, `get_code_version()` |
| [io_sessions](io-sessions.md) | `io_sessions.py` | Session discovery, CSV management, and the `SessionBundle` orchestrator | `load_sessions_csv()`, `get_session_bundle()`, `SessionBundle`, `cache_step()` |
| [io_nwb](io-nwb.md) | `io_nwb.py` | NWB file I/O -- open, inspect, extract, and save neural/behavioral data | `open_nwb_handle()`, `resolve_nwb_path()`, `inspect_modalities()`, `extract_units_and_spikes()`, `extract_trials()`, `extract_behavior_events()`, `extract_eye_tracking()` |
| [io_video](io-video.md) | `io_video.py` | Video asset discovery, download, frame-time alignment, and preview clips | `build_video_assets()`, `load_video_assets()`, `load_frame_times()`, `load_timestamps()`, `create_preview_clip()` |
| [features_pose](features-pose.md) | `features_pose.py` | Pose feature extraction -- velocity, acceleration, body geometry, confidence filtering | `derive_pose_features()`, `filter_by_confidence()`, `sample_frame_indices()`, `export_labeling_frames()`, `export_labeling_video()`, `export_pose_predictions_from_sleap_csv()` |
| [pose_inference](pose-inference.md) | `pose_inference.py` | Automated SLEAP model discovery, batch inference, SLP-to-Parquet conversion, and active learning | `discover_sleap_models()`, `run_sleap_inference()`, `run_batch_inference()`, `slp_to_parquet()`, `auto_discover_sleap_csvs()`, `train_sleap_model()`, `suggest_frames_to_label()` |
| [neural_events](neural-events.md) | `neural_events.py` | Event-aligned neural analysis -- PETHs, trial-averaged rates, population vectors, selectivity | `compute_peth()`, `compute_population_peth()`, `trial_averaged_rates()`, `build_population_vectors()`, `reduce_population()`, `compute_selectivity_index()`, `screen_selective_units()` |
| [cross_correlation](cross-correlation.md) | `cross_correlation.py` | Neural-behavior cross-modal correlation, encoding/decoding models, Granger causality | `crosscorrelation()`, `population_crosscorrelation()`, `sliding_correlation()`, `fit_encoding_model()`, `fit_decoding_model()`, `granger_test()`, `compute_neural_behavior_alignment()` |
| [modeling](modeling.md) | `modeling.py` | Design matrix construction, model fitting, evaluation, and neural-behavior fusion | `DesignMatrixBuilder`, `make_model()`, `prepare_features()`, `encode_for_model()`, `time_blocked_splits()`, `evaluate_model()`, `fit_and_evaluate()`, `build_fusion_table()` |
| [motifs](motifs.md) | `motifs.py` | Behavioral motif discovery via K-Means clustering and Hidden Markov Models | `motifs_kmeans()`, `motifs_hmm()` |
| [timebase](timebase.md) | `timebase.py` | Canonical timebase enforcement, artifact writing, time-grid construction, and binning | `CANONICAL_TIMEBASE`, `ensure_time_column()`, `write_parquet_with_timebase()`, `write_npz_with_provenance()`, `build_time_grid()`, `bin_spike_times()`, `bin_continuous_features()` |
| [viz](viz.md) | `viz.py` | Matplotlib visualization helpers for every analysis stage | `plot_raster()`, `plot_peth()`, `plot_crosscorrelation()`, `plot_sliding_correlation()`, `plot_encoding_decoding()`, `plot_granger_summary()`, and 10 more |
| [qc](qc.md) | `qc.py` | Quality-control checks for timestamps, frame drops, FPS estimation, and eye tracking | `check_monotonic()`, `detect_dropped_frames()`, `estimate_fps()`, `compute_video_qc()`, `eye_qc_summary()` |

## Import Convention

All modules are located in `src/` and are added to `sys.path` by the notebook
setup cell. Import them directly:

```python
from config import get_config
from io_sessions import get_session_bundle
from neural_events import compute_peth
from viz import plot_peth
```

## Timebase Convention

Every time column (`t`) throughout the pipeline is in **NWB seconds** -- the
same clock used by the Allen Brain Observatory NWB files. The constant
`timebase.CANONICAL_TIMEBASE` (`"nwb_seconds"`) is embedded in every Parquet
artifact's metadata to make this explicit.
