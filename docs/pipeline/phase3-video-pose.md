# Phase 3: Video & Pose

Phase 3 spans **Notebooks 05-07** and covers the full journey from raw video files to rich behavioral features derived from pose estimation. This is the most complex phase, involving S3 downloads, frame-level timestamp alignment, SLEAP model inference, and multi-step feature engineering.

---

## Notebook 05: Video I/O and Frame Timebase

### Video Asset Discovery

The pipeline supports three video cameras per session: `eye`, `face`, and `side`. The `build_video_assets()` function is the central entry point that discovers, optionally downloads, and registers all video assets.

```python title="src/io_video.py:build_video_assets() (simplified)"
def build_video_assets(
    session_id,
    video_dir=None,
    outputs_dir=None,
    download_missing=None,
):
    cfg = get_config()
    if download_missing is None:
        download_missing = cfg.video_source in {"auto", "s3"}

    s3_assets = list_s3_assets(session_id, cameras=cfg.video_cameras)  # (1)!

    asset_rows = []
    frame_times_rows = []

    for camera in cfg.video_cameras:                    # (2)!
        local_assets = _resolve_local_assets(
            session_id, camera, video_dir, cfg.video_cache_dir
        )
        qc_flags = []

        # Download from S3 if missing and allowed
        if download_missing:
            if local_assets["video"] is None:           # (3)!
                local_video = (
                    Path(cfg.video_cache_dir)
                    / str(session_id)
                    / "behavior_videos"
                    / f"{camera}.mp4"
                )
                download_asset(s3_assets[camera]["s3_uri_video"], local_video)

            # Same for timestamps and metadata ...

        timestamps = load_timestamps(local_ts)
        frame_times_df, metrics, ts_flags = _compute_frame_metrics(
            session_id, camera, timestamps
        )
        qc_flags.extend(ts_flags)

        asset_rows.append({
            "session_id": session_id,
            "camera": camera,
            "source": "s3" if downloaded else "local",
            "s3_uri_video": s3_assets[camera]["s3_uri_video"],
            "http_url_video": s3_assets[camera]["http_url_video"],
            "local_video_path": str(local_video),
            "local_timestamps_path": str(local_ts),
            "n_frames": metrics["n_frames"],
            "fps_est": metrics["fps_est"],
            "t0": metrics["t0"],
            "tN": metrics["tN"],
            "qc_flags": _join_flags(qc_flags),
        })
```

1. Queries the Allen S3 bucket to build URIs for each camera's video, timestamp, and metadata files
2. Iterates over all configured cameras (default: `["eye", "face", "side"]`)
3. Downloads missing assets to the local cache directory

### Local Asset Resolution

The pipeline searches multiple candidate directories to find existing video files:

```python title="src/io_video.py:_candidate_roots()"
def _candidate_roots(session_id, video_dir, cache_dir):
    roots = []
    if video_dir:
        video_dir = Path(video_dir)
        roots.extend([
            video_dir,                             # <video_dir>/
            video_dir / str(session_id),           # <video_dir>/<session_id>/
            video_dir / str(session_id) / "behavior_videos",
        ])
    if cache_dir:
        roots.append(
            Path(cache_dir) / str(session_id) / "behavior_videos"
        )
    return roots  # de-duplicated
```

Within each root, it looks for files matching strict naming conventions:

| Asset | Expected filename(s) |
|-------|---------------------|
| Video | `<camera>.mp4` |
| Timestamps | `<camera>_timestamps.npy` or `<camera>_timestamps.npz` |
| Metadata | `<camera>_metadata.json` |

### Timestamp Loading

Timestamps can come in multiple formats:

```python title="src/io_video.py:load_timestamps()"
def load_timestamps(path):
    if path.suffix.lower() == ".npy":
        return np.load(path)                       # (1)!
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return data[data.files[0]]                 # (2)!
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        if "t" in df.columns:
            return df["t"].to_numpy()              # (3)!
        return df.iloc[:, 0].to_numpy()
    return None
```

1. NumPy binary: most common Allen format
2. Compressed NumPy: first array in the archive
3. CSV: looks for a `t` column, falls back to the first column

### Frame Metrics and QC

For each camera, `_compute_frame_metrics()` validates the timestamps and computes quality metrics:

```python title="src/io_video.py:_compute_frame_metrics() (core logic)"
ts = np.asarray(timestamps, dtype=float)
finite = np.isfinite(ts)
if not np.all(finite):
    qc_flags.append("TIMESTAMP_NAN_PRESENT")      # (1)!

valid_ts = ts[finite]
frame_times_df = pd.DataFrame({
    "session_id": session_id,
    "camera": camera,
    "frame_idx": np.arange(len(ts))[finite],
    "t": valid_ts,
})

metrics["n_frames"] = int(valid_ts.size)
metrics["t0"] = float(valid_ts[0])
metrics["tN"] = float(valid_ts[-1])

diffs = np.diff(valid_ts)
metrics["fps_est"] = 1.0 / float(np.median(diffs))  # (2)!
```

1. NaN timestamps are flagged but not fatal
2. Frame rate is estimated from the median inter-frame interval

!!! warning "QC flags to watch for"

    | Flag | Meaning |
    |------|---------|
    | `NO_TIMESTAMPS` | No timestamp file found for this camera |
    | `TIMESTAMP_NAN_PRESENT` | Some timestamps are NaN (dropped frames) |
    | `NO_VALID_TIMESTAMPS` | All timestamps are NaN |
    | `NON_MONOTONIC` | Timestamps are not strictly increasing |
    | `DROPPED_FRAMES` | Gaps detected in the frame sequence |
    | `DOWNLOAD_FAILED_VIDEO` | S3 download of video failed |
    | `DOWNLOAD_FAILED_TIMESTAMPS` | S3 download of timestamps failed |

### Schema of `video_assets.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | `int` | Session identifier |
| `camera` | `str` | `"eye"`, `"face"`, or `"side"` |
| `source` | `str` | `"local"` or `"s3"` |
| `s3_uri_video` | `str` | S3 URI for the video file |
| `http_url_video` | `str` | HTTPS URL for streaming access |
| `local_video_path` | `str` | Absolute path to local video file |
| `local_timestamps_path` | `str` | Absolute path to local timestamps file |
| `n_frames` | `int` | Total number of valid frames |
| `fps_est` | `float` | Estimated frame rate (Hz) |
| `t0` | `float` | First frame timestamp (NWB seconds) |
| `tN` | `float` | Last frame timestamp (NWB seconds) |
| `qc_flags` | `str` | Pipe-delimited QC flags |

### Schema of `frame_times.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | `int` | Session identifier |
| `camera` | `str` | Camera name |
| `frame_idx` | `int` | Zero-based frame index in the video file |
| `t` | `float64` | Frame timestamp in NWB seconds |

### Upsert Semantics

Both `video_assets.parquet` and `frame_times.parquet` use upsert logic: when you re-run for a session, existing rows for that `(session_id, camera)` pair are replaced:

```python title="src/io_video.py:_upsert_assets()"
def _upsert_assets(new_rows, path):
    if path.exists():
        existing = pd.read_parquet(path)
        keys = set(map(tuple, new_rows[["session_id", "camera"]].values))
        mask = [key not in keys for key in existing.set_index(
            ["session_id", "camera"]
        ).index]
        existing = existing.loc[mask].reset_index(drop=True)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_parquet(path, index=False)
```

---

## Notebook 06: Pose Estimation Setup (SLEAP or DLC)

### Frame Sampling

Before you can train a pose model, you need labeled frames. The pipeline samples frames uniformly across the session:

```python title="src/features_pose.py:sample_frame_indices()"
def sample_frame_indices(frame_times, n_samples=50):
    if frame_times is None or frame_times.empty:
        return np.array([], dtype=int)

    df = frame_times.copy()
    total = len(df)
    n_samples = min(int(n_samples), total)
    row_idx = np.linspace(0, total - 1, n_samples, dtype=int)  # (1)!
    sampled = df.iloc[row_idx]["frame_idx"].astype(int).to_numpy()
    return np.unique(np.sort(sampled))
```

1. `np.linspace` ensures uniform temporal coverage, so you get frames from the beginning, middle, and end of the session

### Labeling Export: PNG Mode

For SLEAP GUI labeling, the pipeline exports individual PNG frames:

```python title="src/features_pose.py:export_labeling_frames() (simplified)"
def export_labeling_frames(video_path, frame_indices, output_dir,
                           frame_times, session_id, camera):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_map = _build_time_map(frame_times)               # (1)!
    rows = []

    for seq_i, idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"{seq_i:06d}.png"
        img_path = frames_dir / filename
        cv2.imwrite(str(img_path), frame)

        rows.append({
            "image_path": str(Path("frames") / filename),
            "session_id": session_id,
            "camera": camera,
            "seq_idx": seq_i,
            "frame_idx": int(idx),
            "t": t_map.get(int(idx), np.nan),          # (2)!
        })

    cap.release()
    labels_path = output_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    return output_dir
```

1. Pre-builds a `{frame_idx: timestamp}` lookup from the frame_times DataFrame for O(1) access
2. Every exported frame retains its NWB-seconds timestamp

### Labeling Export: MP4 Mode

For workflows that prefer a video file (e.g., SLEAP video import), the pipeline also supports MP4 export with optional simultaneous PNG extraction:

```python title="src/features_pose.py:export_labeling_video() (key logic)"
def export_labeling_video(video_path, frame_indices, output_dir,
                          frame_times, session_id, camera,
                          label_fps=30.0, write_pngs=False):
    import cv2
    cap = cv2.VideoCapture(str(video_path))

    # Try MP4 first, fall back to AVI
    mp4_path = output_dir / "labeling.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(label_fps), (width, height),
    )
    if not writer.isOpened():                          # (1)!
        avi_path = output_dir / "labeling.avi"
        writer = cv2.VideoWriter(
            str(avi_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            float(label_fps), (width, height),
        )

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)

        if write_pngs:                                 # (2)!
            cv2.imwrite(str(frames_dir / f"{seq_i:06d}.png"), frame)

    writer.release()
    cap.release()

    # Write labels.csv mapping seq_idx -> frame_idx -> t
    pd.DataFrame(rows).to_csv(output_dir / "labels.csv", index=False)
```

1. Automatic codec fallback: MP4v -> XVID
2. Optionally writes PNGs alongside the video for maximum flexibility

### Schema of `labels.csv`

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | `str` | Relative path to frame image (PNG mode) |
| `video_path` | `str` | Relative path to labeling video (MP4 mode) |
| `session_id` | `int` | Session identifier |
| `camera` | `str` | Camera name |
| `seq_idx` | `int` | Sequential index in the exported set (1-based) |
| `frame_idx` | `int` | Original frame index in the full video |
| `t` | `float64` | Timestamp in NWB seconds |

---

## Notebook 07: Pose-to-Features Engineering

### Auto-Discovery of SLEAP Predictions

The pipeline can automatically find SLEAP prediction files in multiple formats:

```python title="src/pose_inference.py:auto_discover_sleap_csvs()"
def auto_discover_sleap_csvs(session_id=None):
    cfg = get_config()
    csvs = []
    search_dirs = [
        cfg.outputs_dir / "pose",
        cfg.outputs_dir / "labeling",
    ]

    for base in search_dirs:
        if not base.exists():
            continue
        for csv_path in base.rglob("*.csv"):
            sid = _extract_session_id(str(csv_path))   # (1)!
            cam = _extract_camera(str(csv_path), csv_path.stem)  # (2)!

            if session_id is not None and sid != session_id:
                continue

            csvs.append({
                "path": str(csv_path),
                "session_id": sid,
                "camera": cam,
                "filename": csv_path.name,
            })
    return csvs
```

1. Extracts session ID from the file path using regex (looks for 10-digit numbers)
2. Infers camera name from path components (`"side"`, `"face"`, `"eye"`)

### SLEAP CSV to Parquet Conversion

```python title="src/features_pose.py:export_pose_predictions_from_sleap_csv()"
def export_pose_predictions_from_sleap_csv(csv_path, session_id, camera,
                                            frame_times=None, output_path=None):
    df = pd.read_csv(csv_path)

    if "frame_idx" not in df.columns:
        if "frame" in df.columns:
            df = df.rename(columns={"frame": "frame_idx"})     # (1)!

    rename_map = {
        col: col.replace(".", "_")
        for col in df.columns if "." in col
    }
    df = df.rename(columns=rename_map)                         # (2)!

    df["session_id"] = session_id
    df["camera"] = camera

    timestamps = None
    if frame_times is None or frame_times.empty:
        timestamps = _load_camera_timestamps(session_id, camera)  # (3)!
    df = _attach_timestamps(df, frame_times, timestamps)       # (4)!

    front = ["session_id", "camera", "frame_idx", "t"]
    remaining = [c for c in df.columns if c not in front]
    df = df[front + remaining]

    write_parquet_with_timebase(
        df, output_path,
        provenance=make_provenance(session_id, "nwb"),
        required_columns=["t", "frame_idx"],
    )
    return output_path
```

1. Normalizes column name: SLEAP exports use `frame`, pipeline uses `frame_idx`
2. Replaces dots with underscores (e.g., `nose.x` becomes `nose_x`)
3. Falls back to loading camera timestamps directly from `.npy` files
4. Attaches NWB-seconds timestamps to each frame

### Automated SLEAP Inference Pipeline

For sessions where you have a trained model but no predictions yet, the pipeline can run inference automatically.

#### Step 1: Model Discovery

```python title="src/pose_inference.py:discover_sleap_models()"
def discover_sleap_models(search_dirs=None):
    cfg = get_config()
    if search_dirs is None:
        search_dirs = [
            cfg.pose_projects_dir,             # pose_projects/
            cfg.outputs_dir / "models",        # outputs/models/
            cfg.data_dir / "sleap_models",     # data/sleap_models/
        ]

    models = []
    for base in search_dirs:
        # Pattern 1: directory with training_config.json
        for config_path in base.rglob("training_config.json"):
            meta = _parse_training_config(config_path)
            models.append({
                "path": str(config_path.parent),
                "type": "directory",
                **meta,
            })

        # Pattern 2: exported .zip or .pkg.slp packages
        for ext in ("*.zip", "*.pkg.slp"):
            for pkg in base.rglob(ext):
                models.append({
                    "path": str(pkg),
                    "type": "package",
                    "name": pkg.stem,
                })

        # Pattern 3: bare .h5 / .keras weight files
        for ext in ("*.h5", "*.keras"):
            for wf in base.rglob(ext):
                models.append({
                    "path": str(wf),
                    "type": "weights",
                    "name": wf.stem,
                })
    return models
```

#### Step 2: Batch Inference

```python title="src/pose_inference.py:run_batch_inference() (core loop)"
def run_batch_inference(session_ids=None, cameras=None,
                        model_paths=None, batch_size=4,
                        skip_existing=True):
    cfg = get_config()

    # Auto-discover model if not specified
    if model_paths is None:
        models = discover_sleap_models()
        if not models:
            raise FileNotFoundError(
                "No SLEAP models found. Train a model first."
            )
        model_paths = [models[0]["path"]]

    video_assets = load_video_assets()                 # (1)!

    results = []
    for _, row in video_assets.iterrows():
        sid = int(row["session_id"])
        cam = str(row["camera"])
        video_path = Path(row["local_video_path"])

        slp_path = (
            cfg.outputs_dir / "pose" / "predictions"
            / f"session_{sid}_{cam}.predictions.slp"
        )

        if skip_existing and slp_path.exists():        # (2)!
            continue

        # Run SLEAP inference
        run_sleap_inference(                           # (3)!
            video_path=video_path,
            model_paths=model_paths,
            output_path=slp_path,
            batch_size=batch_size,
        )

        # Convert to parquet
        n_frames = slp_to_parquet(                     # (4)!
            slp_path=slp_path,
            session_id=sid,
            camera=cam,
        )
```

1. Reads the video asset registry from Phase 3 / Notebook 05
2. Skips sessions that already have predictions
3. Calls SLEAP's inference API with the model
4. Converts `.slp` predictions to the standard parquet format

#### Step 3: SLP to Parquet Conversion

```python title="src/pose_inference.py:slp_to_parquet() (core logic)"
def slp_to_parquet(slp_path, session_id, camera,
                   output_path=None, confidence_threshold=0.0):
    import sleap
    labels = sleap.load_file(str(slp_path))

    rows = []
    skeleton = labels.skeletons[0]
    node_names = [n.name for n in skeleton.nodes]

    for lf in labels.labeled_frames:
        frame_idx = lf.frame_idx
        for inst_idx, inst in enumerate(lf.instances):
            row = {
                "frame_idx": int(frame_idx),
                "instance": int(inst_idx),
            }
            if hasattr(inst, "score") and inst.score is not None:
                row["instance_score"] = float(inst.score)

            for node_name, point in zip(node_names, inst.points):
                if point is None or not hasattr(point, "x"):
                    row[f"{node_name}_x"] = np.nan
                    row[f"{node_name}_y"] = np.nan
                    row[f"{node_name}_score"] = 0.0
                else:
                    row[f"{node_name}_x"] = float(point.x)
                    row[f"{node_name}_y"] = float(point.y)
                    row[f"{node_name}_score"] = float(point.score)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Attach timestamps
    frame_times = load_frame_times(session_id=session_id, camera=camera)
    if not frame_times.empty:
        ft = frame_times[["frame_idx", "t"]].drop_duplicates("frame_idx")
        df = df.merge(ft, on="frame_idx", how="left")

    df["session_id"] = session_id
    df["camera"] = camera

    write_parquet_with_timebase(
        df, output_path,
        provenance=make_provenance(session_id, "sleap_inference"),
        required_columns=["frame_idx", "t"],
    )
    return len(df)
```

### Schema of `pose_predictions.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | `int` | Session identifier |
| `camera` | `str` | Camera name |
| `frame_idx` | `int` | Frame index in the video |
| `instance` | `int` | Instance index (for multi-animal tracking) |
| `t` | `float64` | Timestamp (NWB seconds) |
| `instance_score` | `float64` | Overall detection confidence |
| `<keypoint>_x` | `float64` | X coordinate in pixels |
| `<keypoint>_y` | `float64` | Y coordinate in pixels |
| `<keypoint>_score` | `float64` | Per-keypoint detection confidence |

---

### Confidence Filtering

Before computing features, low-confidence detections can be filtered:

```python title="src/features_pose.py:filter_by_confidence()"
def filter_by_confidence(df, threshold=0.3, method="nan"):
    df = df.copy()
    keypoints = _find_keypoints(df)                    # (1)!

    if method == "nan":
        for kp in keypoints:
            score_col = f"{kp}_score"
            if score_col in df.columns:
                low = df[score_col] < threshold
                df.loc[low, f"{kp}_x"] = np.nan        # (2)!
                df.loc[low, f"{kp}_y"] = np.nan

    elif method == "drop":
        score_cols = [f"{kp}_score" for kp in keypoints
                      if f"{kp}_score" in df.columns]
        mean_score = df[score_cols].mean(axis=1)
        df = df[mean_score >= threshold]               # (3)!

    return df
```

1. Auto-discovers keypoint names by finding columns ending in `_x` that have a matching `_y`
2. **NaN method**: replaces only the low-confidence coordinates with NaN (preserves high-confidence keypoints in the same frame)
3. **Drop method**: removes entire rows where the mean confidence is below threshold

!!! tip "Which filtering method to use"
    - Use `method="nan"` (default) for feature extraction, as it preserves frames where most keypoints are good
    - Use `method="drop"` for visualization or when you need every keypoint to be valid

---

### Rich Feature Extraction: `derive_pose_features()`

This is the core feature engineering function. It takes raw pose predictions and computes a comprehensive set of behavioral features:

```python title="src/features_pose.py:derive_pose_features()"
def derive_pose_features(pose_df, confidence_threshold=0.0):
    if pose_df is None or pose_df.empty:
        return None
    df = pose_df.copy()
    t = df["t"].to_numpy(dtype=float)
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6                                # (1)!

    if confidence_threshold > 0:
        df = filter_by_confidence(df, confidence_threshold, method="nan")

    keypoints = _find_keypoints(df)

    # === Per-keypoint velocity and acceleration ===
    all_speeds = []
    for kp in keypoints:
        x, y = _get_keypoint_xy(df, kp)
        vx = np.gradient(x, t)                        # (2)!
        vy = np.gradient(y, t)
        speed = np.sqrt(vx**2 + vy**2)
        accel = np.gradient(speed, t)                  # (3)!

        df[f"{kp}_vel"] = speed
        df[f"{kp}_accel"] = accel
        all_speeds.append(speed)

    # === Overall pose speed ===
    speed_matrix = np.column_stack(all_speeds)
    df["pose_speed"] = np.nanmean(speed_matrix, axis=1)      # (4)!
    df["pose_speed_std"] = np.nanstd(speed_matrix, axis=1)   # (5)!

    # === Body length ===
    if len(keypoints) >= 2:
        x0, y0 = _get_keypoint_xy(df, keypoints[0])
        x1, y1 = _get_keypoint_xy(df, keypoints[-1])
        df["body_length"] = np.sqrt((x1-x0)**2 + (y1-y0)**2)  # (6)!

    # === Head angle ===
    if len(keypoints) >= 2:
        x0, y0 = _get_keypoint_xy(df, keypoints[0])
        x1, y1 = _get_keypoint_xy(df, keypoints[1])
        df["head_angle"] = np.arctan2(y0-y1, x0-x1)           # (7)!
        df["head_angular_vel"] = np.gradient(
            np.unwrap(df["head_angle"].to_numpy()), t          # (8)!
        )

    # === Inter-keypoint distances (adjacent pairs) ===
    for i in range(len(keypoints) - 1):
        xi, yi = _get_keypoint_xy(df, keypoints[i])
        xj, yj = _get_keypoint_xy(df, keypoints[i+1])
        df[f"dist_{keypoints[i]}_{keypoints[i+1]}"] = np.sqrt(
            (xj-xi)**2 + (yj-yi)**2
        )                                                      # (9)!

    # === Stillness detection ===
    speed_threshold = np.nanpercentile(
        df["pose_speed"].to_numpy(), 10
    )
    df["is_still"] = (
        df["pose_speed"] < max(speed_threshold, 1.0)
    ).astype(int)                                              # (10)!

    return df[output_cols]
```

1. Prevent division by zero in time derivatives
2. Per-keypoint velocity via numerical gradient: $v = \sqrt{(\frac{dx}{dt})^2 + (\frac{dy}{dt})^2}$
3. Acceleration is the time derivative of speed: $a = \frac{dv}{dt}$
4. Overall body speed: mean velocity across all keypoints (NaN-robust)
5. Speed variability: how differently body parts are moving (indicates articulated motion)
6. Body length: Euclidean distance between the first and last keypoint (proxy for body stretch)
7. Head angle: angle from second to first keypoint using `arctan2`
8. Head angular velocity: computed on the **unwrapped** angle to avoid discontinuities at $\pm\pi$
9. Adjacent keypoint distances: captures limb extension, body curvature
10. Stillness: binary flag, True when overall speed is below the 10th percentile (minimum 1.0 px/s)

### Feature Summary Table

| Feature | Column(s) | Unit | Description |
|---------|-----------|------|-------------|
| Keypoint velocity | `<kp>_vel` | px/s | Per-keypoint speed |
| Keypoint acceleration | `<kp>_accel` | px/s^2 | Per-keypoint acceleration |
| Overall body speed | `pose_speed` | px/s | Mean across all keypoints |
| Speed variability | `pose_speed_std` | px/s | Std across all keypoints |
| Body length | `body_length` | px | First-to-last keypoint distance |
| Head angle | `head_angle` | rad | Direction the head is pointing |
| Head angular velocity | `head_angular_vel` | rad/s | Turning rate |
| Inter-keypoint distance | `dist_<kp1>_<kp2>` | px | Adjacent keypoint distances |
| Stillness | `is_still` | 0/1 | Binary stillness indicator |

---

### Active Learning Suggestions

After initial inference, the pipeline can suggest which frames to label next for the biggest improvement:

```python title="src/pose_inference.py:suggest_frames_to_label()"
def suggest_frames_to_label(slp_path, n_suggestions=20,
                            strategy="low_confidence"):
    import sleap
    labels = sleap.load_file(str(slp_path))

    frame_scores = []
    for lf in labels.labeled_frames:
        scores = []
        for inst in lf.instances:
            for pt in inst.points:
                if pt is not None and hasattr(pt, "score"):
                    scores.append(float(pt.score))
        mean_score = np.mean(scores) if scores else 0.0
        frame_scores.append((lf.frame_idx, mean_score))

    frame_scores.sort(key=lambda x: x[1])              # (1)!

    if strategy == "low_confidence":
        return np.array([
            fs[0] for fs in frame_scores[:n_suggestions]
        ])                                              # (2)!

    elif strategy == "spread":
        candidates = frame_scores[:n_suggestions * 3]   # (3)!
        indices = np.linspace(
            0, len(candidates)-1, n_suggestions, dtype=int
        )
        return np.array([candidates[i][0] for i in indices])
```

1. Sort frames by confidence, lowest first
2. **Low-confidence strategy**: pick the N worst frames, which are where the model struggles most
3. **Spread strategy**: pick from the worst frames but spread evenly across the session to avoid labeling a cluster of similar-looking frames

!!! info "Active learning loop"
    The recommended workflow is:

    1. Label ~50 frames manually
    2. Train initial SLEAP model
    3. Run inference on all videos
    4. Call `suggest_frames_to_label()` to find the 20 most uncertain frames
    5. Label those frames in SLEAP GUI
    6. Retrain and repeat until quality is satisfactory

---

### Output Files from Phase 3

| File | Location | Format | Purpose |
|------|----------|--------|---------|
| `video_assets.parquet` | `outputs/video/` | Parquet | Registry of all video files and their metadata |
| `frame_times.parquet` | `outputs/video/` | Parquet | Frame-level timestamps for all cameras |
| `labels.csv` | `outputs/labeling/session_<id>/` | CSV | Frame export manifest for SLEAP labeling |
| `pose_predictions.parquet` | `outputs/pose/` | Parquet | Raw keypoint coordinates with confidence scores |
| `pose_features.parquet` | `outputs/pose/` | Parquet | Derived behavioral features from pose |
