# Module API Reference

Complete API documentation for every public module in the VBN Analysis Suite. All modules live under `src/` and are imported by the notebook pipeline.

---

## By Pipeline Phase

### Phase 1: Setup & Discovery

<ul class="vbn-links" markdown>
<li markdown>[`config.py`](config.md) <span class="vbn-link-desc">- configuration singleton, paths, environment variables, provenance</span></li>
<li markdown>[`io_sessions.py`](io-sessions.md) <span class="vbn-link-desc">- session discovery, CSV management, SessionBundle, caching</span></li>
</ul>

### Phase 2: Signal Extraction

<ul class="vbn-links" markdown>
<li markdown>[`io_nwb.py`](io-nwb.md) <span class="vbn-link-desc">- NWB file I/O: open, inspect, extract spikes, trials, eye tracking</span></li>
<li markdown>[`io_video.py`](io-video.md) <span class="vbn-link-desc">- video asset discovery, download, frame-time alignment, preview clips</span></li>
</ul>

### Phase 3: Video & Pose

<ul class="vbn-links" markdown>
<li markdown>[`features_pose.py`](features-pose.md) <span class="vbn-link-desc">- velocity, acceleration, body geometry, confidence filtering</span></li>
<li markdown>[`pose_inference.py`](pose-inference.md) <span class="vbn-link-desc">- automated SLEAP inference, model discovery, active learning</span></li>
<li markdown>[`motifs.py`](motifs.md) <span class="vbn-link-desc">- behavioral motif discovery via K-Means clustering and HMMs</span></li>
</ul>

### Phase 4: Correlation & Modeling

<ul class="vbn-links" markdown>
<li markdown>[`neural_events.py`](neural-events.md) <span class="vbn-link-desc">- PETHs, trial-averaged rates, population vectors, unit selectivity</span></li>
<li markdown>[`cross_correlation.py`](cross-correlation.md) <span class="vbn-link-desc">- cross-correlation, encoding/decoding models, Granger causality</span></li>
<li markdown>[`modeling.py`](modeling.md) <span class="vbn-link-desc">- design matrix, XGBoost fitting, time-blocked CV, fusion table</span></li>
</ul>

### Infrastructure & Utilities

<ul class="vbn-links" markdown>
<li markdown>[`timebase.py`](timebase.md) <span class="vbn-link-desc">- canonical timebase enforcement, artifact writing, time-grid construction</span></li>
<li markdown>[`viz.py`](viz.md) <span class="vbn-link-desc">- Matplotlib visualization for every analysis stage (17+ plot functions)</span></li>
<li markdown>[`qc.py`](qc.md) <span class="vbn-link-desc">- quality control: timestamp monotonicity, frame drops, FPS estimation</span></li>
</ul>

---

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

Every time column (`t`) throughout the pipeline is in **NWB seconds**, the
same clock used by the Allen Brain Observatory NWB files. The constant
`timebase.CANONICAL_TIMEBASE` (`"nwb_seconds"`) is embedded in every Parquet
artifact's metadata to make this explicit.
