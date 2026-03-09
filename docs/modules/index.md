# Module API Reference

Complete API documentation for every public module in the VBN Analysis Suite. All modules live under `src/` and are imported by the notebook pipeline.

---

## By Pipeline Phase

<div class="vbn-module-group vbn-animate" markdown>
<div class="vbn-module-group-title">Phase 1: Setup & Discovery</div>
<div class="vbn-modules-grid" markdown>
<a href="config/" class="vbn-module-card" markdown>
<code>config.py</code>
<div class="vbn-module-desc">Configuration singleton, paths, environment variable loading, provenance tracking</div>
</a>
<a href="io-sessions/" class="vbn-module-card" markdown>
<code>io_sessions.py</code>
<div class="vbn-module-desc">Session discovery, CSV management, SessionBundle orchestrator, caching</div>
</a>
</div>
</div>

<div class="vbn-module-group vbn-animate" markdown>
<div class="vbn-module-group-title">Phase 2: Signal Extraction</div>
<div class="vbn-modules-grid" markdown>
<a href="io-nwb/" class="vbn-module-card" markdown>
<code>io_nwb.py</code>
<div class="vbn-module-desc">NWB file I/O: open, inspect, extract spikes, trials, eye tracking</div>
</a>
<a href="io-video/" class="vbn-module-card" markdown>
<code>io_video.py</code>
<div class="vbn-module-desc">Video asset discovery, download, frame-time alignment, preview clips</div>
</a>
</div>
</div>

<div class="vbn-module-group vbn-animate" markdown>
<div class="vbn-module-group-title">Phase 3: Video & Pose</div>
<div class="vbn-modules-grid" markdown>
<a href="features-pose/" class="vbn-module-card" markdown>
<code>features_pose.py</code>
<div class="vbn-module-desc">Pose feature extraction: velocity, acceleration, body geometry, confidence filtering</div>
</a>
<a href="pose-inference/" class="vbn-module-card" markdown>
<code>pose_inference.py</code>
<div class="vbn-module-desc">Automated SLEAP inference, model discovery, active learning, SLP-to-Parquet</div>
</a>
<a href="motifs/" class="vbn-module-card" markdown>
<code>motifs.py</code>
<div class="vbn-module-desc">Behavioral motif discovery via K-Means clustering and HMMs</div>
</a>
</div>
</div>

<div class="vbn-module-group vbn-animate" markdown>
<div class="vbn-module-group-title">Phase 4: Correlation & Modeling</div>
<div class="vbn-modules-grid" markdown>
<a href="neural-events/" class="vbn-module-card" markdown>
<code>neural_events.py</code>
<div class="vbn-module-desc">PETHs, trial-averaged rates, population vectors, unit selectivity screening</div>
</a>
<a href="cross-correlation/" class="vbn-module-card" markdown>
<code>cross_correlation.py</code>
<div class="vbn-module-desc">Cross-correlation, encoding/decoding models, Granger causality</div>
</a>
<a href="modeling/" class="vbn-module-card" markdown>
<code>modeling.py</code>
<div class="vbn-module-desc">Design matrix, XGBoost fitting, time-blocked CV, fusion table building</div>
</a>
</div>
</div>

<div class="vbn-module-group vbn-animate" markdown>
<div class="vbn-module-group-title">Infrastructure & Utilities</div>
<div class="vbn-modules-grid" markdown>
<a href="timebase/" class="vbn-module-card" markdown>
<code>timebase.py</code>
<div class="vbn-module-desc">Canonical timebase enforcement, artifact writing, time-grid construction</div>
</a>
<a href="viz/" class="vbn-module-card" markdown>
<code>viz.py</code>
<div class="vbn-module-desc">Matplotlib visualization for every analysis stage (17+ plot functions)</div>
</a>
<a href="qc/" class="vbn-module-card" markdown>
<code>qc.py</code>
<div class="vbn-module-desc">Quality control: timestamp monotonicity, frame drops, FPS estimation</div>
</a>
</div>
</div>

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
