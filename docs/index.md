---
hero: true
hide:
  - navigation
  - toc
---

# VBN Analysis Suite

## Pipeline

The pipeline runs in four phases. Each one builds on the previous.

<div class="vbn-pipeline vbn-animate" markdown>
<div class="vbn-pipeline-node phase-1" data-href="pipeline/phase1-setup/" markdown>
<div class="vbn-pipeline-icon">1</div>
<div class="vbn-pipeline-label">Setup</div>
<div class="vbn-pipeline-notebooks">NB 00, 01</div>
<div class="vbn-pipeline-summary">Config, sessions, environment</div>
</div>
<div class="vbn-pipeline-node phase-2" data-href="pipeline/phase2-signals/" markdown>
<div class="vbn-pipeline-icon">2</div>
<div class="vbn-pipeline-label">Extraction</div>
<div class="vbn-pipeline-notebooks">NB 02, 03, 04</div>
<div class="vbn-pipeline-summary">Spikes, trials, eye tracking</div>
</div>
<div class="vbn-pipeline-node phase-3" data-href="pipeline/phase3-video-pose/" markdown>
<div class="vbn-pipeline-icon">3</div>
<div class="vbn-pipeline-label">Video & Pose</div>
<div class="vbn-pipeline-notebooks">NB 05, 06, 07</div>
<div class="vbn-pipeline-summary">Video, SLEAP, pose features</div>
</div>
<div class="vbn-pipeline-node phase-4" data-href="pipeline/phase4-correlation/" markdown>
<div class="vbn-pipeline-icon">4</div>
<div class="vbn-pipeline-label">Correlation</div>
<div class="vbn-pipeline-notebooks">NB 08, 09</div>
<div class="vbn-pipeline-summary">Fusion, 6 analyses, QC</div>
</div>
</div>

---

## What it does

This toolkit takes Neuropixels recordings of mice doing a visual change-detection task and answers one question: **do changes in behavior align with changes in neural activity?**

It runs six analyses to get there:

- **PETHs** - how firing rates change around behavioral events
- **Cross-correlation** - the time lag between neural and behavioral signals
- **Sliding-window correlation** - when during the session coupling is strongest
- **Encoding model** - can behavior predict neural firing?
- **Decoding model** - can neural activity predict behavior?
- **Granger causality** - does one actually cause the other?

All analyses use time-blocked cross-validation. Every artifact shares a single NWB-seconds timebase.

---

## Documentation

<ul class="vbn-links" markdown>
<li markdown>[Getting started](getting-started/index.md) <span class="vbn-link-desc">- install, quickstart, configuration</span></li>
<li markdown>[Pipeline](pipeline/index.md) <span class="vbn-link-desc">- phase-by-phase walkthrough</span></li>
<li markdown>[Modules](modules/index.md) <span class="vbn-link-desc">- API reference for all 14 src modules</span></li>
<li markdown>[SLEAP workflow](guides/sleap-workflow.md) <span class="vbn-link-desc">- labeling, training, inference, active learning</span></li>
<li markdown>[Correlation guide](guides/neural-behavior-correlation.md) <span class="vbn-link-desc">- interpreting PETHs, Granger, encoding/decoding</span></li>
<li markdown>[Data access](guides/data-access.md) <span class="vbn-link-desc">- Allen SDK, S3 downloads, disk management</span></li>
<li markdown>[Troubleshooting](guides/troubleshooting.md) <span class="vbn-link-desc">- common errors and fixes</span></li>
<li markdown>[Artifacts](reference/artifacts.md) <span class="vbn-link-desc">- every file the pipeline produces</span></li>
<li markdown>[Timebase contract](reference/timebase.md) <span class="vbn-link-desc">- the nwb_seconds clock alignment</span></li>
<li markdown>[Notebook map](reference/notebook-map.md) <span class="vbn-link-desc">- inputs, outputs, and dependencies for each notebook</span></li>
</ul>

---

## Requirements

Python 3.10, conda, 16+ GB RAM, 50-150 GB disk per session. GPU optional (speeds up SLEAP). Works on macOS, Linux, and Windows (WSL2).

??? info "Full details"

    **Data**: Access to the [Allen Institute VBN dataset](https://portal.brain-map.org/). SDK mode downloads automatically, or provide local NWB files.

    **Compute**: 4+ cores, 16 GB RAM minimum (32 recommended). CUDA GPU optional for SLEAP inference.

    **Software**: Python 3.10, Conda (Miniconda or Mambaforge), Git, JupyterLab.
