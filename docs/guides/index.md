# Guides

Practical, step-by-step guides for working with the VBN Analysis Suite. Each guide assumes you have completed the [Getting Started](../getting-started/index.md) setup and can run Notebook 00 without errors.

---

## Available Guides

<div class="vbn-nav-cards vbn-animate" markdown>
<a href="sleap-workflow/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">SLEAP Workflow</div>
<div class="vbn-nav-card-desc">The complete labeling, training, inference, and active-learning loop. How many frames to label, where to put models, and how to run batch inference.</div>
</a>
<a href="neural-behavior-correlation/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Neural-Behavior Correlation</div>
<div class="vbn-nav-card-desc">How to interpret PETHs, cross-correlation, encoding/decoding models, Granger causality, and unit selectivity. What makes a "good" result.</div>
</a>
<a href="data-access/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Data Access & Storage</div>
<div class="vbn-nav-card-desc">How the Allen VBN dataset is organized, SDK vs manual mode, S3 video downloads, NWB file inspection, and working with limited disk space.</div>
</a>
<a href="troubleshooting/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Troubleshooting</div>
<div class="vbn-nav-card-desc">Solutions for every common error: installation conflicts, import failures, NWB loading issues, video codec problems, SLEAP GPU OOM, and more.</div>
</a>
</div>

---

## Recommended Reading Order

If you are new to the project, read the guides in this order:

1. **Data Access**: understand where data lives and how to get it
2. **SLEAP Workflow**: set up pose estimation (the most hands-on phase)
3. **Neural-Behavior Correlation**: interpret the results from Notebook 08
4. **Troubleshooting**: bookmark this for when things go wrong

## See Also

- [Reference: Artifacts](../reference/artifacts.md): every file the pipeline produces
- [Reference: Timebase](../reference/timebase.md): the canonical `nwb_seconds` contract
- [Reference: Notebook Map](../reference/notebook-map.md): what each notebook does, its inputs and outputs
