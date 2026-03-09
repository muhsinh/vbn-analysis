# Guides

Practical, step-by-step guides for working with the VBN Analysis Suite. Each guide assumes you have completed the [Getting Started](../getting-started/index.md) setup and can run Notebook 00 without errors.

---

## Available Guides

<ul class="vbn-links" markdown>
<li markdown>[SLEAP Workflow](sleap-workflow.md) <span class="vbn-link-desc">- labeling, training, inference, and active-learning loop</span></li>
<li markdown>[Neural-Behavior Correlation](neural-behavior-correlation.md) <span class="vbn-link-desc">- interpreting PETHs, cross-correlation, encoding/decoding, Granger causality</span></li>
<li markdown>[Data Access & Storage](data-access.md) <span class="vbn-link-desc">- Allen VBN dataset organization, SDK vs manual mode, S3 downloads</span></li>
<li markdown>[Troubleshooting](troubleshooting.md) <span class="vbn-link-desc">- solutions for common errors: installation, imports, NWB, video, SLEAP</span></li>
</ul>

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
