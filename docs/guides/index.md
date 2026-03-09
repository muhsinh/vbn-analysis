# Guides

Practical, step-by-step guides for working with the VBN Analysis Suite.  Each guide assumes you have completed the [Getting Started](../getting-started/index.md) setup and can run Notebook 00 without errors.

## Available Guides

| Guide | What you will learn |
|---|---|
| **[SLEAP Workflow](sleap-workflow.md)** | The complete labeling, training, inference, and active-learning loop for automated pose estimation. How many frames to label, where to put models, and how to run batch inference from Notebook 07. |
| **[Neural-Behavior Correlation](neural-behavior-correlation.md)** | How to interpret PETHs, cross-correlation, encoding/decoding models, Granger causality, and unit selectivity. What constitutes a "good" result and what to do when results are weak. |
| **[Data Access](data-access.md)** | How the Allen Institute VBN dataset is organized, SDK vs Manual mode, S3 video downloads, NWB file inspection, and working with limited disk space. |
| **[Troubleshooting](troubleshooting.md)** | Solutions for every common error: installation conflicts, import failures, NWB loading issues, video codec problems, SLEAP GPU OOM, matplotlib backends, and more. |

## Recommended Reading Order

If you are new to the project, read the guides in this order:

1. **Data Access** -- understand where data lives and how to get it
2. **SLEAP Workflow** -- set up pose estimation (the most hands-on phase)
3. **Neural-Behavior Correlation** -- interpret the results from Notebook 08
4. **Troubleshooting** -- bookmark this for when things go wrong

## See Also

- [Reference: Artifacts](../reference/artifacts.md) -- every file the pipeline produces
- [Reference: Timebase](../reference/timebase.md) -- the canonical `nwb_seconds` contract
- [Reference: Notebook Map](../reference/notebook-map.md) -- what each notebook does, its inputs and outputs
