# Reference

Technical reference documentation for the VBN Analysis Suite. These pages describe the precise contracts, schemas, and structures that the pipeline relies on.

## Reference Pages

| Page | Description |
|---|---|
| **[Artifacts](artifacts.md)** | Every file produced by the pipeline: format, schema, location, and how to read each one programmatically. |
| **[Timebase Contract](timebase.md)** | The `nwb_seconds` canonical timebase: what it means, how each data source gets aligned, and how to verify alignment. |
| **[Notebook Map](notebook-map.md)** | Detailed reference for all 10 notebooks (00--09): purpose, inputs, outputs, configuration options, and the dependency graph. |

## Quick Links

- Looking for the schema of a specific parquet file? See [Artifacts](artifacts.md).
- Need to understand why timestamps might not match? See [Timebase Contract](timebase.md).
- Want to know which notebooks to re-run after changing config? See [Notebook Map](notebook-map.md).
