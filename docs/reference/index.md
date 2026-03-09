# Reference

Technical reference documentation for the VBN Analysis Suite. These pages describe the precise contracts, schemas, and structures that the pipeline relies on.

---

## Reference Pages

<div class="vbn-nav-cards vbn-animate" markdown>
<a href="artifacts/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Artifacts</div>
<div class="vbn-nav-card-desc">Every file produced by the pipeline: format, schema, location, and how to read each one programmatically.</div>
</a>
<a href="timebase/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Timebase Contract</div>
<div class="vbn-nav-card-desc">The nwb_seconds canonical timebase: what it means, how each data source gets aligned, and how to verify alignment.</div>
</a>
<a href="notebook-map/" class="vbn-nav-card" markdown>
<div class="vbn-nav-card-title">Notebook Map</div>
<div class="vbn-nav-card-desc">Detailed reference for all 10 notebooks (00-09): purpose, inputs, outputs, configuration options, and the dependency graph.</div>
</a>
</div>

---

## Quick Links

- Looking for the schema of a specific parquet file? See [Artifacts](artifacts.md).
- Need to understand why timestamps might not match? See [Timebase Contract](timebase.md).
- Want to know which notebooks to re-run after changing config? See [Notebook Map](notebook-map.md).
