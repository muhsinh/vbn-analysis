# Getting Started

Welcome to the VBN Analysis Suite setup guide. This section walks you through everything you need to go from a fresh clone to running a full neural-behavior correlation analysis.

---

## Where to Start

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up your Python environment with conda, install SLEAP and AllenSDK, and verify everything works.

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run the pipeline end-to-end on a single session in under 15 minutes, or use mock mode to test without any data.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Full reference for every environment variable, Config dataclass field, and provenance tracking mechanism.

    [:octicons-arrow-right-24: Configuration Reference](configuration.md)

</div>

---

## Recommended Reading Order

1. **Installation**: get your environment running
2. **Quickstart**: run a test session (mock mode or real data)
3. **Configuration**: understand and customize every knob

!!! tip "Already comfortable with conda?"

    If you have a working Python 3.10 environment with the scientific stack, you can skip straight to the [Quickstart](quickstart.md) and install missing packages as needed. The `environment.yml` file is the canonical dependency list.

---

## Quick Reference

| Topic | Page | Key content |
|-------|------|-------------|
| Conda setup | [Installation](installation.md) | `conda env create`, SLEAP, AllenSDK |
| First run | [Quickstart](quickstart.md) | `sessions.csv`, mock mode, one-session walkthrough |
| Env vars | [Configuration](configuration.md) | `ACCESS_MODE`, `POSE_TOOL`, `BIN_SIZE_S`, etc. |
| Config object | [Configuration](configuration.md) | `Config` dataclass, `get_config()`, singleton pattern |
| Provenance | [Configuration](configuration.md) | `make_provenance()`, `config_snapshot.json` |
