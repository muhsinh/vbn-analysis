"""Reporting, artifact registry, and logging utilities."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config import get_config

_LOGGERS: Dict[int, logging.Logger] = {}


def setup_session_logger(session_id: int) -> logging.Logger:
    if session_id in _LOGGERS:
        return _LOGGERS[session_id]

    cfg = get_config()
    cfg.ensure_dirs()
    log_path = cfg.outputs_dir / "reports" / "logs" / f"session_{session_id}.log"

    logger = logging.getLogger(f"session_{session_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[session_id] = logger
    return logger


def build_artifact_registry(outputs_dir: Path | None = None) -> pd.DataFrame:
    cfg = get_config()
    if outputs_dir is None:
        outputs_dir = cfg.outputs_dir

    records: List[Dict[str, Any]] = []
    for path in outputs_dir.rglob("*"):
        if path.is_dir():
            continue
        step = path.parent.name
        match = re.search(r"session_(\d+)", path.name)
        session_id = int(match.group(1)) if match else None
        records.append(
            {
                "step": step,
                "artifact_path": str(path),
                "exists": path.exists(),
                "last_modified": pd.Timestamp.fromtimestamp(path.stat().st_mtime),
                "session_id": session_id,
                "notes": "",
            }
        )

    df = pd.DataFrame(records)
    return df


def write_artifact_registry(outputs_dir: Path | None = None) -> Path:
    cfg = get_config()
    if outputs_dir is None:
        outputs_dir = cfg.outputs_dir
    df = build_artifact_registry(outputs_dir)
    registry_path = outputs_dir / "reports" / "artifact_registry.parquet"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    from timebase import write_parquet_with_timebase
    from config import make_provenance
    write_parquet_with_timebase(
        df,
        registry_path,
        provenance=make_provenance(None, "nwb"),
    )
    return registry_path


def parse_notebook_header(nb_path: Path) -> Dict[str, Any]:
    try:
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        if not nb.cells:
            return {}
        first = nb.cells[0]
        if first.cell_type != "markdown":
            return {}
        content = first.source
    except Exception:
        # fallback: try json
        try:
            data = json.loads(nb_path.read_text(encoding="utf-8"))
            cells = data.get("cells", [])
            if not cells:
                return {}
            content = "".join(cells[0].get("source", []))
        except Exception:
            return {}

    # Extract YAML frontmatter block even if it is wrapped in an HTML comment.
    # We look for the first two '---' delimiters on their own line.
    try:
        import yaml
    except ImportError:
        return {}

    lines = content.splitlines()
    delim_idxs = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(delim_idxs) < 2:
        return {}
    start, end = delim_idxs[0], delim_idxs[1]
    if end <= start + 1:
        return {}
    header_text = "\n".join(lines[start + 1 : end])
    try:
        return yaml.safe_load(header_text) or {}
    except Exception:
        return {}


def validate_prerequisites(required_paths: List[str], base_dir: Path | None = None) -> List[str]:
    if base_dir is None:
        base_dir = get_config().outputs_dir
    missing = []
    for rel in required_paths:
        path = Path(rel)
        if not path.is_absolute():
            path = base_dir.parent / rel
        if not path.exists():
            missing.append(str(path))
    return missing


def validate_artifact_schema(path: Path, required_columns: List[str]) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return False
    except Exception:
        return False
    # metadata sidecar
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("timebase") != "nwb_seconds":
        return False
    if "provenance" not in meta:
        return False
    return True


def write_run_summary(summary: pd.DataFrame, outputs_dir: Path | None = None) -> Path:
    cfg = get_config()
    if outputs_dir is None:
        outputs_dir = cfg.outputs_dir
    path = outputs_dir / "reports" / "run_summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    from timebase import write_parquet_with_timebase
    from config import make_provenance
    write_parquet_with_timebase(
        summary,
        path,
        provenance=make_provenance(None, "nwb"),
    )
    return path
