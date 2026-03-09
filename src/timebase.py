"""Timebase utilities and artifact writing helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


CANONICAL_TIMEBASE = "nwb_seconds"


def ensure_time_column(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Expected time column '{time_col}' in dataframe")
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    return df


def write_parquet_with_timebase(
    df: pd.DataFrame,
    path: Path,
    timebase: str = CANONICAL_TIMEBASE,
    provenance: Dict[str, Any] | None = None,
    required_columns: Iterable[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_write = df.copy()

    if required_columns:
        missing = [col for col in required_columns if col not in df_to_write.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} for artifact {path.name}")

    metadata = {"timebase": timebase}
    if provenance:
        metadata["provenance"] = provenance

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df_to_write, preserve_index=False)
        existing = table.schema.metadata or {}
        merged = {**{k.encode(): v.encode() for k, v in metadata.items() if isinstance(v, str)}, **existing}
        # Store provenance as JSON in metadata
        if provenance:
            merged[b"provenance"] = json.dumps(provenance).encode("utf-8")
        merged[b"timebase"] = timebase.encode("utf-8")
        table = table.replace_schema_metadata(merged)
        pq.write_table(table, path)
    except Exception:
        df_to_write.to_parquet(path, index=False)

    # Sidecar metadata for tool-agnostic access
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def write_npz_with_provenance(data: Dict[str, Any], path: Path, provenance: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump({"provenance": provenance, "timebase": CANONICAL_TIMEBASE}, f, indent=2)
    return path


def build_time_grid(start: float, end: float, bin_size_s: float) -> np.ndarray:
    if end <= start:
        return np.array([])
    n_bins = int(np.floor((end - start) / bin_size_s))
    return start + np.arange(n_bins) * bin_size_s


def bin_spike_times(spike_times: Dict[str, np.ndarray], time_grid: np.ndarray, bin_size_s: float) -> pd.DataFrame:
    if spike_times is None:
        return pd.DataFrame()
    counts = {}
    for unit_id, times in spike_times.items():
        if len(time_grid) == 0:
            counts[unit_id] = np.array([])
            continue
        bins = np.append(time_grid, time_grid[-1] + bin_size_s)
        counts[unit_id], _ = np.histogram(times, bins=bins)
    df = pd.DataFrame(counts)
    df.insert(0, "t", time_grid)
    return df


def bin_continuous_features(df: pd.DataFrame, time_grid: np.ndarray, agg: str = "mean") -> pd.DataFrame:
    if df is None or df.empty or len(time_grid) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["bin"] = np.searchsorted(time_grid, df["t"].to_numpy(), side="right") - 1
    df = df[df["bin"].between(0, len(time_grid) - 1)]
    grouped = df.groupby("bin").agg(agg)
    if "t" in grouped.columns:
        grouped = grouped.drop(columns=["t"])
    grouped = grouped.reindex(range(len(time_grid)), fill_value=np.nan)
    grouped.insert(0, "t", time_grid)
    return grouped.reset_index(drop=True)
