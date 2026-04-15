"""Shared type aliases for the VBN analysis suite.

Only types that are genuinely used across two or more modules live here.
Module-local types stay in their own files.
"""
from __future__ import annotations

import numpy as np
from typing import TypedDict


# Spike times: maps unit_id (str) → 1-D array of spike times in NWB seconds.
# Used in: neural_events, cross_correlation, viz, modeling, timebase.
SpikeTimesDict = dict[str, np.ndarray]


class Provenance(TypedDict):
    """Metadata written into every parquet / npz sidecar.

    Produced by config.make_provenance(); consumed by timebase writers,
    io_nwb save helpers, io_video, io_sessions, reports, features_pose,
    and pose_inference.
    """
    session_id: int | None
    code_version: str
    created_at: str
    alignment_method: str
