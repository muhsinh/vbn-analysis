"""Pose estimation subpackage for VBN analysis.

Provides standardized interfaces for SLEAP and DeepLabCut pose estimation.
"""

from .schema import (
    PoseOutput,
    POSE_SCHEMA_COLUMNS,
    load_pose_outputs,
    save_pose_outputs,
    validate_pose_schema,
)
from .sleap import (
    run_sleap_inference,
    convert_sleap_to_standard,
    get_sleap_node_names,
)
from .dlc import (
    run_dlc_inference,
    convert_dlc_to_standard,
    get_dlc_bodyparts,
)

__all__ = [
    # Schema
    "PoseOutput",
    "POSE_SCHEMA_COLUMNS",
    "load_pose_outputs",
    "save_pose_outputs",
    "validate_pose_schema",
    # SLEAP
    "run_sleap_inference",
    "convert_sleap_to_standard",
    "get_sleap_node_names",
    # DLC
    "run_dlc_inference",
    "convert_dlc_to_standard",
    "get_dlc_bodyparts",
]
