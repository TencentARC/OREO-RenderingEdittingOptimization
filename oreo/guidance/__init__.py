"""
Guidance package for OREO: FlowEdit-based image editing losses for 3D generation training.

Main entry: create_guidance(cfg, train_device) -> FlowEdit (optional pipeline-parallel).
"""

from oreo.guidance.base import (
    create_guidance,
    GuidanceResult,
    BaseGuidance,
)
from oreo.guidance.pipeline_parallel import PipelineParallelMixin

__all__ = [
    "create_guidance",
    "GuidanceResult",
    "BaseGuidance",
    "PipelineParallelMixin",
]
