"""
Guidance module.

provide retained FlowEdit Guidance, used for 3D Generate training:
- FlowEdit: Edit image → Calculate latent/pixel loss

Main interface:
- create_guidance(cfg, train_device, use_pp): create normal FlowEdit Guidance Example
- GuidanceResult: Unified return result format
- BaseGuidance: abstract base class

Equipment allocation:
- Guidance The model automatically runs on train_device + 1 of GPU on
- For example: training in cuda:0, then Guidance in cuda:1

all Guidance Use true uniformly Loss Mode:
- Pipeline Return Tracker(includes x0 / z_edits etc.)
- Pass Tracker.loss(src) Compute true loss
- direct loss.backward(), no need SpecifyGradient

Usage:
    from oreo.guidance import create_guidance

    guidance = create_guidance(cfg.guidance, accelerator.device)
    result = guidance.compute_guidance(
        comp_rgb, condition_images,
        guidance_cfg=cfg.shape.guidance,
    )
    result.loss.backward()  # Use uniformly result.loss
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
