"""
TrellisContrastiveState — contrastive FlowEdit training state (independent of TrellisState).

Covers Stage-1 structure and Stage-2 SLAT contrastive intermediates (c_src/c_tgt, pos/neg, ...).
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from oreo.systems.base import BaseState
from .stage_latent import StageLatent


@dataclass
class TrellisContrastiveState(BaseState):
    """Contrastive FlowEdit state container (not a subclass of TrellisState)."""

    @dataclass
    class ViewsGenerated:
        """Generate perspective cache (override base class, add DINOv2 encoding).

        - image_tensor:    (B, V, H, W, C) Rendering results
        - image_condition: (B, S, C) DINOv2 patch token encoding
        """
        image_tensor: Any = None
        image_condition: Any = None

    @dataclass
    class ViewsEdited:
        """Edit result cache (overrides base class, adds DINOv2 encoding).

        - image_tensor:    (B, V, H, W, C) Edit results
        - image_condition: (B, S, C) DINOv2 patch token encoding
        """
        image_tensor: Any = None
        image_condition: Any = None

    @dataclass
    class Trackers:
        """Tracker container."""
        velocity: Any = None   # VelocityTracker
        guidance: Any = None   # List[StateTracker](FlowEdit intermediate step)

    # ============== batch key mapping ==============
    _CAMERA_KEYS: ClassVar[List[str]] = [
        "c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions",
    ]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]

    # ============== substate container ==============
    stage1: StageLatent = field(default_factory=StageLatent)
    stage2: StageLatent = field(default_factory=StageLatent)
    trackers: Trackers = field(default_factory=Trackers)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)
    dense_fallback: bool = False

    # ============================================================
    # attach_batch
    # ============================================================

    def attach_batch(
        self, batch: Dict[str, Any], pipeline: Any = None,
    ) -> "TrellisContrastiveState":
        """Extract and mount conditions, cameras, and other information from data batches."""
        # 1. views_conditioned
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])

        if "image_pils" in batch and pipeline is not None:
            cond = pipeline.prepare_image_conditions(batch["image_pils"])
            self.views_conditioned.cond_embed = cond["cond"]
            self.views_conditioned.uncond_embed = cond.get(
                "neg_cond", torch.zeros_like(cond["cond"]),
            )

        # 2. Guidance data
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]

        # 3. Camera parameters
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])

        return self

    # ============================================================
    # visual_packs — supply VisualIO.save_contrastive_train Use
    # ============================================================

    def visual_packs(self) -> List[Dict[str, Any]]:
        """
        Returns the visual packet list.

        each pack: {"suffix", "render_tensor", "edit_tensor", "trackers"}
        """
        packs = []
        if self.views_generated.image_tensor is not None:
            packs.append({
                "suffix": "_contrastive",
                "render_tensor": self.views_generated.image_tensor,   # (B,V,H,W,C)
                "edit_tensor": self.views_edited.image_tensor,       # (B,V,C,H,W)
                "trackers": self.trackers.guidance,
            })
        return packs
