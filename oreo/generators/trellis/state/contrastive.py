"""
TrellisContrastiveState - Contrastive FlowEdit Training state container

Stores all intermediate states in the comparison learning process and supports multiple stages (Stage 1: structure, Stage 2: SLAT):
- StageLatent: each stage latent Status (z0 / zt / z0_pred / z0_teacher / reg_loss)
- sparse structure coordinates (BaseState.coords)
- Camera parameters (cameras)
- condition information (views_conditioned): input image + DINOv2/CLIP Embed (inherited from BaseState)
- Generate results (views_generated): Teacher z₀ rendering + DINOv2 condition
- Edit results (views_edited): FlowEdit Edit + DINOv2 condition
- Trackers (velocity + guidance)
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from oreo.systems.base import BaseState
from .stage_latent import StageLatent


@dataclass
class TrellisContrastiveState(BaseState):
    """
    Contrastive FlowEdit Training state container.

    independent from TrellisState, not inherited TrellisState(Avoid field conflicts).
    inheritance BaseState get cameras / views_conditioned / extract_embeddings.

    Use StageLatent Unified management of two stages latent Status:
        stage1: structure flow model latent — Tensor (B, 8, 16, 16, 16)
        stage2: SLAT flow model latent — SparseTensor (N, C)

    Training process mounting convention:
        P0  → views_conditioned (attach_batch), cameras (attach_batch)
        P1  → stage2.z0 = teacher z₀, stage2.z0_teacher = same reference
              BaseState.coords = sparse structure coordinates (by dense sampling output)
        P3  → z0_hat_norm remain as local variable
        P4a → views_generated.image_tensor (rendering teacher z₀)
        P4b → views_edited.image_tensor (FlowEdit Edit results)
              trackers.guidance = FlowEdit middle step trackers
        P4c → views_generated.image_condition / views_edited.image_condition (DINOv2)
        P5  → trackers.velocity = VelocityTracker
    """

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
