"""TrellisState — generation-process state (coords, multi-stage latents, embeddings, renders)."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional
import torch

from oreo.systems.base import BaseState
from .stage_latent import StageLatent


@dataclass
class TrellisState(BaseState):
    """Holds all intermediate tensors for one Trellis generation forward."""

    @dataclass
    class Guidance:
        """Guidance Result container."""
        loss: Any = None                  # Lord loss(can be directly backward)
        loss_dict: Any = None             # Segmentation loss Dictionary (for logging)

    @dataclass
    class ViewsEdited:
        """Edit results container (overrides base class, supports Hybrid Dual-pass rendering).

        - color_tensor / color_trackers: GS Color edit + trackers
        - normal_tensor / normal_trackers: Mesh Normal edit + trackers(Hybrid use)

        Only used in single channel mode color_tensor / color_trackers.
        attach_guidance_result Default write color_* field.
        """
        color_tensor: Any = None         # (B, V, C, H, W) GS Color edit
        color_trackers: Any = None       # List[StateTracker]
        normal_tensor: Any = None        # (B, V, C, H, W) Mesh Normal edit(Hybrid use)
        normal_trackers: Any = None      # List[StateTracker](Hybrid use)

    @dataclass
    class ViewsGenerated:
        """Generate perspective cache (overrides base class, supports Hybrid Dual-pass rendering).

        - image_tensor: single path GS Color render the result, or Hybrid mode GS Color.
        - normal_tensor: Hybrid mode Mesh Normal Rendering results.
        """
        image_tensor: Any = None   # (B, V, H, W, C) GS Color Rendering results
        normal_tensor: Any = None  # (B, V, H, W, C) Mesh Normal Rendering result (Hybrid use)

    # batch key -> state Mapping of attributes (class constants)
    _CAMERA_KEYS: ClassVar[List[str]] = ["c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions"]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]

    # ============== Trellis Dedicated substate container ==============
    stage1: StageLatent = field(default_factory=StageLatent)
    stage2: StageLatent = field(default_factory=StageLatent)
    guidance: Guidance = field(default_factory=Guidance)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)  # Cover BaseState
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)           # Cover BaseState

    def attach_batch(self, batch: Dict[str, Any], pipeline: Any = None) -> "TrellisState":
        """
        Extract and mount all data from data batch to state.

        Args:
            batch: DataLoader The returned batch data includes images, camera parameters, etc.
            pipeline: Required for calling prepare_image_conditions from image_pils Generate conditional embeddings.

        Returns:
            self: Support chain calls
        """
        # ---- 1. views_conditioned(image, path, embed) ----
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])

        # from image_pils Generate conditional encoding
        if "image_pils" in batch and pipeline is not None:
            # pipeline.prepare_image_conditions Responsible for preprocessing images (such as back removal, scaling) and calculating Embedding
            cond = pipeline.prepare_image_conditions(batch["image_pils"])
            self.views_conditioned.cond_embed = cond["cond"]
            # if not neg_cond, using all-zero tensors as unconditional embeddings
            self.views_conditioned.uncond_embed = cond.get("neg_cond", torch.zeros_like(cond["cond"]))

        # ---- 2. guidance signal (Guidance data) ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]

        # ---- 3. Camera parameters ----
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])

        return self

    def attach_guidance_result(self, guidance_result: Any) -> "TrellisState":
        """
        will GuidanceResult mount to state.

        Default write color_tensor / color_trackers(GS Color road).
        Hybrid mode,mesh pass of edit Moved by the caller to normal_tensor / normal_trackers.

        Args:
            guidance_result: GuidanceResult object, including loss and optional edited_imgs.

        Returns:
            self: Support chain calls
        """
        # Loss mount to guidance
        self.guidance.loss = guidance_result.loss
        self.guidance.loss_dict = guidance_result.loss_dict
        # Edited image and trackers write color_* buffer (default)
        self.views_edited.color_tensor = guidance_result.edited_imgs
        self.views_edited.color_trackers = guidance_result.trackers
        return self

    def prepare_for_vjp(self) -> "TrellisState":
        """
        Phase 2→3 Transition Cleanup: Release decode/render intermediate product, reduced VJP Stage memory water level.

        VJP Just need:
        - stage2.z0.coords(via .replace() build x_t)
        - views_conditioned(Conditional encoding)
        The rest decode products,spatial_cache All can be released.

        Returns:
            self: Support chain calls
        """
        # SparseTensor._spatial_cache storage neighbor maps / window partition indices Wait
        # Use .clear() do in-place Clear:replace() will share the same dict quote,
        # in-place In order to allow all sharing parties to release the cache
        if self.stage2.z0 is not None:
            self.stage2.z0._spatial_cache.clear()
        torch.cuda.empty_cache()
        return self