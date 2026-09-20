"""
Guidance Module infrastructure.

Provided by:
- GuidanceResult: Unified return format
- BaseGuidance: Abstract base class, containing shared logic
- create_guidance(): Factory function, create retained FlowEdit Guidance

Equipment allocation:
- Guidance By default the model runs on training equipment + 1 of GPU on
- For example: training in cuda:0, then Guidance in cuda:1

unified framework (true Loss mode):
- all Guidance Subclasses follow the same compute_guidance Process:
    1. format conversion
    2. encode to latent(once, with gradient)
    3. call Pipeline(no gradient)
    4. Pass Tracker.loss() Compute true loss(can be directly backward)

★ The configuration is divided into two layers (Init / Runtime):
- Init configure(cfg.guidance): Model loading parameters, passed to __init__, shared at all stages
- Runtime configure(cfg.stage.guidance): prompt / loss weights etc., each time compute_guidance incoming
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, List, Dict, Tuple
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from oreo.systems.base import compute_guidance_device

if TYPE_CHECKING:
    from oreo.guidance.pipelines.qwen_image_edit.trackers import StateTracker


# =====================================================================
# GuidanceResult - Unified return format
# =====================================================================

@dataclass
class GuidanceResult:
    """
    Guidance Calculation result (generic format).

    Attributes:
        loss: Lord loss(can be directly backward)
        edited_imgs: edited image (B,V,C,H,W),FlowEdit dedicated
        loss_dict: Segmentation loss Dictionary, used for logging
        trackers: FlowEdit List of intermediate state trackers (for multi-step supervision)
    """
    loss: Optional[torch.Tensor] = None                                 # Lord loss(edit() mode is None)
    edited_imgs: Optional[torch.Tensor] = None                          # (B,V,C,H,W) FlowEdit dedicated
    loss_dict: Optional[Dict[str, torch.Tensor]] = field(default=None)  # Segmentation loss
    trackers: Optional[List["StateTracker"]] = None             # FlowEdit dedicated


# =====================================================================
# BaseGuidance - Abstract base class (contains shared logic)
# =====================================================================

class BaseGuidance(ABC):
    """
    Guidance abstract base class(true Loss mode).

    ★ The configuration is divided into two levels:
    - __init__(guidance_cfg, train_device): Only load the model,guidance_cfg Contains only init parameters
    - compute_guidance(..., guidance_cfg=...): Runtime parameters passed guidance_cfg incoming

    Provide sharing functions:
    - Device initialization
    - Format conversion (tensor_to_pil, pils_to_tensor)
    - Input format processing (_reshape_input)
    - Latent encoding(encode_to_latent)
    - unified compute_guidance template method

    Subclasses require:
    - settings loss_key class attributes
    - initialization self.pipe(Pipeline Example)
    - realize _run_pipeline() call pipeline
    - realize _compute_loss() Compute true loss(via Tracker.loss())
    - Optional override _build_result() Customized return results
    """

    # Class attributes: used for loss_dict of key Name (can be overridden by subclasses)
    loss_key: str = "guidance"

    # Instance properties (subclass initialization)
    device: torch.device
    train_device: torch.device
    pipe: Any  # Pipeline Example
    edit_resolution: int

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        """
        Base class initialization (only loads the model, does not bind runtime parameters).

        Args:
            guidance_cfg: Guidance Initialization configuration (cfg.guidance), including:
                - model_path: model path
                - edit_resolution: VAE Encoding resolution
                - type: Guidance Type
            train_device: training equipment
        """
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        self.edit_resolution = guidance_cfg.edit_resolution

        # Subclasses need to be in __init__ Medium initialization self.pipe

    # =========================================================================
    # Format conversion (sharing)
    # =========================================================================

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Tensor turn PIL images.

        Args:
            tensor: [C, H, W] float [0,1]

        Returns:
            PIL.Image
        """
        arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr.transpose(1, 2, 0))

    def pils_to_tensor(self, pils: List[Image.Image], size: Tuple[int, int]) -> torch.Tensor:
        """
        PIL Image list transfer Tensor.

        Args:
            pils: List[PIL.Image]
            size: (W, H) target size

        Returns:
            [N, C, H, W] tensor
        """
        tensors = [TF.to_tensor(p.resize(size, Image.LANCZOS)) for p in pils]
        return torch.stack(tensors).to(self.device)

    # =========================================================================
    # Input format processing (shared)
    # =========================================================================

    def _reshape_input(self, comp_rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int, int, torch.device]:
        """
        Uniform handling of input formats.

        Args:
            comp_rgb: (B, V, H, W, C) or (B, V, C, H, W) or (N, C, H, W)

        Returns:
            (comp_rgb, B, V, C, H, W, source_device)
            Among them comp_rgb converted to [N, C, H, W] and move to self.device
        """
        source_device = comp_rgb.device

        if comp_rgb.dim() == 5:
            if comp_rgb.shape[-1] == 3:
                # (B, V, H, W, C) → (N, C, H, W)
                B, V, H, W, C = comp_rgb.shape
                comp_rgb = comp_rgb.permute(0, 1, 4, 2, 3).reshape(B * V, C, H, W)
            else:
                # (B, V, C, H, W) → (N, C, H, W)
                B, V, C, H, W = comp_rgb.shape
                comp_rgb = comp_rgb.reshape(B * V, C, H, W)
        else:
            # (N, C, H, W)
            B, V = 1, 1
            _, C, H, W = comp_rgb.shape

        comp_rgb = comp_rgb.to(self.device)

        return comp_rgb, B, V, C, H, W, source_device

    # =========================================================================
    # Latent Coding (sharing)
    # =========================================================================

    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        encode image to packed latent(differentiable).

        Args:
            images: [B, C, H, W] float [0,1]

        Returns:
            [B, seq, C_lat] packed latent
        """
        B = images.shape[0]

        # Resize to editing resolution
        resized = F.interpolate(
            images,
            size=(self.edit_resolution, self.edit_resolution),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )  # [B, C, edit_res, edit_res]

        # [0,1] → [-1,1]
        normalized = resized * 2 - 1  # [B, C, H, W]
        images_5d = normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # [B, C, 1, H, W]

        # VAE encode(differentiable)
        latent_5d = self.pipe._encode_vae_image_differentiable(images_5d)  # [B, C_lat, 1, H_lat, W_lat]

        # Pack
        _, C_lat, _, H_lat, W_lat = latent_5d.shape
        latent = self.pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)  # [B, seq, C_lat]

        return latent.to(dtype=images.dtype)

    # =========================================================================
    # Abstract method (subclasses must implement it)
    # =========================================================================

    @abstractmethod
    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        guidance_cfg: Any,
        B: int,
        V: int,
    ) -> Any:
        """
        call Pipeline(no gradient).

        Args:
            comp_rgb: [N, C, H, W] Rendering
            condition_images: [B] Conditional diagram
            src_latent: [N, seq, C] latent(Already detach)
            guidance_cfg: Runtime configuration (prompt / steps / cfg scales etc.)
            B, V: batch size and views

        Returns:
            Pipeline output(FlowEditPipelineOutput)
        """
        pass

    @abstractmethod
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: Any,
        comp_rgb: torch.Tensor,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss(True loss, can be directly backward).

        Args:
            src_latent: [N, seq, C] Gradient latent
            pipeline_output: Pipeline output (contains Tracker)
            comp_rgb: [N, C, H, W] Rendering (required by some subclasses)
            guidance_cfg: Runtime configuration (loss weight / reduce strategy, etc.)

        Returns:
            (loss, loss_dict)
            - loss: scalar loss(There is a gradient, which can be backward)
            - loss_dict: various items loss Dictionary (for logging)
        """
        pass

    # =========================================================================
    # Main entrance (template method)
    # =========================================================================

    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        *,
        guidance_cfg: Any,
        **kwargs,
    ) -> GuidanceResult:
        """
        Calculate Guidance loss(template method, true Loss mode).

        Unified process:
            1. format conversion
            2. encode to latent(once, with gradient)
            3. call Pipeline(no gradient, passed in guidance_cfg)
            4. Pass Tracker.loss() Compute true loss(incoming guidance_cfg)

        If you only need the edited image without loss, please use FlowEditGuidance.edit().

        Args:
            comp_rgb: render image (B,V,H,W,C) or (B,V,C,H,W),float [0,1]
            condition_images: Conditional image list [len=B] of PIL.Image
            guidance_cfg: Runtime configuration (per-stage), must be passed in.
                          trellis2: cfg.shape.guidance / cfg.tex.guidance
                          trellis:  cfg.train.guidance
            **kwargs: Additional parameters (such as rank)

        Returns:
            GuidanceResult: contains loss and optional edited_imgs
        """
        # 1. format conversion
        comp_rgb, B, V, C, H, W, source_device = self._reshape_input(comp_rgb)

        # 2. encode to latent(No gradient, just as FlowEdit starting point)
        with torch.no_grad():
            src_latent = self.encode_to_latent(comp_rgb)  # [N, seq, C_lat]

        # 3. call Pipeline(no gradient)
        with torch.no_grad():
            pipeline_output = self._run_pipeline(
                comp_rgb,
                condition_images,
                src_latent=src_latent.detach(),
                guidance_cfg=guidance_cfg,
                B=B, V=V,
            )

        # 4. Calculate loss(True loss, with gradient)
        loss, loss_dict = self._compute_loss(
            src_latent,
            pipeline_output,
            comp_rgb,
            guidance_cfg=guidance_cfg,
        )

        # 5. Move to training device
        loss = loss.to(self.train_device)

        # 6. Return results (subclasses can pass _build_result Customized)
        return self._build_result(
            loss, loss_dict, pipeline_output, B, V, C, H, W, source_device
        )

    def _build_result(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        pipeline_output: Any,
        B: int, V: int, C: int, H: int, W: int,
        source_device: torch.device,
    ) -> GuidanceResult:
        """
        Build returns results. Subclasses can override to add additional fields.

        Args:
            loss: Lord loss
            loss_dict: various items loss dictionary
            pipeline_output: Pipeline output
            B, V, C, H, W: Dimension information
            source_device: original equipment

        Returns:
            GuidanceResult
        """
        return GuidanceResult(
            loss=loss,
            edited_imgs=None,
            loss_dict={self.loss_key: loss.detach(), **loss_dict},
            trackers=None,
        )

    # =========================================================================
    # Helper methods
    # =========================================================================

    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get each item loss weight.

        Return by default {loss_key: 1.0}.
        FlowEdit This method can be overridden to return the segment loss weight.

        Returns:
            Dict[str, float]: {loss_name: weight}
        """
        return {self.loss_key: 1.0}

    def cleanup(self) -> None:
        """Release resources."""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()


# =====================================================================
# Factory
# =====================================================================

def create_guidance(
    guidance_cfg: Any,
    train_device: torch.device,
    use_pp: bool = False,
) -> BaseGuidance:
    """Create the retained ordinary Qwen FlowEdit guidance.

    use_pp selects the retained pipeline-parallel FlowEdit variant.
    Every other type is rejected so removed experimental stacks cannot be
    reintroduced through a compatibility branch.
    """
    paradigm = guidance_cfg.type
    if paradigm != "flowedit":
        raise ValueError(
            f"Unknown guidance type: {paradigm!r}; only 'flowedit' is retained"
        )

    if use_pp:
        from oreo.guidance.paradigms.flowedit import FlowEditGuidancePP
        return FlowEditGuidancePP(guidance_cfg, train_device)

    from oreo.guidance.paradigms.flowedit import FlowEditGuidance
    return FlowEditGuidance(guidance_cfg, train_device)
