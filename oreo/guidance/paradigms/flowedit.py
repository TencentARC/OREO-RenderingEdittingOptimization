"""
FlowEdit Guidance: load Qwen-Image-Edit on the guidance device alongside Trellis training.

Init/Runtime split: __init__ loads weights; compute_guidance gets prompts / CFG / loss weights.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

import torch

from oreo.systems.utils import composite_alpha
from oreo.guidance.base import GuidanceResult, BaseGuidance
from oreo.guidance.pipeline_parallel import PipelineParallelMixin
from oreo.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
from oreo.guidance.pipelines.qwen_image_edit.trackers import StateTracker
from oreo.guidance.metric import METRIC_REGISTRY, BaseMetric


# =============================================================================
# data class
# =============================================================================

@dataclass
class EditOutput:
    """Single-image edit output container."""
    image: Image.Image                                  # edited image
    latent: torch.Tensor                                # [1, seq, C] edited latent
    tracker_tgt: Optional[StateTracker] = None          # tgt branch tracker
    tracker_src: Optional[StateTracker] = None          # src branch tracker
    pos_prompt_embeds: Optional[torch.Tensor] = None
    pos_prompt_embeds_mask: Optional[torch.Tensor] = None
    neg_prompt_embeds: Optional[torch.Tensor] = None
    neg_prompt_embeds_mask: Optional[torch.Tensor] = None


@dataclass
class FlowEditPipelineOutput:
    """Batch FlowEdit pipeline output."""
    edited_images: List[Image.Image]                    # [N] edited image
    edited_tensor: torch.Tensor                         # [N, C, H, W] edited image tensor(for visualization)
    latent_after: torch.Tensor                          # [N, seq, C] edited latent
    trackers_tgt: Optional[List[StateTracker]] = None   # [N] tgt Branch tracker list
    trackers_src: Optional[List[StateTracker]] = None   # [N] src Branch tracker list
    pos_prompt_embeds: Optional[torch.Tensor] = None
    pos_prompt_embeds_mask: Optional[torch.Tensor] = None
    neg_prompt_embeds: Optional[torch.Tensor] = None
    neg_prompt_embeds_mask: Optional[torch.Tensor] = None
    condition_tensor: Optional[torch.Tensor] = None


# =============================================================================
# FlowEditGuidance
# =============================================================================

class FlowEditGuidance(BaseGuidance):
    """
    True-loss FlowEdit (latent + pixel metrics).
    
    Init loads the edit model; runtime cfg drives prompts and loss aggregation.
    """

    loss_key = "flowedit"

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        """
        initialization FlowEdit Guidance(Only load the model).

        Args:
            guidance_cfg: Guidance Initialization configuration (cfg.guidance), including:
                - type: "flowedit"
                - model_path: model path
                - edit_resolution: VAE Encoding resolution
                - flowedit: FlowEdit Exclusive init Configuration
            train_device: Equipment used for training
        """
        super().__init__(guidance_cfg, train_device)

        self.flowedit_cfg = guidance_cfg[guidance_cfg.type]  # = guidance_cfg.flowedit
        model_path = guidance_cfg.model_path

        logging.info(f"[FlowEditGuidance] Loading pipeline on {self.device}...")
        logging.info(f"[FlowEditGuidance] Model: {model_path}")

        self.pipe = FlowEditFullPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # ---- Pixel Metrics(lazy loading: first time runtime Created on request) ----
        self.metrics: Dict[str, BaseMetric] = {}

        logging.info(f"[FlowEditGuidance] Ready.")

    # =========================================================================
    # Edit-only entrance (no gradient, none loss)
    # =========================================================================

    def edit(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        *,
        guidance_cfg: Any,
        **kwargs,
    ) -> GuidanceResult:
        """
        Just do FlowEdit Edit, not calculated loss, the whole process no_grad.

        Contrastive Training only requires edited images as c_tgt,
        No need Guidance internal CSD/Pixel loss.
        This method skips tracker record and loss Compute and save video memory.

        Args:
            comp_rgb: render image (B,V,H,W,C) or (B,V,C,H,W),float [0,1]
            condition_images: Conditional image list [len=B] of PIL.Image
            guidance_cfg: Runtime configuration (prompt / cfg scale Wait, no need loss field)

        Returns:
            GuidanceResult: loss=None, edited_imgs=(B,V,C,H,W)
        """
        comp_rgb, B, V, C, H, W, source_device = self._reshape_input(comp_rgb)
        with torch.no_grad():
            src_latent = self.encode_to_latent(comp_rgb)
            pipeline_output = self._run_pipeline(
                comp_rgb, condition_images,
                src_latent=src_latent,
                guidance_cfg=guidance_cfg,
                B=B, V=V,
            )
        edited_imgs = pipeline_output.edited_tensor.reshape(B, V, C, H, W).to(source_device)
        return GuidanceResult(
            loss=None,
            edited_imgs=edited_imgs,
            trackers=pipeline_output.trackers_tgt,
        )

    # =========================================================================
    # FlowEdit Leaflet editing
    # =========================================================================

    def _edit_single(
        self,
        rendered_pil: Image.Image,
        condition_pil: Image.Image,
        latent_before: torch.Tensor,
        flowedit_cfg: Any,
    ) -> EditOutput:
        """Execute single image FlowEdit Edit."""
        condition_pil = composite_alpha(condition_pil, tuple(flowedit_cfg.bg_color))
        latent_before = latent_before.to(dtype=torch.bfloat16)

        device = torch.device(self.pipe._execution_device)
        generator = torch.Generator(device=device).manual_seed(flowedit_cfg.seed)
        ic = self.flowedit_cfg

        # src branches only in loss Record when necessary,tgt Always record (visualization of editing process)
        loss_cfg = getattr(flowedit_cfg, 'loss', None)

        output = self.pipe(
            image=[rendered_pil, condition_pil],
            target_prompt=flowedit_cfg.target_prompt,
            source_prompt=flowedit_cfg.source_prompt,
            generator=generator,
            negative_prompt_src=flowedit_cfg.negative_prompt_src,
            negative_prompt_tgt=flowedit_cfg.negative_prompt_tgt,
            num_inference_steps=ic.steps,
            true_cfg_scale_src=flowedit_cfg.true_cfg_scale_src,
            true_cfg_scale_tgt=flowedit_cfg.true_cfg_scale_tgt,
            n_max=ic.n_max,
            noise_mode=ic.noise_mode,
            use_tgt_record=True,
            use_src_record=False,
            csd_pos_mode=ic.csd_pos_mode,
            csd_neg_mode=ic.csd_neg_mode,
            remove_tgt_neg=ic.remove_tgt_neg,
            src_latent=latent_before,
            height=self.edit_resolution,
            width=self.edit_resolution,
        )

        return EditOutput(
            image=output.images[0],
            latent=output.latents,
            tracker_tgt=output.tracker_tgt,
            tracker_src=output.tracker_src,
            pos_prompt_embeds=output.pos_prompt_embeds,
            pos_prompt_embeds_mask=output.pos_prompt_embeds_mask,
            neg_prompt_embeds=output.neg_prompt_embeds,
            neg_prompt_embeds_mask=output.neg_prompt_embeds_mask,
        )

    # =========================================================================
    # Pipeline Call (implement abstract method)
    # =========================================================================

    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        guidance_cfg: Any,
        B: int,
        V: int,
    ) -> FlowEditPipelineOutput:
        """
        execute FlowEdit Pipeline(No gradients, multi-step editing).

        Args:
            comp_rgb: [N, C, H, W] Rendering
            condition_images: [B] Conditional diagram
            src_latent: [N, seq, C] latent(Already detach)
            guidance_cfg: Runtime configuration (prompt / steps / cfg scales etc.)
            B, V: batch size and views

        Returns:
            FlowEditPipelineOutput
        """
        N = B * V
        H, W = comp_rgb.shape[2], comp_rgb.shape[3]

        edited_images, latents = [], []
        trackers_tgt: List[StateTracker] = []
        trackers_src: List[StateTracker] = []
        pos_prompt_embeds = None
        pos_prompt_embeds_mask = None
        neg_prompt_embeds = None
        neg_prompt_embeds_mask = None

        for b in range(B):
            for v in range(V):
                i = b * V + v

                # Use the incoming latent
                rendered_pil = self.tensor_to_pil(comp_rgb[i])
                output = self._edit_single(
                    rendered_pil,
                    condition_images[b],
                    src_latent[i:i+1],
                    flowedit_cfg=guidance_cfg,
                )

                edited_images.append(output.image)
                latents.append(output.latent)
                if output.tracker_tgt is not None:
                    trackers_tgt.append(output.tracker_tgt)
                if output.tracker_src is not None:
                    trackers_src.append(output.tracker_src)
                if pos_prompt_embeds is None:
                    pos_prompt_embeds = output.pos_prompt_embeds
                    pos_prompt_embeds_mask = output.pos_prompt_embeds_mask
                    neg_prompt_embeds = output.neg_prompt_embeds
                    neg_prompt_embeds_mask = output.neg_prompt_embeds_mask

        # Convert the edited image to tensor
        edited_tensor = self.pils_to_tensor(edited_images, (W, H))

        return FlowEditPipelineOutput(
            edited_images=edited_images,
            edited_tensor=edited_tensor,
            latent_after=torch.cat(latents, dim=0),
            trackers_tgt=trackers_tgt if trackers_tgt else None,
            trackers_src=trackers_src if trackers_src else None,
            pos_prompt_embeds=pos_prompt_embeds,
            pos_prompt_embeds_mask=pos_prompt_embeds_mask,
            neg_prompt_embeds=neg_prompt_embeds,
            neg_prompt_embeds_mask=neg_prompt_embeds_mask,
        )

    # =========================================================================
    # Loss Calculation (implementing abstract methods)
    # =========================================================================

    def _compute_branch_loss(
        self,
        src_latent: torch.Tensor,
        trackers: List[StateTracker],
        guidance_cfg: Any,
    ) -> torch.Tensor:
        """Calculate a single branch latent loss(for all view average)."""
        loss_cfg = guidance_cfg.loss
        losses = []
        for i, tracker in enumerate(trackers):
            single = src_latent[i:i+1]  # [1, seq, C]
            loss = tracker.loss(
                src=single,
                csd_weight=loss_cfg.latent_csd,
                mse_weight=loss_cfg.latent_mse,
                reduce=guidance_cfg.reduce_mode,
                ada=guidance_cfg.ada_normalize,
                eps=guidance_cfg.ada_eps,
            )
            losses.append(loss)
        return torch.stack(losses).mean()  # scalar

    def _compute_latent_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate Latent Loss(Branches are calculated independently and summed by weight).

        total = tgt_branch * tgt_loss + src_branch * src_loss

        Gradient path:loss → src_latent → VAE encoder → comp_rgb → 3D model

        Args:
            src_latent: [N, seq, C] Gradient latent
            pipeline_output: Pipeline Output (including Tracker)
            guidance_cfg: Runtime configuration

        Returns:
            (loss, loss_dict)
        """
        loss_cfg = guidance_cfg.loss
        total_loss = torch.tensor(0.0, device=src_latent.device, dtype=src_latent.dtype)
        loss_dict: Dict[str, torch.Tensor] = {}

        # ---- tgt branch ----
        if loss_cfg.tgt_branch > 0 and pipeline_output.trackers_tgt:
            loss_tgt = self._compute_branch_loss(
                src_latent, pipeline_output.trackers_tgt, guidance_cfg,
            )
            total_loss = total_loss + loss_cfg.tgt_branch * loss_tgt
            loss_dict["latent_tgt"] = (loss_cfg.tgt_branch * loss_tgt).detach()

        # ---- src branch ----
        if loss_cfg.src_branch > 0 and pipeline_output.trackers_src:
            loss_src = self._compute_branch_loss(
                src_latent, pipeline_output.trackers_src, guidance_cfg,
            )
            total_loss = total_loss + loss_cfg.src_branch * loss_src
            loss_dict["latent_src"] = (loss_cfg.src_branch * loss_src).detach()

        return total_loss, loss_dict

    def _compute_pixel_loss(
        self,
        comp_rgb: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate Pixel Loss(SSIM / LPIPS / DINO / CLIP).

        will render comp_rgb(with gradient) and edited image edited_tensor(no gradient)
        in pixels/The feature space is used for similarity comparison, and the gradient is directly returned to comp_rgb.

        ★ full runtime Configuration: The weight is declared in guidance_cfg.loss in,
           Models are loaded lazily on demand (for the first time weight > 0 created and cached).

        Gradient path:loss → metric model → comp_rgb → 3D model

        Args:
            comp_rgb: [N, C, H, W] Rendering (with gradient)
            pipeline_output: Pipeline Output (including edited_tensor)
            guidance_cfg: Runtime configuration (loss The weight is in guidance_cfg.loss middle)

        Returns:
            (loss, loss_dict)
        """
        loss_cfg = guidance_cfg.loss
        edited = pipeline_output.edited_tensor.detach()  # [N, C, H, W], no gradient

        total_loss = torch.tensor(0.0, device=comp_rgb.device, dtype=comp_rgb.dtype)
        loss_dict: Dict[str, torch.Tensor] = {}

        for name, cls in METRIC_REGISTRY.items():
            weight = loss_cfg[name]  # must be in runtime config Explicit declaration in
            if weight > 0:
                # Lazy loading: Create and cache when first used
                if name not in self.metrics:
                    self.metrics[name] = cls(weight=weight, device=self.device)
                    logging.info(f"[FlowEditGuidance] Lazy-loaded pixel metric: {name}")
                metric = self.metrics[name]
                pixel_loss = metric.compute(rendered=comp_rgb, target=edited)  # scalar
                total_loss = total_loss + weight * pixel_loss
                loss_dict[name] = (weight * pixel_loss).detach()

        return total_loss, loss_dict

    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        comp_rgb: torch.Tensor,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate the total Loss = Latent Loss + Pixel Loss.

        Args:
            src_latent: [N, seq, C] Gradient latent
            pipeline_output: Pipeline output
            comp_rgb: [N, C, H, W] Rendering (with gradient)
            guidance_cfg: Runtime configuration

        Returns:
            (loss, loss_dict)
        """
        # ---- Pixel Loss(Gradient path:loss → metric → comp_rgb) ----
        total_loss, loss_dict = self._compute_pixel_loss(
            comp_rgb, pipeline_output, guidance_cfg,
        )

        return total_loss, loss_dict

    # =========================================================================
    # Build return results
    # =========================================================================

    def _build_result(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        pipeline_output: FlowEditPipelineOutput,
        B: int, V: int, C: int, H: int, W: int,
        source_device: torch.device,
    ) -> GuidanceResult:
        """
        Build return results, add FlowEdit Exclusive fields.
        """
        edited_for_vis = pipeline_output.edited_tensor.reshape(B, V, C, H, W).to(source_device)

        return GuidanceResult(
            loss=loss,
            edited_imgs=edited_for_vis,
            loss_dict={self.loss_key: loss.detach(), **loss_dict},
            trackers=pipeline_output.trackers_tgt,  # FlowEdit Exclusive (tgt branch)
        )

    def set_accelerator(self, accelerator) -> None:
        pass

    def save_checkpoint(self, ckpt_dir) -> None:
        pass

    def load_checkpoint(self, ckpt_dir, **kwargs) -> None:
        pass

    def cleanup(self) -> None:
        """Release model memory"""
        logging.info("[FlowEditGuidance] Cleaning up...")
        for metric in self.metrics.values():
            metric.cleanup()
        self.metrics.clear()
        del self.pipe
        torch.cuda.empty_cache()
        logging.info("[FlowEditGuidance] Cleanup done.")


# =============================================================================
# FlowEditGuidancePP - Pipeline parallel version
# =============================================================================

class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
    """FlowEditGuidance + PipelineParallelMixin."""

    def __init__(self, guidance_cfg, train_device: torch.device):
        super().__init__(guidance_cfg, train_device)
        self._init_pipeline_parallel(num_streams=2)
        logging.info(f"[FlowEditGuidancePP] Pipeline parallelism enabled.")
