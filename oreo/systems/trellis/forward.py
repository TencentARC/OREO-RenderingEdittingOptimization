"""
Shared Trellis forward, decode+render, and evaluation helpers.

Extracted for Dual-Stage entry and eval scripts to reuse without duplicating system.py logic.
"""

from __future__ import annotations

import torch
import ml_collections
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from accelerate import Accelerator

from oreo.generators.trellis.state import TrellisState
from oreo.generators.trellis.rollout import rollout_sparse, rollout_dense
from oreo.systems.base import EvalModeGuard
from oreo.systems.trellis.system import TrellisSystem
from oreo.systems.utils.visual import TrellisVisualIO


# =====================================================================
# Rendering tool functions - Mesh rendering
# =====================================================================

def decode_and_render_mesh(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # TrellisMeshRasterizer
    device: torch.device,
) -> Dict[str, Any]:
    """Decode latents -> mesh -> multi-view images."""
    # ---- Decode ----
    outputs = pipeline.sparse.decode(latents, formats=['mesh'])  # dict
    meshes = outputs['mesh']  # list[len=B] of MeshExtractResult

    # ---- Get camera parameters ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    batch_size, num_views = extr_all.shape[:2]  # (), ()

    # ---- Sample-by-sample, per-view rendering ----
    all_renders: Dict[str, List[torch.Tensor]] = {}

    for i, mesh in enumerate(meshes):
        view_renders: Dict[str, List[torch.Tensor]] = {}

        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)

            # Mesh renderer returns RenderOutput
            render_out = renderer.render(mesh, ext_iv, intr_iv)  # RenderOutput
            render_dict = {
                "color": render_out.color,  # (H,W,3)
                "normal": render_out.normal,  # (H,W,3)
                "depth": render_out.depth,  # (H,W)
                "mask": render_out.mask,  # (H,W)
            }

            for k, val in render_dict.items():
                if val is None:
                    continue
                view_renders.setdefault(k, []).append(val)  # (H,W,C) or (H,W)

        # stacked view dimensions: list[V] of (H,W,C) -> (V,H,W,C)
        for k, v_list in view_renders.items():
            stacked = torch.stack(v_list, dim=0)  # (V,H,W,C)
            all_renders.setdefault(k, []).append(stacked)

    # stack batch Dimensions: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {}
    for k, b_list in all_renders.items():
        result[k] = torch.stack(b_list, dim=0)  # (B,V,H,W,C)

    result["meshes"] = meshes  # Reserve mesh for export
    return result


# =====================================================================
# Rendering tool functions - Gaussian Splatting rendering
# =====================================================================

def decode_and_render_gs(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # GaussianRenderer
    device: torch.device,
) -> Dict[str, Any]:
    """Decode latents -> Gaussians -> multi-view images."""
    # ---- Decode ----
    outputs = pipeline.sparse.decode(latents, formats=['gaussian'])  # dict
    gaussians = outputs['gaussian']  # list[len=B] of Gaussian

    # ---- Get camera parameters ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    _, num_views = extr_all.shape[:2]  # (), ()

    # ---- Sample-by-sample, per-view rendering ----
    all_colors: List[torch.Tensor] = []

    for i, gs in enumerate(gaussians):
        view_colors: List[torch.Tensor] = []

        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)

            # GS renderer returns RenderOutput
            render_out = renderer.render(gs, ext_iv, intr_iv)  # RenderOutput
            color = render_out.color  # (H,W,3)
            view_colors.append(color)

        # stacked view dimensions: list[V] of (H,W,C) -> (V,H,W,C)
        stacked = torch.stack(view_colors, dim=0)  # (V,H,W,C)
        all_colors.append(stacked)

    # stack batch Dimensions: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {
        "color": torch.stack(all_colors, dim=0),  # (B,V,H,W,C)
        "gaussians": gaussians,  # Reserve GS for other purposes
    }
    return result


# =====================================================================
# forward propagation - shared Trellis forward logic
# =====================================================================

def trellis_forward(
    system: TrellisSystem,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """Dense sample -> rollout -> decode -> render shared forward."""
    pipeline = system.pipeline

    # Use a fixed seed when evaluating (to ensure reproducibility), and press step change
    _seed = int(cfg.seed) if not is_training else int(cfg.seed) + global_step

    # ---- 1. Dense Sampling(structure generation)----
    # if state.coords Already precomputed (e.g. teacher Reuse student of coords), then skip
    if state.coords is None:
        generator_dense = torch.Generator(device='cpu').manual_seed(_seed)
        with torch.no_grad():
            rollout_dense(state, cfg, system, device, generator=generator_dense)
        batch_size = state.stage1.z0.shape[0]
        state.coords = pipeline.dense.decode_to_coords(
            state.stage1.z0, batch_size=batch_size,
        )  # (B*T, 4)

    # ---- 2. Rollout: Perform sparse feature sampling (mount state.stage2.z0 and state.stage2.reg_loss)----
    generator = torch.Generator(device=device).manual_seed(_seed)

    rollout_sparse(
        state, cfg, system, device,
        generator=generator,
        is_training=is_training,
    )
    latents = state.stage2.z0  # SparseTensor (mounted on rollout)

    # release rollout The video memory fragments generated during the stage are decode make room
    torch.cuda.empty_cache()

    # ---- 3. Decode & rendering ----
    renderer_type = cfg.renderer.type
    renderer = system.renderers[renderer_type]  # from renderers dict Find

    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color": (B,V,H,W,C), "gaussians": list
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        render_out["color"] = render_out["normal"]

    state.views_generated.image_tensor = render_out["color"]  # (B,V,H,W,C) Mount generated graph for visualization

    return render_out


# =====================================================================
# Assessment - Reasoning and visual preservation
# =====================================================================

@torch.no_grad()
def evaluate(
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """Run inference and save visualization artifacts."""
    if eval_loader is None:
        return {}

    # ---- Fixed random seeds to ensure that each evaluation result is reproducible ----
    eval_seed = int(cfg.seed)
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)

    pipeline = system.pipeline
    # Get sampling parameters
    # evaluate Sampling parameters are not used directly (by trellis_forward obtained internally)

    # ---- create TrellisVisualIO for saving ----
    visual_io = TrellisVisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution, accelerator=accelerator)

    # =====================================================
    # Use EvalModeGuard Make sure all models are in evaluation mode
    # =====================================================
    pipe_models = pipeline.pipe.models
    # ★ Replace it with None when reasoning DDP original model of packaging
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models['slat_flow_model'],
        pipe_models['slat_decoder_mesh'],
        pipe_models['slat_decoder_gs'],
    ):
        # =====================================================
        # Iterate over the evaluation data set
        # =====================================================
        for batch_idx, batch in enumerate(eval_loader):
            # each batch Create independent states to avoid crossing batch Residue
            state = TrellisState()

            # ---- mount batch data ----
            state.attach_batch(batch, pipeline=pipeline)  # automatically from image_pils Generate conditional encoding and mount

            # ---- Use shared trellis_forward Perform forward pass ----
            render_out = trellis_forward(
                system, state, cfg, accelerator.device, global_step, is_training=False
            )

            # ---- Save results (all processes save the samples they processed)----
            renderer_type = cfg.renderer.type
            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=(renderer_type != "gs"),
            )

    return {"eval_done": 1.0}
