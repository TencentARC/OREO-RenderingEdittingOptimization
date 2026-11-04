"""
Adapter over the reference TrellisImageTo3DPipeline, unifying SparseTensor I/O.

Depends on _reference_codes/TRELLIS and exposes the dense / sparse stage API
expected by oreo.systems.trellis.
"""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.modules.sparse import SparseTensor
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler

from oreo.generators.trellis.scheduler import TrellisFlowScheduler



def build_pipeline_from_reference(cfg: Any, accelerator: Any, device: Optional[torch.device] = None) -> Any:
    """Build a TrellisRefAdapter from config + Accelerator."""
    project_root = torch.__file__  # place a place for mypy, actually reset below
    # will _reference_codes/TRELLIS Join sys.path
    project_root = sys.argv[0]  # Only occupy space to prevent static check alarms
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
    if trellis_ref_root not in sys.path:
        sys.path.insert(0, trellis_ref_root)
    triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
    if triposf_ref_root not in sys.path:
        sys.path.insert(0, triposf_ref_root)


    # Set default CUDA Device (supports passing in custom devices for pipeline parallelism)
    if device is None:
        device = accelerator.device
    if device.type == "cuda":
        # Make sure the device has a specific index
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    pipe_raw = TrellisImageTo3DPipeline.from_pretrained(cfg.pretrained.model)
    pipe_raw.to(device)
    # Note: no longer called pipe_raw.cuda(), as it overrides the device setting to GPU 0
    os.environ["TRELLIS_VERBOSE"] = "1" if bool(getattr(cfg, "verbose", False)) else "0"

    return TrellisRefAdapter(pipe_raw, FlowEulerSampler=FlowEulerSampler)


# =====================================================================
# Sharing tools
# =====================================================================

def _scale_timesteps(
    timesteps: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Scale timesteps from [0,1] to the model's expected [0,1000] range (float / 0-dim / batched)."""
    if torch.is_tensor(timesteps):
        if timesteps.dim() == 0:
            return torch.full(
                (batch_size,), float(timesteps.item()) * 1000,
                device=device, dtype=torch.float32,
            )  # (B,)
        return timesteps * 1000  # (B,)
    return torch.full(
        (batch_size,), float(timesteps) * 1000,
        device=device, dtype=torch.float32,
    )  # (B,)


# =====================================================================
# DenseStage — Stage 1 (sparse_structure_flow_model)
# =====================================================================

class DenseStage:
    """Stage-1 ops on dense tensors (B, C, R, R, R)."""

    def __init__(self, adapter: "TrellisRefAdapter"):
        self._adapter = adapter

    @property
    def pipe(self):
        return self._adapter.pipe

    def get_runtime_params(self) -> tuple[int, float, float, float, float]:
        """
        Return (steps, guidance, rescale_t, cfg_min, cfg_max).
        """
        ss_params = self.pipe.sparse_structure_sampler_params
        return (
            int(ss_params["steps"]),
            float(ss_params["cfg_strength"]),
            float(ss_params["rescale_t"]),
            float(ss_params["cfg_interval"][0]),
            float(ss_params["cfg_interval"][1]),
        )

    def resolve_flow_module(self) -> Any:
        """get sparse_structure_flow_model original model (removed DDP packaging)."""
        model = self.pipe.models["sparse_structure_flow_model"]
        return model.module if hasattr(model, "module") else model

    def decode_to_coords(
        self,
        z_s: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        will Dense latent Decoded into placeholder coordinates.

        z_s (B, C, R, R, R) → decoder → threshold > 0 → coords (B*T, 4) int32

        Args:
            z_s: Dense rollout Output latent (B, C, R, R, R)
            batch_size: used for batch index expansion

        Returns:
            coords: (B*T, 4) int32, listed as [batch_idx, x, y, z]
        """
        decoder = self.pipe.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()  # (T, 4)

        if batch_size <= 1:
            return coords

        coords_list = []
        for b in range(batch_size):
            cb = coords.clone()  # (T, 4)
            cb[:, 0] = b
            coords_list.append(cb)
        return torch.cat(coords_list, dim=0)  # (B*T, 4)

    def init_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """generate Dense initial noise (B, C, R, R, R).

        with native sample_sparse_structure Be consistent: first in CPU Noise is generated on
        Again .to(device), to ensure that the values under the same seed are exactly the same.
        """
        flow_model = self.resolve_flow_module()
        reso = flow_model.resolution  # 16
        in_ch = flow_model.in_channels  # 8
        noise = torch.randn(
            batch_size, in_ch, reso, reso, reso,
            dtype=torch.float32,
            generator=generator,
        )  # CPU
        return noise.to(self.pipe.device)  # (B, C, R, R, R)

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: Any,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """Dense single step v predict, return (B, C, R, R, R)."""
        model = self.pipe.models["sparse_structure_flow_model"]
        t_scaled = _scale_timesteps(timesteps, x_t.shape[0], x_t.device)
        return model(x_t, t_scaled, cond_embeddings)  # (B, C, R, R, R)

    def scheduler(
        self,
        steps: int,
        rescale_t: float,
    ) -> Tuple[np.ndarray, list]:
        """
        Time step sequence generation.

        Returns:
            (t_seq, t_pairs): t_seq for (steps+1,),t_pairs for [(t, t_prev), ...]
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        return t_seq, t_pairs


# =====================================================================
# SparseStage — Stage 2 (slat_flow_model)
# =====================================================================

class SparseStage:
    """Stage-2 ops on SparseTensor (feats (N,C), coords (N,4))."""

    def __init__(self, adapter: "TrellisRefAdapter"):
        self._adapter = adapter

    @property
    def pipe(self):
        return self._adapter.pipe

    def get_runtime_params(self) -> tuple[int, float, float, float, float, float]:
        """
        Return (steps, guidance, rescale_t, cfg_min, cfg_max, mc_threshold).
        """
        slat_params = self.pipe.slat_sampler_params
        return (
            int(slat_params["steps"]),
            float(slat_params["cfg_strength"]),
            float(slat_params["rescale_t"]),
            float(slat_params["cfg_interval"][0]),
            float(slat_params["cfg_interval"][1]),
            float(slat_params.get("mc_threshold", 0.0)),
        )

    def resolve_flow_module(self) -> Any:
        """get slat_flow_model original model (removed DDP packaging)."""
        model = self.pipe.models["slat_flow_model"]
        return model.module if hasattr(model, "module") else model

    def init_latents(
        self,
        coords: torch.Tensor,
        in_channels: int,
        generator: Optional[torch.Generator] = None,
    ) -> Any:
        """Generate initial SparseTensor latent (feats: (N, C))."""
        feats = torch.randn(
            coords.shape[0],
            int(in_channels),
            device=coords.device,
            dtype=torch.float32,
            generator=generator,
        )  # (N, C)
        return SparseTensor(coords=coords, feats=feats)

    def sampling_step(
        self,
        x_t_sparse: Any,
        timesteps: Any,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> Any:
        """
        Sparse single step v predict, return SparseTensor.

        model expectations t ∈ [0, 1000], external input [0, 1], internal scaling.
        CFG Handled externally.
        """
        model = self.pipe.models["slat_flow_model"]
        t_scaled = _scale_timesteps(timesteps, cond_embeddings.shape[0], x_t_sparse.device)
        return model(x_t_sparse, t_scaled, cond_embeddings)

    def scheduler(self) -> TrellisFlowScheduler:
        """Return Trellis Flow Matching scheduler."""
        return TrellisFlowScheduler()

    def decode(
        self,
        latents: Any,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Unified packaging decode_slat."""
        fmt = formats if formats is not None else ["mesh"]
        return self.pipe.decode_slat(latents, formats=fmt)


# =====================================================================
# TrellisRefAdapter — combination + general tools
# =====================================================================

class TrellisRefAdapter:
    """
    Wraps TrellisImageTo3DPipeline with .dense and .sparse sub-objects.
    
    Usage:
        pipeline.dense.sampling_step(...)
        pipeline.sparse.sampling_step(...)
    """

    def __init__(self, pipe_raw: Any, FlowEulerSampler: Any):
        self.pipe = pipe_raw
        self.FlowEulerSampler = FlowEulerSampler
        self.dense = DenseStage(self)
        self.sparse = SparseStage(self)

    # === Compatible entrance ===

    def get_sampler_runtime_params(self) -> tuple[int, float, int, float, float, float]:
        """
        Compatible entry: return (ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold).
        """
        ss_steps, ss_guidance, _, _, _ = self.dense.get_runtime_params()
        slat_steps, slat_guidance, slat_rescale_t, _, _, slat_mc_threshold = self.sparse.get_runtime_params()
        return ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold

    # === Conditions / image ===

    def prepare_image_conditions(self, images: List[Any]) -> Dict[str, Any]:
        """Preprocess the image and generate cond/neg_cond."""
        images_proc = [self.pipe.preprocess_image(img) for img in images]
        cond_dict = self.pipe.get_cond(images_proc)

        cond = cond_dict.get("cond")
        if cond is None:
            raise ValueError("prepare_image_conditions: get_cond returned cond is empty.")
        neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))  # (B, S, C)

        return {"cond": cond, "neg_cond": neg_cond}

    # === Mesh Export ===

    def export_mesh_obj(self, mesh: Any, out_path: str) -> None:
        """Export MeshExtractResult for OBJ."""
        if mesh is None:
            return
        mesh_np = trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh_np.export(out_path)

    # === LoRA control ===

    @contextmanager
    def disable_lora_context(self, model_key: str = "slat_flow_model"):
        """Temporarily disable the specified model LoRA adapter."""
        raw = self.pipe.models[model_key]
        model = raw.module if hasattr(raw, "module") else raw
        if hasattr(model, 'disable_adapters'):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            yield
