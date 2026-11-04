#!/usr/bin/env python3
"""
TRELLIS training adapter: LoRA on SparseLinear, optimizers, and train strategies.

Core pieces:
- SparseLinearLora: LoRA only on feats path
- register_sparse_linear_with_peft: register LoRA into PEFT dispatch
- build_optimizer_for_sparse / dense: stage-wise optimizers
- TrellisFullFinetuneStrategy / LoRA / Frozen
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import torch
import torch.nn as nn

from trellis.modules import sparse as sp  # noqa: E402
from trellis.modules.sparse.linear import SparseLinear  # noqa: E402

from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402
from peft.tuners.lora.layer import LoraLayer  # noqa: E402
import torch.optim as optim


class SparseLinearLora(nn.Module, LoraLayer):
    """LoRA wrapper for TRELLIS SparseLinear: forward takes SparseTensor, applies LoRA on feats only, then replaces feats."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def forward(self, x: sp.SparseTensor, *args: Any, **kwargs: Any) -> sp.SparseTensor:
        # Baseline output (hold SparseTensor structure)
        if self.disable_adapters or self.merged:
            return self.get_base_layer()(x, *args, **kwargs)

        out: sp.SparseTensor = self.get_base_layer()(x, *args, **kwargs)  # SparseTensor feats shape (N, Cout)
        out_dtype = out.feats.dtype

        # chase adapter Overlay LoRA Contribution, only affects feats
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            feats = x.feats.to(lora_A.weight.dtype)  # (N, Cin)
            lora_feats = lora_B(lora_A(dropout(feats))) * scaling  # (N, Cout)
            out = out.replace(out.feats + lora_feats.to(out_dtype))  # SparseTensor feats (N, Cout)

        return out

    # merge/Split Logic: Directly to Baseline nn.Linear The weight is incremented, and peft Linear consistent
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = list(self.active_adapters) if adapter_names is None else adapter_names
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                delta = self.get_delta_weight(active_adapter)
                base = self.get_base_layer()
                base.weight.data = base.weight.data + delta
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                delta = self.get_delta_weight(active_adapter)
                base = self.get_base_layer()
                base.weight.data = base.weight.data - delta

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        return (weight_B @ weight_A) * self.scaling[adapter]


def register_sparse_linear_with_peft() -> None:
    """Register SparseLinear LoRA into PEFT's dispatch table."""
    from peft.tuners.lora import layer as lora_layer_mod

    orig_dispatch = lora_layer_mod.dispatch_default

    def _dispatch(target: torch.nn.Module, adapter_name: str, lora_config, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base = target.get_base_layer()
        else:
            target_base = target

        if isinstance(target_base, SparseLinear):
            return SparseLinearLora(
                target,
                adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=False,
                **kwargs,
            )

        return orig_dispatch(target, adapter_name, lora_config, **kwargs)

    lora_layer_mod.dispatch_default = _dispatch


def _build_optimizer(
    trainable_params: list,
    opt_cfg: Any,
) -> Optional[optim.Optimizer]:
    """Internal helper to build an optimizer from opt config."""
    if not trainable_params:
        return None

    opt_type = str(opt_cfg.type).lower()

    if opt_type == 'adam_8bit':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            trainable_params,
            lr=float(opt_cfg.lr),
            betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )

    from timm.optim.optim_factory import create_optimizer_v2

    if opt_type == "sgd":
        return create_optimizer_v2(trainable_params, opt="sgd", lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))

    if opt_type == "adan":
        return create_optimizer_v2(
            trainable_params,
            opt="adan",
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
            betas=(0.98, 0.92, 0.99),
            eps=float(opt_cfg.eps),
        )

    return create_optimizer_v2(trainable_params, opt=opt_type, lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay), betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)), eps=float(opt_cfg.eps))


def build_optimizer_for_sparse(
    slat_model: nn.Module,
    opt_cfg: Any,
) -> Optional[optim.Optimizer]:
    """Optimizer for slat_flow_model (Sparse Stage 2)."""
    trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    return _build_optimizer(trainable_params, opt_cfg)


def build_optimizer_for_dense(
    ss_model: nn.Module,
    opt_cfg: Any,
) -> Optional[optim.Optimizer]:
    """Optimizer for sparse_structure_flow_model (Dense Stage 1)."""
    trainable_params = [p for p in ss_model.parameters() if p.requires_grad]
    return _build_optimizer(trainable_params, opt_cfg)


# =====================================================================
# Model trainable settings
# =====================================================================

def set_slat_trainable(pipeline: Any, trainable: bool = True) -> None:
    """Toggle trainable flags on slat_flow_model."""
    slat_model = pipeline.pipe.models["slat_flow_model"]
    for p in slat_model.parameters():
        p.requires_grad = trainable

    status = "trainable" if trainable else "Freeze"
    n_params = sum(p.numel() for p in slat_model.parameters())
    print(f"[set_slat_trainable] slat_flow_model {status} ({n_params:,} parameters)")


def inject_lora_to_slat(pipeline: Any, lora_cfg: Any) -> None:
    """Inject LoRA layers into slat_flow_model (call register_sparse_linear_with_peft first)."""
    from peft import LoraConfig, get_peft_model

    slat_model = pipeline.pipe.models["slat_flow_model"]

    # build LoRA Configuration
    config = LoraConfig(
        r=int(lora_cfg.lora_rank),
        lora_alpha=int(lora_cfg.get("lora_alpha", lora_cfg.lora_rank)),
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
    )

    # Inject LoRA
    slat_model_lora = get_peft_model(slat_model, config)
    pipeline.pipe.models["slat_flow_model"] = slat_model_lora

    # Statistical parameters
    trainable = sum(p.numel() for p in slat_model_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in slat_model_lora.parameters())
    print(f"[inject_lora_to_slat] LoRA Injection completed: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# =====================================================================
# Trellis Training strategy (all parameters / LoRA / frozen)
# =====================================================================

from oreo.systems.utils.strategy import TrainingStrategy, SpconvInferenceMixin


class TrellisFullFinetuneStrategy(SpconvInferenceMixin, TrainingStrategy):
    """Full fine-tune: unfreeze all params; teacher loaded frozen from pretrained."""

    def __init__(
        self,
        pipeline: Any,
        train_device: torch.device,
        teacher_device: torch.device,  # Ignore, force the same device
        pretrained_path: str,
    ):
        # ★ Force the same equipment to avoid spconv Across devices indice_key caching problem
        super().__init__(pipeline, train_device, train_device)
        self._sparse_teacher: Optional[nn.Module] = None
        self._dense_teacher: Optional[nn.Module] = None
        self._pretrained_path = pretrained_path

    def setup(self) -> None:
        """Full parameter settings: Unfreeze students and load teachers from pre-training (same device)."""
        from trellis import models as trellis_models

        # Unfreeze students
        for p in self._student.parameters():
            p.requires_grad = True

        # Load Sparse Teacher to the same device (spconv Cross-device support is not supported)
        model_args = self.pipeline.pipe._pretrained_args['models']
        slat_model_path = f"{self._pretrained_path}/{model_args['slat_flow_model']}"
        self._sparse_teacher = trellis_models.from_pretrained(slat_model_path)
        self._sparse_teacher.to(self.teacher_device).eval().requires_grad_(False)

        slat_mem = sum(p.numel() * p.element_size() for p in self._sparse_teacher.parameters()) / 1e6
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)

        # Load Dense Teacher (sparse_structure_flow_model, used for on-policy dense rollout fallback)
        ss_model_path = f"{self._pretrained_path}/{model_args['sparse_structure_flow_model']}"
        self._dense_teacher = trellis_models.from_pretrained(ss_model_path)
        self._dense_teacher.to(self.train_device).eval().requires_grad_(False)

        ss_mem = sum(p.numel() * p.element_size() for p in self._dense_teacher.parameters()) / 1e6

        print(f"[TrellisFullFinetuneStrategy] Full parameter fine-tuning: {trainable:,} Parameters are trainable")
        print(f"[TrellisFullFinetuneStrategy] Sparse teacher → {self.teacher_device} ({slat_mem:.0f} MB)")
        print(f"[TrellisFullFinetuneStrategy] Dense teacher  → {self.train_device} ({ss_mem:.0f} MB)")
        print(f"[TrellisFullFinetuneStrategy] ⚠️ Double the memory (spconv Cross-device inference is not supported)")

    @contextmanager
    def sparse_teacher_context(self) -> Generator[None, None, None]:
        """temporary replacement pipeline in sparse The model is a frozen teacher."""
        original = self.pipeline.pipe.models["slat_flow_model"]
        self.pipeline.pipe.models["slat_flow_model"] = self._sparse_teacher
        try:
            yield
        finally:
            self.pipeline.pipe.models["slat_flow_model"] = original

    @contextmanager
    def dense_teacher_context(self) -> Generator[None, None, None]:
        """temporary replacement pipeline in dense The model is a frozen teacher."""
        original = self.pipeline.pipe.models["sparse_structure_flow_model"]
        self.pipeline.pipe.models["sparse_structure_flow_model"] = self._dense_teacher
        try:
            yield
        finally:
            self.pipeline.pipe.models["sparse_structure_flow_model"] = original


class TrellisLoRAStrategy(SpconvInferenceMixin, TrainingStrategy):
    """LoRA fine-tune: freeze base weights, train LoRA only."""

    def setup(self) -> None:
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._student.parameters())
        logging.info(f"[TrellisLoRAStrategy] trainable {trainable:,} / Total parameters {total:,} ({100*trainable/total:.2f}%)")

    @contextmanager
    def sparse_teacher_context(self) -> Generator[None, None, None]:
        with self.pipeline.disable_lora_context():
            yield

    def save_student(self, ckpt_dir: Path) -> None:
        """save only LoRA adapter parameters."""
        from safetensors.torch import save_file
        lora_sd = {k: v for k, v in self._unwrap().state_dict().items() if "lora_" in k}
        save_file(lora_sd, str(ckpt_dir / "slat_flow_model.safetensors"))

    def load_student(self, ckpt_dir: Path) -> None:
        """load only LoRA adapter Parameters (strict=False)."""
        p = ckpt_dir / "slat_flow_model.safetensors"
        if p.exists():
            from safetensors.torch import load_file
            self._unwrap().load_state_dict(load_file(str(p), device="cpu"), strict=False)

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        """LoRA Export: load first adapter,merge Arrive base, export the complete weights."""
        from safetensors.torch import save_file

        export_dir.mkdir(parents=True, exist_ok=True)
        self.load_student(ckpt_dir)
        merged_sd = self._merge_lora_weights()
        dst = export_dir / "slat_flow_model.safetensors"
        save_file(merged_sd, str(dst))
        logging.info(f"[export] slat_flow_model (LoRA merged): → {dst}")
        return {"slat_flow_model": dst}

    def _merge_lora_weights(self) -> Dict[str, torch.Tensor]:
        """will lora_A/lora_B fold in base weight, returns None LoRA key clean state_dict."""
        model = self._unwrap()
        merged: Dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                W = module.weight.data                      # (out_features, in_features)
                A = module.lora_A.weight.data               # (rank, in_features)
                B = module.lora_B.weight.data               # (out_features, rank)
                scaling = getattr(module, "scaling", 1.0)
                merged[f"{name}.weight"] = W + (B @ A) * scaling  # (out_features, in_features)
                if module.bias is not None:
                    merged[f"{name}.bias"] = module.bias.data

        for k, v in model.state_dict().items():
            if "lora_" not in k and k not in merged:
                merged[k] = v

        return merged


class TrellisFrozenStrategy(TrainingStrategy):
    """Frozen / inference strategy (no SpconvInferenceMixin required)."""

    def setup(self) -> None:
        for p in self._student.parameters():
            p.requires_grad = False
        logging.info("[TrellisFrozenStrategy] Model freeze (inference mode)")

    @contextmanager
    def sparse_teacher_context(self) -> Generator[None, None, None]:
        raise RuntimeError("TrellisFrozenStrategy Regularization is not supported")
        yield  # type hint

    @contextmanager
    def dense_teacher_context(self) -> Generator[None, None, None]:
        raise RuntimeError("TrellisFrozenStrategy Not supported dense teacher")
        yield  # type hint

    @property
    def has_teacher(self) -> bool:
        return False

    def prepare_sparse(self, accelerator, optimizer_sparse):
        """No need for freeze mode DDP packaging."""
        self._accelerator = accelerator
        return optimizer_sparse

    def save_student(self, ckpt_dir: Path) -> None:
        pass

    def load_student(self, ckpt_dir: Path) -> None:
        pass

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        return {}

