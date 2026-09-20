"""Training Strategy: Base Class + spconv Reasoning compatible mixin.

Specific strategies (TrellisLoRAStrategy etc.) are defined in each generator of training_adpter in.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Generator
import torch
import torch.nn as nn


# =====================================================================
# base class
# =====================================================================

class TrainingStrategy(ABC):
    """Training strategy base class."""

    def __init__(self, pipeline: Any, train_device: torch.device, teacher_device: torch.device):
        self.pipeline = pipeline
        self.train_device = train_device
        self.teacher_device = teacher_device
        self._student = pipeline.pipe.models["slat_flow_model"]
        self._accelerator = None

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    @contextmanager
    def sparse_teacher_context(self) -> Generator[None, None, None]: ...

    @contextmanager
    def dense_teacher_context(self) -> Generator[None, None, None]:
        """temporary replacement pipeline in dense The model is a frozen teacher.

        Default no-op(Subclasses override to provide dense teacher swap).
        """
        yield

    @property
    def student(self) -> nn.Module:
        return self._student

    @property
    def has_teacher(self) -> bool:
        return True

    # ---- DDP Register ----

    def prepare_sparse(self, accelerator, optimizer_sparse=None):
        """use accelerator.prepare packaging Sparse model(+optimizer), write back to pipeline. Return prepared optimizer.

        When optimizer_sparse for None time (eval_only mode), only prepare model,
        ensure accelerator.load_state() can be restored correctly checkpoint weight.
        """
        self._accelerator = accelerator
        if optimizer_sparse is not None:
            self._student, optimizer_sparse = accelerator.prepare(self._student, optimizer_sparse)
        else:
            self._student = accelerator.prepare(self._student)
        self.pipeline.pipe.models["slat_flow_model"] = self._student
        return optimizer_sparse

    def prepare_dense(self, accelerator, optimizer_dense=None):
        """use accelerator.prepare packaging Dense model(+optimizer), write back to pipeline.

        only Dual-Stage Entry call.Sparse-only No need to call the entrance.
        """
        self._dense_student = self.pipeline.pipe.models["sparse_structure_flow_model"]
        if optimizer_dense is not None:
            self._dense_student, optimizer_dense = accelerator.prepare(
                self._dense_student, optimizer_dense,
            )
        else:
            self._dense_student = accelerator.prepare(self._dense_student)
        self.pipeline.pipe.models["sparse_structure_flow_model"] = self._dense_student
        return optimizer_dense

    @contextmanager
    def inference_context(self):
        """Reasoning context. base class default no-op, subclass / mixin Can be overwritten."""
        yield

    def _unwrap(self) -> nn.Module:
        """remove DDP wrapper Get the original model."""
        return self._accelerator.unwrap_model(self._student)

    # ---- Checkpoint (for training recovery) ----

    def save_student(self, ckpt_dir: Path) -> None:
        """Save all weights of the student model."""
        from safetensors.torch import save_file
        save_file(self._unwrap().state_dict(), str(ckpt_dir / "slat_flow_model.safetensors"))

    def load_student(self, ckpt_dir: Path) -> None:
        """Load all weights of the student model."""
        p = ckpt_dir / "slat_flow_model.safetensors"
        if p.exists():
            from safetensors.torch import load_file
            self._unwrap().load_state_dict(load_file(str(p), device="cpu"))

    # ---- Inference derived ----

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        """
        from training checkpoint Export inference-compatible weights files to export_dir.

        Default behavior: direct copy slat_flow_model.safetensors.
        LoRA Subclasses override this method to implement merge.
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        src = ckpt_dir / "slat_flow_model.safetensors"
        dst = export_dir / "slat_flow_model.safetensors"
        shutil.copy2(src, dst)
        logging.info(f"[export] slat_flow_model: {src} → {dst}")
        return {"slat_flow_model": dst}


# =====================================================================
# Mixin:spconv Reasoning compatible
# =====================================================================

class SpconvInferenceMixin:
    """
    Mixin: stripping during inference DDP package, make pipeline Use the original model directly.
    imitate the official TRELLIS self.models(reasoning)vs self.training_models(training) design.
    """

    @contextmanager
    def inference_context(self):
        """Temporarily switch to None during reasoning DDP Original mockup of packaging."""
        if self._accelerator is None:
            yield
            return

        pipe_models = self.pipeline.pipe.models
        saved_model = pipe_models["slat_flow_model"]
        inner = self._accelerator.unwrap_model(self._student)

        pipe_models["slat_flow_model"] = inner
        try:
            yield
        finally:
            pipe_models["slat_flow_model"] = saved_model
