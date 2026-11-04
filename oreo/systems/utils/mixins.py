"""Reusable training mixins (device moves, freeze helpers, etc.)."""
import csv
import torch
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image


# =====================================================================
# Mixin base class - Reusable function modules
# =====================================================================

class AcceleratorMixin:
    """Accelerator Basic function encapsulation.

    provide accelerator Related common operations, such as main process judgment, device acquisition, synchronization, etc.
    Subclasses need to provide self.accelerator properties.
    """

    accelerator: Accelerator = None

    @property
    def is_main_process(self) -> bool:
        """Is it the main process?"""
        return self.accelerator is None or self.accelerator.is_main_process

    @property
    def device(self) -> torch.device:
        """current device"""
        return self.accelerator.device if self.accelerator else torch.device("cpu")

    def wait_for_everyone(self) -> None:
        """Wait for all processes to sync"""
        if self.accelerator:
            self.accelerator.wait_for_everyone()


class DistributedMixin(AcceleratorMixin):
    """Distributed aggregation capabilities.

    Provides cross-process reduce Operation, used for indicator aggregation during distributed training.
    """

    def reduce_sum(self, value: float) -> float:
        """Sum across processes"""
        if self.accelerator is None:
            return value
        t = torch.tensor([value], device=self.device)
        return self.accelerator.reduce(t, reduction="sum").item()

    def reduce_mean(self, total_sum: float, total_count: float) -> float:
        """Average across processes (aggregate first sum and count, and then calculate the mean)"""
        global_sum = self.reduce_sum(total_sum)
        global_count = self.reduce_sum(total_count)
        return global_sum / max(global_count, 1e-8)


class WandbMixin(AcceleratorMixin):
    """Wandb Logging function.

    Pass accelerator.log() Log metrics and images to wandb.
    need to be in Accelerator Set during initialization log_with=["wandb"].
    """

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Record the indicator to wandb"""
        if self.is_main_process and self.accelerator:
            self.accelerator.log(metrics, step=step)

    def log_images(self, images: Dict[str, Image.Image], step: int, prefix: str = "") -> None:
        """record image to wandb"""
        if self.is_main_process and self.accelerator:
            import wandb
            log_dict = {}
            for k, v in images.items():
                key = f"{prefix}/{k}" if prefix else k
                log_dict[key] = wandb.Image(v)
            self.accelerator.log(log_dict, step=step)


class CSVMixin(AcceleratorMixin):
    """CSV Logging function.

    provide CSV File logging, written only by the main process.
    Subclasses need to provide self.csv_path properties.
    """

    csv_path: Path = None

    def log_csv(self, row: Dict[str, Any]) -> None:
        """Append a line to CSV File (main process only)"""
        if not self.is_main_process or not self.csv_path:
            return

        path = self.csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fieldnames = list(row.keys())
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


class AccumulatorMixin:
    """Indicator accumulation function.

    Metrics for accumulating multiple steps, support batch Weighted average.
    """

    def reset(self) -> None:
        """reset totalizer"""
        self.sums: Dict[str, float] = {}
        self.count: float = 0.0

    def accumulate(self, logs: Dict[str, Any], batch_size: int) -> None:
        """Cumulative metrics for a step"""
        bs = float(batch_size)
        self.count += bs
        for k, v in logs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = float(v.detach().item())
            self.sums.setdefault(k, 0.0)
            self.sums[k] += v * bs

    def average(self) -> Optional[Dict[str, float]]:
        """Calculate the local cumulative average"""
        if self.count <= 0:
            return None
        return {k: v / self.count for k, v in self.sums.items()}
