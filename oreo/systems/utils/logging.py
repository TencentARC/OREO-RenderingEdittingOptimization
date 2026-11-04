"""MetricLogger and autograd step logging helpers."""
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .mixins import AccumulatorMixin, DistributedMixin, CSVMixin, WandbMixin


# =====================================================================
# Autograd Three-stage log merging
# =====================================================================

def build_autograd_step_log(
    guidance_log: Dict[str, Any],
    reg_weight: float,
    phase3_log: Dict[str, Any] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Format one autograd training step's metrics for logging."""
    merged = {**guidance_log, **(phase3_log or {})}

    total = merged.get("loss/guidance", 0.0)
    reg = merged.get("loss/reg")
    if reg is not None:
        total += reg_weight * reg
    merged["loss/total"] = total

    if prefix:
        merged = {f"{prefix}{k}": v for k, v in merged.items()}

    return merged


# =====================================================================
# MetricLogger - Unified Metrics Logger (using Mixin combination)
# =====================================================================

class MetricLogger(AccumulatorMixin, DistributedMixin, CSVMixin, WandbMixin):
    """Aggregates and writes train/eval scalars (optionally to Accelerate trackers)."""

    def __init__(self, accelerator: Accelerator = None, csv_path: Path = None):
        """
        initialization.

        Args:
            accelerator: Accelerate accelerator (for distributed aggregation and wandb)
            csv_path: CSV Log output path
        """
        self.accelerator = accelerator
        self.csv_path = csv_path
        self.reset()

    def _distributed_average(self) -> Optional[Dict[str, float]]:
        """
        Calculate the average after distributed aggregation.

        Security implementation: first all_gather all rank of key Take the union and package it into a single tensor
        do it once all_reduce. Even if each rank of log key inconsistent (such as a rank OOM cause
        guidance key missing) will not deadlock.
        """
        if self.count <= 0.0:
            return None

        # ★ collect all rank of key set, union → any rank Some key Won't lose it
        if dist.is_initialized():
            local_keys = set(self.sums.keys())
            all_keys_list = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys_list, local_keys)
            keys = sorted(set().union(*all_keys_list))
        else:
            keys = sorted(self.sums.keys())

        # Pack [count, v0, v1, ...] → Missing key supplement 0.0, single all_reduce
        vals = [self.count] + [float(self.sums.get(k, 0.0)) for k in keys]
        t = torch.tensor(vals, device=self.device)  # (len(keys)+1,)
        t = self.accelerator.reduce(t, reduction="sum")  # only 1 times all_reduce

        total_count = t[0].item()
        if total_count <= 0.0:
            return None
        return {k: t[i + 1].item() / total_count for i, k in enumerate(keys)}

    def flush(self, global_step: int, epoch: int) -> Optional[Dict[str, float]]:
        """
        Emit the accumulated average log and reset.

        In distributed training, all processes are synchronized first and then aggregated.
        """
        self.wait_for_everyone()

        avg_log = self._distributed_average()
        if avg_log:
            # CSV log (utilizing CSVMixin)
            self.log_csv({"global_step": global_step, "epoch": epoch, **avg_log})
            # Wandb log (utilizing WandbMixin)
            self.log_metrics(avg_log, step=global_step)

        self.reset()
        return avg_log

    def log_step(
        self,
        logs: Dict[str, Any],
        batch_size: int,
        global_step: int,
        epoch: int,
    ) -> None:
        """
        Record single-step logs (automatically handle gradient accumulation).

        Accumulated at each step, only in sync_gradients fired and reset.
        """
        self.accumulate(logs, batch_size)

        if self.accelerator is None or self.accelerator.sync_gradients:
            avg_log = self._distributed_average()
            if avg_log:
                self.log_csv({"global_step": global_step, "epoch": epoch, **avg_log})
                self.log_metrics(avg_log, step=global_step)
            self.reset()

