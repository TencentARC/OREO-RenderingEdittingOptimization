"""
Shared training-system primitives: device mapping, CFG mix, config/state bases, checkpoint I/O.

Inherited by Trellis OREO entries; keeps multi-backend hooks in one place.
"""

# =====================================================================
# Standard library import
# =====================================================================
import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, ClassVar

# =====================================================================
# Third-party library import
# =====================================================================
import numpy as np
import yaml
import torch
from accelerate import Accelerator
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


# =====================================================================
# Device allocation tool - training and Guidance of GPU mapping
# =====================================================================

def compute_guidance_device(train_device: torch.device) -> torch.device:
    """
    Map train CUDA device -> guidance device (node-local offset, fallback to shared).
    
    Formula: guidance = LOCAL_WORLD_SIZE + LOCAL_RANK (if in range).
    """
    if train_device.type != "cuda":
        raise ValueError(f"Training equipment must be CUDA, currently: {train_device}")

    # ---- Get the number of processes in a node ----
    # priority use LOCAL_WORLD_SIZE(torchrun/accelerate launch will be set)
    # Alternatives:WORLD_SIZE / NNODES Or fall back to the WORLD_SIZE
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")

    if local_world_size is not None:
        n_train_gpus = int(local_world_size)
    else:
        # Fallback plan: Assume a single machine, or based on NNODES Calculate
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        nnodes = int(os.environ.get("NNODES", 1))
        n_train_gpus = world_size // nnodes

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Guidance Equipment = In-node trainingGPUnumber + LOCAL_RANK
    guidance_idx = n_train_gpus + local_rank

    # Check whether the device exists; if insufficient, fall back to training with the same device
    n_gpus = torch.cuda.device_count()
    if guidance_idx >= n_gpus:
        logging.warning(
            f"Guidance need cuda:{guidance_idx}, but this node only has {n_gpus} a GPU."
            f"Fallback to training device {train_device}(shared mode)."
        )
        return train_device

    return torch.device(f"cuda:{guidance_idx}")


# =====================================================================
# Utility function - CFG mix
# =====================================================================

def mix_cfg(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    scale: float,
    uncond_mode: str = "detach"
) -> torch.Tensor:
    """Generic CFG mix: cond + scale*(cond-uncond)."""
    if uncond_pred is None:
        return cond_pred  # (B,T,C)
    if uncond_mode == "detach":
        uncond_pred = uncond_pred.detach()  # (B,T,C)
    if uncond_mode == "mirror":
        cond_pred = cond_pred.detach()  # (B,T,C)
    return cond_pred + scale * (cond_pred - uncond_pred)  # (B,T,C)


# =====================================================================
# ModeGuard - Module mode context manager
# =====================================================================

class ModeGuard:
    """Context manager to temporarily set module train/eval."""

    def __init__(self, *modules: Any, training: bool = False):
        """
        Args:
            *modules: To be managed nn.Module Example (automatic filtering None)
            training: target mode,True For training mode,False for evaluation mode
        """
        self.modules = [m for m in modules if m is not None]
        self.training = training
        self.states: List[bool] = []

    def __enter__(self):
        """Enter context: Save state and switch to target mode."""
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.train(self.training)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original training state."""
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


def TrainModeGuard(*modules: Any) -> ModeGuard:
    """ModeGuard(..., training=True)."""
    return ModeGuard(*modules, training=True)


def EvalModeGuard(*modules: Any) -> ModeGuard:
    """ModeGuard(..., training=False)."""
    return ModeGuard(*modules, training=False)


# =====================================================================
# BaseConfig - Configure base class
# =====================================================================

@dataclass
class BaseConfig:
    """Shared training config fields across backends."""
    config_path: Optional[str] = None
    run_name: str = "run"
    logdir: str = "./logs"
    seed: int = 42
    eval_only: bool = False
    num_epochs: int = 100
    batch_size: int = 1
    eval_batch_size: int = 1
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    checkpoint: Optional[str] = None

    # Subconfiguration dictionary (subclasses can cover specific structures)
    train: Dict[str, Any] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    renderer: Dict[str, Any] = field(default_factory=dict)
    guidance: Dict[str, Any] = field(default_factory=dict)
    freq: Dict[str, Any] = field(default_factory=dict)

    def save_yaml(self, path: str) -> None:
        """Save current configuration as YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_from_file(cls, path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> "BaseConfig":
        """Construct configuration from file (subclasses need to implement specific logic)."""
        raise NotImplementedError("load_from_file Need to be implemented by subclasses.")


# =====================================================================
# BaseState - State base class
# =====================================================================

@dataclass
class BaseState:
    """Shared generation-state base."""

    @dataclass
    class Cameras:
        """Camera parameter container."""
        c2w: Any = None         # (B,V,4,4) camera-to-world
        w2c: Any = None         # (B,V,4,4) world-to-camera
        mvp: Any = None         # (B,V,4,4) MVP matrix
        positions: Any = None   # (B,V,3) camera position
        intrinsics: Any = None  # (B,V,3,3) Internal parameter matrix
        light_positions: Any = None  # (B,V,3) light source position

    @dataclass
    class ViewsGenerated:
        """Generate perspective cache. store from 3D Represents the rendered multi-view image."""
        image_tensor: Any = None  # (B,V,H,W,C) or (B,V,C,H,W)

    @dataclass
    class ViewsEdited:
        """View cache after editing. Store the edited perspective image."""
        image_tensor: Any = None      # (B,V,C,H,W)
        trackers: Any = None          # List[StateTracker], used for intermediate step visualization

    @dataclass
    class ViewsConditioned:
        """Conditional perspective caching. Stores the input conditional image and its embedding."""
        image_pils: Any = None   # list[len=B] of PIL.Image
        paths: Any = None        # list[len=B] of str
        cond_embed: Any = None   # (B,S,C) conditional embedding
        uncond_embed: Any = None # (B,S,C) Unconditional embedding (for CFG)

    @dataclass
    class Guidance:
        """Guidance cache. Store guidance signals for supervision."""

    # ============== core status field ==============
    coords: Any = None  # sparse structure coordinates
    feats: Any = None   # sparse features

    # ============== substate container ==============
    cameras: Cameras = field(default_factory=Cameras)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)
    views_conditioned: ViewsConditioned = field(default_factory=ViewsConditioned)
    guidance: Guidance = field(default_factory=Guidance)

    # ============== Data mount field ==============
    guidances_data: Any = None

    def attach_batch(self, batch: Dict[str, Any]) -> "BaseState":
        """
        Extract and mount conditions, cameras, and other information from data batches.
        Subclasses should override this method to handle model-specific batch structure.
        """
        raise NotImplementedError("attach_batch Need to be implemented by subclasses.")

    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        from views_conditioned Extract conditional and unconditional embeddings.

        Returns:
            tuple: (cond_embed, uncond_embed)
        """
        cond = self.views_conditioned.cond_embed
        uncond = self.views_conditioned.uncond_embed

        if cond is None:
            raise ValueError("views_conditioned.cond_embed Empty and cannot be extracted embeddings.")

        # Process list Format
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=0)  # (B,S,C)
        if isinstance(uncond, list):
            uncond = torch.cat(uncond, dim=0)  # (B,S,C)

        # Deal with redundant dimensions
        if isinstance(cond, torch.Tensor) and cond.dim() == 4 and cond.shape[1] == 1:
            cond = cond.squeeze(1)  # (B,S,C)
        if isinstance(uncond, torch.Tensor) and uncond.dim() == 4 and uncond.shape[1] == 1:
            uncond = uncond.squeeze(1)  # (B,S,C)

        return cond, uncond



# =====================================================================
# Environment setup tool - Standalone function (for use by all systems)
# =====================================================================

def setup_env_and_seed(cfg: Any) -> None:
    """Seed RNGs and optional deterministic flags."""
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Run directory with CSV Tools
# =====================================================================

class _TeeWriter:
    """Write to both original stream and a log file."""
    def __init__(self, original_stream, log_file_handle):
        self.original_stream = original_stream
        self.log_file_handle = log_file_handle

    def write(self, message):
        self.original_stream.write(message)
        if self.log_file_handle and message.strip():
            self.log_file_handle.write(message)
            self.log_file_handle.flush()

    def flush(self):
        self.original_stream.flush()
        if self.log_file_handle:
            self.log_file_handle.flush()


def _setup_file_logging(log_file: Path, accelerator: Accelerator) -> None:
    """Per-rank file logging next to console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    rank = accelerator.process_index

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console handler(all processes)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler(each rank One file each)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    rank_log = log_file.parent / f"run_rank{rank}.log"
    file_handler = logging.FileHandler(rank_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"[Rank {rank}] The log will be saved to: {rank_log}")

    # stderr Redirect (main process only, catches unhandled exceptions traceback)
    if accelerator.is_main_process:
        stderr_log_file = log_file.parent / "stderr.log"
        stderr_file = open(stderr_log_file, "a", encoding="utf-8")
        sys.stderr = _TeeWriter(sys.__stderr__, stderr_file)
        logging.info(f"stderr will save to: {stderr_log_file}")


def build_run_paths(cfg: Any, accelerator: Accelerator) -> Tuple[Path, Path, Path, Path]:
    """Create run dirs and dump config YAML."""
    run_root = Path(cfg.logdir) / (cfg.run_name if cfg.run_name else "run")
    logs_dir = run_root / "logs"
    visuals_train_dir = run_root / "visualizations" / "train"
    visuals_eval_dir = run_root / "visualizations" / "eval"

    if accelerator.is_main_process:
        run_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        visuals_train_dir.mkdir(parents=True, exist_ok=True)
        visuals_eval_dir.mkdir(parents=True, exist_ok=True)

        # Save the configuration (using ml_collections.ConfigDict of to_dict method)
        with (run_root / "config.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(cfg.to_dict(), sort_keys=False))

        # Save startup command
        with (run_root / "run_command.txt").open("w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))

    # Initialize file log
    _setup_file_logging(logs_dir / "run.log", accelerator)

    return run_root, logs_dir, visuals_train_dir, visuals_eval_dir


# =====================================================================
# CheckpointIO - Checkpoint management
# =====================================================================

@dataclass
class CheckpointIO:
    """Thin Accelerate save_state/load_state wrapper."""

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(self, system: Any, state: BaseState, cfg: Any, epoch: int, global_step: int) -> None:
        """save checkpoint(model + optimizer + random_states + meta)."""
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"
        target.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(target))

        if self.accelerator.is_main_process:
            meta = {"epoch": int(epoch), "global_step": int(global_step)}
            with (target / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.accelerator.wait_for_everyone()

    def load(self, path: str, mode: str = "train") -> int:
        """
        load checkpoint(model + optimizer + random_states).

        Args:
            path: Checkpoint directory path.
            mode: "train" time start_epoch = epoch + 1; otherwise 0.
        """
        cp = path
        if not (isinstance(cp, str) and cp):
            self.start_epoch = 0
            return 0

        root = Path(cp)
        meta_path = root / "meta.json"
        if not (root.is_dir() and meta_path.exists()):
            self.start_epoch = 0
            self.start_global_step = 0
            return 0

        # accelerator One-time recovery: model weights + optimizer + random states
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(str(root))
        self.accelerator.wait_for_everyone()

        meta = json.load(meta_path.open("r", encoding="utf-8"))
        self.start_epoch = int(meta["epoch"]) + 1 if mode == "train" else 0
        self.start_global_step = int(meta["global_step"])
        logging.info(
            f"[CheckpointIO] Checkpoint restored: {root} "
            f"(epoch={meta['epoch']}, step={meta['global_step']}, After recovery start_epoch={self.start_epoch})"
        )
        return self.start_epoch


# =====================================================================
# from utils Import common tools (avoid duplication of definitions)
# =====================================================================

from oreo.systems.utils import MetricLogger, VisualIO


