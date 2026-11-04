"""oreo.systems.utils — shared logging, mixins, profiler, StageOps ABC, strategy, visual I/O."""
from .mixins import AcceleratorMixin, DistributedMixin, WandbMixin, CSVMixin, AccumulatorMixin
from .logging import MetricLogger, build_autograd_step_log
from .visual import VisualIO, TrellisVisualIO, Trellis2VisualIO, composite_alpha, composite_alpha_to_black, composite_alpha_to_white
from .strategy import TrainingStrategy, SpconvInferenceMixin
from .profiler import PhaseProfiler, AsyncPhaseProfiler
from .stage_ops import StageOps, StageSkipError

__all__ = [
    "AcceleratorMixin", "DistributedMixin", "WandbMixin", "CSVMixin", "AccumulatorMixin",
    "MetricLogger", "build_autograd_step_log",
    "VisualIO", "TrellisVisualIO", "Trellis2VisualIO", "composite_alpha", "composite_alpha_to_black", "composite_alpha_to_white",
    "TrainingStrategy", "SpconvInferenceMixin",
    "PhaseProfiler",
    "AsyncPhaseProfiler",
    "StageOps", "StageSkipError",
]
