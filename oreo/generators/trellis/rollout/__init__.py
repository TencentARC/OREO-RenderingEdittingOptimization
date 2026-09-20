"""
Trellis Rollout module

Provides sampling mode:
- ODE: Standard Euler sampling, for inference and ReFL/DRaFT style training
- FlowEdit: Differential two-branch sampling, for inference only/Assessment

Tracker class:
- RolloutTracker (rollout.autograd_tracker): three stages Autograd of cond-level proxy logger
"""

from .base import mix_cfg_sparse, _predict_sparse_cond_velocity, auto_device_sparse, auto_device_dense, predict_sparse_velocity_with_cfg
from .ode import rollout_sparse, rollout_dense
from .flowedit import rollout_sparse_flowedit, rollout_dense_flowedit
from .autograd_tracker import RolloutTracker, VelocityTracker

__all__ = [
    # Base utilities
    "mix_cfg_sparse",
    "_predict_sparse_cond_velocity",
    "auto_device_sparse",
    "auto_device_dense",
    "predict_sparse_velocity_with_cfg",
    # ODE rollout
    "rollout_sparse",
    "rollout_dense",
    # FlowEdit rollout
    "rollout_sparse_flowedit",
    "rollout_dense_flowedit",
    # Autograd tracker
    "RolloutTracker",
    "VelocityTracker",
]
