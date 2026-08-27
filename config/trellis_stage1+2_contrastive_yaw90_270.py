"""Formal yaw-limited variant of the dual-stage contrastive config.

This module is intentionally derived from the reference config so the formal
run inherits every setting except the training camera yaw range.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_CONFIG_PATH = Path(__file__).with_name("trellis_stage1+2_contrastive.py")
_BASE_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "_trellis_stage1_2_contrastive_base",
    str(_BASE_CONFIG_PATH),
)
if _BASE_CONFIG_SPEC is None or _BASE_CONFIG_SPEC.loader is None:
    raise ImportError(f"unable to load base config: {_BASE_CONFIG_PATH}")

_BASE_CONFIG_MODULE = importlib.util.module_from_spec(_BASE_CONFIG_SPEC)
_BASE_CONFIG_SPEC.loader.exec_module(_BASE_CONFIG_MODULE)


def get_config():
    """Return the baseline config with only the training yaw range changed."""
    cfg = _BASE_CONFIG_MODULE.get_config()
    cfg.data.train.yaw_range = [90.0, 270.0]
    return cfg
