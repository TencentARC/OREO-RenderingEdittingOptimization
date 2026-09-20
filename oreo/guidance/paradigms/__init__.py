"""
Guidance Paradigms 模块。

提供 retained Guidance 范式实现：
- FlowEdit: 多步编辑式训练/推理（普通与 pipeline-parallel 变体）
"""

from oreo.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditGuidancePP

__all__ = [
    "FlowEditGuidance",
    "FlowEditGuidancePP",
]
