"""
Import-light helpers for flat Accelerate Dual-Stage checkpoints (strict load + student/teacher diff checks).

File-to-model mapping is explicit (Accelerate registration order).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, Optional


FULL_DUAL_LAYOUT = (
    ("slat_flow_model", "model.safetensors"),
    ("sparse_structure_flow_model", "model_1.safetensors"),
)


def unwrap_module(model: Any) -> Any:
    """Strip nested .module wrappers to the inner nn.Module."""
    seen = set()
    while hasattr(model, "module") and id(model) not in seen:
        seen.add(id(model))
        inner = model.module
        if inner is model:
            break
        model = inner
    return model


def _default_state_loader(path: str) -> Mapping[str, Any]:
    # Keep safetensors optional for mapping tests and other import-light users.
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def _state_report(target: str, filename: str, path: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    tensor_count = 0
    numel = 0
    for value in state.values():
        value_numel = getattr(value, "numel", None)
        if callable(value_numel):
            tensor_count += 1
            numel += int(value_numel())
    return {
        "target": target,
        "file": filename,
        "path": str(path),
        "key_count": len(state),
        "tensor_count": tensor_count,
        "numel": numel,
        "strict": True,
    }


def load_full_dual_checkpoint(
    checkpoint_dir: Any,
    *,
    slat_flow_model: Any,
    sparse_structure_flow_model: Any,
    state_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """strict=True load of both required flat Accelerate model files."""
    root = Path(checkpoint_dir)
    paths = {target: root / filename for target, filename in FULL_DUAL_LAYOUT}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "full Dual-Stage checkpoint requires both flat files; missing: "
            + ", ".join(missing)
        )

    models = {
        "slat_flow_model": slat_flow_model,
        "sparse_structure_flow_model": sparse_structure_flow_model,
    }
    loader = state_loader or _default_state_loader
    stages = []
    for target, filename in FULL_DUAL_LAYOUT:
        path = paths[target]
        state = loader(str(path))
        if not isinstance(state, Mapping):
            raise TypeError(f"state loader returned non-mapping for {path}: {type(state).__name__}")
        if not state:
            raise ValueError(f"checkpoint state is empty: {path}")
        model = unwrap_module(models[target])
        model.load_state_dict(state, strict=True)
        stages.append(_state_report(target, filename, path, state))

    return {
        "format": "accelerate_flat_full_dual",
        "checkpoint_dir": str(root),
        "stages": stages,
    }


def _shape_tuple(value: Any) -> tuple:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"parameter has no shape: {type(value).__name__}")
    return tuple(int(dim) for dim in shape)


def _float_value(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    to_float = getattr(value, "float", None)
    if callable(to_float):
        value = to_float()
    return value


def _max_abs_diff(student: Any, base: Any) -> float:
    difference = _float_value(student) - _float_value(base)
    absolute = difference.abs()
    maximum = absolute.max()
    item = getattr(maximum, "item", None)
    return float(item() if callable(item) else maximum)


def validate_student_diff(
    student_model: Any,
    base_model: Any,
    *,
    target: str,
    threshold: float = 1e-8,
) -> Dict[str, Any]:
    """Assert student params differ from teacher/base (sanity after load)."""
    student = dict(unwrap_module(student_model).named_parameters())
    base = dict(unwrap_module(base_model).named_parameters())
    if not student or not base:
        raise ValueError(f"{target}: student/base parameter sets must be non-empty")
    if set(student) != set(base):
        missing_from_student = sorted(set(base) - set(student))
        extra_in_student = sorted(set(student) - set(base))
        raise ValueError(
            f"{target}: parameter key mismatch; missing={missing_from_student}, "
            f"extra={extra_in_student}"
        )

    different = 0
    max_abs_diff = 0.0
    parameter_numel = 0
    for name in sorted(student):
        student_param = student[name]
        base_param = base[name]
        student_shape = _shape_tuple(student_param)
        base_shape = _shape_tuple(base_param)
        if student_shape != base_shape:
            raise ValueError(
                f"{target}: shape mismatch for {name}: "
                f"student={student_shape}, base={base_shape}"
            )
        value_numel = getattr(student_param, "numel", None)
        if callable(value_numel):
            parameter_numel += int(value_numel())
        difference = _max_abs_diff(student_param, base_param)
        max_abs_diff = max(max_abs_diff, difference)
        if difference > threshold:
            different += 1

    if different == 0:
        raise ValueError(
            f"{target}: student parameters do not differ from base/teacher "
            f"above threshold={threshold:.3g} (max_abs_diff={max_abs_diff:.3g})"
        )

    return {
        "target": target,
        "parameter_count": len(student),
        "parameter_numel": parameter_numel,
        "different_parameter_count": different,
        "max_abs_diff": max_abs_diff,
        "threshold": float(threshold),
        "keys_match": True,
        "shapes_match": True,
    }
