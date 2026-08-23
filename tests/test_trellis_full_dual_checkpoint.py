"""Import-light regression tests for full Dual-Stage checkpoint loading."""

import ast
import json
import tempfile
import unittest
from pathlib import Path


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Difference:
    def __init__(self, values):
        self._values = values

    def abs(self):
        return _Difference([abs(value) for value in self._values])

    def max(self):
        return _Scalar(max(self._values))


class _Tensor:
    def __init__(self, values, shape=None):
        self.values = tuple(values)
        self.shape = tuple(shape or (len(self.values),))

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def float(self):
        return self

    def __sub__(self, other):
        return _Difference([a - b for a, b in zip(self.values, other.values)])


class _Model:
    def __init__(self, parameters):
        self._parameters = dict(parameters)
        self.load_calls = []

    def load_state_dict(self, state_dict, strict=False):
        self.load_calls.append({"strict": strict, "keys": tuple(state_dict)})
        self._parameters = dict(state_dict)

    def named_parameters(self):
        return iter(self._parameters.items())


class _Wrapper:
    def __init__(self, module):
        self.module = module


class FullDualCheckpointTest(unittest.TestCase):
    def setUp(self):
        from edit4shape.systems.trellis.checkpoint import (
            load_full_dual_checkpoint,
            validate_student_diff,
        )

        self.load_full_dual_checkpoint = load_full_dual_checkpoint
        self.validate_student_diff = validate_student_diff

    def _checkpoint(self, include_model=True, include_model_1=True):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        if include_model:
            (root / "model.safetensors").touch()
        if include_model_1:
            (root / "model_1.safetensors").touch()
        self.addCleanup(tempdir.cleanup)
        return root

    def test_flat_mapping_loads_both_models_strictly_and_reports_it(self):
        root = self._checkpoint()
        slat = _Model({"weight": _Tensor([0.0])})
        dense_inner = _Model({"weight": _Tensor([0.0, 0.0])})
        dense = _Wrapper(dense_inner)
        slat_base = _Model({"weight": _Tensor([0.0])})
        dense_base = _Model({"weight": _Tensor([0.0, 0.0])})
        states = {
            "model.safetensors": {"weight": _Tensor([2.0])},
            "model_1.safetensors": {"weight": _Tensor([3.0, 4.0])},
        }

        report = self.load_full_dual_checkpoint(
            root,
            slat_flow_model=slat,
            sparse_structure_flow_model=dense,
            state_loader=lambda path: states[Path(path).name],
        )

        self.assertEqual(slat._parameters["weight"].values, (2.0,))
        self.assertEqual(dense_inner._parameters["weight"].values, (3.0, 4.0))
        self.assertEqual(slat.load_calls, [{"strict": True, "keys": ("weight",)}])
        self.assertEqual(dense_inner.load_calls, [{"strict": True, "keys": ("weight",)}])
        by_target = {stage["target"]: stage for stage in report["stages"]}
        self.assertEqual(by_target["slat_flow_model"]["file"], "model.safetensors")
        self.assertEqual(
            by_target["sparse_structure_flow_model"]["file"],
            "model_1.safetensors",
        )
        self.assertEqual(by_target["sparse_structure_flow_model"]["numel"], 2)
        json.dumps(report)
        self.assertGreater(
            self.validate_student_diff(
                slat, slat_base, target="slat_flow_model"
            )["different_parameter_count"],
            0,
        )
        self.assertGreater(
            self.validate_student_diff(
                dense, dense_base, target="sparse_structure_flow_model"
            )["different_parameter_count"],
            0,
        )

    def test_missing_either_file_fails_before_loading_any_model(self):
        for include_model, include_model_1 in ((False, True), (True, False)):
            with self.subTest(include_model=include_model, include_model_1=include_model_1):
                root = self._checkpoint(include_model, include_model_1)
                slat = _Model({"weight": _Tensor([0.0])})
                dense = _Model({"weight": _Tensor([0.0])})
                with self.assertRaises(FileNotFoundError):
                    self.load_full_dual_checkpoint(
                        root,
                        slat_flow_model=slat,
                        sparse_structure_flow_model=dense,
                        state_loader=lambda path: {},
                    )
                self.assertEqual(slat.load_calls, [])
                self.assertEqual(dense.load_calls, [])

    def test_both_student_models_must_differ_from_their_base(self):
        slat_base = _Model({"weight": _Tensor([0.0, 0.0])})
        dense_base = _Model({"weight": _Tensor([1.0])})
        slat_student = _Model({"weight": _Tensor([0.0, 2.0])})
        dense_student = _Model({"weight": _Tensor([4.0])})

        slat_report = self.validate_student_diff(
            slat_student, slat_base, target="slat_flow_model"
        )
        dense_report = self.validate_student_diff(
            _Wrapper(dense_student), dense_base, target="sparse_structure_flow_model"
        )

        self.assertEqual(slat_report["different_parameter_count"], 1)
        self.assertEqual(dense_report["different_parameter_count"], 1)
        self.assertGreater(slat_report["max_abs_diff"], 0.0)
        self.assertGreater(dense_report["max_abs_diff"], 0.0)
        json.dumps([slat_report, dense_report])

    def test_student_equal_to_base_fails(self):
        model = _Model({"weight": _Tensor([1.0])})
        with self.assertRaises(ValueError):
            self.validate_student_diff(model, model, target="slat_flow_model")

    def test_key_or_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            self.validate_student_diff(
                _Model({"student_key": _Tensor([1.0])}),
                _Model({"base_key": _Tensor([0.0])}),
                target="slat_flow_model",
            )
        with self.assertRaises(ValueError):
            self.validate_student_diff(
                _Model({"weight": _Tensor([1.0, 2.0], shape=(2,))}),
                _Model({"weight": _Tensor([0.0, 0.0], shape=(1, 2))}),
                target="slat_flow_model",
            )


class PrepareRegistrationContractTest(unittest.TestCase):
    def test_dense_eval_registration_is_config_driven_and_after_sparse(self):
        system_path = (
            Path(__file__).parents[1]
            / "edit4shape"
            / "systems"
            / "trellis"
            / "system.py"
        )
        source = system_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "prepare_models_and_optimizers"
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        sparse_call = next(node for node in calls if node.func.attr == "prepare_sparse")
        dense_call = next(node for node in calls if node.func.attr == "prepare_dense")
        self.assertLess(sparse_call.lineno, dense_call.lineno)

        dense_if = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and any(candidate is dense_call for candidate in ast.walk(node))
            and "cfg.train.dense_optimizer" in ast.unparse(node.test)
        )
        condition = ast.unparse(dense_if.test)
        self.assertIn("cfg.train.dense_optimizer", condition)
        self.assertNotIn("optimizer_dense", condition)


if __name__ == "__main__":
    unittest.main()
