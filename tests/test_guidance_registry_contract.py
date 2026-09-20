"""Import-light contract tests for the retained Qwen FlowEdit registry."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "edit4shape/guidance/base.py"
PARADIGMS_PATH = ROOT / "edit4shape/guidance/paradigms/__init__.py"
QWEN_PATH = ROOT / "edit4shape/guidance/pipelines/qwen_image_edit/__init__.py"

REMOVED_TYPES = {
    "distillation",
    "bilevel_distillation",
    "flowedit_gan",
    "flowedit_latent_gan",
    "flowedit_latent_gan_cfgdiff",
    "flowedit_latent_gan_cfgdiff_bt",
    "flowedit_triimage_gan",
    "flowedit_latent_gan_cfgdiff_triimage",
}
REMOVED_EXPORTS = {
    "QwenImageDistillationPipeline",
    "QwenImageBilevelDistillationPipeline",
    "DistillationOutput",
    "BilevelDistillationOutput",
    "InversionStateTracker",
    "create_distillation_tracker",
}


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _string_values(value: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


class GuidanceRegistryContractTest(unittest.TestCase):
    def test_factory_routes_only_flowedit_and_fails_closed(self):
        source = BASE_PATH.read_text(encoding="utf-8")
        factory = _function(BASE_PATH, "create_guidance")

        comparisons = []
        for node in ast.walk(factory):
            if not isinstance(node, ast.Compare):
                continue
            if (
                isinstance(node.left, ast.Name)
                and node.left.id == "paradigm"
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
            ):
                comparisons.append((type(node.ops[0]).__name__, node.comparators[0].value))
        self.assertEqual(comparisons, [("NotEq", "flowedit")])

        returns = {
            node.value.func.id
            for node in ast.walk(factory)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        }
        self.assertEqual(returns, {"FlowEditGuidance", "FlowEditGuidancePP"})
        self.assertTrue(
            any(
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "ValueError"
                for node in ast.walk(factory)
            )
        )
        self.assertFalse(REMOVED_TYPES & _string_values(factory))
        for removed_type in REMOVED_TYPES:
            self.assertNotIn(f'"{removed_type}"', source)

    def test_paradigm_exports_are_flowedit_only(self):
        tree = ast.parse(PARADIGMS_PATH.read_text(encoding="utf-8"))
        exports = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        self.assertEqual(_string_values(exports), {"FlowEditGuidance", "FlowEditGuidancePP"})
        source = PARADIGMS_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"distillation|latent_gan|triimage|discriminator")

    def test_qwen_package_exports_only_retained_pipeline_and_tracker(self):
        tree = ast.parse(QWEN_PATH.read_text(encoding="utf-8"))
        exports = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        export_names = _string_values(exports)
        self.assertTrue({"FlowEditFullPipeline", "FlowEditPipelineOutput", "StateTracker"} <= export_names)
        self.assertIn("create_flowedit_tracker", export_names)
        self.assertFalse(REMOVED_EXPORTS & export_names)

        imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertEqual(
            imports,
            {
                "edit4shape.guidance.pipelines.qwen_image_edit.flowedit_full",
                "edit4shape.guidance.pipelines.qwen_image_edit.trackers",
                "edit4shape.guidance.pipelines.utils",
            },
        )


if __name__ == "__main__":
    unittest.main()
