"""Contract tests for unconditional-only FlowEdit source guidance.

These tests inspect the lightweight Python syntax tree so they do not import
Diffusers or load Qwen weights.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "edit4shape/guidance/pipelines/qwen_image_edit/flowedit_full.py"
PARADIGM_PATH = ROOT / "edit4shape/guidance/paradigms/flowedit.py"
CONFIG_PATHS = tuple((ROOT / "config").glob("*.py"))
REMOVED_SOURCE_CFG_NAMES = {
    "source_prompt",
    "negative_prompt_src",
    "true_cfg_scale_src",
}


def _call_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )


def _assignment_line(method: ast.FunctionDef, name: str) -> int:
    return next(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


def _assigned_name(method: ast.FunctionDef, name: str) -> str:
    assignment = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Name)
    return assignment.value.id


def _transformer_calls(method: ast.FunctionDef, start: int, end: int) -> list[ast.Call]:
    return [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "transformer"
        and start < node.lineno < end
    ]


def _keyword_name(call: ast.Call, keyword: str) -> str:
    value = next(item.value for item in call.keywords if item.arg == keyword)
    assert isinstance(value, ast.Name)
    return value.id


class FlowEditSourceUnconditionalTest(unittest.TestCase):
    def test_source_branch_has_one_unconditional_transformer_call_per_step(self):
        method = _call_method(PIPELINE_PATH, "FlowEditPipeline", "__call__")
        self.assertEqual(
            _assigned_name(method, "source_uncond_embeds"),
            "negative_prompt_embeds_tgt",
        )
        self.assertEqual(
            _assigned_name(method, "source_uncond_embeds_mask"),
            "negative_prompt_embeds_mask_tgt",
        )

        source_start = _assignment_line(method, "latents_src")
        target_start = _assignment_line(method, "latents_tgt")
        source_calls = _transformer_calls(method, source_start, target_start)

        self.assertEqual(len(source_calls), 1)
        self.assertEqual(
            _keyword_name(source_calls[0], "encoder_hidden_states"),
            "source_uncond_embeds",
        )
        self.assertEqual(
            _keyword_name(source_calls[0], "encoder_hidden_states_mask"),
            "source_uncond_embeds_mask",
        )

        source_nodes = [
            node for node in ast.walk(method)
            if source_start < getattr(node, "lineno", -1) < target_start
        ]
        source_names = {node.id for node in source_nodes if isinstance(node, ast.Name)}
        self.assertFalse(source_names & REMOVED_SOURCE_CFG_NAMES)
        self.assertFalse(
            any(name.startswith("v_cond_src") or name.startswith("v_cfg_src") for name in source_names)
        )

    def test_target_branch_retains_conditional_unconditional_cfg(self):
        method = _call_method(PIPELINE_PATH, "FlowEditPipeline", "__call__")
        target_start = _assignment_line(method, "latents_tgt")
        update_start = _assignment_line(method, "v_delta")
        target_calls = _transformer_calls(method, target_start, update_start)

        self.assertEqual(len(target_calls), 2)
        self.assertEqual(
            {
                _keyword_name(call, "encoder_hidden_states") for call in target_calls
            },
            {"prompt_embeds_tgt", "negative_prompt_embeds_tgt"},
        )

        target_names = {
            node.id
            for node in ast.walk(method)
            if isinstance(node, ast.Name) and target_start < node.lineno < update_start
        }
        self.assertLessEqual(
            {"true_cfg_scale_tgt", "v_cond_tgt", "v_uncond_tgt", "v_cfg_tgt"},
            target_names,
        )

    def test_removed_source_cfg_options_are_absent_from_runtime_interfaces(self):
        pipeline_call = _call_method(PIPELINE_PATH, "FlowEditPipeline", "__call__")
        self.assertFalse(
            set(arg.arg for arg in pipeline_call.args.args) & REMOVED_SOURCE_CFG_NAMES
        )

        paradigm_source = PARADIGM_PATH.read_text(encoding="utf-8")
        self.assertTrue(all(name not in paradigm_source for name in REMOVED_SOURCE_CFG_NAMES))

        config_source = "\n".join(path.read_text(encoding="utf-8") for path in CONFIG_PATHS)
        self.assertTrue(all(name not in config_source for name in REMOVED_SOURCE_CFG_NAMES))


if __name__ == "__main__":
    unittest.main()
