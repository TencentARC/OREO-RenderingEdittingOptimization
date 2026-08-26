"""Import-light contracts for the reference dual-stage training setup."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "trellis_stage1+2_contrastive.py"
ENTRY_PATH = (
    ROOT
    / "edit4shape"
    / "systems"
    / "trellis"
    / "entries"
    / "contrastive_dualstage_autograd.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignment(source: ast.AST, dotted_name: str) -> ast.Assign:
    """Return the assignment whose target is the requested dotted name."""
    for node in ast.walk(source):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if ast.unparse(target) == dotted_name:
                return node
    raise AssertionError(f"missing assignment: {dotted_name}")


def _constant_assignment(source: ast.AST, dotted_name: str):
    assignment = _assignment(source, dotted_name)
    return ast.literal_eval(assignment.value)


class DualStageConfigContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _parse(CONFIG_PATH)
        cls.entry = _parse(ENTRY_PATH)

    def test_reference_defaults_match_reference_run(self) -> None:
        self.assertEqual(_constant_assignment(self.config, "cfg.mixed_precision"), "bf16")
        self.assertEqual(
            _constant_assignment(self.config, "tr.student_denoise_cfg"), False
        )
        self.assertEqual(
            _constant_assignment(self.config, "tr.loss.contrastive.teacher_cfg"), True
        )

    def test_flowedit_defaults_keep_reverse_cfg_and_white_prompt(self) -> None:
        self.assertEqual(
            _constant_assignment(self.config, "cfg.target_prompt"),
            "Rotate the camera. White background.",
        )
        self.assertEqual(
            ast.unparse(_assignment(self.config, "cfg.source_prompt").value),
            "cfg.target_prompt",
        )
        source_cfg = _assignment(self.config, "cfg.true_cfg_scale_src")
        self.assertEqual(ast.unparse(source_cfg.value), "-1 * cfg.true_cfg_scale_tgt")
        self.assertEqual(_constant_assignment(self.config, "cfg.true_cfg_scale_tgt"), 4)

    def test_dualstage_accelerator_reads_top_level_precision(self) -> None:
        calls = [
            node
            for node in ast.walk(self.entry)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Accelerator"
        ]
        self.assertEqual(len(calls), 1)
        precision = next(
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "mixed_precision"
        )
        self.assertEqual(ast.unparse(precision), "cfg.mixed_precision")


if __name__ == "__main__":
    unittest.main()
