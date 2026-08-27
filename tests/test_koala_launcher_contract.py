"""Import-light contracts for the formal Koala launch profiles."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
KOALA_DIR = ROOT / "scripts" / "koala"
COMMON = KOALA_DIR / "run_8gpu_teacher_cfg_mp_no.sh"
TRUE = KOALA_DIR / "run_8gpu_teacher_cfg_true_mp_no.sh"
FALSE = KOALA_DIR / "run_8gpu_teacher_cfg_false_mp_no.sh"
YAW = KOALA_DIR / "run_8gpu_teacher_cfg_true_yaw90_270_mp_no.sh"
YAW_CONFIG = ROOT / "config" / "trellis_stage1+2_contrastive_yaw90_270.py"


class KoalaLauncherContractTest(unittest.TestCase):
    def test_all_launchers_have_valid_bash_syntax(self) -> None:
        for launcher in (COMMON, TRUE, FALSE, YAW):
            result = subprocess.run(
                ["bash", "-n", str(launcher)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_yaw_profile_is_a_fixed_teacher_cfg_true_wrapper(self) -> None:
        source = YAW.read_text(encoding="utf-8")
        self.assertIn(
            '"--config=config/trellis_stage1+2_contrastive_yaw90_270.py"',
            source,
        )
        self.assertRegex(
            source,
            re.compile(
                r'run_8gpu_teacher_cfg_mp_no\.sh" true \\\n+\s+"--config=config/trellis_stage1\+2_contrastive_yaw90_270\.py" "\$@"'
            ),
        )

    def test_derived_config_changes_only_training_yaw_range(self) -> None:
        source = YAW_CONFIG.read_text(encoding="utf-8")
        self.assertIn('"trellis_stage1+2_contrastive.py"', source)
        self.assertIn("cfg.data.train.yaw_range = [90.0, 270.0]", source)
        self.assertNotIn("cfg.data.eval.yaw_range", source)

    def test_common_launcher_selects_config_and_records_effective_range(self) -> None:
        source = COMMON.read_text(encoding="utf-8")
        self.assertIn(
            'BASE_CONFIG_PATH="config/trellis_stage1+2_contrastive.py"',
            source,
        )
        self.assertIn(
            'YAW_CONFIG_PATH="config/trellis_stage1+2_contrastive_yaw90_270.py"',
            source,
        )
        self.assertIn('TRAIN_YAW_RANGE_EFFECTIVE="[0.0,360.0]"', source)
        self.assertIn('TRAIN_YAW_RANGE_EFFECTIVE="[90.0,270.0]"', source)
        self.assertIn(
            '--config="${CONFIG_PATH}"',
            source,
        )
        self.assertIn("config=${CONFIG_PATH}", source)
        self.assertIn("train_yaw_range=${TRAIN_YAW_RANGE_EFFECTIVE}", source)
        self.assertIn('\\"train_yaw_range\\":${TRAIN_YAW_RANGE_EFFECTIVE}', source)

    def test_existing_true_and_false_wrappers_remain_baseline_delegates(self) -> None:
        for launcher, teacher_cfg in ((TRUE, "true"), (FALSE, "false")):
            source = launcher.read_text(encoding="utf-8")
            self.assertRegex(
                source,
                re.compile(
                    rf'run_8gpu_teacher_cfg_mp_no\.sh" {teacher_cfg} "\$@"'
                ),
            )
            self.assertNotIn("yaw_range", source)


if __name__ == "__main__":
    unittest.main()
