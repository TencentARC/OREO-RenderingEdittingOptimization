"""Import-light contract tests for the Qwen/TRELLIS environment definitions."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
SETUP = ROOT / "scripts" / "setup_koala.sh"

EXPECTED_REQUIREMENTS = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "diffusers": "0.38.0",
    "transformers": "4.57.6",
    "tokenizers": "0.22.0",
    "accelerate": "1.4.0",
    "safetensors": "0.8.0",
    "flash-attn": "2.7.3",
    "spconv-cu124": "2.3.8",
}


def requirement_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("# build-with-no-isolation: "):
            line = line.removeprefix("# build-with-no-isolation: ")
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


class QwenEnvironmentContractTest(unittest.TestCase):
    def test_requirements_use_exact_verified_versions(self) -> None:
        requirements = REQUIREMENTS.read_text()
        pins = requirement_pins(requirements)
        for package, expected in EXPECTED_REQUIREMENTS.items():
            self.assertEqual(pins.get(package), expected, package)
        self.assertNotRegex(requirements, re.compile(r"^flash-attn==", re.MULTILINE))
        self.assertIn(
            "# build-with-no-isolation: flash-attn==2.7.3", requirements
        )

    def test_setup_matches_requirements_and_rejects_transformers_5(self) -> None:
        setup = SETUP.read_text()
        self.assertIn("uv venv --python 3.12", setup)
        for package, expected in EXPECTED_REQUIREMENTS.items():
            self.assertIn(f'"{package}": "{expected}"', setup)
            self.assertIn(f"{package}=={expected}", setup)
        self.assertNotRegex(setup, r"transformers\s*[><~!]=")
        self.assertNotIn('"transformers>=', setup)

    def test_setup_never_restores_the_legacy_mutable_venv_cache(self) -> None:
        setup = SETUP.read_text()
        self.assertIn("flow_grpo_v2/uv-venv.tar", setup)
        self.assertNotRegex(setup, re.compile(r'^VENV_TAR=.*?/uv-venv\.tar"$', re.MULTILINE))
        self.assertIn("ENV_CACHE_ID=", setup)
        self.assertIn('VENV_TAR="${S3_V2}/envs/${ENV_CACHE_ID}.tar"', setup)


if __name__ == "__main__":
    unittest.main()
