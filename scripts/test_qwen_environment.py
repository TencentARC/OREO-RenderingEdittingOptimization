#!/usr/bin/env python3
"""Explicit runtime diagnostic for the pinned Qwen/TRELLIS environment.

This script is intentionally not called by a training entry point. Run it directly
after constructing or restoring an environment.
"""

from importlib.metadata import PackageNotFoundError, version
import sys


EXPECTED = {
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


def main() -> int:
    failures: list[str] = []
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info[:2] != (3, 12):
        failures.append(f"python: expected 3.12, found {actual_python}")

    for package, expected in EXPECTED.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            failures.append(f"{package}: not installed (expected {expected})")
            continue
        if actual != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")

    if failures:
        print("Qwen/TRELLIS environment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    from diffusers import QwenImageEditPlusPipeline  # noqa: F401

    print("Qwen/TRELLIS environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
