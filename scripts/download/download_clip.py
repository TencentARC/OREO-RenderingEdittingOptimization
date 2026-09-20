#!/usr/bin/env python3
"""
Download CLIP Model and processor to the project root directory pretrained_weights/clip/clip-vit-large-patch14
Convenient for offline loading through local paths (model_id with processor_id pointing to the same directory):
  - clip_model_id      -> pretrained_weights/clip/clip-vit-large-patch14
  - clip_processor_id  -> pretrained_weights/clip/clip-vit-large-patch14
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def snapshot_to_dir(repo_id: str, out_dir: Path, revision: str = "main") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CLIP weight to pretrained_weights/clip for offline loading")
    parser.add_argument("--repo", default="openai/clip-vit-large-patch14", help="CLIP The warehouse where the model and processor are located")
    parser.add_argument("--revision", default="main", help="HF warehouse branch/revision")
    parser.add_argument(
        "--out",
        default=None,
        help="Output root directory (default writes to <project_root>/pretrained_weights/clip/clip-vit-large-patch14)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = Path(args.out) if args.out is not None else (project_root / "pretrained_weights" / "clip" / "clip-vit-large-patch14")
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"➡️  Download CLIP: {args.repo} -> {out_root}")
    path = snapshot_to_dir(args.repo, out_root, args.revision)
    print(f"✅ Complete: {path}")

    # Shows the relative path that should be set in the configuration file
    try:
        rel_dir = out_root.relative_to(project_root)
    except ValueError:
        rel_dir = out_root

    print("\n📌 Please use the following local paths in your configuration (same model as processor):")
    print(f"clip_model_id: {rel_dir}")
    print(f"clip_processor_id: {rel_dir}")
    print("\n🎉 All done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


