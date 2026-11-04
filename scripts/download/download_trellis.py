#!/usr/bin/env python3
"""Download TRELLIS-image-large pretrained weights via huggingface_hub.snapshot_download."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TRELLIS weights via snapshot_download.")
    parser.add_argument(
        "--repo-id",
        default="microsoft/TRELLIS-image-large",
        help="HuggingFace Warehouse name, default microsoft/TRELLIS-image-large",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Download directory, default project root/pretrained_weights/TRELLIS-image-large",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only use local cache (not connected to the Internet); the cache must be complete, otherwise an error will be reported",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dest = Path(args.dest) if args.dest else project_root / "pretrained_weights" / "TRELLIS-image-large"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo_id}")
    print(f"Dest: {dest}")

    # enable HF transfer to increase download speed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
        local_files_only=bool(args.local_only),
    )
    print(f"✅ Downloaded {args.repo_id} to {dest}")
    print(f"config.pretrained.model Pointable: {dest}")


if __name__ == "__main__":
    main()
