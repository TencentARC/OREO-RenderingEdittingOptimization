"""Small, network-free tests for the pinned Qwen snapshot archive contract."""

from __future__ import annotations

import io
import hashlib
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.qwen_snapshot_archive import (
    EXPECTED_COMMIT,
    MODEL_CACHE_NAME,
    SNAPSHOT_MANIFEST,
    SnapshotContractError,
    build_archive,
    extract_archive,
    verify_archive,
    verify_cache,
)

WEIGHT_SHARD = "transformer/diffusion_pytorch_model-00001-of-00005.safetensors"
TEST_CONTENTS = {
    "model_index.json": b"index-1",
    "transformer/config.json": b"config-12",
    WEIGHT_SHARD: b"weight-data",
}


def git_blob_id(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


TEST_MANIFEST = {
    "model_index.json": (git_blob_id(TEST_CONTENTS["model_index.json"]), 7),
    "transformer/config.json": (
        git_blob_id(TEST_CONTENTS["transformer/config.json"]),
        9,
    ),
    WEIGHT_SHARD: (hashlib.sha256(TEST_CONTENTS[WEIGHT_SHARD]).hexdigest(), 11),
}


def make_cache(parent: Path, manifest: dict[str, int] = TEST_MANIFEST) -> Path:
    cache = parent / MODEL_CACHE_NAME
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / EXPECTED_COMMIT
    (cache / "refs").mkdir(parents=True)
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    (cache / "refs" / "main").write_text(f"{EXPECTED_COMMIT}\n")
    for relative, (blob_id, size) in manifest.items():
        blob = blobs / blob_id
        content = TEST_CONTENTS[relative]
        assert len(content) == size
        blob.write_bytes(content)
        link = snapshot / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return cache


class QwenSnapshotArchiveTest(unittest.TestCase):
    def test_build_verify_extract_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root / "source")
            archive = root / "qwen.tar"
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                build_archive(cache, archive)
                verify_archive(archive)
                snapshot = extract_archive(archive, root / "restored-hub")
                self.assertEqual(snapshot.name, EXPECTED_COMMIT)
                verify_cache(root / "restored-hub" / MODEL_CACHE_NAME)

    def test_cache_rejects_wrong_commit_and_broken_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache(Path(directory))
            (cache / "refs" / "main").write_text("wrong-commit\n")
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                with self.assertRaises(SnapshotContractError):
                    verify_cache(cache)
                (cache / "refs" / "main").write_text(f"{EXPECTED_COMMIT}\n")
                broken = cache / "snapshots" / EXPECTED_COMMIT / "model_index.json"
                broken.unlink()
                broken.symlink_to("../../blobs/missing")
                with self.assertRaises(SnapshotContractError):
                    verify_cache(cache)

    def test_cache_rejects_missing_required_transformer_weight_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache(Path(directory))
            shard = cache / "snapshots" / EXPECTED_COMMIT / WEIGHT_SHARD
            shard.unlink()
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                with self.assertRaisesRegex(SnapshotContractError, WEIGHT_SHARD):
                    verify_cache(cache)

    def test_cache_rejects_wrong_blob_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache(Path(directory))
            shard = cache / "snapshots" / EXPECTED_COMMIT / WEIGHT_SHARD
            shard.resolve().write_bytes(b"short")
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                with self.assertRaisesRegex(SnapshotContractError, "wrong size"):
                    verify_cache(cache)

    def test_cache_rejects_same_size_blob_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache(Path(directory))
            shard = cache / "snapshots" / EXPECTED_COMMIT / WEIGHT_SHARD
            shard.resolve().write_bytes(b"x" * TEST_MANIFEST[WEIGHT_SHARD][1])
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                with self.assertRaisesRegex(SnapshotContractError, "digest mismatch"):
                    verify_cache(cache)

    def test_cache_rejects_extra_snapshot_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache(Path(directory))
            extra_blob = cache / "blobs" / git_blob_id(b"extra")
            extra_blob.write_bytes(b"extra")
            extra = cache / "snapshots" / EXPECTED_COMMIT / "extra.json"
            extra.symlink_to(os.path.relpath(extra_blob, extra.parent))
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                with self.assertRaisesRegex(SnapshotContractError, "extra"):
                    verify_cache(cache)

    def test_official_manifest_has_all_35_files_and_pinned_shard_size(self) -> None:
        self.assertEqual(len(SNAPSHOT_MANIFEST), 35)
        self.assertEqual(
            SNAPSHOT_MANIFEST[WEIGHT_SHARD],
            (
                "2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
                9973578592,
            ),
        )

    def test_archive_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.tar"
            with tarfile.open(archive_path, "w") as archive:
                payload = b"bad"
                member = tarfile.TarInfo("../outside")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with self.assertRaises(SnapshotContractError):
                verify_archive(archive_path)

    def test_archive_rejects_same_size_blob_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root / "source")
            valid_archive = root / "valid.tar"
            corrupt_archive = root / "corrupt.tar"
            weight_blob_id, weight_size = TEST_MANIFEST[WEIGHT_SHARD]
            weight_member = f"{MODEL_CACHE_NAME}/blobs/{weight_blob_id}"
            with patch(
                "scripts.qwen_snapshot_archive.SNAPSHOT_MANIFEST", TEST_MANIFEST
            ):
                build_archive(cache, valid_archive)
                with tarfile.open(valid_archive) as source, tarfile.open(
                    corrupt_archive, "w"
                ) as destination:
                    for member in source.getmembers():
                        if member.name == weight_member:
                            replacement = tarfile.TarInfo(member.name)
                            replacement.size = weight_size
                            replacement.mode = member.mode
                            destination.addfile(
                                replacement, io.BytesIO(b"x" * weight_size)
                            )
                        else:
                            destination.addfile(member, source.extractfile(member))
                with self.assertRaisesRegex(SnapshotContractError, "digest mismatch"):
                    verify_archive(corrupt_archive)


class QwenFastSetupContractTest(unittest.TestCase):
    def test_fast_mode_is_offline_and_never_uses_legacy_qwen_tar(self) -> None:
        setup = (
            Path(__file__).resolve().parents[1] / "scripts" / "setup_koala.sh"
        ).read_text()
        self.assertIn(f'QWEN_COMMIT="{EXPECTED_COMMIT}"', setup)
        self.assertIn("cache_dir='${HF_HOME}/hub'", setup)
        self.assertNotIn("cache_dir='${HF_HOME}',", setup)
        self.assertIn('QWEN_TAR="${S3_V2}/models/${QWEN_ARCHIVE_ID}.tar"', setup)
        self.assertNotIn('QWEN_TAR="${S3_SHARED}/qwen-image-edit-2511.tar"', setup)
        self.assertIn("export HF_HUB_OFFLINE=1", setup)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", setup)
        self.assertIn("export DIFFUSERS_OFFLINE=1", setup)
        self.assertIn("Required immutable Qwen archive not found", setup)
        self.assertIn("Skipped in --fast mode", setup)
        self.assertIn("--qwen-only requires --fast", setup)
        self.assertIn("Skipped by explicit --qwen-only mode", setup)
        self.assertIn("Required DINOv2 cache not found", setup)


if __name__ == "__main__":
    unittest.main()
