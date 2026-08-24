#!/usr/bin/env python3
"""Build, verify, or safely extract the pinned Qwen HF snapshot archive.

Offline archive workflow (run on the Linux/Koala builder that owns the cache)::

    python scripts/qwen_snapshot_archive.py verify-cache "$CACHE_ROOT"
    python scripts/qwen_snapshot_archive.py build "$CACHE_ROOT" /local-ssd/qwen.tar
    python scripts/qwen_snapshot_archive.py verify-archive /local-ssd/qwen.tar

Only after all three commands pass should an operator upload the new file to the
immutable URI printed by ``scripts/setup_koala.sh``. The helper itself never
accesses S3 or Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


MODEL_CACHE_NAME = "models--Qwen--Qwen-Image-Edit-2511"
EXPECTED_COMMIT = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
SNAPSHOT_MANIFEST = {
    ".gitattributes": ("b2ef89be59c5792bf037a43d1bb69d922d28e9d8", 1580),
    "README.md": ("277c39d3f5bb2c3908b435727e80c98af75f934a", 7179),
    "model_index.json": ("a2e083f28f4f6c958c08f3882c89038b8f6da872", 516),
    "processor/added_tokens.json": ("482ced4679301bf287ebb310bdd1790eb4514232", 605),
    "processor/chat_template.jinja": ("6c226632394ae7474b0d4b13e15793eac2e21ee9", 1017),
    "processor/merges.txt": ("31349551d90c7606f325fe0f11bbb8bd5fa0d7c7", 1671853),
    "processor/preprocessor_config.json": ("9a24c5165e5de348ba27d0d00dd52b27c6044ab2", 826),
    "processor/special_tokens_map.json": ("ac23c0aaa2434523c494330aeb79c58395378103", 613),
    "processor/tokenizer.json": ("9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa", 11421896),
    "processor/tokenizer_config.json": ("203eb2267a9b927c93a97b67990818164bf3c692", 4727),
    "processor/video_preprocessor_config.json": ("ef8f979d86a2255c8fc115327eefce152e5a7cf3", 910),
    "processor/vocab.json": ("4783fe10ac3adce15ac8f358ef5462739852c569", 2776833),
    "scheduler/scheduler_config.json": ("f400c338630cdaa261c6575648f374eba547b9c6", 485),
    "text_encoder/config.json": ("52b1556c2927e08c13529f2a107b08c87c7a4e65", 3146),
    "text_encoder/generation_config.json": ("5fdb82f3b3f12d83074ce7b45e330d8ee4efb7d7", 244),
    "text_encoder/model-00001-of-00004.safetensors": ("d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4", 4968243304),
    "text_encoder/model-00002-of-00004.safetensors": ("b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3", 4991495816),
    "text_encoder/model-00003-of-00004.safetensors": ("09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6", 4932751040),
    "text_encoder/model-00004-of-00004.safetensors": ("5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a", 1691924384),
    "text_encoder/model.safetensors.index.json": ("92b467639266f4aa72a9b5ab73dbdf5a1f936ee4", 57655),
    "tokenizer/added_tokens.json": ("482ced4679301bf287ebb310bdd1790eb4514232", 605),
    "tokenizer/chat_template.jinja": ("28028c056af412405debd878cdda0171e35fa5d1", 2427),
    "tokenizer/merges.txt": ("31349551d90c7606f325fe0f11bbb8bd5fa0d7c7", 1671853),
    "tokenizer/special_tokens_map.json": ("ac23c0aaa2434523c494330aeb79c58395378103", 613),
    "tokenizer/tokenizer_config.json": ("eaed590d62aaf0ba31e284b66ddcb18222f066c2", 4686),
    "tokenizer/vocab.json": ("6bce3a0a3866c4791a74d83d78f6824c3af64ec3", 3383407),
    "transformer/config.json": ("199398df180d275eee9da066489f9bb6e2b97609", 362),
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": ("2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a", 9973578592),
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": ("54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259", 9987326072),
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": ("c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16", 9987307440),
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": ("ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d", 9930685712),
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": ("2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940", 982130472),
    "transformer/diffusion_pytorch_model.safetensors.index.json": ("2fdb0449e2b2299b77e056c95b1a7caec5a3f438", 198887),
    "vae/config.json": ("ebc479e2fe5a2265f4bcaf1e1073df64a2d4e33e", 730),
    "vae/diffusion_pytorch_model.safetensors": ("0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344", 253806966),
}


class SnapshotContractError(RuntimeError):
    """The cache or archive does not satisfy the pinned snapshot contract."""


def _digest_stream(stream, blob_id: str, size: int) -> str:
    if len(blob_id) == 64:
        digest = hashlib.sha256()
    elif len(blob_id) == 40:
        digest = hashlib.sha1(f"blob {size}\0".encode())
    else:
        raise SnapshotContractError(f"unsupported blob ID: {blob_id}")
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def verify_cache(cache_root: Path) -> Path:
    cache_root = cache_root.resolve()
    refs_main = cache_root / "refs" / "main"
    snapshot = cache_root / "snapshots" / EXPECTED_COMMIT
    blobs = cache_root / "blobs"

    if not refs_main.is_file():
        raise SnapshotContractError(f"missing refs/main: {refs_main}")
    if refs_main.read_text().strip() != EXPECTED_COMMIT:
        raise SnapshotContractError("refs/main does not name the expected commit")
    if not snapshot.is_dir() or not blobs.is_dir():
        raise SnapshotContractError("missing exact snapshots/<commit> or blobs directory")

    snapshot_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if snapshot_files != set(SNAPSHOT_MANIFEST):
        missing = sorted(set(SNAPSHOT_MANIFEST) - snapshot_files)
        extra = sorted(snapshot_files - set(SNAPSHOT_MANIFEST))
        raise SnapshotContractError(
            f"snapshot manifest mismatch; missing={missing}, extra={extra}"
        )

    verified_blobs: set[str] = set()
    for relative, (blob_id, expected_size) in SNAPSHOT_MANIFEST.items():
        path = snapshot / relative
        if not path.is_symlink() or not path.exists():
            raise SnapshotContractError(f"missing or broken snapshot file: {relative}")
        target = Path(os.readlink(path))
        if target.is_absolute():
            raise SnapshotContractError(f"absolute snapshot symlink: {path}")
        resolved = (path.parent / target).resolve(strict=False)
        expected_blob = (blobs / blob_id).resolve(strict=False)
        if resolved != expected_blob or not _inside(resolved, cache_root):
            raise SnapshotContractError(
                f"snapshot file targets the wrong blob: {relative}"
            )
        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise SnapshotContractError(
                f"snapshot file has wrong size: {relative} "
                f"(expected {expected_size}, found {actual_size})"
            )
        if blob_id not in verified_blobs:
            with resolved.open("rb") as blob:
                actual_digest = _digest_stream(blob, blob_id, expected_size)
            if actual_digest != blob_id:
                raise SnapshotContractError(f"blob digest mismatch: {relative}")
            verified_blobs.add(blob_id)

    for path in snapshot.rglob("*"):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        if target.is_absolute():
            raise SnapshotContractError(f"absolute snapshot symlink: {path}")
        resolved = (path.parent / target).resolve(strict=False)
        if not _inside(resolved, cache_root) or not resolved.is_file():
            raise SnapshotContractError(f"unsafe or broken snapshot symlink: {path}")
        try:
            resolved.relative_to(blobs.resolve())
        except ValueError as exc:
            raise SnapshotContractError(
                f"snapshot symlink does not resolve into blobs: {path}"
            ) from exc
    return snapshot


def _normalized_link_target(member: tarfile.TarInfo) -> str | None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise SnapshotContractError(f"unsafe archive member path: {member.name}")
    if not path.parts or path.parts[0] != MODEL_CACHE_NAME:
        raise SnapshotContractError(f"unexpected archive root: {member.name}")
    if member.issym():
        target = PurePosixPath(member.linkname)
        if target.is_absolute():
            raise SnapshotContractError(f"absolute archive link: {member.name}")
        resolved = PurePosixPath(*path.parent.parts, *target.parts)
    elif member.islnk():
        resolved = PurePosixPath(member.linkname)
        if resolved.is_absolute():
            raise SnapshotContractError(f"absolute archive link: {member.name}")
    else:
        return None
    if member.issym() or member.islnk():
        normalized: list[str] = []
        for part in resolved.parts:
            if part == "..":
                if not normalized:
                    raise SnapshotContractError(f"escaping archive link: {member.name}")
                normalized.pop()
            elif part not in ("", "."):
                normalized.append(part)
        if not normalized or normalized[0] != MODEL_CACHE_NAME:
            raise SnapshotContractError(f"escaping archive link: {member.name}")
        return PurePosixPath(*normalized).as_posix()
    return None


def verify_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        if not members:
            raise SnapshotContractError("empty archive")
        link_targets: list[str] = []
        for member in members:
            target = _normalized_link_target(member)
            if target is not None:
                link_targets.append(target)
        names = {member.name.rstrip("/") for member in members}
        required = {
            f"{MODEL_CACHE_NAME}/refs/main",
            f"{MODEL_CACHE_NAME}/snapshots/{EXPECTED_COMMIT}",
            f"{MODEL_CACHE_NAME}/blobs",
        }
        manifest_members = {
            f"{MODEL_CACHE_NAME}/snapshots/{EXPECTED_COMMIT}/{relative}"
            for relative in SNAPSHOT_MANIFEST
        }
        required.update(manifest_members)
        missing = required - names
        if missing:
            raise SnapshotContractError(f"archive missing roots: {sorted(missing)}")
        missing_targets = set(link_targets) - names
        if missing_targets:
            raise SnapshotContractError(
                f"archive contains broken links: {sorted(missing_targets)[:3]}"
            )
        member_by_name = {member.name.rstrip("/"): member for member in members}
        snapshot_prefix = f"{MODEL_CACHE_NAME}/snapshots/{EXPECTED_COMMIT}/"
        snapshot_file_members = {
            name.removeprefix(snapshot_prefix)
            for name, member in member_by_name.items()
            if name.startswith(snapshot_prefix) and not member.isdir()
        }
        if snapshot_file_members != set(SNAPSHOT_MANIFEST):
            extra = sorted(snapshot_file_members - set(SNAPSHOT_MANIFEST))
            raise SnapshotContractError(f"archive has extra snapshot entries: {extra}")
        verified_blobs: set[str] = set()
        for relative, (blob_id, expected_size) in SNAPSHOT_MANIFEST.items():
            name = f"{MODEL_CACHE_NAME}/snapshots/{EXPECTED_COMMIT}/{relative}"
            member = member_by_name[name]
            if not member.issym():
                raise SnapshotContractError(f"manifest entry is not a symlink: {relative}")
            target = _normalized_link_target(member)
            expected_target = f"{MODEL_CACHE_NAME}/blobs/{blob_id}"
            if target != expected_target:
                raise SnapshotContractError(
                    f"manifest entry targets the wrong blob: {relative}"
                )
            if member_by_name[target].size != expected_size:
                actual_size = None if target is None else member_by_name[target].size
                raise SnapshotContractError(
                    f"archive blob has wrong size for {relative}: "
                    f"expected {expected_size}, found {actual_size}"
                )
            if blob_id not in verified_blobs:
                blob_file = archive.extractfile(member_by_name[target])
                if blob_file is None:
                    raise SnapshotContractError(f"cannot read archive blob: {blob_id}")
                actual_digest = _digest_stream(blob_file, blob_id, expected_size)
                if actual_digest != blob_id:
                    raise SnapshotContractError(f"archive blob digest mismatch: {relative}")
                verified_blobs.add(blob_id)
        refs_member = archive.getmember(f"{MODEL_CACHE_NAME}/refs/main")
        refs_file = archive.extractfile(refs_member)
        if refs_file is None or refs_file.read().decode().strip() != EXPECTED_COMMIT:
            raise SnapshotContractError("archive refs/main has the wrong commit")


def build_archive(cache_root: Path, output: Path) -> None:
    snapshot = verify_cache(cache_root)
    cache_root = cache_root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    linked_blobs: set[Path] = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            linked_blobs.add(path.resolve(strict=True))

    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, "w") as archive:
            for directory in ("refs", "snapshots", "blobs"):
                info = tarfile.TarInfo(f"{MODEL_CACHE_NAME}/{directory}")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            archive.add(
                cache_root / "refs" / "main",
                arcname=f"{MODEL_CACHE_NAME}/refs/main",
                recursive=False,
            )
            archive.add(
                snapshot,
                arcname=f"{MODEL_CACHE_NAME}/snapshots/{EXPECTED_COMMIT}",
                recursive=True,
            )
            for blob in sorted(linked_blobs):
                archive.add(
                    blob,
                    arcname=f"{MODEL_CACHE_NAME}/blobs/{blob.name}",
                    recursive=False,
                )
        verify_archive(temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_archive(archive_path: Path, hub_root: Path) -> Path:
    verify_archive(archive_path)
    destination = hub_root.resolve()
    cache_root = destination / MODEL_CACHE_NAME
    if cache_root.exists():
        raise SnapshotContractError(f"refusing to replace existing cache: {cache_root}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination, filter="data")
    return verify_cache(cache_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_cache_parser = subparsers.add_parser("verify-cache")
    verify_cache_parser.add_argument("cache_root", type=Path)

    verify_archive_parser = subparsers.add_parser("verify-archive")
    verify_archive_parser.add_argument("archive", type=Path)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("cache_root", type=Path)
    build_parser.add_argument("output", type=Path)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("archive", type=Path)
    extract_parser.add_argument("hub_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "verify-cache":
            verify_cache(args.cache_root)
        elif args.command == "verify-archive":
            verify_archive(args.archive)
        elif args.command == "build":
            build_archive(args.cache_root, args.output)
        else:
            extract_archive(args.archive, args.hub_root)
    except (OSError, tarfile.TarError, SnapshotContractError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
