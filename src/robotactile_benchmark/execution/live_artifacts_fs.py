"""No-follow inventory, read, and streaming hash operations for live bundles."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_MAX_BUNDLE_BYTES,
    LIVE_MAX_MEMBER_BYTES,
    LIVE_MAX_MEMBER_COUNT,
    LiveArtifactValidationError,
    validate_live_member_path,
)


@dataclass(frozen=True)
class LiveFileMetadata:
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class LiveBundleSnapshot:
    root: Path
    root_device: int
    root_inode: int
    files: Mapping[str, LiveFileMetadata]
    directories: frozenset[str]


def scan_live_bundle(root: Path) -> LiveBundleSnapshot:
    """Snapshot a bounded all-regular, no-symlink directory tree without reads."""

    root = Path(root)
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise LiveArtifactValidationError(
            "live artifact root cannot be stated"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise LiveArtifactValidationError("live artifact root must be a real directory")
    files: dict[str, LiveFileMetadata] = {}
    directories: set[str] = set()
    total = 0

    def visit(directory: Path) -> None:
        nonlocal total
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise LiveArtifactValidationError(
                "live artifact cannot be scanned"
            ) from error
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            validate_live_member_path(relative)
            if entry.is_symlink():
                raise LiveArtifactValidationError(f"symlink is forbidden: {relative}")
            if entry.is_dir(follow_symlinks=False):
                directories.add(relative)
                if len(directories) > LIVE_MAX_MEMBER_COUNT:
                    raise LiveArtifactValidationError("too many live directories")
                visit(Path(entry.path))
                continue
            if not entry.is_file(follow_symlinks=False):
                raise LiveArtifactValidationError(f"non-regular member: {relative}")
            member_stat = entry.stat(follow_symlinks=False)
            if not 1 <= member_stat.st_size <= LIVE_MAX_MEMBER_BYTES:
                raise LiveArtifactValidationError(
                    f"member size outside cap: {relative}"
                )
            files[relative] = LiveFileMetadata(
                member_stat.st_size, member_stat.st_dev, member_stat.st_ino
            )
            total += member_stat.st_size
            if len(files) > LIVE_MAX_MEMBER_COUNT + 1 or total > LIVE_MAX_BUNDLE_BYTES:
                raise LiveArtifactValidationError(
                    "live inventory exceeds count/byte cap"
                )

    visit(root)
    return LiveBundleSnapshot(
        root=root,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        files=files,
        directories=frozenset(directories),
    )


def _open_snapshot_member(
    snapshot: LiveBundleSnapshot, relative: str
) -> tuple[int, LiveFileMetadata, os.stat_result]:
    metadata = snapshot.files.get(validate_live_member_path(relative))
    if metadata is None:
        raise LiveArtifactValidationError(f"member is absent: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(snapshot.root / relative, flags)
    except OSError as error:
        raise LiveArtifactValidationError(
            f"member cannot be opened: {relative}"
        ) from error
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or (
        before.st_size,
        before.st_dev,
        before.st_ino,
    ) != (metadata.size_bytes, metadata.device, metadata.inode):
        os.close(descriptor)
        raise LiveArtifactValidationError("member changed before descriptor read")
    return descriptor, metadata, before


def _ensure_unchanged(descriptor: int, before: os.stat_result) -> None:
    after = os.fstat(descriptor)
    if (after.st_size, after.st_dev, after.st_ino) != (
        before.st_size,
        before.st_dev,
        before.st_ino,
    ):
        raise LiveArtifactValidationError("member changed during descriptor read")


def read_live_member(
    snapshot: LiveBundleSnapshot, relative: str, maximum: int
) -> bytes:
    """Read exactly one snapshotted regular file through O_NOFOLLOW."""

    descriptor, metadata, before = _open_snapshot_member(snapshot, relative)
    try:
        if metadata.size_bytes > maximum:
            raise LiveArtifactValidationError(f"member is too large: {relative}")
        chunks: list[bytes] = []
        remaining = metadata.size_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise LiveArtifactValidationError("member was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LiveArtifactValidationError("member grew during read")
        _ensure_unchanged(descriptor, before)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def hash_live_member(snapshot: LiveBundleSnapshot, relative: str) -> str:
    """Stream-hash one member without materializing a potentially large NPY."""

    descriptor, metadata, before = _open_snapshot_member(snapshot, relative)
    try:
        digest = hashlib.sha256()
        total = 0
        while total < metadata.size_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, metadata.size_bytes - total))
            if not chunk:
                raise LiveArtifactValidationError("member truncated while hashing")
            digest.update(chunk)
            total += len(chunk)
        if os.read(descriptor, 1):
            raise LiveArtifactValidationError("member grew while hashing")
        _ensure_unchanged(descriptor, before)
        return digest.hexdigest()
    finally:
        os.close(descriptor)
