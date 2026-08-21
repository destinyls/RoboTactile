"""Canonical filesystem helpers for generated primary matrix bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
)
from robotactile_benchmark.closed_loop.artifact_io import strict_json_bytes
from robotactile_benchmark.matrix.io import canonical_matrix_json_bytes
from robotactile_benchmark.matrix.primary_generation_contracts import (
    PrimaryMatrixGenerationError,
)

_MAX_FILE_BYTES = 16 * 1024 * 1024


def file_sha256(path: Path) -> str:
    """Hash one regular file after rejecting symlinks."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PrimaryMatrixGenerationError("bundle member must be a regular file")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Create one canonical JSON member without replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_matrix_json_bytes(value))


def read_json(path: Path, label: str) -> object:
    """Read exact canonical JSON with bounded size and duplicate-key rejection."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PrimaryMatrixGenerationError(f"{label} must be a regular file")
    before = candidate.stat()
    if not 1 <= before.st_size <= _MAX_FILE_BYTES:
        raise PrimaryMatrixGenerationError(f"{label} size is invalid")
    raw = candidate.read_bytes()
    after = candidate.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PrimaryMatrixGenerationError(f"{label} changed while reading")
    try:
        value = strict_json_bytes(raw, label)
    except ArtifactValidationError as error:
        raise PrimaryMatrixGenerationError(str(error)) from error
    if canonical_matrix_json_bytes(value) != raw:
        raise PrimaryMatrixGenerationError(f"{label} is not canonical JSON")
    return value


def scan_files(root: Path, *, exclude: Optional[str] = None) -> dict[str, str]:
    """Return a sorted content inventory and reject all non-files/symlinks."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise PrimaryMatrixGenerationError("bundle root must be a real directory")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise PrimaryMatrixGenerationError("bundle cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PrimaryMatrixGenerationError("bundle members must be regular files")
        if relative != exclude:
            result[relative] = file_sha256(path)
    return result


def copy_regular_file(source: Path, destination: Path) -> None:
    """Copy a bounded regular member while preserving only its bytes."""

    origin = Path(source)
    if origin.is_symlink() or not origin.is_file():
        raise PrimaryMatrixGenerationError("copy source must be a regular file")
    if not 1 <= origin.stat().st_size <= _MAX_FILE_BYTES:
        raise PrimaryMatrixGenerationError("copy source size is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(origin.read_bytes())


def publish_staging(staging: Path, output: Path) -> str:
    """Atomically publish one tree, accepting only an identical existing tree."""

    source = Path(staging)
    target = Path(output).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise PrimaryMatrixGenerationError(
            "another bundle publication is active"
        ) from error
    try:
        if target.exists() or target.is_symlink():
            if (
                not target.is_symlink()
                and target.is_dir()
                and scan_files(target) == scan_files(source)
            ):
                return "already_present"
            raise FileExistsError(f"refusing to replace existing output: {target}")
        os.rename(source, target)
        return "created"
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def staging_directory(output: Path) -> Path:
    """Allocate a private staging sibling for atomic directory publication."""

    target = Path(output).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))


def discard_staging(path: Path) -> None:
    """Remove only a known staging directory after generation."""

    candidate = Path(path)
    if candidate.exists():
        shutil.rmtree(candidate)


def relative_run_config_document(
    document: Mapping[str, object], resources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Copy a typed run-config document with portable bundled resource paths."""

    result = dict(document)
    result["resources"] = {key: dict(value) for key, value in resources.items()}
    return result
