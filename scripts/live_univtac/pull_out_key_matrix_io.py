"""Canonical atomic persistence helpers for the pull-out-key request matrix."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_json


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one canonical JSON object exactly once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    """Hash a regular file without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Return a path-sorted file map and reject symlinked content."""

    if root.is_symlink() or not root.is_dir():
        raise FileExistsError(f"refusing to overwrite non-directory output: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FileExistsError(f"refusing to trust symlink in output: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_sha256(path)
    return result


def publish_tree(staging: Path, output: Path) -> str:
    """Atomically publish without replacement; identical content is a no-op."""

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if output.exists() or output.is_symlink():
            if tree_hashes(output) == tree_hashes(staging):
                return "already_present"
            raise FileExistsError(f"refusing to overwrite existing matrix: {output}")
        os.rename(staging, output)
        return "created"
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
