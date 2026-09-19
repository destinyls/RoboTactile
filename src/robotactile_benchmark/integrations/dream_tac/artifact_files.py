"""Stable file hashing helpers for Dream-Tac artifact manifests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def absolute_path(value: Path, name: str) -> Path:
    """Require one explicit absolute artifact path."""

    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path")
    return value.absolute()


def require_below(root: Path, path: Path, name: str) -> None:
    """Require a component to reside strictly below its artifact root."""

    if path == root or root not in path.parents:
        raise ValueError(f"{name} must be below bundle_root")


@dataclass(frozen=True)
class DreamTacArtifactFile:
    """One path-independent file identity below the checkpoint root."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(_string(self.path, "checkpoint file path"))
        if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
            raise ValueError("checkpoint file path must be safe and relative")
        object.__setattr__(self, "path", path.as_posix())
        object.__setattr__(self, "sha256", _sha256(self.sha256, "file sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> "DreamTacArtifactFile":
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError("checkpoint file identity fields mismatch")
        return cls(
            path=_string(value["path"], "checkpoint file path"),
            sha256=_sha256(value["sha256"], "checkpoint file sha256"),
        )


def stable_file_sha256(path: Path, name: str) -> str:
    """Hash one stable, non-symlink regular file."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"Dream-Tac {name} must be a non-symlink regular file")
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = selected.stat()
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
        raise ValueError(f"Dream-Tac {name} changed while hashing")
    return digest.hexdigest()


def checkpoint_file_hashes(root: Path) -> tuple[tuple[str, str], ...]:
    """Return a stable, relative inventory of one checkpoint directory."""

    selected = Path(root)
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("Dream-Tac checkpoint_root must be a real directory")
    files: list[tuple[str, str]] = []
    for path in sorted(selected.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dream-Tac checkpoint tree cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Dream-Tac checkpoint tree contains a non-regular entry")
        relative = path.relative_to(selected).as_posix()
        files.append((relative, stable_file_sha256(path, relative)))
    if not files:
        raise ValueError("Dream-Tac checkpoint tree cannot be empty")
    return tuple(files)


__all__ = [
    "DreamTacArtifactFile",
    "absolute_path",
    "checkpoint_file_hashes",
    "require_below",
    "stable_file_sha256",
]
