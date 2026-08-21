"""Strict canonical JSON and exact on-disk bundle inventory checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from robotactile_benchmark.closed_loop.artifact_contracts import (
    MAX_BUNDLE_BYTES,
    MAX_MEMBER_BYTES,
    MAX_MEMBER_COUNT,
    ROOT_RECEIPT_PATH,
    ArtifactValidationError,
    RootReceipt,
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict canonical UTF-8 JSON with exactly one trailing newline."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("value is outside canonical JSON") from error
    return (payload + "\n").encode("utf-8")


def strict_json_bytes(raw: bytes, name: str) -> object:
    """Parse canonical JSON while rejecting duplicates, constants, BOM, and CRLF."""

    if not raw or len(raw) > MAX_MEMBER_BYTES:
        raise ArtifactValidationError(f"{name} size is outside bounds")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactValidationError(f"{name} is not UTF-8 JSON") from error
    if text.startswith("\ufeff") or "\r" in text:
        raise ArtifactValidationError(f"{name} contains a BOM or CRLF")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactValidationError(f"{name} contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ArtifactValidationError(f"{name} contains non-finite constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ArtifactValidationError(f"{name} is invalid JSON") from error
    if canonical_json_bytes(value) != raw:
        raise ArtifactValidationError(f"{name} is not canonical JSON")
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bundle_inventory(
    root: Path,
) -> tuple[RootReceipt, str, dict[str, bytes]]:
    """Load an exact no-symlink inventory pinned by the canonical root receipt."""

    if root.is_symlink() or not root.is_dir():
        raise ArtifactValidationError("bundle root must be a real directory")
    actual_files, actual_dirs = _scan_directory(root)
    if ROOT_RECEIPT_PATH not in actual_files:
        raise ArtifactValidationError("bundle is missing root_receipt.json")
    root_raw = actual_files[ROOT_RECEIPT_PATH]
    root_receipt = RootReceipt.from_dict(strict_json_bytes(root_raw, ROOT_RECEIPT_PATH))
    expected_files = {member.path for member in root_receipt.members} | {
        ROOT_RECEIPT_PATH
    }
    if set(actual_files) != expected_files:
        raise ArtifactValidationError("actual files do not match the root inventory")
    expected_dirs = {
        parent for path in expected_files for parent in _parent_directories(path)
    }
    if actual_dirs != expected_dirs:
        raise ArtifactValidationError("bundle contains unknown or missing directories")
    member_by_path = {member.path: member for member in root_receipt.members}
    total_size = len(root_raw)
    for path, member in member_by_path.items():
        raw = actual_files[path]
        total_size += len(raw)
        if len(raw) != member.size_bytes or sha256_bytes(raw) != member.sha256:
            raise ArtifactValidationError(f"member bytes do not match receipt: {path}")
    if total_size > MAX_BUNDLE_BYTES:
        raise ArtifactValidationError("bundle bytes exceed the total size cap")
    return root_receipt, sha256_bytes(root_raw), actual_files


def _scan_directory(root: Path) -> tuple[dict[str, bytes], set[str]]:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    total_size = 0

    def visit(directory: Path) -> None:
        nonlocal total_size
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ArtifactValidationError(
                "bundle directory cannot be scanned"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ArtifactValidationError(f"symlink is forbidden: {relative}")
            if entry.is_dir(follow_symlinks=False):
                directories.add(relative)
                if len(directories) > 1:
                    raise ArtifactValidationError("bundle has too many directories")
                visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ArtifactValidationError(
                    f"non-regular member is forbidden: {relative}"
                )
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ArtifactValidationError(
                    "bundle member cannot be stated"
                ) from error
            if stat.st_size < 1 or stat.st_size > MAX_MEMBER_BYTES:
                raise ArtifactValidationError(
                    f"member size is outside bounds: {relative}"
                )
            raw = path.read_bytes()
            if len(raw) != stat.st_size:
                raise ArtifactValidationError("bundle member changed while reading")
            files[relative] = raw
            total_size += len(raw)
            if len(files) > MAX_MEMBER_COUNT + 1 or total_size > MAX_BUNDLE_BYTES:
                raise ArtifactValidationError(
                    "bundle inventory exceeds count or size caps"
                )

    visit(root)
    return files, directories


def _parent_directories(path: str) -> tuple[str, ...]:
    parts = path.split("/")[:-1]
    return tuple("/".join(parts[:index]) for index in range(1, len(parts) + 1))
