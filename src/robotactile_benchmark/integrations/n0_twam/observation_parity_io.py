"""Strict persistence and member verification for observation parity evidence."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationParityArtifact,
    ObservationParityError,
)

MAX_OBSERVATION_PARITY_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash one stable regular file without following a final symlink."""

    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file():
        raise ObservationParityError("observation evidence must be a regular file")
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = target.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ObservationParityError("observation evidence changed while hashing")
    return digest.hexdigest()


def resolve_evidence_member(root: Path, relative: object, name: str) -> Path:
    """Resolve a no-symlink deployment-relative evidence path."""

    deployment = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or "\\" in relative:
        raise ObservationParityError(f"{name} must be a POSIX path")
    member = PurePosixPath(relative)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ObservationParityError(f"{name} is unsafe")
    if not member.parts or member.parts[0] not in {
        "artifacts",
        "outputs",
        "requests",
    }:
        raise ObservationParityError(f"{name} is outside deployment evidence")
    target = deployment / Path(*member.parts)
    current = deployment
    for part in member.parts:
        current = current / part
        if current.is_symlink():
            raise ObservationParityError(f"{name} cannot traverse a symlink")
    return target


def load_observation_parity_artifact(
    deployment_root: Path,
    path: Path,
    *,
    verify_members: bool = True,
) -> ObservationParityArtifact:
    """Load canonical source-bound parity evidence and verify every input hash."""

    root = Path(deployment_root).resolve(strict=True)
    target = Path(path).absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ObservationParityError(
            "observation parity artifact must remain below deployment root"
        ) from error
    expected_prefixes = (("artifacts",), ("outputs",))
    if not any(relative.parts[: len(prefix)] == prefix for prefix in expected_prefixes):
        raise ObservationParityError(
            "observation parity artifact must be in artifacts/ or outputs/"
        )
    if target.is_symlink() or not target.is_file():
        raise ObservationParityError(
            "observation parity artifact must be a regular file"
        )
    raw = target.read_bytes()
    if not 0 < len(raw) <= MAX_OBSERVATION_PARITY_BYTES:
        raise ObservationParityError("observation parity artifact size is invalid")
    try:
        document = strict_json_bytes(raw, "observation parity artifact")
    except (TypeError, ValueError) as error:
        raise ObservationParityError(
            "observation parity artifact must be canonical JSON"
        ) from error
    artifact = ObservationParityArtifact.from_dict(document)
    if canonical_json_bytes(artifact.to_dict()) != raw:
        raise ObservationParityError(
            "observation parity artifact canonical roundtrip mismatch"
        )
    if verify_members:
        _verify_evidence_members(root, artifact)
    return artifact


def _verify_evidence_members(root: Path, artifact: ObservationParityArtifact) -> None:
    for item in artifact.evidence:
        path = resolve_evidence_member(root, item.relpath, "observation evidence path")
        if sha256_file(path) != item.sha256:
            raise ObservationParityError(
                f"observation evidence file hash mismatch: {item.kind}"
            )
        if item.content_sha256 is None:
            continue
        raw = path.read_bytes()
        try:
            value = strict_json_bytes(raw, f"{item.kind} evidence")
        except (TypeError, ValueError) as error:
            raise ObservationParityError(
                f"{item.kind} evidence is not canonical JSON"
            ) from error
        if not isinstance(value, Mapping):
            raise ObservationParityError(f"{item.kind} evidence must be an object")
        document = cast(Mapping[str, object], value)
        if document.get("content_sha256") != item.content_sha256:
            raise ObservationParityError(
                f"observation evidence content hash mismatch: {item.kind}"
            )


def write_observation_parity_artifact(
    path: Path, artifact: ObservationParityArtifact
) -> bool:
    """Publish one canonical artifact without replacing different evidence."""

    if type(artifact) is not ObservationParityArtifact:
        raise TypeError("artifact must be an exact ObservationParityArtifact")
    target = Path(path).absolute()
    payload = canonical_json_bytes(artifact.to_dict())
    _reject_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    if target.is_symlink():
        raise ObservationParityError("observation parity output cannot be a symlink")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(
                "refusing to replace a different observation parity artifact"
            )
        return False
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise FileExistsError(
                    "refusing to replace a different observation parity artifact"
                ) from None
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _reject_symlink_components(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.is_symlink():
            raise ObservationParityError(
                "observation parity output cannot traverse a symlink"
            )


__all__ = [
    "MAX_OBSERVATION_PARITY_BYTES",
    "load_observation_parity_artifact",
    "resolve_evidence_member",
    "sha256_file",
    "write_observation_parity_artifact",
]
