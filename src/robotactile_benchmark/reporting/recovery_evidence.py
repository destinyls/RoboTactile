"""Atomic persistence for evaluator-side behavioral recovery evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, cast

from robotactile_benchmark.matrix.io import canonical_matrix_json_bytes
from robotactile_benchmark.reporting.contracts import require_sha256
from robotactile_benchmark.reporting.recovery_evidence_contracts import (
    CLEAN_ENVELOPE_ALIGNMENT,
    POLICY_TASK_PROGRESS_SIGNAL_ID,
    RECOVERY_EVIDENCE_LEVEL,
    RECOVERY_EVIDENCE_SEMANTIC_VERSION,
    REGISTERED_RECOVERY_SIGNAL_IDS,
    RecoveryEvidence,
    RecoveryEvidenceBinding,
)

MAX_RECOVERY_EVIDENCE_BYTES = 4 * 1024 * 1024


class RecoveryEvidenceValidationError(ValueError):
    """Raised when recovery evidence cannot be independently trusted."""


def recovery_evidence_path(directory: Path, restored_root_sha256: str) -> Path:
    """Return the canonical sidecar location for one restored live artifact."""

    root = require_sha256(restored_root_sha256, "restored_root_sha256")
    return Path(directory) / f"{root}.json"


def write_recovery_evidence(
    directory: Path, evidence: RecoveryEvidence
) -> RecoveryEvidence:
    """Atomically publish one canonical sidecar without replacing other bytes."""

    if not isinstance(evidence, RecoveryEvidence):
        raise TypeError("evidence must be a RecoveryEvidence")
    directory = Path(directory)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise FileExistsError("recovery evidence output must be a real directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = recovery_evidence_path(
        directory, evidence.binding.restored_live_artifact_root_sha256
    )
    raw = canonical_matrix_json_bytes(evidence.to_dict())
    if len(raw) > MAX_RECOVERY_EVIDENCE_BYTES:
        raise RecoveryEvidenceValidationError("recovery evidence exceeds size cap")
    if path.exists() or path.is_symlink():
        return _reuse_exact(path, evidence, raw)
    _publish_no_clobber(path, raw)
    loaded = load_recovery_evidence(path, expected=evidence.binding)
    if loaded != evidence or loaded.sha256 != evidence.sha256:
        raise RecoveryEvidenceValidationError(
            "published recovery evidence changed during publication"
        )
    return loaded


def load_recovery_evidence(
    path: Path,
    *,
    expected: Optional[RecoveryEvidenceBinding] = None,
) -> RecoveryEvidence:
    """Strict-load canonical evidence and independently recompute recovery lag."""

    path = Path(path)
    try:
        before, raw = _read_regular(path)
        document = _strict_json(raw)
        if canonical_matrix_json_bytes(document) != raw:
            raise RecoveryEvidenceValidationError(
                "recovery evidence is not canonical JSON"
            )
        evidence = RecoveryEvidence.from_dict(document)
        expected_name = f"{evidence.binding.restored_live_artifact_root_sha256}.json"
        if path.name != expected_name:
            raise RecoveryEvidenceValidationError(
                "recovery evidence filename is stale for its restored root"
            )
        if expected is not None and evidence.binding != expected:
            raise RecoveryEvidenceValidationError(
                "recovery evidence binding disagrees with matrix cell"
            )
        if _file_identity(path.stat()) != before:
            raise RecoveryEvidenceValidationError(
                "recovery evidence changed during typed validation"
            )
        if evidence.sha256 != _sha256(raw):
            raise RecoveryEvidenceValidationError(
                "recovery evidence content hash is inconsistent"
            )
        return evidence
    except RecoveryEvidenceValidationError:
        raise
    except (KeyError, OSError, OverflowError, TypeError, ValueError) as error:
        raise RecoveryEvidenceValidationError(
            "recovery evidence failed strict validation"
        ) from error


def _reuse_exact(
    path: Path, evidence: RecoveryEvidence, raw: bytes
) -> RecoveryEvidence:
    try:
        loaded = load_recovery_evidence(path, expected=evidence.binding)
    except RecoveryEvidenceValidationError as error:
        raise FileExistsError("target belongs to another recovery evidence") from error
    if loaded != evidence or path.read_bytes() != raw:
        raise FileExistsError("target belongs to another recovery evidence")
    return loaded


def _read_regular(path: Path) -> tuple[tuple[int, ...], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryEvidenceValidationError(
            "recovery evidence must be a regular file"
        )
    before = path.stat()
    if not 0 < before.st_size <= MAX_RECOVERY_EVIDENCE_BYTES:
        raise RecoveryEvidenceValidationError(
            "recovery evidence size is outside bounds"
        )
    raw = path.read_bytes()
    identity = _file_identity(before)
    if len(raw) != before.st_size or _file_identity(path.stat()) != identity:
        raise RecoveryEvidenceValidationError("recovery evidence changed during read")
    return identity, raw


def _file_identity(stat: os.stat_result) -> tuple[int, ...]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _strict_json(raw: bytes) -> object:
    if not raw or raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise RecoveryEvidenceValidationError(
            "recovery evidence has noncanonical encoding"
        )

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryEvidenceValidationError(
                    "recovery evidence has duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise RecoveryEvidenceValidationError(
            f"recovery evidence has non-finite constant {token}"
        )

    try:
        return cast(
            object,
            json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=pairs_hook,
                parse_constant=reject_constant,
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryEvidenceValidationError(
            "recovery evidence is not strict UTF-8 JSON"
        ) from error


def _publish_no_clobber(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise FileExistsError(
                    "target belongs to another recovery evidence"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "CLEAN_ENVELOPE_ALIGNMENT",
    "MAX_RECOVERY_EVIDENCE_BYTES",
    "POLICY_TASK_PROGRESS_SIGNAL_ID",
    "RECOVERY_EVIDENCE_LEVEL",
    "RECOVERY_EVIDENCE_SEMANTIC_VERSION",
    "REGISTERED_RECOVERY_SIGNAL_IDS",
    "RecoveryEvidence",
    "RecoveryEvidenceBinding",
    "RecoveryEvidenceValidationError",
    "load_recovery_evidence",
    "recovery_evidence_path",
    "write_recovery_evidence",
]
