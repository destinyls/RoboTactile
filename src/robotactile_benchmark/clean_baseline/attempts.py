"""Strict successful-command receipt gate for clean campaign artifacts."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional

from robotactile_benchmark.clean_baseline.campaign_attempt_io import (
    ATTEMPT_FIELDS_BY_VERSION,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignManifest,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.io import read_canonical_json_file
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V3_SEMANTIC_VERSION,
    verify_all_task_qualification,
)
from robotactile_benchmark.closed_loop.artifact_io import sha256_bytes
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacRuntimeAttestation,
    load_isaac_runtime_attestation,
)
from robotactile_benchmark.execution.lifecycle_watchdog import (
    LifecycleIdentity,
    verify_close_after_export_timeout,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.runtime_attestation import (
    N0ServerRuntimeAttestation,
    load_n0_server_runtime_attestation,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

_CLOSE_AFTER_EXPORT_HARD_TIMEOUT = "close_after_export_hard_timeout"

_ATTEMPT_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_root_sha256",
        "execution_status",
        "score_eligible",
        "score_success",
        "terminal_status",
    }
)


class CleanAttemptDisposition(str, Enum):
    """Official candidate outcomes declared by a v2 command receipt."""

    VALID_OUTCOME = "valid_outcome"
    EXCEPTION_REPLACED = "exception_replaced"


@dataclass(frozen=True)
class LoadedCleanAttempt:
    """One identity-bound legacy or source-attested attempt receipt."""

    receipt_sha256: str
    semantic_version: str
    disposition: CleanAttemptDisposition
    exception_code: Optional[str]
    qualification_relpath: Optional[str] = None
    qualification_sha256: Optional[str] = None
    runtime_source_binding: Optional[RuntimeSourceBinding] = None
    isaac_attestation_relpath: Optional[str] = None
    isaac_attestation_sha256: Optional[str] = None
    n0_server_attestation_relpath: Optional[str] = None
    n0_server_attestation_sha256: Optional[str] = None


@dataclass(frozen=True)
class _SourceBoundAttemptFields:
    qualification_relpath: str
    qualification_sha256: str
    runtime_source_binding: RuntimeSourceBinding
    isaac_attestation_relpath: str
    isaac_attestation_sha256: str
    n0_server_attestation_relpath: str
    n0_server_attestation_sha256: str


def load_successful_attempt(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    artifact: LoadedLiveUniVTACArtifact,
) -> tuple[Optional[str], Optional[str]]:
    """Return the receipt hash or one stable protocol-invalid reason code."""

    attempt, reason = load_candidate_attempt(root, manifest, spec, artifact)
    if attempt is None:
        return None, reason
    if attempt.disposition is not CleanAttemptDisposition.VALID_OUTCOME:
        return None, "attempt_receipt_invalid"
    return attempt.receipt_sha256, None


def load_candidate_attempt(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    artifact: Optional[LoadedLiveUniVTACArtifact],
) -> tuple[Optional[LoadedCleanAttempt], Optional[str]]:
    """Load one exact v1/v2 candidate receipt without manufacturing evidence."""

    directory = (
        root
        / "outputs"
        / "clean-campaigns"
        / manifest.campaign_id
        / "attempts"
        / spec.task
    )
    if _traverses_symlink(root, directory):
        return None, "attempt_receipt_invalid"
    if directory.exists() and not directory.is_dir():
        return None, "attempt_receipt_invalid"
    if not directory.exists():
        return None, "attempt_receipt_missing"
    prefix = f"{spec.ordinal:04d}-"
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return None, "attempt_receipt_invalid"
    candidates = [
        Path(entry.path)
        for entry in entries
        if entry.name.startswith(prefix) and entry.name.endswith(".json")
    ]
    if not candidates:
        return None, "attempt_receipt_missing"
    if len(candidates) != 1:
        return None, "multiple_attempt_receipts"
    candidate = candidates[0]
    if candidate.is_symlink() or not candidate.is_file():
        return None, "attempt_receipt_invalid"
    try:
        document, raw = read_canonical_json_file(candidate, "clean attempt receipt")
        return_code = _validate_attempt_receipt(root, manifest, spec, document)
    except (OSError, TypeError, ValueError):
        return None, "attempt_receipt_invalid"
    semantic_version = document["semantic_version"]
    if semantic_version == "1.0":
        if return_code != 0:
            return None, "attempt_return_code_nonzero"
        if artifact is None or not _attempt_artifact_matches(document, artifact):
            return None, "attempt_artifact_binding_mismatch"
        if document["artifact_validation_error_type"] is not None:
            return None, "attempt_receipt_invalid"
        return (
            LoadedCleanAttempt(
                receipt_sha256=sha256_bytes(raw),
                semantic_version="1.0",
                disposition=CleanAttemptDisposition.VALID_OUTCOME,
                exception_code=None,
            ),
            None,
        )
    try:
        disposition, exception_code = _validate_v2_disposition(
            root,
            manifest,
            spec,
            document,
            artifact,
            return_code=return_code,
        )
        source_fields = (
            _validate_v3_source_binding(root, manifest, spec, candidate, document)
            if semantic_version == "3.0"
            else None
        )
    except (OSError, TypeError, ValueError):
        return None, "attempt_receipt_invalid"
    return (
        LoadedCleanAttempt(
            receipt_sha256=sha256_bytes(raw),
            semantic_version=str(semantic_version),
            disposition=disposition,
            exception_code=exception_code,
            qualification_relpath=(
                None if source_fields is None else source_fields.qualification_relpath
            ),
            qualification_sha256=(
                None if source_fields is None else source_fields.qualification_sha256
            ),
            runtime_source_binding=(
                None if source_fields is None else source_fields.runtime_source_binding
            ),
            isaac_attestation_relpath=(
                None
                if source_fields is None
                else source_fields.isaac_attestation_relpath
            ),
            isaac_attestation_sha256=(
                None
                if source_fields is None
                else source_fields.isaac_attestation_sha256
            ),
            n0_server_attestation_relpath=(
                None
                if source_fields is None
                else source_fields.n0_server_attestation_relpath
            ),
            n0_server_attestation_sha256=(
                None
                if source_fields is None
                else source_fields.n0_server_attestation_sha256
            ),
        ),
        None,
    )


def candidate_attempt_present(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
) -> bool:
    """Return whether any receipt-shaped entry exists for one candidate."""

    directory = (
        root
        / "outputs"
        / "clean-campaigns"
        / manifest.campaign_id
        / "attempts"
        / spec.task
    )
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        return True
    prefix = f"{spec.ordinal:04d}-"
    try:
        return any(
            entry.name.startswith(prefix) and entry.name.endswith(".json")
            for entry in os.scandir(directory)
        )
    except OSError:
        return True


def _validate_attempt_receipt(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    document: Mapping[str, object],
) -> int:
    semantic_version = document.get("semantic_version")
    try:
        expected_fields = ATTEMPT_FIELDS_BY_VERSION[str(semantic_version)]
    except KeyError as error:
        raise CleanCampaignError("unsupported clean attempt version") from error
    if set(document) != expected_fields:
        raise CleanCampaignError("clean attempt receipt fields mismatch")
    return_code = document["return_code"]
    duration = document["duration_s"]
    ordinal = document["ordinal"]
    if (
        isinstance(return_code, bool)
        or not isinstance(return_code, int)
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        raise CleanCampaignError("clean attempt numeric fields are invalid")
    if (
        document["campaign_id"] != manifest.campaign_id
        or document["campaign_manifest_sha256"] != manifest.sha256
        or document["command_kind"] != "robotactile_live_univtac_run_v1"
        or (
            semantic_version in {"1.0", "2.0"}
            and semantic_version != manifest.semantic_version
        )
        or (
            semantic_version == "3.0"
            and (
                manifest.semantic_version != "2.0"
                or document["campaign_semantic_version"] != "2.0"
            )
        )
        or ordinal != spec.ordinal
        or document["task_id"] != spec.task
        or document["request_file_sha256"] != spec.request_file_sha256
        or document["trial_manifest_sha256"] != spec.trial_manifest_sha256
    ):
        raise CleanCampaignError("clean attempt identity mismatch")
    started = _timestamp(document["started_at_utc"])
    finished = _timestamp(document["finished_at_utc"])
    if finished < started:
        raise CleanCampaignError("clean attempt timestamps are reversed")
    log_path = _strict_log_path(root, document["log_relpath"])
    log_sha256 = document["log_sha256"]
    if not isinstance(log_sha256, str) or _sha256_file(log_path) != log_sha256:
        raise CleanCampaignError("clean attempt log hash mismatch")
    artifact_error = document["artifact_validation_error_type"]
    if artifact_error is not None and (
        not isinstance(artifact_error, str) or not artifact_error
    ):
        raise CleanCampaignError("artifact validation error type is invalid")
    if semantic_version == "1.0" and return_code == 0 and artifact_error is not None:
        raise CleanCampaignError(
            "successful attempt cannot report artifact validation error"
        )
    return return_code


def _validate_v2_disposition(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    document: Mapping[str, object],
    artifact: Optional[LoadedLiveUniVTACArtifact],
    *,
    return_code: int,
) -> tuple[CleanAttemptDisposition, Optional[str]]:
    disposition = CleanAttemptDisposition(document["candidate_disposition"])
    exception = document["exception_code"]
    artifact_error = document["artifact_validation_error_type"]
    bound_artifact = document["artifact"]
    if disposition is CleanAttemptDisposition.VALID_OUTCOME:
        if (
            return_code != 0
            or artifact_error is not None
            or exception is not None
            or artifact is None
            or artifact.evidence.result.terminal_status.value == "crash"
            or not artifact.evidence.result.score_eligible
        ):
            raise CleanCampaignError("valid outcome receipt is incoherent")
        if not _attempt_artifact_matches(document, artifact):
            raise CleanCampaignError("valid outcome artifact binding mismatch")
        return disposition, None
    if not isinstance(exception, str) or not exception:
        raise CleanCampaignError("replacement receipt requires exception_code")
    if artifact is None:
        if bound_artifact is not None:
            raise CleanCampaignError("replacement receipt binds an absent artifact")
        if return_code == 0 and artifact_error is None:
            raise CleanCampaignError("replacement receipt lacks exception evidence")
    else:
        if not _attempt_artifact_matches(document, artifact):
            raise CleanCampaignError("replacement artifact binding mismatch")
        if artifact.evidence.result.terminal_status.value != "crash":
            if exception != _CLOSE_AFTER_EXPORT_HARD_TIMEOUT:
                raise CleanCampaignError("only a crash artifact can be replaced")
            _validate_close_timeout_attempt(root, manifest, spec, document, return_code)
    return disposition, exception


def _validate_v3_source_binding(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    receipt_path: Path,
    document: Mapping[str, object],
) -> _SourceBoundAttemptFields:
    qualification_relpath = _evidence_relpath(
        document["qualification_relpath"], "qualification"
    )
    qualification_sha256 = _string(
        document["qualification_sha256"], "qualification SHA256"
    )
    source = RuntimeSourceBinding.from_dict(document["runtime_source_binding"])
    qualification_path = _strict_evidence_path(root, qualification_relpath)
    qualification = verify_all_task_qualification(root, qualification_path)
    if (
        qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION
        or qualification.sha256 != qualification_sha256
        or qualification.campaign_manifest_sha256 != manifest.sha256
    ):
        raise CleanCampaignError("attempt qualification binding mismatch")
    try:
        task_index = qualification.tasks.index(spec.task)
    except ValueError as error:
        raise CleanCampaignError("attempt qualification lacks task") from error
    if qualification.task_source_bindings[task_index] != source:
        raise CleanCampaignError("attempt task source binding mismatch")

    server_relpath = _evidence_relpath(
        document["n0_server_attestation_relpath"], "N0 server attestation"
    )
    server_sha256 = _string(
        document["n0_server_attestation_sha256"],
        "N0 server attestation SHA256",
    )
    server = load_n0_server_runtime_attestation(
        root,
        _strict_evidence_path(root, server_relpath),
        expected_sha256=server_sha256,
        verify_members=True,
    )
    if (
        server.task_id != spec.task
        or server.qualification_sha256 != qualification_sha256
        or server.task_source_binding != source
    ):
        raise CleanCampaignError("attempt N0 server binding mismatch")

    isaac_relpath = _evidence_relpath(
        document["isaac_attestation_relpath"], "Isaac attestation"
    )
    isaac_sha256 = _string(
        document["isaac_attestation_sha256"], "Isaac attestation SHA256"
    )
    isaac = load_isaac_runtime_attestation(
        root,
        _strict_evidence_path(root, isaac_relpath),
        expected_sha256=isaac_sha256,
        verify_members=True,
    )
    expected_identity = _attempt_lifecycle_identity(manifest, spec, receipt_path)
    _validate_v3_isaac_identity(
        isaac,
        expected_identity=expected_identity,
        campaign_id=manifest.campaign_id,
        qualification_sha256=qualification_sha256,
        source=source,
        server_relpath=server_relpath,
        server_sha256=server_sha256,
        server=server,
    )
    return _SourceBoundAttemptFields(
        qualification_relpath=qualification_relpath,
        qualification_sha256=qualification_sha256,
        runtime_source_binding=source,
        isaac_attestation_relpath=isaac_relpath,
        isaac_attestation_sha256=isaac_sha256,
        n0_server_attestation_relpath=server_relpath,
        n0_server_attestation_sha256=server_sha256,
    )


def _validate_v3_isaac_identity(
    isaac: IsaacRuntimeAttestation,
    *,
    expected_identity: LifecycleIdentity,
    campaign_id: str,
    qualification_sha256: str,
    source: RuntimeSourceBinding,
    server_relpath: str,
    server_sha256: str,
    server: N0ServerRuntimeAttestation,
) -> None:
    if (
        isaac.campaign_id != campaign_id
        or isaac.lifecycle_identity != expected_identity
        or isaac.qualification_sha256 != qualification_sha256
        or isaac.runtime_source_binding != source
        or isaac.n0_server_attestation_relpath != server_relpath
        or isaac.n0_server_attestation_sha256 != server_sha256
        or isaac.n0_server_content_sha256 != server.content_sha256
    ):
        raise CleanCampaignError("attempt Isaac attestation binding mismatch")


def _attempt_lifecycle_identity(
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    receipt_path: Path,
) -> LifecycleIdentity:
    prefix = f"{spec.ordinal:04d}-"
    if not receipt_path.name.startswith(prefix) or receipt_path.suffix != ".json":
        raise CleanCampaignError("attempt receipt filename is invalid")
    attempt_id = receipt_path.name[len(prefix) : -len(".json")]
    if not attempt_id:
        raise CleanCampaignError("attempt receipt lacks attempt ID")
    return LifecycleIdentity(
        campaign_manifest_sha256=manifest.sha256,
        request_file_sha256=spec.request_file_sha256,
        trial_manifest_sha256=spec.trial_manifest_sha256,
        task_id=spec.task,
        ordinal=spec.ordinal,
        attempt_id=attempt_id,
    )


def _evidence_relpath(value: object, name: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise CleanCampaignError(f"{name} path must be POSIX relative")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] not in {"artifacts", "outputs", "requests"}
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CleanCampaignError(f"{name} path is unsafe")
    return value


def _strict_evidence_path(root: Path, relative: str) -> Path:
    path = root / Path(*PurePosixPath(relative).parts)
    if _traverses_symlink(root, path) or not path.is_file():
        raise CleanCampaignError("attempt evidence member is unavailable")
    return path


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CleanCampaignError(f"{name} must be a non-empty string")
    return value


def _validate_close_timeout_attempt(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
    document: Mapping[str, object],
    return_code: int,
) -> None:
    log_path = _strict_log_path(root, document["log_relpath"])
    prefix = f"{spec.ordinal:04d}-"
    if not log_path.stem.startswith(prefix):
        raise CleanCampaignError("close-timeout log name does not match ordinal")
    identity = LifecycleIdentity(
        campaign_manifest_sha256=manifest.sha256,
        request_file_sha256=spec.request_file_sha256,
        trial_manifest_sha256=spec.trial_manifest_sha256,
        task_id=spec.task,
        ordinal=spec.ordinal,
        attempt_id=log_path.stem[len(prefix) :],
    )
    proof = verify_close_after_export_timeout(
        root=root,
        campaign_id=manifest.campaign_id,
        identity=identity,
        log_path=log_path,
    )
    if (
        proof.return_code != return_code
        or proof.log_sha256 != document["log_sha256"]
        or proof.started_at_utc != document["started_at_utc"]
        or proof.finished_at_utc != document["finished_at_utc"]
        or proof.duration_s != document["duration_s"]
    ):
        raise CleanCampaignError("close-timeout attempt/watchdog binding mismatch")


def _attempt_artifact_matches(
    document: Mapping[str, object], artifact: LoadedLiveUniVTACArtifact
) -> bool:
    value = document["artifact"]
    if not isinstance(value, Mapping) or set(value) != _ATTEMPT_ARTIFACT_FIELDS:
        return False
    result = artifact.evidence.result
    return value == {
        "artifact_root_sha256": artifact.external_root_sha256,
        "execution_status": (
            None if result.execution_status is None else result.execution_status.value
        ),
        "score_eligible": result.score_eligible,
        "score_success": result.score_success,
        "terminal_status": result.terminal_status.value,
    }


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise CleanCampaignError("clean attempt timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise CleanCampaignError("clean attempt timestamp must include timezone")
    return parsed


def _strict_log_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise CleanCampaignError("clean attempt log path must be POSIX relative")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "logs"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CleanCampaignError("clean attempt log path is unsafe")
    path = root / Path(*relative.parts)
    if _traverses_symlink(root, path):
        raise CleanCampaignError("clean attempt log path traverses a symlink")
    return path


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CleanCampaignError("attempt log must be a regular non-symlink file")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise CleanCampaignError("attempt log changed while being hashed")
    return digest.hexdigest()


def _traverses_symlink(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


__all__ = [
    "CleanAttemptDisposition",
    "LoadedCleanAttempt",
    "candidate_attempt_present",
    "load_candidate_attempt",
    "load_successful_attempt",
]
