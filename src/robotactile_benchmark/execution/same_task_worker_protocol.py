"""Strict framed IPC contracts for a same-task execution worker."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from typing import BinaryIO, ClassVar, Literal, Mapping, TypeVar, cast

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes

SAME_TASK_WORKER_CONTRACT = "same_task_worker_v1"
SAME_TASK_WORKER_SEMANTIC_VERSION = "1.0"
MAX_FRAME_BYTES = 1024 * 1024

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FRAME_HEADER = struct.Struct(">I")
_T = TypeVar("_T")


def _require_exact_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_identifier(value: object, name: str) -> str:
    selected = _require_exact_string(value, name)
    if _IDENTIFIER.fullmatch(selected) is None:
        raise ValueError(f"{name} contains unsupported characters")
    return selected


def _require_sha256(value: object, name: str) -> str:
    selected = _require_exact_string(value, name)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return selected


def _require_ordinal(value: object) -> int:
    if type(value) is not int:
        raise TypeError("ordinal must be an integer")
    if value < 0:
        raise ValueError("ordinal must be non-negative")
    return value


def _require_process_id(value: object) -> int:
    if type(value) is not int:
        raise TypeError("process_id must be an integer")
    if value <= 0:
        raise ValueError("process_id must be positive")
    return value


def _require_return_code(value: object) -> int:
    if type(value) is not int:
        raise TypeError("return_code must be an integer")
    if not 0 <= value <= 255:
        raise ValueError("return_code must be in [0,255]")
    return value


def _require_contract(contract: object, semantic_version: object) -> None:
    if contract != SAME_TASK_WORKER_CONTRACT:
        raise ValueError("worker_contract mismatch")
    if semantic_version != SAME_TASK_WORKER_SEMANTIC_VERSION:
        raise ValueError("semantic_version mismatch")


def _exact_document(
    document: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(document)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def compute_request_id(
    *,
    campaign_manifest_sha256: str,
    task_id: str,
    ordinal: int,
    attempt_id: str,
    request_file_sha256: str,
) -> str:
    """Derive the immutable command identity from frozen campaign fields."""

    identity = {
        "attempt_id": _require_identifier(attempt_id, "attempt_id"),
        "campaign_manifest_sha256": _require_sha256(
            campaign_manifest_sha256, "campaign_manifest_sha256"
        ),
        "ordinal": _require_ordinal(ordinal),
        "request_file_sha256": _require_sha256(
            request_file_sha256, "request_file_sha256"
        ),
        "task_id": _require_identifier(task_id, "task_id"),
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


@dataclass(frozen=True)
class SameTaskWorkerReadyIdentity:
    """Identity emitted exactly once after a task-local worker is ready."""

    worker_contract: str
    semantic_version: str
    session_id: str
    task_id: str
    process_id: int
    campaign_manifest_sha256: str
    source_binding_sha256: str
    integration_config_sha256: str
    action_execution_contract: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "worker_contract",
            "semantic_version",
            "session_id",
            "task_id",
            "process_id",
            "campaign_manifest_sha256",
            "source_binding_sha256",
            "integration_config_sha256",
            "action_execution_contract",
        }
    )

    def __post_init__(self) -> None:
        _require_contract(self.worker_contract, self.semantic_version)
        _require_identifier(self.session_id, "session_id")
        _require_identifier(self.task_id, "task_id")
        _require_process_id(self.process_id)
        _require_sha256(self.campaign_manifest_sha256, "campaign_manifest_sha256")
        _require_sha256(self.source_binding_sha256, "source_binding_sha256")
        _require_sha256(self.integration_config_sha256, "integration_config_sha256")
        _require_identifier(self.action_execution_contract, "action_execution_contract")

    def to_document(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_document(
        cls, document: Mapping[str, object]
    ) -> SameTaskWorkerReadyIdentity:
        _exact_document(document, cls._FIELDS, "ready identity")
        return cls(
            worker_contract=_require_exact_string(
                document["worker_contract"], "worker_contract"
            ),
            semantic_version=_require_exact_string(
                document["semantic_version"], "semantic_version"
            ),
            session_id=_require_identifier(document["session_id"], "session_id"),
            task_id=_require_identifier(document["task_id"], "task_id"),
            process_id=_require_process_id(document["process_id"]),
            campaign_manifest_sha256=_require_sha256(
                document["campaign_manifest_sha256"], "campaign_manifest_sha256"
            ),
            source_binding_sha256=_require_sha256(
                document["source_binding_sha256"], "source_binding_sha256"
            ),
            integration_config_sha256=_require_sha256(
                document["integration_config_sha256"], "integration_config_sha256"
            ),
            action_execution_contract=_require_identifier(
                document["action_execution_contract"], "action_execution_contract"
            ),
        )


@dataclass(frozen=True)
class SameTaskWorkerRequest:
    """One immutable episode dispatch sent to a task-local worker."""

    worker_contract: str
    semantic_version: str
    request_id: str
    campaign_manifest_sha256: str
    task_id: str
    ordinal: int
    attempt_id: str
    request_file_sha256: str
    trial_manifest_sha256: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "worker_contract",
            "semantic_version",
            "request_id",
            "campaign_manifest_sha256",
            "task_id",
            "ordinal",
            "attempt_id",
            "request_file_sha256",
            "trial_manifest_sha256",
        }
    )

    def __post_init__(self) -> None:
        _require_contract(self.worker_contract, self.semantic_version)
        _require_sha256(self.request_id, "request_id")
        _require_sha256(self.campaign_manifest_sha256, "campaign_manifest_sha256")
        _require_identifier(self.task_id, "task_id")
        _require_ordinal(self.ordinal)
        _require_identifier(self.attempt_id, "attempt_id")
        _require_sha256(self.request_file_sha256, "request_file_sha256")
        _require_sha256(self.trial_manifest_sha256, "trial_manifest_sha256")
        expected = compute_request_id(
            campaign_manifest_sha256=self.campaign_manifest_sha256,
            task_id=self.task_id,
            ordinal=self.ordinal,
            attempt_id=self.attempt_id,
            request_file_sha256=self.request_file_sha256,
        )
        if self.request_id != expected:
            raise ValueError("request_id does not match the request identity")

    def to_document(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> SameTaskWorkerRequest:
        _exact_document(document, cls._FIELDS, "worker request")
        return cls(
            worker_contract=_require_exact_string(
                document["worker_contract"], "worker_contract"
            ),
            semantic_version=_require_exact_string(
                document["semantic_version"], "semantic_version"
            ),
            request_id=_require_sha256(document["request_id"], "request_id"),
            campaign_manifest_sha256=_require_sha256(
                document["campaign_manifest_sha256"], "campaign_manifest_sha256"
            ),
            task_id=_require_identifier(document["task_id"], "task_id"),
            ordinal=_require_ordinal(document["ordinal"]),
            attempt_id=_require_identifier(document["attempt_id"], "attempt_id"),
            request_file_sha256=_require_sha256(
                document["request_file_sha256"], "request_file_sha256"
            ),
            trial_manifest_sha256=_require_sha256(
                document["trial_manifest_sha256"], "trial_manifest_sha256"
            ),
        )


@dataclass(frozen=True)
class SameTaskWorkerResponse:
    """Identity-bound terminal response for one worker dispatch."""

    worker_contract: str
    semantic_version: str
    request_id: str
    task_id: str
    ordinal: int
    attempt_id: str
    status: Literal["completed", "failed"]
    return_code: int
    artifact_root_sha256: str | None
    exception_code: str | None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "worker_contract",
            "semantic_version",
            "request_id",
            "task_id",
            "ordinal",
            "attempt_id",
            "status",
            "return_code",
            "artifact_root_sha256",
            "exception_code",
        }
    )

    def __post_init__(self) -> None:
        _require_contract(self.worker_contract, self.semantic_version)
        _require_sha256(self.request_id, "request_id")
        _require_identifier(self.task_id, "task_id")
        _require_ordinal(self.ordinal)
        _require_identifier(self.attempt_id, "attempt_id")
        if self.status not in {"completed", "failed"}:
            raise ValueError("status must be completed or failed")
        _require_return_code(self.return_code)
        if self.artifact_root_sha256 is not None:
            _require_sha256(self.artifact_root_sha256, "artifact_root_sha256")
        if self.exception_code is not None:
            _require_identifier(self.exception_code, "exception_code")
        if self.status == "completed" and self.return_code != 0:
            raise ValueError("completed response requires return_code=0")
        if self.status == "completed" and self.artifact_root_sha256 is None:
            raise ValueError("completed response requires an artifact digest")
        if self.status == "failed" and self.exception_code is None:
            raise ValueError("failed response requires an exception_code")

    def validate_against(self, request: SameTaskWorkerRequest) -> None:
        if (
            self.request_id != request.request_id
            or self.task_id != request.task_id
            or self.ordinal != request.ordinal
            or self.attempt_id != request.attempt_id
        ):
            raise ValueError("worker response identity mismatch")

    def to_document(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> SameTaskWorkerResponse:
        _exact_document(document, cls._FIELDS, "worker response")
        status = _require_exact_string(document["status"], "status")
        if status not in {"completed", "failed"}:
            raise ValueError("status must be completed or failed")
        artifact_digest = document["artifact_root_sha256"]
        exception_code = document["exception_code"]
        return cls(
            worker_contract=_require_exact_string(
                document["worker_contract"], "worker_contract"
            ),
            semantic_version=_require_exact_string(
                document["semantic_version"], "semantic_version"
            ),
            request_id=_require_sha256(document["request_id"], "request_id"),
            task_id=_require_identifier(document["task_id"], "task_id"),
            ordinal=_require_ordinal(document["ordinal"]),
            attempt_id=_require_identifier(document["attempt_id"], "attempt_id"),
            status=cast(Literal["completed", "failed"], status),
            return_code=_require_return_code(document["return_code"]),
            artifact_root_sha256=(
                None
                if artifact_digest is None
                else _require_sha256(artifact_digest, "artifact_root_sha256")
            ),
            exception_code=(
                None
                if exception_code is None
                else _require_identifier(exception_code, "exception_code")
            ),
        )


def write_frame(stream: BinaryIO, document: Mapping[str, object]) -> None:
    """Write one length-prefixed canonical JSON document."""

    payload = canonical_json_bytes(dict(document))
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("frame payload size is outside the allowed range")
    stream.write(_FRAME_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int, *, allow_clean_eof: bool) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if chunk is None:
            raise OSError("IPC stream returned no bytes")
        if not chunk:
            if allow_clean_eof and not chunks:
                raise EOFError("IPC stream reached EOF")
            raise EOFError("IPC stream ended inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def read_frame(stream: BinaryIO) -> dict[str, object]:
    """Read and validate one length-prefixed canonical JSON document."""

    header = _read_exact(stream, _FRAME_HEADER.size, allow_clean_eof=True)
    (size,) = _FRAME_HEADER.unpack(header)
    if size == 0 or size > MAX_FRAME_BYTES:
        raise ValueError("frame payload size is outside the allowed range")
    payload = _read_exact(stream, size, allow_clean_eof=False)
    try:
        decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frame payload is not valid JSON") from error
    if type(decoded) is not dict:
        raise TypeError("frame payload must be a JSON object")
    document = cast(dict[str, object], decoded)
    if canonical_json_bytes(document) != payload:
        raise ValueError("frame payload is not canonical JSON")
    return document


__all__ = [
    "MAX_FRAME_BYTES",
    "SAME_TASK_WORKER_CONTRACT",
    "SAME_TASK_WORKER_SEMANTIC_VERSION",
    "SameTaskWorkerReadyIdentity",
    "SameTaskWorkerRequest",
    "SameTaskWorkerResponse",
    "compute_request_id",
    "read_frame",
    "write_frame",
]
