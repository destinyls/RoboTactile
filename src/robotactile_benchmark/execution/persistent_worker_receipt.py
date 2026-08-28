"""Canonical evidence for one task-local persistent execution worker."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, Tuple, cast

from robotactile_benchmark.clean_baseline.io import (
    read_canonical_json_file,
    write_canonical_no_clobber,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

PERSISTENT_WORKER_EVIDENCE_LEVEL = "persistent_worker_session_receipt_v1"
PERSISTENT_WORKER_SEMANTIC_VERSION = "1.0"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHUTDOWN_STATUSES = frozenset({"clean", "forced", "crashed"})


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] not in {"artifacts", "outputs", "requests", "sources"}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must remain below the deployment root")
    return value


def _timestamp(value: object, name: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value, parsed


@dataclass(frozen=True)
class EpisodeDispatch:
    """Identity and timing of one episode leased to the worker."""

    task_id: str
    campaign_id: str
    sequence: int
    ordinal: int
    attempt_id: str
    initial_seed: int
    exogenous_seed: int
    request_file_sha256: str
    trial_manifest_sha256: str
    dispatched_at_utc: str
    completed_at_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        object.__setattr__(
            self, "attempt_id", _identifier(self.attempt_id, "attempt_id")
        )
        for name in ("sequence", "ordinal", "initial_seed", "exogenous_seed"):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name, minimum=0),
            )
        for name in ("request_file_sha256", "trial_manifest_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        dispatched, dispatched_time = _timestamp(
            self.dispatched_at_utc, "dispatched_at_utc"
        )
        completed, completed_time = _timestamp(
            self.completed_at_utc, "completed_at_utc"
        )
        if completed_time < dispatched_time:
            raise ValueError("episode completion precedes dispatch")
        object.__setattr__(self, "dispatched_at_utc", dispatched)
        object.__setattr__(self, "completed_at_utc", completed)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "EpisodeDispatch":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("episode dispatch fields mismatch")
        return cls(**cast(dict[str, Any], dict(value)))


@dataclass(frozen=True)
class PersistentWorkerSessionReceipt:
    """Hash-bound final receipt for one task-local persistent worker."""

    task_id: str
    campaign_id: str
    worker_session_id: str
    worker_pid: int
    worker_process_group_id: int
    worker_posix_session_id: int
    restart_generation: int
    runtime_source_binding: RuntimeSourceBinding
    qualification_relpath: str
    qualification_sha256: str
    n0_server_attestation_relpath: str
    n0_server_attestation_sha256: str
    reset_equivalence_receipt_relpath: str
    reset_equivalence_receipt_sha256: str
    episode_dispatches: Tuple[EpisodeDispatch, ...]
    started_at_utc: str
    finished_at_utc: str
    shutdown_status: str
    content_sha256: str
    evidence_level: str = PERSISTENT_WORKER_EVIDENCE_LEVEL
    semantic_version: str = PERSISTENT_WORKER_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in ("task_id", "campaign_id", "worker_session_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "worker_pid",
            "worker_process_group_id",
            "worker_posix_session_id",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        object.__setattr__(
            self,
            "restart_generation",
            _positive_integer(self.restart_generation, "restart_generation", minimum=0),
        )
        if type(self.runtime_source_binding) is not RuntimeSourceBinding:
            raise TypeError("runtime_source_binding must be RuntimeSourceBinding")
        for name in (
            "qualification_relpath",
            "n0_server_attestation_relpath",
            "reset_equivalence_receipt_relpath",
        ):
            object.__setattr__(self, name, _relative_path(getattr(self, name), name))
        for name in (
            "qualification_sha256",
            "n0_server_attestation_sha256",
            "reset_equivalence_receipt_sha256",
            "content_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        dispatches = tuple(self.episode_dispatches)
        if not dispatches or any(
            type(item) is not EpisodeDispatch for item in dispatches
        ):
            raise ValueError("episode_dispatches must contain typed episodes")
        sequences = tuple(item.sequence for item in dispatches)
        if sequences != tuple(range(len(dispatches))):
            raise ValueError("episode dispatch sequence must be unique and contiguous")
        if any(
            item.task_id != self.task_id or item.campaign_id != self.campaign_id
            for item in dispatches
        ):
            raise ValueError("persistent worker episodes must share task and campaign")
        if len({item.attempt_id for item in dispatches}) != len(dispatches):
            raise ValueError("episode attempt IDs must be unique")
        started, started_time = _timestamp(self.started_at_utc, "started_at_utc")
        finished, finished_time = _timestamp(self.finished_at_utc, "finished_at_utc")
        if finished_time < started_time:
            raise ValueError("worker finish precedes start")
        if any(
            _timestamp(item.dispatched_at_utc, "dispatched_at_utc")[1] < started_time
            or _timestamp(item.completed_at_utc, "completed_at_utc")[1] > finished_time
            for item in dispatches
        ):
            raise ValueError("episode dispatch lies outside worker lifetime")
        if self.shutdown_status not in _SHUTDOWN_STATUSES:
            raise ValueError("shutdown_status is unsupported")
        if (
            self.evidence_level != PERSISTENT_WORKER_EVIDENCE_LEVEL
            or self.semantic_version != PERSISTENT_WORKER_SEMANTIC_VERSION
        ):
            raise ValueError("persistent worker receipt version mismatch")
        object.__setattr__(self, "episode_dispatches", dispatches)
        object.__setattr__(self, "started_at_utc", started)
        object.__setattr__(self, "finished_at_utc", finished)
        if self.content_sha256 != canonical_hash(self._content_dict()):
            raise ValueError("persistent worker receipt content hash mismatch")

    def _content_dict(self) -> dict[str, object]:
        return {
            name: (
                self.runtime_source_binding.to_dict()
                if name == "runtime_source_binding"
                else [item.to_dict() for item in self.episode_dispatches]
                if name == "episode_dispatches"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "content_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def build(cls, **values: Any) -> "PersistentWorkerSessionReceipt":
        content = {
            name: (
                values[name].to_dict()
                if name == "runtime_source_binding"
                else [item.to_dict() for item in values[name]]
                if name == "episode_dispatches"
                else values[name]
            )
            for name in cls.__dataclass_fields__
            if name != "content_sha256"
        }
        return cls(**values, content_sha256=canonical_hash(content))

    @classmethod
    def from_dict(cls, value: object) -> "PersistentWorkerSessionReceipt":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("persistent worker receipt fields mismatch")
        parsed = dict(value)
        parsed["runtime_source_binding"] = RuntimeSourceBinding.from_dict(
            parsed["runtime_source_binding"]
        )
        dispatches = parsed["episode_dispatches"]
        if not isinstance(dispatches, Sequence) or isinstance(dispatches, (str, bytes)):
            raise TypeError("episode_dispatches must be a sequence")
        parsed["episode_dispatches"] = tuple(
            EpisodeDispatch.from_dict(item) for item in dispatches
        )
        return cls(**cast(dict[str, Any], parsed))


def persistent_worker_receipt_sha256(path: Path) -> str:
    """Return the byte-level SHA256 of one stable receipt file."""

    _, raw = read_canonical_json_file(path, "persistent worker receipt")
    return hashlib.sha256(raw).hexdigest()


def write_persistent_worker_session_receipt(
    path: Path, receipt: PersistentWorkerSessionReceipt
) -> bool:
    """Publish a typed receipt without replacing different existing bytes."""

    if type(receipt) is not PersistentWorkerSessionReceipt:
        raise TypeError("receipt must be an exact PersistentWorkerSessionReceipt")
    return write_canonical_no_clobber(path, receipt.to_dict())


def load_persistent_worker_session_receipt(
    path: Path, *, expected_sha256: str | None = None
) -> PersistentWorkerSessionReceipt:
    """Load a canonical receipt and optionally verify its file digest."""

    document, raw = read_canonical_json_file(path, "persistent worker receipt")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, "expected_sha256"
    ):
        raise ValueError("persistent worker receipt SHA256 mismatch")
    return PersistentWorkerSessionReceipt.from_dict(document)


__all__ = [
    "EpisodeDispatch",
    "PersistentWorkerSessionReceipt",
    "load_persistent_worker_session_receipt",
    "persistent_worker_receipt_sha256",
    "write_persistent_worker_session_receipt",
]
