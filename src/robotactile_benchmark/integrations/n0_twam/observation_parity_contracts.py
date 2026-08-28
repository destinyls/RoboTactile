"""Typed source-bound observation-contract parity evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional, Tuple, cast

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

OBSERVATION_PARITY_EVIDENCE_LEVEL = (
    "source_bound_n0_univtac_observation_contract_parity_v2"
)
OBSERVATION_PARITY_SEMANTIC_VERSION = "2.0"
REQUIRED_EVIDENCE_KINDS = (
    "experiment_lock",
    "teacher_forced_probe",
    "isaac_install_receipt",
    "n0_client_install_receipt",
    "n0_runtime_receipt",
    "tacex_install_receipt",
    "n0_artifact_manifest",
)
REQUIRED_GATE_IDS = (
    "source_identity",
    "experiment_lock",
    "teacher_forced_probe",
    "camera_renderer",
    "model_boundary",
    "state_registration",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class ObservationParityError(ValueError):
    """Observation parity evidence is malformed or contradictory."""


class ObservationParityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ObservationParityError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise ObservationParityError(f"{name} must be a lowercase SHA256")
    return normalized


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ObservationParityError(f"{name} must be a boolean")
    return cast(bool, value)


def _relative_path(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if "\\" in normalized:
        raise ObservationParityError(f"{name} must be a POSIX path")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ObservationParityError(f"{name} is unsafe")
    if not path.parts or path.parts[0] not in {
        "artifacts",
        "outputs",
        "requests",
    }:
        raise ObservationParityError(f"{name} must stay in deployment evidence")
    return normalized


@dataclass(frozen=True)
class ObservationEvidenceBinding:
    """One immutable input file used by an observation parity decision."""

    kind: str
    relpath: str
    sha256: str
    content_sha256: Optional[str]

    def __post_init__(self) -> None:
        kind = _nonempty(self.kind, "evidence kind")
        if kind not in REQUIRED_EVIDENCE_KINDS:
            raise ObservationParityError("unknown observation evidence kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "relpath",
            _relative_path(self.relpath, "evidence path"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "evidence SHA256"))
        if self.content_sha256 is not None:
            object.__setattr__(
                self,
                "content_sha256",
                _sha256(self.content_sha256, "evidence content SHA256"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "kind": self.kind,
            "relpath": self.relpath,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ObservationEvidenceBinding":
        fields = {"content_sha256", "kind", "relpath", "sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ObservationParityError("observation evidence fields mismatch")
        document = cast(Mapping[str, object], value)
        return cls(
            kind=_nonempty(document["kind"], "evidence kind"),
            relpath=_relative_path(document["relpath"], "evidence path"),
            sha256=_sha256(document["sha256"], "evidence SHA256"),
            content_sha256=(
                None
                if document["content_sha256"] is None
                else _sha256(
                    document["content_sha256"],
                    "evidence content SHA256",
                )
            ),
        )


@dataclass(frozen=True)
class ObservationParityGate:
    """One explicit PASS/FAIL/UNKNOWN observation-contract decision."""

    gate_id: str
    status: ObservationParityStatus
    code: str
    detail: str

    def __post_init__(self) -> None:
        gate_id = _nonempty(self.gate_id, "gate ID")
        code = _nonempty(self.code, "gate code")
        if gate_id not in REQUIRED_GATE_IDS or _IDENTIFIER.fullmatch(code) is None:
            raise ObservationParityError("observation parity gate identity is invalid")
        object.__setattr__(self, "gate_id", gate_id)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "detail", _nonempty(self.detail, "gate detail"))
        object.__setattr__(
            self,
            "status",
            self.status
            if isinstance(self.status, ObservationParityStatus)
            else ObservationParityStatus(self.status),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "detail": self.detail,
            "gate_id": self.gate_id,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ObservationParityGate":
        fields = {"code", "detail", "gate_id", "status"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ObservationParityError("observation parity gate fields mismatch")
        document = cast(Mapping[str, object], value)
        return cls(
            gate_id=_nonempty(document["gate_id"], "gate ID"),
            status=ObservationParityStatus(document["status"]),
            code=_nonempty(document["code"], "gate code"),
            detail=_nonempty(document["detail"], "gate detail"),
        )


@dataclass(frozen=True)
class ObservationParityArtifact:
    """Canonical result that binds parity decisions to exact source artifacts."""

    task_id: str
    source_binding: RuntimeSourceBinding
    evidence: Tuple[ObservationEvidenceBinding, ...]
    gates: Tuple[ObservationParityGate, ...]
    passed: bool
    failure_codes: Tuple[str, ...]
    limitations: Tuple[str, ...]
    content_sha256: str
    evidence_level: str = OBSERVATION_PARITY_EVIDENCE_LEVEL
    semantic_version: str = OBSERVATION_PARITY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task ID"))
        if type(self.source_binding) is not RuntimeSourceBinding:
            raise TypeError("source_binding must be an exact RuntimeSourceBinding")
        evidence = tuple(self.evidence)
        if tuple(item.kind for item in evidence) != REQUIRED_EVIDENCE_KINDS:
            raise ObservationParityError("observation evidence inventory mismatch")
        gates = tuple(self.gates)
        if tuple(item.gate_id for item in gates) != REQUIRED_GATE_IDS:
            raise ObservationParityError("observation parity gate inventory mismatch")
        expected_pass = all(
            item.status is ObservationParityStatus.PASS for item in gates
        )
        expected_codes = tuple(
            item.code
            for item in gates
            if item.status is not ObservationParityStatus.PASS
        )
        if type(self.passed) is not bool or self.passed is not expected_pass:
            raise ObservationParityError("observation parity pass flag mismatch")
        if tuple(self.failure_codes) != expected_codes:
            raise ObservationParityError("observation parity failure codes mismatch")
        limitations = tuple(_nonempty(item, "limitation") for item in self.limitations)
        if not limitations or len(limitations) != len(set(limitations)):
            raise ObservationParityError("observation parity limitations are invalid")
        if (
            self.evidence_level != OBSERVATION_PARITY_EVIDENCE_LEVEL
            or self.semantic_version != OBSERVATION_PARITY_SEMANTIC_VERSION
        ):
            raise ObservationParityError("observation parity version mismatch")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "gates", gates)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content SHA256"),
        )
        if self.content_sha256 != canonical_hash(self._content_dict()):
            raise ObservationParityError("observation parity content hash mismatch")

    def _content_dict(self) -> dict[str, object]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_level": self.evidence_level,
            "failure_codes": list(self.failure_codes),
            "gates": [item.to_dict() for item in self.gates],
            "limitations": list(self.limitations),
            "passed": self.passed,
            "semantic_version": self.semantic_version,
            "source_binding": self.source_binding.to_dict(),
            "task_id": self.task_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        source_binding: RuntimeSourceBinding,
        evidence: Sequence[ObservationEvidenceBinding],
        gates: Sequence[ObservationParityGate],
        limitations: Sequence[str],
    ) -> "ObservationParityArtifact":
        evidence_tuple = tuple(evidence)
        gate_tuple = tuple(gates)
        failure_codes = tuple(
            item.code
            for item in gate_tuple
            if item.status is not ObservationParityStatus.PASS
        )
        content = {
            "evidence": [item.to_dict() for item in evidence_tuple],
            "evidence_level": OBSERVATION_PARITY_EVIDENCE_LEVEL,
            "failure_codes": list(failure_codes),
            "gates": [item.to_dict() for item in gate_tuple],
            "limitations": list(limitations),
            "passed": not failure_codes,
            "semantic_version": OBSERVATION_PARITY_SEMANTIC_VERSION,
            "source_binding": source_binding.to_dict(),
            "task_id": task_id,
        }
        return cls(
            task_id=task_id,
            source_binding=source_binding,
            evidence=evidence_tuple,
            gates=gate_tuple,
            passed=not failure_codes,
            failure_codes=failure_codes,
            limitations=tuple(limitations),
            content_sha256=canonical_hash(content),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ObservationParityArtifact":
        fields = {
            "content_sha256",
            "evidence",
            "evidence_level",
            "failure_codes",
            "gates",
            "limitations",
            "passed",
            "semantic_version",
            "source_binding",
            "task_id",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ObservationParityError("observation parity artifact fields mismatch")
        document = cast(Mapping[str, object], value)
        for name in ("evidence", "gates", "failure_codes", "limitations"):
            if not isinstance(document[name], Sequence) or isinstance(
                document[name], (str, bytes)
            ):
                raise ObservationParityError(f"{name} must be a sequence")
        evidence = cast(Sequence[object], document["evidence"])
        gates = cast(Sequence[object], document["gates"])
        failure_codes = cast(Sequence[object], document["failure_codes"])
        limitations = cast(Sequence[object], document["limitations"])
        return cls(
            task_id=_nonempty(document["task_id"], "task ID"),
            source_binding=RuntimeSourceBinding.from_dict(document["source_binding"]),
            evidence=tuple(
                ObservationEvidenceBinding.from_dict(item) for item in evidence
            ),
            gates=tuple(ObservationParityGate.from_dict(item) for item in gates),
            passed=_boolean(document["passed"], "passed"),
            failure_codes=tuple(
                _nonempty(item, "failure code") for item in failure_codes
            ),
            limitations=tuple(_nonempty(item, "limitation") for item in limitations),
            content_sha256=_sha256(document["content_sha256"], "content SHA256"),
            evidence_level=_nonempty(document["evidence_level"], "evidence level"),
            semantic_version=_nonempty(
                document["semantic_version"], "semantic version"
            ),
        )


__all__ = [
    "OBSERVATION_PARITY_EVIDENCE_LEVEL",
    "OBSERVATION_PARITY_SEMANTIC_VERSION",
    "ObservationEvidenceBinding",
    "ObservationParityArtifact",
    "ObservationParityError",
    "ObservationParityGate",
    "ObservationParityStatus",
    "REQUIRED_EVIDENCE_KINDS",
    "REQUIRED_GATE_IDS",
]
