"""Typed cell outcomes and deterministic matrix summaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.matrix.contracts import (
    MATRIX_SEMANTIC_VERSION,
    MatrixCellSpec,
    require_integer,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.trials import Condition, TerminalStatus


class MatrixCellStatus(str, Enum):
    """Stable state for every requested matrix execution."""

    PENDING = "pending"
    COMPLETED = "completed"
    CRASH = "crash"
    UNSUPPORTED = "unsupported"
    VALIDATOR_REJECTED = "validator_rejected"


@dataclass(frozen=True)
class CellArtifactReference:
    """Content addresses of a single-cell result and artifact root receipt."""

    result_sha256: str
    root_receipt_sha256: str
    evidence_level: str

    def __post_init__(self) -> None:
        for name in ("result_sha256", "root_receipt_sha256"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "evidence_level",
            require_nonempty(self.evidence_level, "evidence_level"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "result_sha256": self.result_sha256,
            "root_receipt_sha256": self.root_receipt_sha256,
            "evidence_level": self.evidence_level,
        }

    @classmethod
    def from_dict(cls, value: object) -> CellArtifactReference:
        if not isinstance(value, Mapping) or set(value) != {
            "result_sha256",
            "root_receipt_sha256",
            "evidence_level",
        }:
            raise ValueError("cell artifact reference fields mismatch")
        return cls(
            result_sha256=require_sha256(value["result_sha256"], "result_sha256"),
            root_receipt_sha256=require_sha256(
                value["root_receipt_sha256"], "root_receipt_sha256"
            ),
            evidence_level=require_nonempty(value["evidence_level"], "evidence_level"),
        )


@dataclass(frozen=True)
class MatrixCellExecution:
    """Executor-returned terminal outcome before matrix metadata is attached."""

    status: MatrixCellStatus
    artifact: Optional[CellArtifactReference]
    failure_code: Optional[str]

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, MatrixCellStatus)
            else MatrixCellStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        if status is MatrixCellStatus.PENDING:
            raise ValueError("an executor cannot return a pending outcome")
        if self.artifact is not None and not isinstance(
            self.artifact, CellArtifactReference
        ):
            raise TypeError("artifact must be a CellArtifactReference or None")
        failure = self.failure_code
        if failure is not None:
            failure = require_nonempty(failure, "failure_code")
            object.__setattr__(self, "failure_code", failure)
        if status is MatrixCellStatus.COMPLETED:
            if self.artifact is None or failure is not None:
                raise ValueError(
                    "completed execution requires an artifact and no failure"
                )
        elif status is MatrixCellStatus.VALIDATOR_REJECTED:
            if self.artifact is None or failure is None:
                raise ValueError(
                    "validator rejection requires artifact and failure code"
                )
        elif failure is None:
            raise ValueError("crash/unsupported execution requires a failure code")

    @classmethod
    def completed(cls, artifact: CellArtifactReference) -> MatrixCellExecution:
        return cls(MatrixCellStatus.COMPLETED, artifact, None)

    @classmethod
    def crash(cls, failure_code: str) -> MatrixCellExecution:
        return cls(MatrixCellStatus.CRASH, None, failure_code)

    @classmethod
    def unsupported(cls, failure_code: str) -> MatrixCellExecution:
        return cls(MatrixCellStatus.UNSUPPORTED, None, failure_code)

    @classmethod
    def validator_rejected(
        cls, artifact: CellArtifactReference, failure_code: str
    ) -> MatrixCellExecution:
        return cls(MatrixCellStatus.VALIDATOR_REJECTED, artifact, failure_code)

    @classmethod
    def from_closed_loop(
        cls,
        result: ClosedLoopTrialResult,
        *,
        root_receipt_sha256: str,
        evidence_level: str,
    ) -> MatrixCellExecution:
        """Map a strict single-trial result without caching any score fields."""

        artifact = CellArtifactReference(
            result_sha256=result.sha256,
            root_receipt_sha256=root_receipt_sha256,
            evidence_level=evidence_level,
        )
        status = result.terminal_status
        if status is TerminalStatus.CRASH:
            return cls(MatrixCellStatus.CRASH, artifact, result.failure_code or "crash")
        if status is TerminalStatus.UNSUPPORTED_CONTRACT:
            return cls(
                MatrixCellStatus.UNSUPPORTED,
                artifact,
                result.failure_code or "unsupported_contract",
            )
        if status is TerminalStatus.VALIDATOR_REJECTED:
            return cls.validator_rejected(
                artifact, result.failure_code or "validator_rejected"
            )
        return cls.completed(artifact)


@dataclass(frozen=True)
class MatrixCellReceipt:
    """Persisted terminal cell record with reporting-safe static metadata."""

    cell_sha256: str
    trial_manifest_sha256: str
    task: str
    pair_key: str
    condition: Condition
    operator_id: Optional[str]
    severity_level: Optional[int]
    status: MatrixCellStatus
    artifact: Optional[CellArtifactReference]
    failure_code: Optional[str]
    semantic_version: str = MATRIX_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in ("cell_sha256", "trial_manifest_sha256", "pair_key"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        object.__setattr__(self, "task", require_nonempty(self.task, "task"))
        condition = (
            self.condition
            if isinstance(self.condition, Condition)
            else Condition(self.condition)
        )
        status = (
            self.status
            if isinstance(self.status, MatrixCellStatus)
            else MatrixCellStatus(self.status)
        )
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "status", status)
        if status is MatrixCellStatus.PENDING:
            raise ValueError("persisted cell receipt cannot be pending")
        if self.semantic_version != MATRIX_SEMANTIC_VERSION:
            raise ValueError("unsupported matrix receipt semantic version")
        _validate_fault_metadata(condition, self.operator_id, self.severity_level)
        MatrixCellExecution(status, self.artifact, self.failure_code)

    @classmethod
    def from_execution(
        cls, cell: MatrixCellSpec, execution: MatrixCellExecution
    ) -> MatrixCellReceipt:
        return cls(
            cell_sha256=cell.sha256,
            trial_manifest_sha256=cell.trial.sha256,
            task=cell.task,
            pair_key=cell.pair_key,
            condition=cell.trial.condition,
            operator_id=cell.operator_id,
            severity_level=cell.severity_level,
            status=execution.status,
            artifact=execution.artifact,
            failure_code=execution.failure_code,
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell_sha256,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "task": self.task,
            "pair_key": self.pair_key,
            "condition": self.condition.value,
            "operator_id": self.operator_id,
            "severity_level": self.severity_level,
            "status": self.status.value,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "failure_code": self.failure_code,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixCellReceipt:
        fields = {
            "cell_sha256",
            "trial_manifest_sha256",
            "task",
            "pair_key",
            "condition",
            "operator_id",
            "severity_level",
            "status",
            "artifact",
            "failure_code",
            "semantic_version",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("matrix cell receipt fields mismatch")
        artifact = value["artifact"]
        return cls(
            cell_sha256=require_sha256(value["cell_sha256"], "cell_sha256"),
            trial_manifest_sha256=require_sha256(
                value["trial_manifest_sha256"], "trial_manifest_sha256"
            ),
            task=require_nonempty(value["task"], "task"),
            pair_key=require_sha256(value["pair_key"], "pair_key"),
            condition=Condition(require_nonempty(value["condition"], "condition")),
            operator_id=_optional_string(value["operator_id"], "operator_id"),
            severity_level=_optional_integer(value["severity_level"], "severity_level"),
            status=MatrixCellStatus(require_nonempty(value["status"], "status")),
            artifact=None
            if artifact is None
            else CellArtifactReference.from_dict(artifact),
            failure_code=_optional_string(value["failure_code"], "failure_code"),
            semantic_version=require_nonempty(
                value["semantic_version"], "semantic_version"
            ),
        )


def _validate_fault_metadata(
    condition: Condition, operator_id: Optional[str], severity_level: Optional[int]
) -> None:
    faulted = condition in {Condition.FAULTED, Condition.RESTORED}
    if faulted != (operator_id is not None and severity_level is not None):
        raise ValueError("fault metadata must exist exactly for faulted/restored cells")
    if operator_id is not None:
        require_nonempty(operator_id, "operator_id")
    if severity_level is not None:
        level = require_integer(severity_level, "severity_level", minimum=1)
        if level > 5:
            raise ValueError("severity_level must be in [1, 5]")


def _optional_string(value: object, name: str) -> Optional[str]:
    return None if value is None else require_nonempty(value, name)


def _optional_integer(value: object, name: str) -> Optional[int]:
    return None if value is None else require_integer(value, name)
