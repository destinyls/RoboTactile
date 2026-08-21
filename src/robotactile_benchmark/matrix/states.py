"""Explicit pending and terminal state views for matrix summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from robotactile_benchmark.matrix.contracts import (
    MatrixCellSpec,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.matrix.results import (
    CellArtifactReference,
    MatrixCellExecution,
    MatrixCellReceipt,
    MatrixCellStatus,
    _optional_integer,
    _optional_string,
    _validate_fault_metadata,
)
from robotactile_benchmark.trials import Condition


@dataclass(frozen=True)
class MatrixCellState:
    """Summary state, including explicit pending cells during partial runs."""

    cell_sha256: str
    trial_manifest_sha256: str
    task: str
    pair_key: str
    condition: Condition
    operator_id: Optional[str]
    severity_level: Optional[int]
    status: MatrixCellStatus
    receipt_sha256: Optional[str]
    artifact: Optional[CellArtifactReference]
    failure_code: Optional[str]

    @classmethod
    def pending(cls, cell: MatrixCellSpec) -> MatrixCellState:
        return cls._from_cell(cell, MatrixCellStatus.PENDING, None, None, None)

    @classmethod
    def from_receipt(
        cls, cell: MatrixCellSpec, receipt: MatrixCellReceipt
    ) -> MatrixCellState:
        return cls._from_cell(
            cell, receipt.status, receipt.sha256, receipt.artifact, receipt.failure_code
        )

    @classmethod
    def _from_cell(
        cls,
        cell: MatrixCellSpec,
        status: MatrixCellStatus,
        receipt_sha256: Optional[str],
        artifact: Optional[CellArtifactReference],
        failure_code: Optional[str],
    ) -> MatrixCellState:
        return cls(
            cell.sha256,
            cell.trial.sha256,
            cell.task,
            cell.pair_key,
            cell.trial.condition,
            cell.operator_id,
            cell.severity_level,
            status,
            receipt_sha256,
            artifact,
            failure_code,
        )

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
        _validate_fault_metadata(condition, self.operator_id, self.severity_level)
        if status is MatrixCellStatus.PENDING:
            if any(
                value is not None
                for value in (self.receipt_sha256, self.artifact, self.failure_code)
            ):
                raise ValueError("pending state cannot reference a terminal receipt")
        else:
            require_sha256(self.receipt_sha256, "receipt_sha256")
            MatrixCellExecution(status, self.artifact, self.failure_code)

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
            "receipt_sha256": self.receipt_sha256,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixCellState:
        fields = {
            "cell_sha256",
            "trial_manifest_sha256",
            "task",
            "pair_key",
            "condition",
            "operator_id",
            "severity_level",
            "status",
            "receipt_sha256",
            "artifact",
            "failure_code",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("matrix cell state fields mismatch")
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
            receipt_sha256=(
                None
                if value["receipt_sha256"] is None
                else require_sha256(value["receipt_sha256"], "receipt_sha256")
            ),
            artifact=(
                None if artifact is None else CellArtifactReference.from_dict(artifact)
            ),
            failure_code=_optional_string(value["failure_code"], "failure_code"),
        )
