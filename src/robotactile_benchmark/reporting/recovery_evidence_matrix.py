"""Matrix identity bridge for optional evaluator-side recovery evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from robotactile_benchmark.matrix.contracts import MatrixCellSpec
from robotactile_benchmark.matrix.io import MatrixResumeError
from robotactile_benchmark.matrix.states import MatrixCellState
from robotactile_benchmark.reporting.recovery_evidence import (
    RecoveryEvidenceBinding,
    RecoveryEvidenceValidationError,
    load_recovery_evidence,
    recovery_evidence_path,
)
from robotactile_benchmark.trials import Condition

RECOVERY_DIRECTORY = "recovery"


def clean_artifact_root(
    cells: Tuple[MatrixCellSpec, ...], states: Tuple[MatrixCellState, ...]
) -> Optional[str]:
    """Resolve the unique clean artifact root required by every sidecar."""

    clean_states = [
        state
        for cell, state in zip(cells, states)
        if cell.trial.condition is Condition.CLEAN
    ]
    if len(clean_states) != 1:
        raise MatrixResumeError("matrix lost its unique clean baseline")
    artifact = clean_states[0].artifact
    return None if artifact is None else artifact.root_receipt_sha256


def recovery_outcome_fields(
    matrix_output: Path,
    cell: MatrixCellSpec,
    clean_root_sha256: Optional[str],
    restored_root_sha256: str,
) -> tuple[bool, Optional[int], str]:
    """Return eligibility, recomputed lag, and the report-bound source hash."""

    if cell.trial.condition is not Condition.RESTORED:
        return False, None, restored_root_sha256
    directory = Path(matrix_output) / RECOVERY_DIRECTORY
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise RecoveryEvidenceValidationError(
            "matrix recovery evidence directory must be a real directory"
        )
    path = recovery_evidence_path(directory, restored_root_sha256)
    if not path.exists() and not path.is_symlink():
        return False, None, restored_root_sha256
    if clean_root_sha256 is None:
        raise RecoveryEvidenceValidationError(
            "recovery evidence exists without a clean artifact root"
        )
    operator_id = cell.operator_id
    severity_level = cell.severity_level
    restoration_index = cell.trial.restoration_index
    if operator_id is None or severity_level is None or restoration_index is None:
        raise RecoveryEvidenceValidationError(
            "restored matrix cell lost recovery identity"
        )
    expected = RecoveryEvidenceBinding(
        clean_live_artifact_root_sha256=clean_root_sha256,
        restored_live_artifact_root_sha256=restored_root_sha256,
        task=cell.task,
        pair_key=cell.pair_key,
        operator_id=operator_id,
        severity_level=severity_level,
        restoration_index=restoration_index,
    )
    evidence = load_recovery_evidence(path, expected=expected)
    return True, evidence.recovery_lag_steps, evidence.sha256


__all__ = [
    "RECOVERY_DIRECTORY",
    "clean_artifact_root",
    "recovery_outcome_fields",
]
