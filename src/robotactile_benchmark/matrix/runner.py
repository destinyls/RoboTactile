"""Deterministic, resumable execution of one materialized matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from robotactile_benchmark.matrix.contracts import MatrixCellSpec
from robotactile_benchmark.matrix.io import (
    MATRIX_SUMMARY_PATH,
    cell_receipt_path,
    load_cell_receipt,
    load_matrix_summary,
    prepare_matrix_output,
    write_cell_receipt,
    write_matrix_summary,
)
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.matrix.results import (
    MatrixCellExecution,
    MatrixCellReceipt,
    MatrixCellStatus,
)
from robotactile_benchmark.matrix.states import MatrixCellState
from robotactile_benchmark.matrix.summary import MatrixRunResult, MatrixSummary


class MatrixCellExecutor(Protocol):
    """Injected single-cell runner used by CPU fakes and live backends alike."""

    def __call__(self, cell: MatrixCellSpec) -> MatrixCellExecution: ...


def run_matrix(
    output: Path,
    manifest: MatrixManifest,
    executor: MatrixCellExecutor,
    *,
    max_new_cells: Optional[int] = None,
) -> MatrixRunResult:
    """Execute missing cells, reuse only strict receipts, and never drop a cell."""

    if not isinstance(manifest, MatrixManifest):
        raise TypeError("manifest must be a MatrixManifest")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if max_new_cells is not None and (
        isinstance(max_new_cells, bool)
        or not isinstance(max_new_cells, int)
        or max_new_cells < 0
    ):
        raise ValueError("max_new_cells must be a non-negative integer or None")
    output = Path(output)
    prepare_matrix_output(output, manifest)
    summary_path = output / MATRIX_SUMMARY_PATH
    if summary_path.exists() or summary_path.is_symlink():
        loaded_summary = load_matrix_summary(output, manifest)
        return MatrixRunResult(
            states=loaded_summary.cells,
            summary=loaded_summary,
            executed_cell_count=0,
            reused_cell_count=len(loaded_summary.cells),
            pending_cell_count=0,
        )

    states: list[MatrixCellState] = []
    executed_count = 0
    reused_count = 0
    for cell in manifest.cells:
        receipt_path = cell_receipt_path(output, cell)
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = load_cell_receipt(output, cell)
            states.append(MatrixCellState.from_receipt(cell, receipt))
            reused_count += 1
            continue
        if max_new_cells is not None and executed_count >= max_new_cells:
            states.append(MatrixCellState.pending(cell))
            continue
        execution = _execute_cell(executor, cell)
        receipt = MatrixCellReceipt.from_execution(cell, execution)
        write_cell_receipt(output, cell, receipt)
        states.append(MatrixCellState.from_receipt(cell, receipt))
        executed_count += 1

    frozen_states = tuple(states)
    pending_count = sum(
        state.status is MatrixCellStatus.PENDING for state in frozen_states
    )
    summary: Optional[MatrixSummary] = None
    if pending_count == 0:
        summary = MatrixSummary(
            matrix_id=manifest.matrix_id,
            kind=manifest.kind,
            manifest_sha256=manifest.sha256,
            pair_key=manifest.pair_key,
            cells=frozen_states,
        )
        write_matrix_summary(output, summary)
        summary = load_matrix_summary(output, manifest)
    return MatrixRunResult(
        states=frozen_states,
        summary=summary,
        executed_cell_count=executed_count,
        reused_cell_count=reused_count,
        pending_cell_count=pending_count,
    )


def _execute_cell(
    executor: MatrixCellExecutor, cell: MatrixCellSpec
) -> MatrixCellExecution:
    try:
        result = executor(cell)
        if not isinstance(result, MatrixCellExecution):
            raise TypeError("matrix executor returned an untyped result")
        return result
    except Exception as error:
        error_type = type(error)
        return MatrixCellExecution.crash(
            f"executor_exception:{error_type.__module__}.{error_type.__qualname__}"
        )
