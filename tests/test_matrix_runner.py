from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    CellArtifactReference,
    MatrixCellExecution,
    MatrixCellSpec,
    MatrixCellStatus,
    MatrixResumeError,
    build_primary_matrix_manifest,
    load_matrix_summary,
    run_matrix,
)
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TrialManifest,
    system_manifest_hash,
)


def _manifest():
    clean = TrialManifest(
        task="pull_out_key",
        initial_seed=11,
        exogenous_seed=29,
        condition=Condition.CLEAN,
        base_system_id="univtac-act-tactile",
        executed_system_id="univtac-act-tactile",
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            "univtac-act-tactile", "b" * 64, "c" * 64, "qpos8_next_step"
        ),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=None,
        matched_no_touch_system_id=None,
        restoration_index=None,
        restoration_mode=None,
    )
    faults = tuple(
        FaultManifest(
            operator_id=operator_id,
            severity_level=severity,
            operator_seed=1000 + severity,
            start_index=20,
            stop_index=100,
            sensor_slots=("left", "right")
            if operator_id == "C1_sensor_identity_misrouting"
            else ("left",),
            observability=Observability.BLIND,
            parameters={
                **(
                    {"rest_reference_sha256": "d" * 64}
                    if operator_id in REST_REFERENCE_OPERATOR_IDS
                    else {}
                ),
                **(
                    {"realization": "registered_pixels"}
                    if operator_id == "C2_frame_misregistration"
                    else {}
                ),
            },
        )
        for operator_id in sorted(CORE_OPERATOR_IDS)
        for severity in range(1, 6)
    )
    return build_primary_matrix_manifest(
        matrix_id="runner-primary-v1",
        clean=clean,
        fault_manifests=faults,
        restoration_index=60,
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, cell: MatrixCellSpec) -> MatrixCellExecution:
        self.calls.append(cell.sha256)
        fault = cell.fault_manifest
        if (
            fault is not None
            and fault.operator_id == "A1_stream_absence"
            and fault.severity_level == 5
            and cell.trial.condition is Condition.FAULTED
        ):
            return MatrixCellExecution.unsupported("tactile_required")
        if (
            fault is not None
            and fault.operator_id == "F4_local_nonresponsive_patch"
            and fault.severity_level == 4
            and cell.trial.condition is Condition.RESTORED
        ):
            return MatrixCellExecution.validator_rejected(
                CellArtifactReference(
                    result_sha256="7" * 64,
                    root_receipt_sha256="8" * 64,
                    evidence_level="cpu_fake_matrix_test",
                ),
                "operator_validator_failed",
            )
        if (
            fault is not None
            and fault.operator_id == "T2_held_last_freeze"
            and fault.severity_level == 2
            and cell.trial.condition is Condition.FAULTED
        ):
            raise RuntimeError("simulated backend crash")
        return MatrixCellExecution.completed(
            CellArtifactReference(
                result_sha256="1" * 64,
                root_receipt_sha256="2" * 64,
                evidence_level="cpu_fake_matrix_test",
            )
        )


def test_cpu_fake_primary_e2e_preserves_all_cells_and_executes_baselines_once(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    executor = FakeExecutor()

    result = run_matrix(tmp_path / "matrix", manifest, executor)
    summary = load_matrix_summary(tmp_path / "matrix", manifest)

    assert result.summary == summary
    assert result.executed_cell_count == 142
    assert result.reused_cell_count == 0
    assert result.pending_cell_count == 0
    assert len(summary.cells) == 142
    assert Counter(state.status for state in summary.cells) == {
        MatrixCellStatus.COMPLETED: 139,
        MatrixCellStatus.CRASH: 1,
        MatrixCellStatus.UNSUPPORTED: 1,
        MatrixCellStatus.VALIDATOR_REJECTED: 1,
    }
    assert len(executor.calls) == len(set(executor.calls)) == 142
    assert sum(state.condition is Condition.CLEAN for state in summary.cells) == 1
    assert sum(state.condition is Condition.NO_TOUCH for state in summary.cells) == 1
    assert all(state.task == "pull_out_key" for state in summary.cells)
    assert all(state.pair_key == manifest.pair_key for state in summary.cells)


def test_resume_strictly_reuses_every_verified_cell_without_executor_calls(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    first = FakeExecutor()
    run_matrix(tmp_path / "matrix", manifest, first)
    resumed = FakeExecutor()

    result = run_matrix(tmp_path / "matrix", manifest, resumed)

    assert result.executed_cell_count == 0
    assert result.reused_cell_count == 142
    assert result.pending_cell_count == 0
    assert resumed.calls == []


def test_partial_run_returns_every_unexecuted_cell_as_pending_then_resumes(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    first = FakeExecutor()
    partial = run_matrix(tmp_path / "matrix", manifest, first, max_new_cells=3)

    assert partial.summary is None
    assert partial.executed_cell_count == 3
    assert partial.reused_cell_count == 0
    assert partial.pending_cell_count == 139
    assert len(partial.states) == 142
    assert Counter(state.status for state in partial.states) == {
        MatrixCellStatus.COMPLETED: 3,
        MatrixCellStatus.PENDING: 139,
    }

    resumed = FakeExecutor()
    complete = run_matrix(tmp_path / "matrix", manifest, resumed)
    assert complete.executed_cell_count == 139
    assert complete.reused_cell_count == 3
    assert complete.pending_cell_count == 0
    assert len(resumed.calls) == 139


def test_corrupt_existing_cell_fails_closed_instead_of_silently_recomputing(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    run_matrix(tmp_path / "matrix", manifest, FakeExecutor(), max_new_cells=1)
    receipt_path = next((tmp_path / "matrix" / "cells").glob("*.json"))
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    document["failure_code"] = "tampered"
    receipt_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    resumed = FakeExecutor()

    with pytest.raises(MatrixResumeError, match="verified cell"):
        run_matrix(tmp_path / "matrix", manifest, resumed)
    assert resumed.calls == []


def test_manifest_and_summary_are_atomic_no_clobber_boundaries(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    output = tmp_path / "matrix"
    run_matrix(output, manifest, FakeExecutor())
    summary_before = (output / "matrix_summary.json").read_bytes()

    run_matrix(output, manifest, FakeExecutor())
    assert (output / "matrix_summary.json").read_bytes() == summary_before

    manifest_document = json.loads(
        (output / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    manifest_document["matrix_id"] = "tampered"
    (output / "matrix_manifest.json").write_text(
        json.dumps(manifest_document) + "\n", encoding="utf-8"
    )
    with pytest.raises(MatrixResumeError, match="matrix manifest"):
        run_matrix(output, manifest, FakeExecutor())


def test_summary_loader_does_not_create_a_missing_output(tmp_path: Path) -> None:
    manifest = _manifest()
    output = tmp_path / "missing"

    with pytest.raises(MatrixResumeError):
        load_matrix_summary(output, manifest)
    assert not output.exists()
