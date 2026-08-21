from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.result_hashes import terminal_trace_sha256
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.execution.live_univtac import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    CellArtifactReference,
    MatrixCellExecution,
    MatrixCellReceipt,
    MatrixCellSpec,
    MatrixCellState,
)
from robotactile_benchmark.matrix.io import canonical_matrix_json_bytes
from robotactile_benchmark.reporting import ReportingSpec, matrix_adapter
from robotactile_benchmark.reporting.recovery_evidence import (
    POLICY_TASK_PROGRESS_SIGNAL_ID,
    RecoveryEvidence,
    RecoveryEvidenceBinding,
    RecoveryEvidenceValidationError,
    load_recovery_evidence,
    recovery_evidence_path,
    write_recovery_evidence,
)
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(*, clean_root: str = "1" * 64) -> RecoveryEvidenceBinding:
    return RecoveryEvidenceBinding(
        clean_live_artifact_root_sha256=clean_root,
        restored_live_artifact_root_sha256="2" * 64,
        task="pull_out_key",
        pair_key="3" * 64,
        operator_id="F1_global_response_drift",
        severity_level=3,
        restoration_index=2,
    )


def _evidence(
    *,
    clean_root: str = "1" * 64,
    tolerance: float = 0.02,
    quality: tuple[float, ...] = (0.9, 0.8, 0.7, 0.51, 0.49, 0.5),
) -> RecoveryEvidence:
    return RecoveryEvidence.evaluate(
        binding=_binding(clean_root=clean_root),
        signal_id=POLICY_TASK_PROGRESS_SIGNAL_ID,
        clean_envelope=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        restored_quality=quality,
        tolerance=tolerance,
        consecutive_steps=2,
    )


def test_writer_publishes_canonical_content_address_and_loader_recomputes_lag(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    directory = tmp_path / "recovery"

    written = write_recovery_evidence(directory, evidence)
    path = recovery_evidence_path(directory, "2" * 64)
    loaded = load_recovery_evidence(path, expected=_binding())

    assert written == loaded == evidence
    assert loaded.recovery_lag_steps == 1
    assert len(loaded.sha256) == 64
    assert path.name == f"{'2' * 64}.json"
    assert path.read_bytes() == canonical_matrix_json_bytes(evidence.to_dict())


def test_unrecovered_evidence_preserves_none_after_strict_reload(
    tmp_path: Path,
) -> None:
    evidence = _evidence(quality=(0.9, 0.8, 0.7, 0.6, 0.6, 0.6))

    write_recovery_evidence(tmp_path, evidence)
    loaded = load_recovery_evidence(recovery_evidence_path(tmp_path, "2" * 64))

    assert evidence.recovery_lag_steps is None
    assert loaded.recovery_lag_steps is None


def test_unknown_signal_and_unevaluable_criterion_are_rejected() -> None:
    with pytest.raises(ValueError, match="registered"):
        RecoveryEvidence.evaluate(
            binding=_binding(),
            signal_id="private.unregistered.signal",
            clean_envelope=(0.5, 0.5, 0.5),
            restored_quality=(0.9, 0.5, 0.5),
            tolerance=0.01,
            consecutive_steps=1,
        )

    with pytest.raises(ValueError, match="post-restoration"):
        RecoveryEvidence.evaluate(
            binding=_binding(),
            signal_id=POLICY_TASK_PROGRESS_SIGNAL_ID,
            clean_envelope=(0.5, 0.5, 0.5),
            restored_quality=(0.9, 0.5, 0.5),
            tolerance=0.01,
            consecutive_steps=2,
        )


@pytest.mark.parametrize("mutation", ["unknown", "stale_lag", "noncanonical"])
def test_loader_rejects_unknown_stale_and_noncanonical_documents(
    tmp_path: Path, mutation: str
) -> None:
    evidence = _evidence()
    path = recovery_evidence_path(tmp_path, "2" * 64)
    document = evidence.to_dict()
    if mutation == "unknown":
        document["unregistered_field"] = True
        raw = canonical_matrix_json_bytes(document)
    elif mutation == "stale_lag":
        document["recovery_lag_steps"] = 0
        raw = canonical_matrix_json_bytes(document)
    else:
        raw = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)

    with pytest.raises(RecoveryEvidenceValidationError):
        load_recovery_evidence(path)


def test_loader_rejects_stale_filename_and_symlink(tmp_path: Path) -> None:
    evidence = _evidence()
    canonical = recovery_evidence_path(tmp_path, "2" * 64)
    canonical.write_bytes(canonical_matrix_json_bytes(evidence.to_dict()))
    stale = canonical.with_name(f"{'4' * 64}.json")
    stale.write_bytes(canonical.read_bytes())

    with pytest.raises(RecoveryEvidenceValidationError, match="filename"):
        load_recovery_evidence(stale)

    link = canonical.with_name(f"{'5' * 64}.json")
    try:
        link.symlink_to(canonical)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(RecoveryEvidenceValidationError, match="regular file"):
        load_recovery_evidence(link)


def test_writer_is_idempotent_but_never_clobbers_another_evidence(
    tmp_path: Path,
) -> None:
    evidence = _evidence()

    write_recovery_evidence(tmp_path, evidence)
    before = recovery_evidence_path(tmp_path, "2" * 64).read_bytes()
    assert write_recovery_evidence(tmp_path, evidence) == evidence
    assert recovery_evidence_path(tmp_path, "2" * 64).read_bytes() == before

    with pytest.raises(FileExistsError, match="another recovery evidence"):
        write_recovery_evidence(tmp_path, _evidence(tolerance=0.03))


def _restored_cell() -> MatrixCellSpec:
    fault = FaultManifest(
        operator_id="F1_global_response_drift",
        severity_level=3,
        operator_seed=17,
        start_index=1,
        stop_index=2,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters={"rest_reference_sha256": "d" * 64},
    )
    system_id = "univtac-act-tactile"
    checkpoint = "a" * 64
    config = "b" * 64
    trial = TrialManifest(
        task="pull_out_key",
        initial_seed=11,
        exogenous_seed=29,
        condition=Condition.RESTORED,
        base_system_id=system_id,
        executed_system_id=system_id,
        dataset_sha256="c" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            system_id, checkpoint, config, "qpos8_next_step"
        ),
        checkpoint_sha256=checkpoint,
        config_sha256=config,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=fault.sha256,
        matched_no_touch_system_id=None,
        restoration_index=2,
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
    )
    return MatrixCellSpec(trial=trial, fault_manifest=fault)


def _result(trial: TrialManifest) -> ClosedLoopTrialResult:
    values: dict[str, object] = {
        "trial_manifest_sha256": trial.sha256,
        "pair_key": trial.pair_key,
        "run_spec_sha256": _digest("run"),
        "initial_state_sha256": _digest("state"),
        "terminal_status": TerminalStatus.SUCCESS,
        "execution_status": TerminalStatus.SUCCESS,
        "score_eligible": True,
        "score_success": True,
        "validation_passed": True,
        "validation_failure_codes": (),
        "clean_trace_sha256": _digest("clean"),
        "delivered_trace_sha256": _digest("delivered"),
        "action_trace_sha256": _digest("actions"),
        "observation_count": 2,
        "control_cycle_count": 1,
        "failure_stage": None,
        "failure_code": None,
        "semantic_version": "1.0",
    }
    values["terminal_trace_sha256"] = terminal_trace_sha256(
        trial_manifest_sha256=values["trial_manifest_sha256"],
        pair_key=values["pair_key"],
        run_spec_sha256=values["run_spec_sha256"],
        initial_state_sha256=values["initial_state_sha256"],
        terminal_status=values["terminal_status"],
        execution_status=values["execution_status"],
        score_eligible=values["score_eligible"],
        score_success=values["score_success"],
        validation_passed=values["validation_passed"],
        validation_failure_codes=values["validation_failure_codes"],
        clean_trace_sha256=values["clean_trace_sha256"],
        delivered_trace_sha256=values["delivered_trace_sha256"],
        action_trace_sha256_value=values["action_trace_sha256"],
        observation_count=values["observation_count"],
        control_cycle_count=values["control_cycle_count"],
        failure_stage=values["failure_stage"],
        failure_code=values["failure_code"],
        semantic_version=values["semantic_version"],
    )
    return ClosedLoopTrialResult(**values)  # type: ignore[arg-type]


def _spec() -> ReportingSpec:
    return ReportingSpec(
        system_id="univtac-act-tactile",
        matched_control_qualified=False,
        minimum_clean_gain=0.05,
        primary_operator_ids=("F1_global_response_drift",),
        bootstrap_resamples=5,
        bootstrap_seed=7,
    )


def test_matrix_adapter_uses_default_bound_evidence_and_missing_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _restored_cell()
    result = _result(cell.trial)
    restored_root = "2" * 64
    clean_root = "1" * 64
    reference = CellArtifactReference(
        result_sha256=result.sha256,
        root_receipt_sha256=restored_root,
        evidence_level=LIVE_ARTIFACT_EVIDENCE_LEVEL,
    )
    receipt = MatrixCellReceipt.from_execution(
        cell, MatrixCellExecution.completed(reference)
    )
    state = MatrixCellState.from_receipt(cell, receipt)
    verified = matrix_adapter._VerifiedArtifact(
        trial=cell.trial,
        fault_manifest=cell.fault_manifest,
        result=result,
        root_receipt_sha256=restored_root,
        evidence_level=LIVE_ARTIFACT_EVIDENCE_LEVEL,
    )
    monkeypatch.setattr(
        matrix_adapter,
        "_load_verified_artifact",
        lambda path, artifact_reference: verified,
    )
    binding = RecoveryEvidenceBinding(
        clean_live_artifact_root_sha256=clean_root,
        restored_live_artifact_root_sha256=restored_root,
        task=cell.task,
        pair_key=cell.pair_key,
        operator_id=cell.operator_id,
        severity_level=cell.severity_level,
        restoration_index=cell.trial.restoration_index,
    )
    evidence = RecoveryEvidence.evaluate(
        binding=binding,
        signal_id=POLICY_TASK_PROGRESS_SIGNAL_ID,
        clean_envelope=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        restored_quality=(0.9, 0.8, 0.7, 0.51, 0.49, 0.5),
        tolerance=0.02,
        consecutive_steps=2,
    )

    missing = matrix_adapter._outcome_from_cell(
        cell,
        state,
        _spec(),
        lambda unused: tmp_path / "artifact",
        matrix_output=tmp_path,
        clean_root_sha256=clean_root,
    )
    assert missing.recovery_eligible is False
    assert missing.recovery_lag_steps is None

    write_recovery_evidence(tmp_path / "recovery", evidence)
    recovered = matrix_adapter._outcome_from_cell(
        cell,
        state,
        _spec(),
        lambda unused: tmp_path / "artifact",
        matrix_output=tmp_path,
        clean_root_sha256=clean_root,
    )
    assert recovered.recovery_eligible is True
    assert recovered.recovery_lag_steps == 1
    assert recovered.source_root_sha256 == evidence.sha256


def test_matrix_adapter_rejects_cross_cell_recovery_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _restored_cell()
    result = _result(cell.trial)
    reference = CellArtifactReference(
        result_sha256=result.sha256,
        root_receipt_sha256="2" * 64,
        evidence_level=LIVE_ARTIFACT_EVIDENCE_LEVEL,
    )
    state = MatrixCellState.from_receipt(
        cell,
        MatrixCellReceipt.from_execution(
            cell, MatrixCellExecution.completed(reference)
        ),
    )
    monkeypatch.setattr(
        matrix_adapter,
        "_load_verified_artifact",
        lambda path, artifact_reference: matrix_adapter._VerifiedArtifact(
            trial=cell.trial,
            fault_manifest=cell.fault_manifest,
            result=result,
            root_receipt_sha256="2" * 64,
            evidence_level=LIVE_ARTIFACT_EVIDENCE_LEVEL,
        ),
    )
    write_recovery_evidence(tmp_path / "recovery", _evidence(clean_root="6" * 64))

    with pytest.raises(RecoveryEvidenceValidationError, match="binding"):
        matrix_adapter._outcome_from_cell(
            cell,
            state,
            _spec(),
            lambda unused: tmp_path / "artifact",
            matrix_output=tmp_path,
            clean_root_sha256="1" * 64,
        )
