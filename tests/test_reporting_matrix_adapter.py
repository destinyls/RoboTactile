from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robotactile_benchmark import cli
from robotactile_benchmark.closed_loop.result_hashes import terminal_trace_sha256
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    REST_REFERENCE_OPERATOR_IDS,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    CellArtifactReference,
    MatrixCellExecution,
    MatrixCellSpec,
    MatrixGridPoint,
    MatrixResumeError,
    build_focused_phase_manifest,
    build_primary_matrix_manifest,
    load_matrix_summary,
    run_matrix,
)
from robotactile_benchmark.matrix.io import canonical_matrix_json_bytes
from robotactile_benchmark.reporting import ReportingSpec, matrix_adapter
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_trial() -> TrialManifest:
    return TrialManifest(
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


def _fault(operator_id: str, severity: int) -> FaultManifest:
    parameters: dict[str, object] = {}
    if operator_id in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = "d" * 64
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=9000 + severity,
        start_index=20,
        stop_index=100,
        sensor_slots=("left", "right")
        if operator_id == "C1_sensor_identity_misrouting"
        else ("left",),
        observability=Observability.BLIND,
        parameters=parameters,
    )


def _primary_manifest():
    return build_primary_matrix_manifest(
        matrix_id="report-primary-v1",
        clean=_clean_trial(),
        fault_manifests=tuple(
            _fault(operator_id, severity)
            for operator_id in sorted(CORE_OPERATOR_IDS)
            for severity in range(1, 6)
        ),
        restoration_index=60,
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )


def _trial_result(
    trial: TrialManifest, terminal_status: TerminalStatus
) -> ClosedLoopTrialResult:
    normal_statuses = {
        TerminalStatus.SUCCESS,
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
    }
    assert terminal_status in normal_statuses
    score_success = terminal_status is TerminalStatus.SUCCESS
    values = {
        "trial_manifest_sha256": trial.sha256,
        "pair_key": trial.pair_key,
        "run_spec_sha256": _digest("run-spec"),
        "initial_state_sha256": _digest("initial-state"),
        "terminal_status": terminal_status,
        "execution_status": terminal_status,
        "score_eligible": True,
        "score_success": score_success,
        "validation_passed": True,
        "validation_failure_codes": (),
        "clean_trace_sha256": _digest(f"clean:{trial.sha256}"),
        "delivered_trace_sha256": _digest(f"delivered:{trial.sha256}"),
        "action_trace_sha256": _digest(f"actions:{trial.sha256}"),
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
    return ClosedLoopTrialResult(**values)


def _materialize_primary(tmp_path: Path, *, receipt_only_failures: bool = False):
    manifest = _primary_manifest()
    output = tmp_path / "matrix"
    verified: dict[str, matrix_adapter._VerifiedArtifact] = {}

    def executor(cell: MatrixCellSpec) -> MatrixCellExecution:
        if (
            receipt_only_failures
            and cell.operator_id == "A1_stream_absence"
            and cell.severity_level == 1
        ):
            if cell.trial.condition is Condition.FAULTED:
                return MatrixCellExecution.unsupported("tactile_required")
            if cell.trial.condition is Condition.RESTORED:
                return MatrixCellExecution.crash("executor_exception:test")
        status = {
            Condition.CLEAN: TerminalStatus.SUCCESS,
            Condition.NO_TOUCH: TerminalStatus.TASK_FAILURE,
            Condition.FAULTED: TerminalStatus.TASK_FAILURE,
            Condition.RESTORED: TerminalStatus.SUCCESS,
        }[cell.trial.condition]
        result = _trial_result(cell.trial, status)
        root = _digest(f"root:{cell.sha256}")
        evidence = "unqualified_live_univtac_execution_v1"
        verified[root] = matrix_adapter._VerifiedArtifact(
            trial=cell.trial,
            fault_manifest=cell.fault_manifest,
            result=result,
            root_receipt_sha256=root,
            evidence_level=evidence,
        )
        return MatrixCellExecution.completed(
            CellArtifactReference(result.sha256, root, evidence)
        )

    run_matrix(output, manifest, executor)
    return manifest, output, verified


def _install_fake_artifact_loader(
    monkeypatch: pytest.MonkeyPatch,
    verified: dict[str, matrix_adapter._VerifiedArtifact],
) -> list[Path]:
    loaded_paths: list[Path] = []

    def load(path: Path, reference: CellArtifactReference):
        loaded_paths.append(path)
        assert path.name == reference.root_receipt_sha256
        return verified[reference.root_receipt_sha256]

    monkeypatch.setattr(matrix_adapter, "_load_verified_artifact", load)
    return loaded_paths


def _spec() -> ReportingSpec:
    return ReportingSpec(
        system_id="univtac-act-tactile",
        matched_control_qualified=True,
        minimum_clean_gain=0.05,
        primary_operator_ids=tuple(sorted(CORE_OPERATOR_IDS)),
        primary_severity_levels=(1, 2, 3, 4, 5),
        bootstrap_resamples=10,
        bootstrap_seed=7,
    )


def test_matrix_outcomes_use_strict_artifact_results_and_content_addressed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, output, verified = _materialize_primary(tmp_path)
    loaded_paths = _install_fake_artifact_loader(monkeypatch, verified)

    outcomes = matrix_adapter.load_matrix_outcomes(
        output / "matrix_manifest.json", output, _spec()
    )

    assert len(outcomes) == len(manifest.cells) == 142
    assert len(loaded_paths) == 142
    assert all(path.parent == output / "artifacts" for path in loaded_paths)
    clean = next(item for item in outcomes if item.condition is Condition.CLEAN)
    no_touch = next(item for item in outcomes if item.condition is Condition.NO_TOUCH)
    assert clean.score_success is True
    assert no_touch.score_success is False
    assert clean.source_root_sha256 == loaded_paths[0].name


def test_matrix_outcomes_fail_closed_on_summary_or_artifact_crosslink_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, verified = _materialize_primary(tmp_path)
    _install_fake_artifact_loader(monkeypatch, verified)
    summary_path = output / "matrix_summary.json"
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    document["matrix_id"] = "tampered"
    summary_path.write_bytes(canonical_matrix_json_bytes(document))

    with pytest.raises(MatrixResumeError, match="summary"):
        matrix_adapter.load_matrix_outcomes(
            output / "matrix_manifest.json", output, _spec()
        )

    _, other_output, other_verified = _materialize_primary(tmp_path / "other")
    first = next(iter(other_verified))
    original = other_verified[first]
    other_verified[first] = matrix_adapter._VerifiedArtifact(
        trial=original.trial,
        fault_manifest=original.fault_manifest,
        result=original.result,
        root_receipt_sha256="9" * 64,
        evidence_level=original.evidence_level,
    )
    _install_fake_artifact_loader(monkeypatch, other_verified)
    with pytest.raises(ValueError, match="artifact root"):
        matrix_adapter.load_matrix_outcomes(
            other_output / "matrix_manifest.json", other_output, _spec()
        )


def test_focused_matrix_reporting_is_explicitly_unsupported(tmp_path: Path) -> None:
    fault = _fault("F1_global_response_drift", 1)
    manifest = build_focused_phase_manifest(
        matrix_id="focused",
        clean=_clean_trial(),
        grid_points=(MatrixGridPoint("contact-rise", fault, 60),),
        restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )
    output = tmp_path / "focused"
    run_matrix(
        output,
        manifest,
        lambda cell: MatrixCellExecution.unsupported("not_materialized"),
    )

    with pytest.raises(
        ValueError,
        match="focused matrix reporting requires registered comparison identity/recovery signal",
    ):
        matrix_adapter.load_matrix_outcomes(
            output / "matrix_manifest.json", output, _spec()
        )


def test_receipt_only_unsupported_and_crash_are_preserved_without_cached_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, output, verified = _materialize_primary(
        tmp_path, receipt_only_failures=True
    )
    _install_fake_artifact_loader(monkeypatch, verified)

    outcomes = matrix_adapter.load_matrix_outcomes(
        output / "matrix_manifest.json", output, _spec()
    )

    unsupported = next(
        item
        for item in outcomes
        if item.condition is Condition.FAULTED
        and item.operator_id == "A1_stream_absence"
        and item.severity_level == 1
    )
    crash = next(
        item
        for item in outcomes
        if item.condition is Condition.RESTORED
        and item.operator_id == "A1_stream_absence"
        and item.severity_level == 1
    )
    summary = load_matrix_summary(output, manifest)
    receipt_roots = {
        state.receipt_sha256 for state in summary.cells if state.artifact is None
    }
    assert unsupported.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
    assert unsupported.score_eligible is False
    assert unsupported.score_success is None
    assert crash.terminal_status is TerminalStatus.CRASH
    assert crash.score_eligible is True
    assert crash.score_success is False
    assert {unsupported.source_root_sha256, crash.source_root_sha256} == receipt_roots


def test_unknown_matrix_evidence_is_not_upgraded_to_a_loadable_artifact(
    tmp_path: Path,
) -> None:
    reference = CellArtifactReference(
        result_sha256="1" * 64,
        root_receipt_sha256="2" * 64,
        evidence_level="cpu_fake_matrix_test",
    )

    with pytest.raises(ValueError, match="supported loadable bundle type"):
        matrix_adapter._load_verified_artifact(tmp_path / "missing", reference)


def test_report_writer_and_cli_emit_source_bound_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, output, verified = _materialize_primary(tmp_path)
    _install_fake_artifact_loader(monkeypatch, verified)
    spec_path = tmp_path / "reporting_spec.json"
    spec_path.write_bytes(canonical_matrix_json_bytes(_spec().to_dict()))
    report_output = tmp_path / "report"

    exported = matrix_adapter.write_matrix_report(
        output / "matrix_manifest.json",
        output,
        spec_path,
        report_output,
    )

    assert exported.manifest_sha256 == manifest.sha256
    assert exported.outcome_count == 142
    assert exported.receipt.source_root_sha256 == exported.summary.source_root_sha256
    assert (report_output / "report_receipt.json").is_file()

    second_output = tmp_path / "report-cli"
    assert (
        cli.main(
            [
                "report-matrix",
                "--matrix-manifest",
                str(output / "matrix_manifest.json"),
                "--matrix-output",
                str(output),
                "--reporting-spec",
                str(spec_path),
                "--output",
                str(second_output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_sha256"] == manifest.sha256
    assert payload["simulator_qualification_claimed"] is False
    assert payload["summary_sha256"] == exported.summary.sha256
