from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from robotactile_benchmark.backends.qualification_checks import qualification_action
from robotactile_benchmark.backends.qualification_fakes import make_fake_runtime
from robotactile_benchmark.backends.univtac_pairing import (
    UniVTACPairedBackendSession,
)
from robotactile_benchmark.closed_loop.contracts import ActionPlan
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.execution import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
    LiveExecutionUnavailableError,
    LivePolicyKind,
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    LiveMatrixCellExecutor,
    LiveMatrixCellResources,
    LiveMatrixExecutionTemplate,
    MatrixCellSpec,
    MatrixCellStatus,
    MatrixGridPoint,
    PairedLiveMatrixExecutor,
    build_focused_phase_manifest,
    live_matrix_artifact_path,
    materialize_live_matrix_request,
)
from robotactile_benchmark.trials import (
    Condition,
    TrialManifest,
    system_manifest_hash,
)


def _matrix_cells() -> tuple[MatrixCellSpec, ...]:
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
    )
    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=700,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters={},
    )
    manifest = build_focused_phase_manifest(
        matrix_id="live-executor-three-condition-v1",
        clean=clean,
        grid_points=(MatrixGridPoint("contact-window", fault),),
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )
    return manifest.cells


def _write_fault(path: Path, fault: FaultManifest) -> None:
    path.write_text(
        json.dumps(fault.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class _Resolver:
    def __init__(self, root: Path, cells: tuple[MatrixCellSpec, ...]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.calls: list[Condition] = []
        by_condition = {cell.trial.condition: cell for cell in cells}
        persistent = by_condition[Condition.FAULTED].fault_manifest
        assert persistent is not None
        self.persistent = root / "persistent.json"
        self.no_touch = root / "vision_only" / "policy_last.ckpt"
        _write_fault(self.persistent, persistent)

    def __call__(self, cell: MatrixCellSpec) -> LiveMatrixCellResources:
        condition = cell.trial.condition
        self.calls.append(condition)
        if condition is Condition.NO_TOUCH:
            return LiveMatrixCellResources(
                fault_manifest_path=None,
                rest_references_path=None,
                matched_no_touch_artifact_path=self.no_touch,
            )
        fault_path = self.persistent if condition is Condition.FAULTED else None
        return LiveMatrixCellResources(
            fault_manifest_path=fault_path,
            rest_references_path=None,
            matched_no_touch_artifact_path=None,
        )


def _template(root: Path, **updates: Any) -> LiveMatrixExecutionTemplate:
    template = LiveMatrixExecutionTemplate(
        policy_kind=LivePolicyKind.ACT,
        max_control_cycles=4,
        max_observation_steps=5,
        execute_action_steps=1,
        wall_timeout_s=30.0,
        upstream_root=root / "UniVTAC",
        runtime_root=root / "runtime",
        act_device_name="cpu",
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
    )
    return replace(template, **updates)


def _backend(loaded: Any) -> DeterministicFakeBackend:
    return DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )


class _BoundedUniVTACPolicy(DeterministicFakePolicy):
    """Keep the generic fake lifecycle but emit a valid frozen qpos8 target."""

    def infer(self, observation: Any) -> ActionPlan:
        inferred = super().infer(observation)
        return ActionPlan(
            action_spec=inferred.action_spec,
            source_step_index=inferred.source_step_index,
            actions=qualification_action(),
        )


def _policy(loaded: Any) -> DeterministicFakePolicy:
    return _BoundedUniVTACPolicy.for_trial(
        loaded.trial, supports_structural_absence=True
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_frozen_template_materializes_exact_three_condition_requests(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    template = _template(tmp_path)

    with pytest.raises(FrozenInstanceError):
        template.max_control_cycles = 9  # type: ignore[misc]
    for cell in cells:
        resources = resolver(cell)
        output = tmp_path / "staging" / cell.sha256
        request = materialize_live_matrix_request(
            cell, template, resources, output_dir=output
        )
        condition = cell.trial.condition
        assert request.condition is condition
        assert request.output_dir == output.absolute()
        assert request.runtime_dir == (template.runtime_root / cell.sha256).absolute()
        assert request.checkpoint_sha256 == cell.trial.checkpoint_sha256
        assert request.config_sha256 == cell.trial.config_sha256
        if condition is Condition.NO_TOUCH:
            assert (
                request.matched_no_touch_artifact_path == resolver.no_touch.absolute()
            )
            assert request.fault_manifest_path is None
        elif condition is Condition.CLEAN:
            assert request.matched_no_touch_artifact_path is None
            assert request.fault_manifest_path is None
        else:
            assert request.fault_manifest_path == resolver.persistent.absolute()
            assert request.matched_no_touch_artifact_path is None


def test_cpu_fake_three_condition_artifacts_are_strict_content_addresses(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    matrix_output = tmp_path / "matrix"
    executor = LiveMatrixCellExecutor(
        matrix_output,
        _template(tmp_path),
        resolver,
        backend_factory=_backend,
        policy_factory=_policy,
    )

    executions = {cell.trial.condition: executor(cell) for cell in cells}

    assert set(executions) == set(Condition)
    assert resolver.calls == [cell.trial.condition for cell in cells]
    for cell in cells:
        execution = executions[cell.trial.condition]
        assert execution.status is MatrixCellStatus.COMPLETED
        assert execution.failure_code is None
        assert execution.artifact is not None
        assert execution.artifact.evidence_level == LIVE_ARTIFACT_EVIDENCE_LEVEL
        artifact_path = live_matrix_artifact_path(
            matrix_output, execution.artifact.root_receipt_sha256
        )
        artifact = load_live_univtac_artifact(artifact_path)
        assert artifact.trial == cell.trial
        assert artifact.root_receipt.result_sha256 == execution.artifact.result_sha256
        assert artifact.external_root_sha256 == artifact_path.name
        assert artifact.root_receipt.simulator_qualification_claimed is False


def test_paired_matrix_uses_one_runtime_and_one_canonical_reset(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    matrix_output = tmp_path / "paired-matrix"
    tasks = []

    def session_factory(loaded: Any) -> UniVTACPairedBackendSession:
        runtime, task = make_fake_runtime(
            loaded.backend_config,
            construction_seed=loaded.trial.initial_seed,
        )
        tasks.append(task)
        return UniVTACPairedBackendSession(loaded.backend_config, runtime)

    executor = PairedLiveMatrixExecutor(
        matrix_output,
        _template(tmp_path),
        resolver,
        policy_factory=_policy,
        session_factory=session_factory,
    )
    executions = executor.execute_batch(cells)

    assert len(executions) == len(cells)
    assert all(item.status is MatrixCellStatus.COMPLETED for item in executions)
    assert len(tasks) == 1
    assert tasks[0].reset_count == 1
    assert tasks[0].capture_count == 1
    assert tasks[0].restore_count == len(cells) - 1
    assert tasks[0].close_count == 1
    receipt = matrix_output / "paired_execution_receipt.json"
    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["reset_receipt"]["all_exact"] is True
    assert len(payload["executions"]) == len(cells)


def test_repeated_execution_reuses_identical_address_without_clobber(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    cell = next(item for item in cells if item.trial.condition is Condition.CLEAN)
    matrix_output = tmp_path / "matrix"
    executor = LiveMatrixCellExecutor(
        matrix_output,
        _template(tmp_path),
        resolver,
        backend_factory=_backend,
        policy_factory=_policy,
    )

    first = executor(cell)
    assert first.artifact is not None
    artifact_path = live_matrix_artifact_path(
        matrix_output, first.artifact.root_receipt_sha256
    )
    before = _tree_bytes(artifact_path)
    second = executor(cell)

    assert second == first
    assert _tree_bytes(artifact_path) == before
    assert sorted(path.name for path in (matrix_output / "artifacts").iterdir()) == [
        first.artifact.root_receipt_sha256
    ]


def test_typed_crash_and_unsupported_results_keep_real_artifacts(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    by_condition = {cell.trial.condition: cell for cell in cells}

    def crashing_policy(loaded: Any) -> DeterministicFakePolicy:
        return DeterministicFakePolicy.for_trial(loaded.trial, fail_on_infer=True)

    crash_resolver = _Resolver(tmp_path / "crash", cells)
    crash = LiveMatrixCellExecutor(
        tmp_path / "crash-matrix",
        _template(tmp_path),
        crash_resolver,
        backend_factory=_backend,
        policy_factory=crashing_policy,
    )(by_condition[Condition.CLEAN])

    def unsupported_policy(loaded: Any) -> DeterministicFakePolicy:
        return DeterministicFakePolicy.for_trial(
            loaded.trial, supports_structural_absence=False
        )

    unsupported_resolver = _Resolver(tmp_path / "unsupported", cells)
    unsupported = LiveMatrixCellExecutor(
        tmp_path / "unsupported-matrix",
        _template(tmp_path),
        unsupported_resolver,
        backend_factory=_backend,
        policy_factory=unsupported_policy,
    )(by_condition[Condition.FAULTED])

    assert crash.status is MatrixCellStatus.CRASH
    assert crash.failure_code == "infer_failed"
    assert crash.artifact is not None
    assert unsupported.status is MatrixCellStatus.UNSUPPORTED
    assert unsupported.failure_code == "structural_absence_unsupported"
    assert unsupported.artifact is not None


def test_live_unavailability_is_unsupported_without_fabricated_artifact(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    cell = next(item for item in cells if item.trial.condition is Condition.CLEAN)

    def unavailable_backend(loaded: Any) -> DeterministicFakeBackend:
        raise LiveExecutionUnavailableError(
            "isaaclab_app_unavailable", "isaaclab.app is unavailable"
        )

    matrix_output = tmp_path / "unavailable-matrix"
    execution = LiveMatrixCellExecutor(
        matrix_output,
        _template(tmp_path),
        resolver,
        backend_factory=unavailable_backend,
        policy_factory=_policy,
    )(cell)

    assert execution.status is MatrixCellStatus.UNSUPPORTED
    assert (
        execution.failure_code == "live_execution_unavailable:isaaclab_app_unavailable"
    )
    assert execution.artifact is None
    assert list((matrix_output / "artifacts").iterdir()) == []

    no_touch = next(c for c in cells if c.trial.condition is Condition.NO_TOUCH)
    missing = LiveMatrixCellExecutor(
        tmp_path / "missing-control-matrix",
        _template(tmp_path),
        resolver,
        backend_factory=_backend,
    )(no_touch)
    assert missing.status is MatrixCellStatus.UNSUPPORTED
    assert (
        missing.failure_code
        == "live_execution_unavailable:official_policy_factory_required"
    )
    assert missing.artifact is None


def test_untyped_preflight_failure_propagates_without_fabricated_success(
    tmp_path: Path,
) -> None:
    cells = _matrix_cells()
    resolver = _Resolver(tmp_path, cells)
    cell = next(item for item in cells if item.trial.condition is Condition.CLEAN)

    def broken_backend(loaded: Any) -> DeterministicFakeBackend:
        raise RuntimeError("unexpected construction failure")

    matrix_output = tmp_path / "broken-matrix"
    executor = LiveMatrixCellExecutor(
        matrix_output,
        _template(tmp_path),
        resolver,
        backend_factory=broken_backend,
        policy_factory=_policy,
    )

    with pytest.raises(RuntimeError, match="unexpected construction failure"):
        executor(cell)
    assert list((matrix_output / "artifacts").iterdir()) == []
