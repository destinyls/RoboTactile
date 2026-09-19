from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robotactile_benchmark.backends.qualification_fakes import make_fake_runtime
from robotactile_benchmark.backends.univtac_pairing import (
    UniVTACPairedBackendSession,
    UniVTACPairingError,
)
from robotactile_benchmark.backends.univtac_reset_witness import UniVTACResetReference
from robotactile_benchmark.closed_loop.fakes import DeterministicFakePolicy
from robotactile_benchmark.execution import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    execute_paired_live_univtac_runs,
    write_paired_execution_receipt,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.paired_live_univtac import (
    PAIRED_EXECUTION_SEMANTIC_VERSION,
    execute_referenced_fault_run,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition


def _fault(root: Path) -> Path:
    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters={},
    )
    path = root / "fault.json"
    path.write_text(json.dumps(fault.to_dict()), encoding="utf-8")
    return path


def _request(
    root: Path,
    condition: Condition,
    *,
    fault_path: Path | None = None,
) -> LiveUniVTACRunRequest:
    return LiveUniVTACRunRequest(
        task_id="pull_out_key",
        condition=condition,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="paired-policy-v1",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        base_system_manifest_sha256=None,
        initial_seed=10,
        exogenous_seed=20,
        max_control_cycles=3,
        max_observation_steps=5,
        execute_action_steps=1,
        wall_timeout_s=5.0,
        upstream_root=root / "upstream",
        runtime_dir=root / "shared-runtime",
        output_dir=None,
        fault_manifest_path=fault_path,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu",
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
    )


class _PairedHarness:
    def __init__(self, *, drift: bool = False) -> None:
        self.drift = drift
        self.tasks: list[Any] = []

    def session(self, loaded: Any) -> UniVTACPairedBackendSession:
        runtime, task = make_fake_runtime(
            loaded.backend_config,
            construction_seed=loaded.trial.initial_seed,
        )
        self.tasks.append(task)
        if self.drift:
            original = runtime.restore_state
            assert original is not None

            def restore(snapshot: object) -> None:
                original(snapshot)
                task._state += 0.5

            runtime = replace(runtime, restore_state=restore)
        return UniVTACPairedBackendSession(loaded.backend_config, runtime)

    @staticmethod
    def policy(loaded: Any) -> DeterministicFakePolicy:
        return DeterministicFakePolicy.for_trial(
            loaded.trial, supports_structural_absence=True
        )


@pytest.mark.parametrize("publisher_raises", [False, True])
def test_failure_published_before_close_preserves_original_exception(
    tmp_path: Path,
    publisher_raises: bool,
) -> None:
    harness = _PairedHarness()
    original = RuntimeError("export failed before publication")
    failures: list[dict[str, object]] = []

    def reject(index: int, execution: Any) -> None:
        raise original

    def publish(failure: dict[str, object]) -> None:
        assert harness.tasks[0].close_count == 0
        failures.append(failure)
        if publisher_raises:
            raise ValueError("failure publisher also failed")

    with pytest.raises(RuntimeError) as caught:
        execute_paired_live_univtac_runs(
            (
                _request(tmp_path, Condition.CLEAN),
                _request(tmp_path, Condition.FAULTED, fault_path=_fault(tmp_path)),
            ),
            session_factory=harness.session,
            policy_factory=harness.policy,
            post_execution_gate=reject,
            failure_publisher=publish,
        )
    assert caught.value is original
    assert len(failures) == 1
    assert harness.tasks[0].close_count == 1


def test_referenced_fault_pair_key_mismatch_rejected_before_factory(
    tmp_path: Path,
) -> None:
    harness = _PairedHarness()
    reference = UniVTACResetReference(
        task_id="pull_out_key",
        initial_seed=10,
        exogenous_seed=20,
        pair_key="f" * 64,
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        source_artifact_root_sha256="d" * 64,
        source_result_sha256="e" * 64,
        source_run_content_sha256="1" * 64,
        expected_simulator_state_sha256="2" * 64,
        expected_native_step=0,
        expected_qpos8=(0.0,) * 8,
        qpos_atol=1e-5,
    )

    def exporter(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("exporter must not be reached")

    def publisher(*args: Any) -> None:
        pytest.fail("publisher must not be reached")

    with pytest.raises(ValueError, match="pair key mismatch"):
        execute_referenced_fault_run(
            _request(tmp_path, Condition.FAULTED, fault_path=_fault(tmp_path)),
            reference,
            policy_factory=harness.policy,
            artifact_exporter=exporter,
            pre_close_publisher=publisher,
            session_factory=harness.session,
        )
    assert harness.tasks == []


def test_paired_execution_runs_two_conditions_from_one_canonical_state(
    tmp_path: Path,
) -> None:
    fault_path = _fault(tmp_path)
    harness = _PairedHarness()
    published = []

    def publish_before_close(result: object) -> None:
        assert harness.tasks[0].close_count == 0
        published.append(result)

    result = execute_paired_live_univtac_runs(
        (
            _request(tmp_path, Condition.CLEAN),
            _request(tmp_path, Condition.FAULTED, fault_path=fault_path),
        ),
        session_factory=harness.session,
        policy_factory=harness.policy,
        pre_close_publisher=publish_before_close,
    )

    assert result.reset_receipt.all_exact
    assert result.witness_indices == (0, 1)
    assert len(result.executions) == 2
    assert {
        item.evidence.result.initial_state_sha256 for item in result.executions
    } == {result.reset_receipt.witnesses[0].simulator_state_sha256}
    task = harness.tasks[0]
    assert task.reset_count == 1
    assert task.capture_count == 1
    assert task.restore_count == 1
    assert task.close_count == 1
    assert published == [result]
    assert result.to_dict()["simulator_qualification_claimed"] is False
    assert result.semantic_version == PAIRED_EXECUTION_SEMANTIC_VERSION
    with pytest.raises(ValueError, match="semantic version"):
        replace(result, semantic_version="1.0")
    receipt_path = tmp_path / "paired-receipt.json"
    first_write = write_paired_execution_receipt(receipt_path, result)
    second_write = write_paired_execution_receipt(receipt_path, result)
    assert first_write == second_write
    assert first_write.group_content_sha256 == result.group_content_sha256


def test_paired_execution_aborts_on_restore_drift(tmp_path: Path) -> None:
    fault_path = _fault(tmp_path)
    harness = _PairedHarness(drift=True)

    with pytest.raises(UniVTACPairingError) as captured:
        execute_paired_live_univtac_runs(
            (
                _request(tmp_path, Condition.CLEAN),
                _request(tmp_path, Condition.FAULTED, fault_path=fault_path),
            ),
            session_factory=harness.session,
            policy_factory=harness.policy,
            require_shared_runtime_dir=True,
        )

    assert captured.value.code == "reset_equivalence_mismatch"
    assert harness.tasks[0].close_count == 1


def test_post_execution_gate_stops_before_faulted_cell(tmp_path: Path) -> None:
    fault_path = _fault(tmp_path)
    harness = _PairedHarness()
    observed: list[Condition] = []

    def gate(index: int, execution: Any) -> None:
        assert index == 0
        observed.append(execution.loaded.trial.condition)
        raise RuntimeError("clean baseline rejected")

    with pytest.raises(RuntimeError, match="clean baseline rejected"):
        execute_paired_live_univtac_runs(
            (
                _request(tmp_path, Condition.CLEAN),
                _request(tmp_path, Condition.FAULTED, fault_path=fault_path),
            ),
            session_factory=harness.session,
            policy_factory=harness.policy,
            post_execution_gate=gate,
        )

    assert observed == [Condition.CLEAN]
    task = harness.tasks[0]
    assert task.reset_count == 1
    assert task.restore_count == 0
    assert task.close_count == 1


def test_paired_execution_requires_clean_first_and_shared_pair_key(
    tmp_path: Path,
) -> None:
    fault_path = _fault(tmp_path)
    clean = _request(tmp_path, Condition.CLEAN)
    faulted = _request(tmp_path, Condition.FAULTED, fault_path=fault_path)
    harness = _PairedHarness()

    with pytest.raises(ValueError, match="start with the clean"):
        execute_paired_live_univtac_runs(
            (faulted, clean),
            session_factory=harness.session,
            policy_factory=harness.policy,
        )
    with pytest.raises(ValueError, match="pair key"):
        execute_paired_live_univtac_runs(
            (clean, replace(faulted, exogenous_seed=21)),
            session_factory=harness.session,
            policy_factory=harness.policy,
        )
    with pytest.raises(ValueError, match="runtime_dir"):
        execute_paired_live_univtac_runs(
            (
                clean,
                replace(faulted, runtime_dir=tmp_path / "different-runtime"),
            ),
            session_factory=harness.session,
            policy_factory=harness.policy,
            require_shared_runtime_dir=True,
        )
    assert harness.tasks == []
