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
        runtime_dir=root / condition.value,
        output_dir=None,
        fault_manifest_path=fault_path,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
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


def test_paired_execution_runs_two_conditions_from_one_canonical_state(
    tmp_path: Path,
) -> None:
    fault_path = _fault(tmp_path)
    harness = _PairedHarness()
    result = execute_paired_live_univtac_runs(
        (
            _request(tmp_path, Condition.CLEAN),
            _request(tmp_path, Condition.FAULTED, fault_path=fault_path),
        ),
        session_factory=harness.session,
        policy_factory=harness.policy,
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
    assert result.to_dict()["simulator_qualification_claimed"] is False
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
        )

    assert captured.value.code == "reset_equivalence_mismatch"
    assert harness.tasks[0].close_count == 1


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
    assert harness.tasks == []
