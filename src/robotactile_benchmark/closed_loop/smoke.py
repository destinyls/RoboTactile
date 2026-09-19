"""Deterministic fake-only closed-loop artifact smoke for CPU qualification."""

from __future__ import annotations

from pathlib import Path

from robotactile_benchmark.closed_loop.artifact_contracts import (
    LoadedClosedLoopBundle,
)
from robotactile_benchmark.closed_loop.artifacts import write_closed_loop_bundle
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition, TrialManifest, system_manifest_hash


def write_cpu_smoke_bundle(output: Path) -> LoadedClosedLoopBundle:
    """Execute and persist one active A1 fault using only deterministic fakes."""

    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=20260821,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    system_id = "deterministic-fake-policy-v1"
    checkpoint_sha256 = "b" * 64
    config_sha256 = "c" * 64
    trial = TrialManifest(
        task="insert_HDMI",
        initial_seed=10,
        exogenous_seed=20,
        condition=Condition.FAULTED,
        base_system_id=system_id,
        executed_system_id=system_id,
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            system_id,
            checkpoint_sha256,
            config_sha256,
            "qpos8_next_step",
        ),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=fault.sha256,
        matched_no_touch_system_id=None,
    )
    run_spec = ClosedLoopRunSpec(
        prompt="insert the cable",
        success_predicate_id="deterministic-fake-success-v1",
        max_control_cycles=5,
        max_observation_steps=6,
        execute_action_steps=1,
        wall_timeout_s=5.0,
    )
    evidence = run_closed_loop_trial_with_evidence(
        trial,
        run_spec,
        DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id=run_spec.success_predicate_id,
        ),
        DeterministicFakePolicy.for_trial(trial),
        fault_manifest=fault,
    )
    return write_closed_loop_bundle(
        output,
        trial=trial,
        run_spec=run_spec,
        fault_manifest=fault,
        evidence=evidence,
    )


def smoke_summary(bundle: LoadedClosedLoopBundle) -> dict[str, object]:
    """Return the bounded, canonical CLI summary without benchmark overclaims."""

    validation = bundle.finalization.validation
    if validation is None:
        raise ValueError("CPU smoke bundle is missing fault validation")
    return {
        "evidence_level": bundle.root_receipt.evidence_level,
        "condition": bundle.trial.condition.value,
        "operator_id": bundle.fault_manifest.operator_id,
        "severity_level": bundle.fault_manifest.severity_level,
        "terminal_status": bundle.result.terminal_status.value,
        "validation_passed": validation.passed,
        "requested_file_count": len(bundle.root_receipt.members) + 1,
        "result_sha256": bundle.result.sha256,
        "root_receipt_sha256": bundle.root_receipt_sha256,
    }
