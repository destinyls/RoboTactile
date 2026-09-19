"""Measured zero-effect structural-absence preflight qualification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Tuple, cast

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    ClosedLoopRunSpec,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.fakes import DeterministicFakeBackend
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import (
    Condition,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)

EffectProbe = Callable[[], Tuple[int, int]]
PolicyProbeFactory = Callable[[], Tuple[ClosedLoopPolicy, EffectProbe]]


def _trial(identity: PolicyIdentity, fault: FaultManifest) -> TrialManifest:
    return TrialManifest(
        task="insert_HDMI",
        initial_seed=1001,
        exogenous_seed=2002,
        condition=Condition.FAULTED,
        base_system_id=identity.system_id,
        executed_system_id=identity.system_id,
        dataset_sha256="f" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            identity.system_id,
            identity.checkpoint_sha256,
            identity.config_sha256,
            ACTION_SPEC,
        ),
        checkpoint_sha256=identity.checkpoint_sha256,
        config_sha256=identity.config_sha256,
        action_spec=ACTION_SPEC,
        fault_manifest_sha256=fault.sha256,
        matched_no_touch_system_id=None,
    )


def run_structural_absence_checks(
    identity: PolicyIdentity, factory: PolicyProbeFactory
) -> Mapping[str, Any]:
    """Run both structural operators and retain measured zero side effects."""

    results = {}
    for operator_id in ("A1_stream_absence", "A2_frame_erasure"):
        fault = FaultManifest(
            operator_id=operator_id,
            severity_level=3,
            operator_seed=23,
            start_index=1,
            stop_index=4,
            sensor_slots=("left",),
            observability=Observability.DECLARED,
            parameters={},
        )
        backend = DeterministicFakeBackend(
            make_synthetic_episode(),
            success_predicate_id="policy-qualification-zero-effect-v1",
        )
        policy, probe = factory()
        result = run_closed_loop_trial(
            _trial(identity, fault),
            ClosedLoopRunSpec(
                prompt="insert HDMI",
                success_predicate_id="policy-qualification-zero-effect-v1",
                max_control_cycles=2,
                max_observation_steps=8,
                execute_action_steps=8,
                wall_timeout_s=5.0,
            ),
            backend,
            policy,
            fault_manifest=fault,
        )
        policy_effect_count, transport_effect_count = probe()
        if result.terminal_status is not TerminalStatus.UNSUPPORTED_CONTRACT:
            raise RuntimeError("structural absence did not fail as unsupported")
        if backend.reset_count or policy_effect_count or transport_effect_count:
            raise RuntimeError("structural absence preflight had forbidden effects")
        results[operator_id] = {
            "terminal_status": result.terminal_status.value,
            "backend_reset_count": backend.reset_count,
            "policy_effect_count": policy_effect_count,
            "transport_effect_count": transport_effect_count,
        }
    return cast(Mapping[str, Any], freeze_value(results))
