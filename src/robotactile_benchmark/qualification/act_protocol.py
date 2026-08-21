"""Deterministic model-free qualification of the strict ACT policy boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Tuple

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord, canonical_hash
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.policies.act import StrictACTPolicy
from robotactile_benchmark.policies.act_loading import (
    ArtifactUnavailableError,
    load_matched_no_touch_policy,
)
from robotactile_benchmark.qualification.absence import (
    EffectProbe,
    run_structural_absence_checks,
)


class _RecordingRuntime:
    def __init__(self) -> None:
        self.reset_count = 0
        self.infer_count = 0
        self.inputs: list[Mapping[str, object]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def get_action(self, observation: Mapping[str, object]) -> Array:
        self.infer_count += 1
        self.inputs.append(observation)
        return np.zeros((1, 8), dtype=np.float32)


def _context(task_id: str) -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="synthetic-episode-v1",
        task=task_id,
        initial_seed=1001,
        exogenous_seed=2002,
        instruction="insert HDMI",
        action_spec=ACTION_SPEC,
    )


def _record(step_index: int, task_id: str) -> ObservationRecord:
    observation = make_synthetic_episode()[step_index].observation
    return replace(
        observation,
        task=task_id,
        proprio=np.arange(8, dtype=np.float32),
    )


def _absence_factory(
    identity: PolicyIdentity, task_id: str
) -> Tuple[StrictACTPolicy, EffectProbe]:
    runtime = _RecordingRuntime()
    policy = StrictACTPolicy(identity, runtime, artifact_task=task_id)

    def probe() -> Tuple[int, int]:
        return runtime.reset_count + runtime.infer_count, 0

    return policy, probe


def run_act_protocol(
    identity: PolicyIdentity, task_id: str
) -> tuple[Mapping[str, object], Mapping[str, Any], str]:
    """Exercise ACT routing/lifecycle without importing or running a model."""

    runtime = _RecordingRuntime()
    policy = StrictACTPolicy(identity, runtime, artifact_task=task_id)
    context = _context(task_id)
    policy.reset(context)
    observation = _record(0, task_id)
    plan = policy.infer(observation)
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=(_record(1, task_id),),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    if len(runtime.inputs) != 1:
        raise RuntimeError("ACT protocol did not route exactly one runtime input")
    try:
        load_matched_no_touch_policy(None)
    except ArtifactUnavailableError as error:
        no_touch_status = error.code
    else:
        raise RuntimeError("matched no-touch artifact unexpectedly became available")
    absence = run_structural_absence_checks(
        identity, lambda: _absence_factory(identity, task_id)
    )
    result = {
        "protocol_status": "passed",
        "runtime_reset_count": runtime.reset_count,
        "runtime_infer_count": runtime.infer_count,
        "runtime_commit_model_calls": 0,
        "action_plan_sha256": plan.sha256,
        "routed_input_sha256": canonical_hash(runtime.inputs[0]),
    }
    return result, absence, no_touch_status
